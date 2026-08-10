from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import numpy as np
import pandas as pd

from config import Settings, load_settings
from data_loader import MarketDataLoader, TransactionLoader
from evaluator import AnomalyEvaluator
from explainer import AnomalyExplainer, ReportGenerator
from feature_engineer import FinancialFeatureEngineer
from models import (
    DBSCANDetector,
    EnsembleDetector,
    IsolationForestDetector,
    LSTMAutoencoderDetector,
    StatisticalDetector,
)
from pipeline import AnomalyDetectionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAMES = (
    "statistical",
    "isolation_forest",
    "lstm_autoencoder",
    "dbscan",
    "ensemble",
)


def model_feature_columns(
    df: pd.DataFrame,
    label_column: Optional[str] = "is_fraud",
) -> list[str]:
    """Return model inputs, including boolean risk flags as 0/1 features."""
    excluded = {
        column
        for column in (label_column, "anomaly_score", "anomaly_label", "anomaly_type")
        if column is not None
    }
    return [
        column
        for column in df.columns
        if column not in excluded
        and (
            pd.api.types.is_numeric_dtype(df[column])
            or pd.api.types.is_bool_dtype(df[column])
        )
    ]


def read_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}. Use CSV or Parquet.")


def load_detector(model_name: str, model_path: Path):
    detector_classes = {
        "statistical": StatisticalDetector,
        "isolation_forest": IsolationForestDetector,
        "lstm_autoencoder": LSTMAutoencoderDetector,
        "dbscan": DBSCANDetector,
        "ensemble": EnsembleDetector,
    }
    try:
        return detector_classes[model_name]().load_model(model_path)
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_name}") from exc


def clickhouse_source(settings: Settings) -> str:
    """Build a SQLAlchemy ClickHouse URL from the configured connection."""
    clickhouse = settings.clickhouse
    credentials = quote(clickhouse.user, safe="")
    if clickhouse.password:
        credentials += f":{quote(clickhouse.password, safe='')}"
    return (
        f"clickhouse+http://{credentials}@{clickhouse.host}:"
        f"{clickhouse.port}/{clickhouse.database}"
    )


def market_anomaly_types(df: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    """Classify detected market events into schema-supported anomaly types."""
    labels = np.asarray(labels, dtype=int)
    anomaly_type = pd.Series(pd.NA, index=df.index, dtype="string")
    return_1 = df.get("return_1", pd.Series(0.0, index=df.index)).astype(float)
    volume_ratio = df.get(
        "volume_anomaly_ratio", pd.Series(1.0, index=df.index)
    ).astype(float)

    alerted = labels == 1
    anomaly_type.loc[alerted & (return_1 <= -0.05)] = "FLASH_CRASH"
    anomaly_type.loc[
        alerted & (return_1 >= 0.05) & (volume_ratio >= 1.5)
    ] = "PUMP_DUMP"
    return anomaly_type


def generate_synthetic_transactions(
    n_transactions: int = 100_000,
    n_users: int = 1_000,
    fraud_rate: float = 0.0005,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates synthetic transaction data with:
    - seasonality
    - correlated merchant category / amount
    - fraud patterns: burst attacks, card testing, account takeover
    """
    rng = np.random.default_rng(seed)

    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    total_seconds = int((end - start).total_seconds())
    timestamps = start + pd.to_timedelta(
        rng.integers(0, total_seconds, size=n_transactions), unit="s"
    )

    user_ids = rng.integers(1, n_users + 1, size=n_transactions)

    merchant_categories = [
        "grocery",
        "restaurant",
        "gas_station",
        "electronics",
        "travel",
        "casino",
        "crypto_exchange",
        "money_transfer",
    ]

    merchant_probs = [0.30, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05]
    merchant_categories = rng.choice(merchant_categories, size=n_transactions, p=merchant_probs)

    # Amount correlated with merchant category
    base_amount = rng.lognormal(mean=3.0, sigma=1.0, size=n_transactions)

    category_multiplier = np.where(
        np.isin(merchant_categories, ["electronics", "travel"]),
        3.0,
        1.0,
    )

    high_risk_multiplier = np.where(
        np.isin(merchant_categories, ["casino", "crypto_exchange", "money_transfer"]),
        5.0,
        1.0,
    )

    amount = base_amount * category_multiplier * high_risk_multiplier

    # Geographic coordinates
    latitude = rng.uniform(25, 48, size=n_transactions)
    longitude = rng.uniform(-125, -70, size=n_transactions)

    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:08d}" for i in range(n_transactions)],
            "user_id": [f"USER{uid:06d}" for uid in user_ids],
            "timestamp": timestamps,
            "amount": amount,
            "currency": "USD",
            "merchant_id": [f"MERCH{rng.integers(1, 10_000):06d}" for _ in range(n_transactions)],
            "merchant_category": merchant_categories,
            "channel": rng.choice(["card", "ach", "wire"], size=n_transactions),
            "latitude": latitude,
            "longitude": longitude,
            "is_fraud": 0,
        }
    )

    # Inject anomalies
    n_fraud = max(1, int(n_transactions * fraud_rate))

    fraud_idx = rng.choice(n_transactions, size=n_fraud, replace=False)
    df.loc[fraud_idx, "is_fraud"] = 1

    # Account takeover pattern: large amount + high-risk category
    df.loc[fraud_idx, "amount"] *= rng.uniform(5, 20, size=len(fraud_idx))
    df.loc[fraud_idx, "merchant_category"] = rng.choice(
        ["casino", "crypto_exchange", "money_transfer"],
        size=len(fraud_idx),
    )

    # Burst attack pattern: repeated small tests then large transaction
    burst_user = f"USER{rng.integers(1, n_users + 1):06d}"
    burst_times = start + pd.to_timedelta(np.arange(20), unit="h")
    burst_df = pd.DataFrame(
        {
            "transaction_id": [f"BURST{i:08d}" for i in range(20)],
            "user_id": burst_user,
            "timestamp": burst_times,
            "amount": np.concatenate([np.full(19, 1.0), [5000.0]]),
            "currency": "USD",
            "merchant_id": [f"MERCH{rng.integers(1, 10_000):06d}" for _ in range(20)],
            "merchant_category": "money_transfer",
            "channel": "card",
            "latitude": rng.uniform(25, 48, size=20),
            "longitude": rng.uniform(-125, -70, size=20),
            "is_fraud": [0] * 19 + [1],
        }
    )

    df = pd.concat([df, burst_df], ignore_index=True)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    return df


def cmd_preprocess(args) -> None:
    settings = load_settings(args.config)

    if args.source and Path(args.source).exists():
        loader = TransactionLoader()
        df = loader.load(args.source)
    else:
        logger.info("No valid source found. Generating synthetic transactions.")
        df = generate_synthetic_transactions(
            n_transactions=args.n_rows,
            fraud_rate=0.0005,
        )

    report = TransactionLoader().validate(df)
    logger.info(f"Data quality report: {report.to_dict()}")

    engineer = FinancialFeatureEngineer(
        windows=settings.features.windows,
        high_risk_categories=settings.features.high_risk_merchant_categories,
    )

    features = engineer.build_transaction_features(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    logger.info(f"Saved features to {output_path}")


def cmd_preprocess_market(args) -> None:
    """Load market ticks/bars, including ClickHouse sources, and build features."""
    settings = load_settings(args.config)
    loader = MarketDataLoader()
    source = args.source or clickhouse_source(settings)
    raw_market_data = loader.load(source, query=args.query)

    report = loader.validate(raw_market_data)
    if report.issues:
        logger.warning("Market data quality report: %s", report.to_dict())

    if "close" in raw_market_data.columns:
        market_bars = raw_market_data
    elif "price" in raw_market_data.columns:
        market_bars = loader.resample_tick_data(raw_market_data, freq=args.bar_frequency)
    else:
        raise ValueError(
            "Market source must contain OHLC bars with a close column or ticks with a price column."
        )

    pipeline = AnomalyDetectionPipeline(settings=settings)
    features = pipeline.run_market_feature_engineering(
        market_bars=market_bars,
        feature_set_name=args.feature_set_name,
        force_recompute=args.force_recompute,
        benchmark_symbol=args.benchmark_symbol,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    logger.info("Saved market features to %s", output_path)


def cmd_train(args) -> None:
    settings = load_settings(args.config)

    if not Path(args.features).exists():
        raise FileNotFoundError(
            f"Feature file not found: {args.features}. Run preprocess first."
        )

    features = read_dataframe(args.features)
    label_column = args.label_column
    if label_column not in features.columns:
        logger.warning(
            "Label column %r is absent; training unsupervised models without labels.",
            label_column,
        )

    feature_cols = model_feature_columns(features, label_column=label_column)
    if not feature_cols:
        raise ValueError("No numeric or boolean feature columns found for training")

    loader = TransactionLoader()
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    val_end = pd.Timestamp(args.val_end, tz="UTC") if args.val_end else None

    train_df, val_df, test_df = loader.time_based_split(
        df=features,
        train_end=train_end,
        val_end=val_end,
    )

    X_train = train_df[feature_cols]
    y_train = train_df[label_column] if label_column in train_df.columns else None

    X_val = val_df[feature_cols] if len(val_df) > 0 else None
    y_val = (
        val_df[label_column]
        if len(val_df) > 0 and label_column in val_df.columns
        else None
    )

    pipeline = AnomalyDetectionPipeline(settings=settings)

    model_names = None if args.model == "all" else [args.model]

    pipeline.train_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_names=model_names,
    )

    # Calibration belongs to training, not to the optional test-evaluation
    # branch.  A normal train/validation split now persists calibrated models.
    if X_val is not None and y_val is not None:
        calibrated = pipeline.calibrate_thresholds(X_val=X_val, y_val=y_val)
        logger.info(f"Calibrated thresholds: {calibrated}")

    if len(test_df) > 0:
        X_test = test_df[feature_cols]
        if label_column in test_df.columns:
            metrics = pipeline.evaluate_models(
                X_test=X_test,
                y_test=test_df[label_column],
            )

            for name, m in metrics.items():
                logger.info(f"{name}: {m.to_dict()}")


def cmd_detect(args) -> None:
    settings = load_settings(args.config)

    df = read_dataframe(args.input)

    feature_cols = model_feature_columns(df, label_column=args.label_column)

    X = df[feature_cols]

    model_path = Path(settings.artifact_dir) / args.model
    model = load_detector(args.model, model_path)

    if args.threshold != "auto":
        model.threshold = float(args.threshold)

    scores, labels = model.predict(X)

    df["anomaly_score"] = scores
    df["anomaly_label"] = labels
    if args.domain == "market":
        df["anomaly_type"] = market_anomaly_types(df, labels)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Wrote detection results to {output_path}")


def cmd_explain(args) -> None:
    settings = load_settings(args.config)

    df = read_dataframe(args.features)

    if "transaction_id" not in df.columns:
        raise ValueError("Feature file must contain transaction_id")

    sample_df = df.loc[df["transaction_id"] == args.sample_id]

    if sample_df.empty:
        raise ValueError(f"Sample ID {args.sample_id} not found")

    feature_cols = model_feature_columns(sample_df, label_column=args.label_column)

    sample = sample_df[feature_cols]

    model_path = Path(settings.artifact_dir) / args.model
    model = load_detector(args.model, model_path)

    explanation = model.get_anomaly_explanation(sample)

    explainer = AnomalyExplainer()
    explanation["natural_language_summary"] = explainer.natural_language_summary(explanation)

    report_generator = ReportGenerator(output_dir=settings.report_dir)
    json_path = report_generator.generate_json(
        report_id=f"explain_{args.sample_id}",
        payload=explanation,
    )

    logger.info(f"Explanation JSON written to {json_path}")
    print(json.dumps(explanation, indent=2, default=str))


def cmd_backtest(args) -> None:
    settings = load_settings(args.config)

    features = read_dataframe(args.features)

    feature_cols = model_feature_columns(features, label_column=args.label_column)

    start = pd.Timestamp(args.start_date, tz="UTC")
    end = pd.Timestamp(args.end_date, tz="UTC")

    backtest_df = features[
        (features["timestamp"] >= start) & (features["timestamp"] <= end)
    ].copy()

    if backtest_df.empty:
        raise ValueError("No data found in backtest date range")

    known_frauds = pd.read_csv(args.event_file)

    model_path = Path(settings.artifact_dir) / args.model
    model = load_detector(args.model, model_path)

    X = backtest_df[feature_cols]
    scores, labels = model.predict(X)

    backtest_df["anomaly_score"] = scores
    backtest_df["anomaly_label"] = labels

    evaluator = AnomalyEvaluator(top_k=settings.calibration.max_alerts_per_day)

    if args.label_column in backtest_df.columns:
        metrics = evaluator.compute_metrics(
            model_name=args.model,
            y_true=backtest_df[args.label_column].to_numpy(),
            y_pred_scores=scores,
            y_pred_labels=labels,
        )

        logger.info(f"Backtest metrics: {metrics.to_dict()}")

    detected_ids = backtest_df.loc[
        backtest_df["anomaly_label"] == 1, "transaction_id"
    ].tolist()

    true_ids = known_frauds["transaction_id"].tolist() if "transaction_id" in known_frauds.columns else []

    matched = len(set(detected_ids) & set(true_ids))

    logger.info(
        f"Backtest matched {matched}/{len(true_ids)} known frauds "
        f"with {len(detected_ids)} alerts"
    )


def simulate_live_mode(
    input_path: str,
    model_name: str = "isolation_forest",
    config_path: Optional[str] = None,
    sleep_seconds: float = 0.1,
) -> None:
    """
    Streams rows and detects anomalies in near-real-time.
    """
    settings = load_settings(config_path)
    model_path = Path(settings.artifact_dir) / model_name

    model = load_detector(model_name, model_path)

    df = read_dataframe(input_path)

    feature_cols = model_feature_columns(df)

    logger.info("Starting simulated live mode...")

    for idx, row in df.iterrows():
        sample = row[feature_cols].to_frame().T

        score, label = model.predict(sample)

        if label[0] == 1:
            logger.warning(
                f"ALERT at row {idx}: score={score[0]:.4f}, id={row.get('transaction_id', 'N/A')}"
            )

        time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Financial Anomaly Detection CLI"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # preprocess
    p_preprocess = subparsers.add_parser("preprocess", help="Preprocess transactions")
    p_preprocess.add_argument("--source", type=str, default=None)
    p_preprocess.add_argument("--output", type=str, default="data/features.parquet")
    p_preprocess.add_argument("--config", type=str, default=None)
    p_preprocess.add_argument("--n-rows", type=int, default=100_000)
    p_preprocess.set_defaults(func=cmd_preprocess)

    # preprocess-market
    p_preprocess_market = subparsers.add_parser(
        "preprocess-market",
        help="Load market ticks/bars and build market anomaly features",
    )
    p_preprocess_market.add_argument(
        "--source",
        type=str,
        default=None,
        help="CSV/Parquet/SQLAlchemy source. Defaults to configured ClickHouse.",
    )
    p_preprocess_market.add_argument(
        "--query",
        type=str,
        default="SELECT * FROM market_ticks",
        help="SQL query used for a database source.",
    )
    p_preprocess_market.add_argument("--config", type=str, default=None)
    p_preprocess_market.add_argument("--output", type=str, default="data/market_features.parquet")
    p_preprocess_market.add_argument("--feature-set-name", type=str, default="market_features")
    p_preprocess_market.add_argument("--bar-frequency", type=str, default="1min")
    p_preprocess_market.add_argument("--benchmark-symbol", type=str, default="SPY")
    p_preprocess_market.add_argument("--force-recompute", action="store_true")
    p_preprocess_market.set_defaults(func=cmd_preprocess_market)

    # train
    p_train = subparsers.add_parser("train", help="Train models")
    p_train.add_argument("--config", type=str, default=None)
    p_train.add_argument("--features", type=str, default="data/features.parquet")
    p_train.add_argument("--model", type=str, choices=("all", *MODEL_NAMES), default="all")
    p_train.add_argument("--label-column", type=str, default="is_fraud")
    p_train.add_argument("--train-end", type=str, required=True)
    p_train.add_argument("--val-end", type=str, default=None)
    p_train.set_defaults(func=cmd_train)

    # detect
    p_detect = subparsers.add_parser("detect", help="Run detection on new data")
    p_detect.add_argument("--config", type=str, default=None)
    p_detect.add_argument("--input", type=str, required=True)
    p_detect.add_argument("--model", type=str, choices=MODEL_NAMES, default="ensemble")
    p_detect.add_argument("--label-column", type=str, default="is_fraud")
    p_detect.add_argument(
        "--domain",
        choices=("transaction", "market"),
        default="transaction",
        help="Market detection emits FLASH_CRASH and PUMP_DUMP event types.",
    )
    p_detect.add_argument("--threshold", type=str, default="auto")
    p_detect.add_argument("--output", type=str, default="data/detection_results.csv")
    p_detect.set_defaults(func=cmd_detect)

    # explain
    p_explain = subparsers.add_parser("explain", help="Explain a sample")
    p_explain.add_argument("--config", type=str, default=None)
    p_explain.add_argument("--sample-id", type=str, required=True)
    p_explain.add_argument("--model", type=str, choices=MODEL_NAMES, default="isolation_forest")
    p_explain.add_argument("--features", type=str, default="data/features.parquet")
    p_explain.add_argument("--label-column", type=str, default="is_fraud")
    p_explain.set_defaults(func=cmd_explain)

    # backtest
    p_backtest = subparsers.add_parser("backtest", help="Backtest over date range")
    p_backtest.add_argument("--config", type=str, default=None)
    p_backtest.add_argument("--features", type=str, default="data/features.parquet")
    p_backtest.add_argument("--start-date", type=str, required=True)
    p_backtest.add_argument("--end-date", type=str, required=True)
    p_backtest.add_argument("--event-file", type=str, required=True)
    p_backtest.add_argument("--model", type=str, choices=MODEL_NAMES, default="ensemble")
    p_backtest.add_argument("--label-column", type=str, default="is_fraud")
    p_backtest.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
