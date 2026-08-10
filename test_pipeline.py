from __future__ import annotations

import numpy as np
import pandas as pd

from config import Settings
from data_loader import TransactionLoader
from pipeline import AnomalyDetectionPipeline, DriftDetector


def make_features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [f"T{i}" for i in range(10)],
            "user_id": [f"U{i % 3}" for i in range(10)],
            "timestamp": pd.date_range(
                "2023-01-01", periods=10, freq="h", tz="UTC"
            ),
            "amount": np.arange(10, dtype=float),
            "is_fraud": [0] * 9 + [1],
        }
    )


def test_temporal_split_no_future_leakage():
    df = make_features_df()

    loader = TransactionLoader()

    train_end = pd.Timestamp("2023-01-01 05:00:00", tz="UTC")
    val_end = pd.Timestamp("2023-01-01 08:00:00", tz="UTC")

    train, val, test = loader.time_based_split(
        df=df,
        train_end=train_end,
        val_end=val_end,
    )

    assert train["timestamp"].max() < train_end
    assert val["timestamp"].min() >= train_end
    assert val["timestamp"].max() < val_end
    assert test["timestamp"].min() >= val_end


def test_drift_detector_triggers_on_distribution_shift():
    rng = np.random.default_rng(42)

    reference = pd.DataFrame(
        {
            "amount": rng.normal(100, 10, size=1000),
            "merchant_category": rng.choice(["grocery", "gas"], size=1000),
        }
    )

    current = pd.DataFrame(
        {
            "amount": rng.normal(200, 30, size=1000),
            "merchant_category": rng.choice(["casino", "crypto"], size=1000),
        }
    )

    drift_detector = DriftDetector(psi_threshold=0.25)

    report = drift_detector.detect_feature_drift(reference, current)

    assert report["trigger"] is True


def test_drift_detector_stable_distribution():
    rng = np.random.default_rng(42)

    reference = pd.DataFrame(
        {
            "amount": rng.normal(100, 10, size=1000),
        }
    )

    current = pd.DataFrame(
        {
            "amount": rng.normal(100, 10, size=1000),
        }
    )

    drift_detector = DriftDetector(psi_threshold=0.25)

    report = drift_detector.detect_feature_drift(reference, current)

    assert report["trigger"] is False


def test_pipeline_trains_and_saves_dbscan(tmp_path):
    settings = Settings(
        artifact_dir=tmp_path / "artifacts",
        feature_store_dir=tmp_path / "feature_store",
        report_dir=tmp_path / "reports",
        training={
            "use_lstm": False,
            "use_dbscan": True,
            "use_isolation_forest": False,
            "use_statistical": False,
            "use_ensemble": False,
        },
        dbscan={"eps_km": 10.0, "min_samples": 2},
    )
    X = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 30.0, 40.0],
            "latitude": [40.0, 40.001, 40.002, 41.0],
            "longitude": [-74.0, -74.001, -74.002, -75.0],
            "high_risk_merchant_flag": [False, False, True, True],
        }
    )

    pipeline = AnomalyDetectionPipeline(settings=settings)
    models = pipeline.train_models(X_train=X, model_names=["dbscan"])

    assert set(models) == {"dbscan"}
    assert models["dbscan"].feature_names == ["latitude", "longitude"]
    assert (tmp_path / "artifacts" / "dbscan" / "state.joblib").exists()
    scores, labels = models["dbscan"].predict(X)
    assert len(scores) == len(X)
    assert len(labels) == len(X)


def test_market_feature_engineering_is_persisted(tmp_path):
    timestamps = pd.date_range("2023-01-01", periods=24, freq="h", tz="UTC")
    bars = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "timestamp": timestamps,
                    "open": np.linspace(100, 110, len(timestamps)),
                    "high": np.linspace(101, 111, len(timestamps)),
                    "low": np.linspace(99, 109, len(timestamps)),
                    "close": np.linspace(100, 110, len(timestamps)),
                    "volume": np.linspace(1_000, 2_000, len(timestamps)),
                }
            )
            for symbol in ("SPY", "ACME")
        ],
        ignore_index=True,
    )
    settings = Settings(
        artifact_dir=tmp_path / "artifacts",
        feature_store_dir=tmp_path / "feature_store",
        report_dir=tmp_path / "reports",
        training={"use_lstm": False},
    )
    pipeline = AnomalyDetectionPipeline(settings=settings)

    features = pipeline.run_market_feature_engineering(
        bars,
        feature_set_name="market_test",
        force_recompute=True,
    )

    assert {"return_1", "volume_anomaly_ratio", "price_gap_flag"}.issubset(features.columns)
    assert pipeline.feature_store.load_latest("market_test") is not None
