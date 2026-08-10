from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

from config import Settings, load_settings
from data_loader import MarketDataLoader, TransactionLoader
from evaluator import AnomalyEvaluator
from explainer import AnomalyExplainer, ReportGenerator
from feature_engineer import FeatureStore, FinancialFeatureEngineer
from models import (
    BaseAnomalyDetector,
    DBSCANDetector,
    EnsembleDetector,
    IsolationForestDetector,
    LSTMAutoencoderDetector,
    StatisticalDetector,
)
from schemas import AnomalyResult, ModelMetrics

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Drift detection using:
    - PSI for feature distributions
    - KL divergence for score distributions
    """

    def __init__(
        self,
        psi_threshold: float = 0.25,
        kl_threshold: float = 0.2,
        bins: int = 10,
    ):
        self.psi_threshold = psi_threshold
        self.kl_threshold = kl_threshold
        self.bins = bins

    @staticmethod
    def _normalize(counts: pd.Series, epsilon: float = 1e-6) -> pd.Series:
        total = counts.sum() + epsilon
        return (counts + epsilon) / total

    def psi_numeric(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        reference = pd.Series(reference).dropna()
        current = pd.Series(current).dropna()

        if len(reference) == 0 or len(current) == 0:
            return 0.0

        bins = pd.qcut(reference, q=self.bins, duplicates="drop")
        bin_edges = pd.unique(bins.cat.categories)

        ref_counts = pd.cut(reference, bins=bin_edges).value_counts()
        cur_counts = pd.cut(current, bins=bin_edges).value_counts()

        ref_pct = self._normalize(ref_counts)
        cur_pct = self._normalize(cur_counts)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def psi_categorical(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> float:
        reference = reference.fillna("__MISSING__").astype(str)
        current = current.fillna("__MISSING__").astype(str)

        ref_counts = reference.value_counts()
        cur_counts = current.value_counts()

        all_categories = ref_counts.index.union(cur_counts.index)

        ref_counts = ref_counts.reindex(all_categories, fill_value=0)
        cur_counts = cur_counts.reindex(all_categories, fill_value=0)

        ref_pct = self._normalize(ref_counts)
        cur_pct = self._normalize(cur_counts)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def kl_divergence_scores(
        self,
        reference_scores: np.ndarray,
        current_scores: np.ndarray,
    ) -> float:
        reference_scores = np.asarray(reference_scores, dtype=float)
        current_scores = np.asarray(current_scores, dtype=float)

        if len(reference_scores) == 0 or len(current_scores) == 0:
            return 0.0

        bins = np.linspace(
            min(reference_scores.min(), current_scores.min()),
            max(reference_scores.max(), current_scores.max()),
            self.bins + 1,
        )

        p, _ = np.histogram(reference_scores, bins=bins, density=True)
        q, _ = np.histogram(current_scores, bins=bins, density=True)

        p = p + 1e-6
        q = q + 1e-6

        p = p / p.sum()
        q = q / q.sum()

        return float(np.sum(p * np.log(p / q)))

    def detect_feature_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "psi": {},
            "trigger": False,
        }

        common_cols = list(set(reference.columns) & set(current.columns))

        for col in common_cols:
            if pd.api.types.is_numeric_dtype(reference[col]):
                psi = self.psi_numeric(reference[col].to_numpy(), current[col].to_numpy())
            else:
                psi = self.psi_categorical(reference[col], current[col])

            report["psi"][col] = psi

        report["trigger"] = any(v > self.psi_threshold for v in report["psi"].values())

        return report


def _fit_one_model(
    name: str,
    detector: BaseAnomalyDetector,
    X_train: Any,
    y_train: Optional[np.ndarray],
    X_val: Optional[Any],
    y_val: Optional[np.ndarray],
) -> tuple[str, Optional[BaseAnomalyDetector], Optional[str]]:
    try:
        detector.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
        return name, detector, None
    except Exception as exc:
        logger.warning(f"Failed to fit {name}: {exc}")
        return name, None, str(exc)


class AnomalyDetectionPipeline:
    """
    Orchestrates:
    - data loading
    - feature engineering
    - temporal splitting
    - model training
    - evaluation
    - threshold calibration
    - alert generation
    - drift-based retraining trigger
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or load_settings()

        self.transaction_loader = TransactionLoader()
        self.market_loader = MarketDataLoader()

        self.feature_engineer = FinancialFeatureEngineer(
            windows=self.settings.features.windows,
            high_risk_categories=self.settings.features.high_risk_merchant_categories,
            impossible_travel_speed_kmh=self.settings.features.impossible_travel_speed_kmh,
            night_start_hour=self.settings.features.night_start_hour,
            night_end_hour=self.settings.features.night_end_hour,
        )

        self.feature_store = FeatureStore(
            root=self.settings.feature_store_dir,
            version="v1",
        )

        self.evaluator = AnomalyEvaluator(top_k=self.settings.calibration.max_alerts_per_day)
        self.explainer = AnomalyExplainer()
        self.report_generator = ReportGenerator(output_dir=self.settings.report_dir)
        self.drift_detector = DriftDetector(psi_threshold=0.25)

        self.models: Dict[str, BaseAnomalyDetector] = {}
        self.metrics: Dict[str, ModelMetrics] = {}

        self.artifact_dir = Path(self.settings.artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def initialize_data_loaders(self) -> None:
        self.transaction_loader = TransactionLoader()
        self.market_loader = MarketDataLoader()

    def run_feature_engineering(
        self,
        transactions: Optional[pd.DataFrame],
        market_bars: Optional[pd.DataFrame] = None,
        feature_set_name: str = "transaction_features",
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        market_features: Optional[pd.DataFrame] = None
        if market_bars is not None:
            market_features = self.run_market_feature_engineering(
                market_bars=market_bars,
                feature_set_name=f"{feature_set_name}_market",
                force_recompute=force_recompute,
            )

        if transactions is None:
            if market_features is None:
                raise ValueError("Provide transactions, market_bars, or both")
            return market_features

        if not force_recompute:
            cached = self.feature_store.load_latest(feature_set_name)
            if cached is not None:
                logger.info(f"Loaded cached feature set: {feature_set_name}")
                return cached

        features = self.feature_engineer.build_transaction_features(transactions)

        self.feature_store.save(feature_set_name, features)
        return features

    def run_market_feature_engineering(
        self,
        market_bars: pd.DataFrame,
        feature_set_name: str = "market_features",
        force_recompute: bool = False,
        benchmark_symbol: str = "SPY",
    ) -> pd.DataFrame:
        """Build and persist market features as a first-class feature set.

        Transaction and market records represent different entities and should
        not be blindly joined.  They are therefore cached separately and can
        be trained/scored by the same detector CLI with their own feature file.
        """
        if not force_recompute:
            cached = self.feature_store.load_latest(feature_set_name)
            if cached is not None:
                logger.info(f"Loaded cached market feature set: {feature_set_name}")
                return cached

        features = self.feature_engineer.build_market_features(market_bars)
        features = self.feature_engineer.build_cross_asset_features(
            features,
            benchmark_symbol=benchmark_symbol,
        )
        features = self.market_loader.detect_price_gaps(features)
        self.feature_store.save(feature_set_name, features)
        return features

    def train_test_split(
        self,
        df: pd.DataFrame,
        train_end: pd.Timestamp,
        val_end: Optional[pd.Timestamp] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.transaction_loader.time_based_split(
            df=df,
            train_end=train_end,
            val_end=val_end,
        )

    def _build_models(self) -> Dict[str, BaseAnomalyDetector]:
        models: Dict[str, BaseAnomalyDetector] = {}

        if self.settings.training.use_statistical:
            models["statistical"] = StatisticalDetector(method="modified_zscore")

        if self.settings.training.use_isolation_forest:
            models["isolation_forest"] = IsolationForestDetector(
                n_estimators=self.settings.iforest.n_estimators,
                contamination=self.settings.iforest.contamination,
                max_samples=self.settings.iforest.max_samples,
                max_features=self.settings.iforest.max_features,
                bootstrap=self.settings.iforest.bootstrap,
                random_state=self.settings.iforest.random_state,
            )

        if self.settings.training.use_lstm:
            try:
                models["lstm_autoencoder"] = LSTMAutoencoderDetector(
                    sequence_length=self.settings.lstm.sequence_length,
                    hidden_units=self.settings.lstm.hidden_units,
                    epochs=self.settings.lstm.epochs,
                    batch_size=self.settings.lstm.batch_size,
                    learning_rate=self.settings.lstm.learning_rate,
                    reconstruction_error_percentile=self.settings.lstm.reconstruction_error_percentile,
                )
            except ImportError as exc:
                logger.warning("LSTM autoencoder is unavailable: %s", exc)

        if self.settings.training.use_dbscan:
            models["dbscan"] = DBSCANDetector(
                eps_km=self.settings.dbscan.eps_km,
                min_samples=self.settings.dbscan.min_samples,
            )

        if self.settings.training.use_ensemble:
            base_detectors = [
                det
                for name, det in models.items()
                if name not in {"lstm_autoencoder", "dbscan"}
            ]

            if base_detectors:
                models["ensemble"] = EnsembleDetector(detectors=base_detectors)

        return models

    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, BaseAnomalyDetector]:
        candidate_models = self._build_models()
        supported_models = {
            "statistical",
            "isolation_forest",
            "lstm_autoencoder",
            "dbscan",
            "ensemble",
        }

        if model_names is not None:
            unknown_models = set(model_names) - supported_models
            if unknown_models:
                raise ValueError(
                    f"Unsupported model(s): {sorted(unknown_models)}. "
                    f"Supported models: {sorted(supported_models)}"
                )

            unavailable_models = set(model_names) - set(candidate_models)
            if unavailable_models:
                raise ValueError(
                    f"Requested model(s) are disabled or unavailable: {sorted(unavailable_models)}. "
                    "Check training settings and optional dependencies."
                )

            candidate_models = {
                k: v for k, v in candidate_models.items() if k in model_names
            }

        if not candidate_models:
            raise ValueError("No models are enabled for training")

        def model_input(name: str, X: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if X is None or name != "dbscan":
                return X

            required = ["latitude", "longitude"]
            missing = [column for column in required if column not in X.columns]
            if missing:
                raise ValueError(
                    "DBSCAN requires geographic features latitude and longitude; "
                    f"missing {missing}."
                )
            return X.loc[:, required]

        training_jobs = [
            (
                name,
                detector,
                model_input(name, X_train),
                model_input(name, X_val),
            )
            for name, detector in candidate_models.items()
        ]

        results = Parallel(n_jobs=self.settings.training.n_jobs, backend="threading")(
            delayed(_fit_one_model)(
                name=name,
                detector=detector,
                X_train=model_X_train,
                y_train=y_train,
                X_val=model_X_val,
                y_val=y_val,
            )
            for name, detector, model_X_train, model_X_val in training_jobs
        )

        fitted_models: Dict[str, BaseAnomalyDetector] = {}

        for name, detector, error in results:
            if detector is not None:
                fitted_models[name] = detector
                model_path = self.artifact_dir / name
                detector.save_model(model_path)
                logger.info(f"Saved model {name} to {model_path}")
            else:
                logger.warning(f"Model {name} failed training: {error}")

        self.models = fitted_models
        return fitted_models

    def evaluate_models(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, ModelMetrics]:
        results: Dict[str, ModelMetrics] = {}

        for name, model in self.models.items():
            scores, labels = model.predict(X_test)

            metrics = self.evaluator.compute_metrics(
                model_name=name,
                y_true=y_test.to_numpy(),
                y_pred_scores=scores,
                y_pred_labels=labels,
                k=self.settings.calibration.max_alerts_per_day,
            )

            results[name] = metrics

        self.metrics = results
        return results

    def calibrate_thresholds(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        objective: Optional[str] = None,
    ) -> Dict[str, float]:
        objective = objective or self.settings.calibration.objective
        grid = self.settings.calibration.search_grid

        calibrated: Dict[str, float] = {}

        y_val_arr = y_val.to_numpy().astype(int)

        for name, model in self.models.items():
            scores, _ = model.predict(X_val)

            finite_scores = np.asarray(scores, dtype=float)
            finite_scores = finite_scores[np.isfinite(finite_scores)]
            if len(finite_scores) == 0:
                logger.warning("Skipping threshold calibration for %s: no finite scores", name)
                continue

            if not np.any(y_val_arr == 1):
                model.threshold = float(np.quantile(finite_scores, 0.995))
                calibrated[name] = model.threshold
                model.save_model(self.artifact_dir / name)
                continue

            # Configured values are retained where they make sense, but every
            # detector is also calibrated on its own score scale.  This avoids
            # applying probability thresholds such as 0.5 to raw IF/DBSCAN
            # scores.
            quantile_grid = np.quantile(finite_scores, np.linspace(0.0, 1.0, 201))
            candidate_thresholds = np.unique(
                np.concatenate(
                    [
                        np.asarray(grid, dtype=float),
                        np.asarray(quantile_grid, dtype=float),
                        np.asarray([model.threshold] if model.threshold is not None else []),
                    ]
                )
            )

            best_threshold = float(model.threshold) if model.threshold is not None else 0.5
            best_value = -np.inf

            for threshold in candidate_thresholds:
                preds = (scores >= threshold).astype(int)

                if objective == "f1":
                    value = f1_score(y_val_arr, preds, zero_division=0)
                elif objective == "precision_at_k":
                    value = self.evaluator.precision_at_top_k(
                        y_val_arr,
                        scores,
                        k=self.settings.calibration.max_alerts_per_day,
                    )
                else:
                    value = f1_score(y_val_arr, preds, zero_division=0)

                if value > best_value:
                    best_value = value
                    best_threshold = threshold

            model.threshold = float(best_threshold)
            calibrated[name] = float(best_threshold)

            # persist updated threshold
            model.save_model(self.artifact_dir / name)

        return calibrated

    def generate_alerts(
        self,
        X: pd.DataFrame,
        model_name: str = "ensemble",
        id_col: str = "transaction_id",
        timestamp_col: str = "timestamp",
        entity_col: str = "user_id",
        output_path: Optional[str | Path] = None,
        webhook_url: Optional[str] = None,
    ) -> List[AnomalyResult]:
        if model_name not in self.models:
            raise KeyError(f"Model {model_name} not found")

        model = self.models[model_name]
        scores, labels = model.predict(X)

        alerts: List[AnomalyResult] = []

        for i in range(len(X)):
            if labels[i] == 0:
                continue

            row = X.iloc[i]

            sample = X.iloc[[i]]
            explanation = model.get_anomaly_explanation(sample)

            alert = AnomalyResult(
                id=str(row.get(id_col, f"row_{i}")),
                detector_name=model_name,
                entity_id=str(row.get(entity_col, "unknown")),
                timestamp=pd.Timestamp(row.get(timestamp_col, pd.Timestamp.utcnow())),
                score=float(scores[i]),
                is_anomaly=True,
                anomaly_type=None,
                explanations=explanation,
            )

            alerts.append(alert)

        payload = [a.to_dict() for a in alerts]

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        if webhook_url:
            try:
                import requests

                requests.post(webhook_url, json=payload, timeout=5)
            except Exception as exc:
                logger.warning(f"Webhook failed: {exc}")

        return alerts

    def schedule_retraining(
        self,
        reference_features: pd.DataFrame,
        current_features: pd.DataFrame,
        reference_scores: Optional[np.ndarray] = None,
        current_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        drift_report = self.drift_detector.detect_feature_drift(
            reference=reference_features,
            current=current_features,
        )

        score_drift = None

        if reference_scores is not None and current_scores is not None:
            score_drift = self.drift_detector.kl_divergence_scores(
                reference_scores=reference_scores,
                current_scores=current_scores,
            )

        retrain_needed = bool(drift_report.get("trigger", False))

        if score_drift is not None and score_drift > self.drift_detector.kl_threshold:
            retrain_needed = True

        return {
            "feature_drift": drift_report,
            "score_drift": score_drift,
            "retrain_needed": retrain_needed,
        }
