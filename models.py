from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
except ImportError:
    tf = None


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return 6371.0 * c


class BaseAnomalyDetector(ABC):
    """
    Base class for all anomaly detectors.

    predict() returns:
    - anomaly scores, higher means more anomalous
    - binary labels, 1 = anomaly, 0 = normal
    """

    name: str = "base"

    def __init__(
        self,
        threshold: Optional[float] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.threshold = threshold
        self.feature_names = feature_names
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X_train: Any,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[Any] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseAnomalyDetector":
        raise NotImplementedError

    @abstractmethod
    def _score(self, X: Any) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X_test: Any) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(self._score(X_test), dtype=float)

        threshold = self.threshold if self.threshold is not None else 0.5
        labels = (scores >= threshold).astype(int)

        return scores, labels

    def _as_numeric_matrix(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            numeric_columns = [
                column
                for column in X.columns
                if pd.api.types.is_numeric_dtype(X[column])
                or pd.api.types.is_bool_dtype(X[column])
            ]

            if self.feature_names is None:
                self.feature_names = numeric_columns

            missing = [column for column in self.feature_names if column not in X.columns]
            if missing:
                raise ValueError(
                    f"Missing required model features for {self.name}: {missing}. "
                    "Use the feature set that was used at training time."
                )

            # Reindexing is intentional: prediction must use precisely the
            # feature names and order captured during fit.  Extra columns (such
            # as identifiers, labels, or newly added features) are ignored.
            return X.reindex(columns=self.feature_names).to_numpy(dtype=float)

        matrix = np.asarray(X, dtype=float)
        if self.feature_names is not None:
            n_columns = matrix.shape[0] if matrix.ndim == 1 else matrix.shape[-1]
            if n_columns != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} model features for {self.name}, "
                    f"received {n_columns}."
                )
        return matrix

    def save_model(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        state = self._get_state()
        joblib.dump(state, path / "state.joblib")

        meta = {
            "name": self.name,
            "threshold": self.threshold,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }

        (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def load_model(self, path: str | Path) -> "BaseAnomalyDetector":
        path = Path(path)

        state = joblib.load(path / "state.joblib")
        self._set_state(state)

        meta_path = path / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.threshold = meta.get("threshold", self.threshold)
            self.feature_names = meta.get("feature_names", self.feature_names)
            self.is_fitted = meta.get("is_fitted", True)
        else:
            self.is_fitted = True

        return self

    def _get_state(self) -> Dict[str, Any]:
        return {}

    def _set_state(self, state: Dict[str, Any]) -> None:
        pass

    def get_anomaly_explanation(self, sample: Any) -> Dict[str, Any]:
        """
        Default explanation: score plus basic shape information.
        Subclasses should override with richer explanations.
        """
        score = float(np.asarray(self._score(sample)).ravel()[0])

        return {
            "detector": self.name,
            "score": score,
            "threshold": self.threshold,
            "is_anomaly": bool(score >= (self.threshold or 0.5)),
        }


class StatisticalDetector(BaseAnomalyDetector):
    """
    Statistical anomaly detection with:
    - z-score
    - modified z-score using MAD
    - optional Grubbs test for batch univariate outliers
    """

    name = "statistical"

    def __init__(
        self,
        method: str = "modified_zscore",
        threshold: Optional[float] = None,
        rolling_window: Optional[str] = None,
    ):
        super().__init__(threshold=threshold)
        self.method = method
        self.rolling_window = rolling_window

        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.median_: Optional[np.ndarray] = None
        self.mad_: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: Any,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[Any] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "StatisticalDetector":
        X = self._as_numeric_matrix(X_train)

        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0) + 1e-9

        self.median_ = np.nanmedian(X, axis=0)
        self.mad_ = np.nanmedian(np.abs(X - self.median_), axis=0) + 1e-9

        scores = self._score(X)

        if self.threshold is None:
            self.threshold = float(np.percentile(scores, 99.5))

        self.is_fitted = True
        return self

    def _score(self, X: Any) -> np.ndarray:
        X = self._as_numeric_matrix(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.method == "zscore":
            z = np.abs((X - self.mean_) / self.std_)
            return np.max(z, axis=1)

        # modified z-score using MAD
        mz = 0.6745 * np.abs((X - self.median_) / self.mad_)
        return np.max(mz, axis=1)

    def grubbs_test(
        self,
        values: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Simple Grubbs test for univariate outliers.
        Uses scipy if available.
        """
        values = np.asarray(values, dtype=float)
        n = len(values)

        if n < 3:
            return {"outlier_index": None, "statistic": None, "critical_value": None}

        mean = np.mean(values)
        std = np.std(values, ddof=1) + 1e-12

        abs_dev = np.abs(values - mean)
        idx = int(np.argmax(abs_dev))
        G = abs_dev[idx] / std

        try:
            from scipy import stats

            t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        except ImportError:
            # rough fallback
            G_crit = 3.0

        return {
            "outlier_index": idx if G > G_crit else None,
            "statistic": float(G),
            "critical_value": float(G_crit),
            "is_outlier": bool(G > G_crit),
        }

    def get_anomaly_explanation(self, sample: Any) -> Dict[str, Any]:
        X = self._as_numeric_matrix(sample)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.method == "zscore":
            contributions = np.abs((X - self.mean_) / self.std_)[0]
        else:
            contributions = 0.6745 * np.abs((X - self.median_) / self.mad_)[0]

        top_idx = np.argsort(contributions)[::-1][:5]

        feature_names = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        return {
            "detector": self.name,
            "score": float(self._score(X)[0]),
            "threshold": self.threshold,
            "is_anomaly": bool(self._score(X)[0] >= (self.threshold or 0.5)),
            "top_contributors": [
                {
                    "feature": feature_names[i],
                    "value": float(X[0, i]),
                    "contribution": float(contributions[i]),
                }
                for i in top_idx
            ],
        }

    def _get_state(self) -> Dict[str, Any]:
        return {
            "mean": self.mean_,
            "std": self.std_,
            "median": self.median_,
            "mad": self.mad_,
            "method": self.method,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        self.mean_ = state["mean"]
        self.std_ = state["std"]
        self.median_ = state["median"]
        self.mad_ = state["mad"]
        self.method = state.get("method", self.method)


class IsolationForestDetector(BaseAnomalyDetector):
    """
    sklearn IsolationForest wrapper with:
    - optional contamination auto-tuning on validation labels
    - feature importance approximation via score perturbation
    """

    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.001,
        max_samples: str | float | int = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: int = 42,
        threshold: Optional[float] = None,
    ):
        super().__init__(threshold=threshold)
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.model: Optional[IsolationForest] = None

    def _build_model(self, contamination: float) -> IsolationForest:
        return IsolationForest(
            n_estimators=self.n_estimators,
            contamination=contamination,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
        )

    def fit(
        self,
        X_train: Any,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[Any] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "IsolationForestDetector":
        X = self._as_numeric_matrix(X_train)

        best_contamination = self.contamination

        if X_val is not None and y_val is not None and np.sum(np.asarray(y_val)) > 0:
            Xv = self._as_numeric_matrix(X_val)
            yv = np.asarray(y_val).astype(int)

            candidates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]
            best_f1 = -1.0

            for cont in candidates:
                model = self._build_model(cont).fit(X)
                val_scores = -model.score_samples(Xv)
                thresh = np.percentile(val_scores, 100 * (1 - cont))
                val_pred = (val_scores >= thresh).astype(int)

                score = f1_score(yv, val_pred, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_contamination = cont

            self.contamination = best_contamination

        self.model = self._build_model(self.contamination).fit(X)

        train_scores = -self.model.score_samples(X)

        if self.threshold is None:
            self.threshold = float(
                np.percentile(train_scores, 100 * (1 - self.contamination))
            )

        self.is_fitted = True
        return self

    def _score(self, X: Any) -> np.ndarray:
        X = self._as_numeric_matrix(X)
        return -self.model.score_samples(X)

    def feature_importances(
        self,
        X_sample: Any,
        n_samples: int = 500,
    ) -> Dict[str, float]:
        """
        Approximate feature importance by measuring mean score change
        when each feature is shuffled.
        """
        X = self._as_numeric_matrix(X_sample)

        if len(X) > n_samples:
            idx = np.random.default_rng(42).choice(len(X), size=n_samples, replace=False)
            X = X[idx]

        base_scores = self._score(X)
        importances = {}

        feature_names = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        rng = np.random.default_rng(42)

        for j in range(X.shape[1]):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, j])
            perm_scores = self._score(X_perm)
            importances[feature_names[j]] = float(np.mean(np.abs(perm_scores - base_scores)))

        total = sum(importances.values()) + 1e-12
        return {k: v / total for k, v in importances.items()}

    def get_anomaly_explanation(self, sample: Any) -> Dict[str, Any]:
        X = self._as_numeric_matrix(sample)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        score = float(self._score(X)[0])

        return {
            "detector": self.name,
            "score": score,
            "threshold": self.threshold,
            "is_anomaly": bool(score >= (self.threshold or 0.5)),
            "note": "Isolation Forest flags samples that are isolated quickly in tree space.",
        }

    def _get_state(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "contamination": self.contamination,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        self.model = state["model"]
        self.contamination = state.get("contamination", self.contamination)


class LSTMAutoencoderDetector(BaseAnomalyDetector):
    """
    LSTM sequence-to-sequence autoencoder.

    Accepts:
    - 3D numpy array: (n_samples, seq_len, n_features)
    - list of variable-length sequences: [ (seq_len_i, n_features), ... ]
    - 2D array/DataFrame: treated as a single sequence
    """

    name = "lstm_autoencoder"

    def __init__(
        self,
        sequence_length: int = 24,
        hidden_units: Optional[List[int]] = None,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        reconstruction_error_percentile: float = 99.0,
        threshold: Optional[float] = None,
    ):
        super().__init__(threshold=threshold)

        if tf is None:
            raise ImportError(
                "TensorFlow is required for LSTMAutoencoderDetector. Install with pip install tensorflow"
            )

        self.sequence_length = sequence_length
        self.hidden_units = hidden_units or [64, 32]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reconstruction_error_percentile = reconstruction_error_percentile

        self.model: Optional[tf.keras.Model] = None
        self.scaler = StandardScaler()
        self.n_features: Optional[int] = None
        self.max_len: Optional[int] = None

    def _prepare_sequences(
        self,
        X: Any,
        fit: bool = False,
    ) -> Tuple[np.ndarray, List[int], bool]:
        """Convert input into fixed-length sequences.

        A 2D table is interpreted as an ordered stream.  Each row receives a
        trailing sequence window, allowing the CLI to train and score one
        anomaly score per record rather than one score for an entire file.
        """
        sequences: List[np.ndarray] = []
        tabular_input = isinstance(X, pd.DataFrame) or (
            isinstance(X, np.ndarray) and X.ndim == 2
        )

        if isinstance(X, list):
            sequences = [
                self._as_numeric_matrix(s)
                if isinstance(s, pd.DataFrame)
                else np.asarray(s, dtype=float)
                for s in X
            ]
        elif isinstance(X, np.ndarray):
            if X.ndim == 3:
                sequences = [X[i] for i in range(X.shape[0])]
            elif X.ndim == 2:
                table = self._as_numeric_matrix(X)
                sequences = [table[max(0, i - self.sequence_length + 1) : i + 1] for i in range(len(table))]
            else:
                raise ValueError("Unsupported array shape for LSTM detector")
        elif isinstance(X, pd.DataFrame):
            table = self._as_numeric_matrix(X)
            sequences = [table[max(0, i - self.sequence_length + 1) : i + 1] for i in range(len(table))]
        else:
            raise ValueError("Unsupported input type for LSTM detector")

        if not sequences:
            raise ValueError("No sequences provided")

        if fit:
            self.n_features = sequences[0].shape[1]
            self.max_len = self.sequence_length

        if self.n_features is None:
            raise RuntimeError("LSTM detector has no fitted feature dimension")

        for sequence in sequences:
            if sequence.ndim != 2 or sequence.shape[1] != self.n_features:
                raise ValueError("All sequences must have the fitted feature dimension")

        max_len = self.max_len or self.sequence_length
        lengths = [min(len(sequence), max_len) for sequence in sequences]

        padded = np.zeros(
            (len(sequences), max_len, self.n_features),
            dtype=float,
        )

        for i, sequence in enumerate(sequences):
            length = lengths[i]
            if length == 0:
                continue

            # Right alignment keeps the most recent observation at the last
            # timestep for every stream row and makes leading zero padding safe
            # for the Keras Masking layer.
            padded[i, -length:, :] = sequence[-length:, :]

        return padded, lengths, tabular_input

    def _transform_sequences(
        self,
        X: Any,
    ) -> Tuple[np.ndarray, List[int], bool]:
        padded_raw, lengths, tabular_input = self._prepare_sequences(X, fit=False)
        padded_scaled = np.zeros_like(padded_raw)

        for i, length in enumerate(lengths):
            if length > 0:
                padded_scaled[i, -length:, :] = self.scaler.transform(
                    padded_raw[i, -length:, :]
                )

        return padded_scaled, lengths, tabular_input

    def _build_model(self, max_len: int, n_features: int) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(max_len, n_features))

        x = tf.keras.layers.Masking(mask_value=0.0)(inputs)

        for units in self.hidden_units:
            x = tf.keras.layers.LSTM(units, return_sequences=True)(x)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )

        return model

    def fit(
        self,
        X_train: Any,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[Any] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LSTMAutoencoderDetector":
        padded_raw, lengths, _ = self._prepare_sequences(X_train, fit=True)

        # Fit scaler on all non-padding timesteps
        non_padding_rows = []
        for i, length in enumerate(lengths):
            if length > 0:
                non_padding_rows.append(padded_raw[i, -length:, :])

        if not non_padding_rows:
            raise ValueError("No non-empty sequences found for scaling")

        flat = np.vstack(non_padding_rows)
        self.scaler.fit(flat)

        _, max_len, n_features = padded_raw.shape
        padded_scaled = np.zeros_like(padded_raw)
        for i, length in enumerate(lengths):
            if length > 0:
                padded_scaled[i, -length:, :] = self.scaler.transform(
                    padded_raw[i, -length:, :]
                )

        self.model = self._build_model(max_len, n_features)

        self.model.fit(
            padded_scaled,
            padded_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=0,
        )

        train_errors = self._score(X_train)

        if self.threshold is None:
            self.threshold = float(
                np.percentile(train_errors, self.reconstruction_error_percentile)
            )

        self.is_fitted = True
        return self

    def _score(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")

        padded_scaled, lengths, tabular_input = self._transform_sequences(X)
        recon = self.model.predict(padded_scaled, verbose=0)

        if tabular_input:
            # One score per source row: the reconstruction error of the newest
            # observation in that row's trailing sequence window.
            return np.mean((padded_scaled[:, -1, :] - recon[:, -1, :]) ** 2, axis=1)

        errors = []

        for i, length in enumerate(lengths):
            if length == 0:
                errors.append(0.0)
                continue

            err = np.mean((padded_scaled[i, -length:, :] - recon[i, -length:, :]) ** 2)
            errors.append(float(err))

        return np.asarray(errors, dtype=float)

    def get_anomaly_explanation(self, sample: Any) -> Dict[str, Any]:
        score = float(self._score(sample)[0])

        return {
            "detector": self.name,
            "score": score,
            "threshold": self.threshold,
            "is_anomaly": bool(score >= (self.threshold or 0.5)),
            "note": "High reconstruction error indicates unusual sequence dynamics.",
        }

    def save_model(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save(path / "keras_model.keras")

        joblib.dump(
            {
                "scaler": self.scaler,
                "threshold": self.threshold,
                "sequence_length": self.sequence_length,
                "hidden_units": self.hidden_units,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "reconstruction_error_percentile": self.reconstruction_error_percentile,
                "n_features": self.n_features,
                "max_len": self.max_len,
                "feature_names": self.feature_names,
            },
            path / "state.joblib",
        )

    def load_model(self, path: str | Path) -> "LSTMAutoencoderDetector":
        path = Path(path)

        state = joblib.load(path / "state.joblib")

        self.scaler = state["scaler"]
        self.threshold = state["threshold"]
        self.sequence_length = state["sequence_length"]
        self.hidden_units = state["hidden_units"]
        self.epochs = state["epochs"]
        self.batch_size = state["batch_size"]
        self.learning_rate = state["learning_rate"]
        self.reconstruction_error_percentile = state["reconstruction_error_percentile"]
        self.n_features = state["n_features"]
        self.max_len = state["max_len"]
        self.feature_names = state.get("feature_names")

        model_path = path / "keras_model.keras"
        if model_path.exists():
            self.model = tf.keras.models.load_model(model_path)

        self.is_fitted = True
        return self


class DBSCANDetector(BaseAnomalyDetector):
    """
    DBSCAN for spatial clustering of transaction locations.

    Input:
    - two columns/features: latitude, longitude in degrees
    """

    name = "dbscan"

    def __init__(
        self,
        eps_km: float = 5.0,
        min_samples: int = 5,
        threshold: Optional[float] = None,
    ):
        super().__init__(threshold=threshold if threshold is not None else 1.0)

        self.eps_km = eps_km
        self.min_samples = min_samples
        self.eps_rad = eps_km / 6371.0

        self.db: Optional[DBSCAN] = None
        self.core_samples_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def _to_latlon(self, X: Any) -> np.ndarray:
        X = self._as_numeric_matrix(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] < 2:
            raise ValueError("DBSCANDetector expects at least two columns: latitude, longitude")

        return X[:, :2]

    def fit(
        self,
        X_train: Any,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[Any] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "DBSCANDetector":
        latlon = self._to_latlon(X_train)
        latlon_rad = np.radians(latlon)

        self.db = DBSCAN(
            eps=self.eps_rad,
            min_samples=self.min_samples,
            metric="haversine",
        )

        self.labels_ = self.db.fit_predict(latlon_rad)

        core_indices = self.db.core_sample_indices_
        self.core_samples_ = latlon[core_indices] if len(core_indices) > 0 else np.empty((0, 2))

        self.is_fitted = True
        return self

    def _score(self, X: Any) -> np.ndarray:
        latlon = self._to_latlon(X)

        if self.core_samples_ is None or len(self.core_samples_) == 0:
            return np.ones(len(latlon), dtype=float) * 2.0

        scores = []

        for lat, lon in latlon:
            distances = haversine_km(
                np.full(len(self.core_samples_), lat, dtype=float),
                np.full(len(self.core_samples_), lon, dtype=float),
                self.core_samples_[:, 0],
                self.core_samples_[:, 1],
            )

            min_dist = float(np.min(distances))
            scores.append(min_dist / self.eps_km)

        return np.asarray(scores, dtype=float)

    def get_anomaly_explanation(self, sample: Any) -> Dict[str, Any]:
        latlon = self._to_latlon(sample)
        lat, lon = latlon[0]

        score = float(self._score(sample)[0])

        if self.core_samples_ is not None and len(self.core_samples_) > 0:
            distances = haversine_km(
                np.full(len(self.core_samples_), lat, dtype=float),
                np.full(len(self.core_samples_), lon, dtype=float),
                self.core_samples_[:, 0],
                self.core_samples_[:, 1],
            )
            nearest_distance_km = float(np.min(distances))
        else:
            nearest_distance_km = None

        return {
            "detector": self.name,
            "score": score,
            "threshold": self.threshold,
            "is_anomaly": bool(score >= (self.threshold or 1.0)),
            "nearest_core_distance_km": nearest_distance_km,
            "note": "Location is anomalous if far from learned dense geographic clusters.",
        }

    def _get_state(self) -> Dict[str, Any]:
        return {
            "eps_km": self.eps_km,
            "min_samples": self.min_samples,
            "core_samples": self.core_samples_,
            "labels": self.labels_,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        self.eps_km = state["eps_km"]
        self.min_samples = state["min_samples"]
        self.core_samples_ = state["core_samples"]
        self.labels_ = state.get("labels")
        self.eps_rad = self.eps_km / 6371.0


class EnsembleDetector(BaseAnomalyDetector):
    """
    Stacking-style ensemble.

    - base detectors produce scores
    - scores are normalized
    - optional LogisticRegression meta-learner is trained on validation labels
    - fallback is weighted voting
    """

    name = "ensemble"

    def __init__(
        self,
        detectors: Optional[List[BaseAnomalyDetector]] = None,
        weights: Optional[List[float]] = None,
        threshold: Optional[float] = None,
    ):
        # The score scale is learned from the base detectors, so a fixed 0.5
        # threshold is not meaningful before fit.  fit() calibrates an
        # operating point from labels when available, otherwise from the
        # training-score tail.
        super().__init__(threshold=threshold)

        self.detectors = detectors or []
        self.detectors_: List[BaseAnomalyDetector] = []

        self.weights = weights
        self.weights_: Optional[np.ndarray] = None

        self.meta_learner: Optional[LogisticRegression] = None

        self.score_min_: Optional[np.ndarray] = None
        self.score_max_: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: Any,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[Any] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "EnsembleDetector":
        if not self.detectors:
            raise ValueError("EnsembleDetector requires at least one detector")

        fitted: List[BaseAnomalyDetector] = []

        for detector in self.detectors:
            try:
                detector.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                )
                fitted.append(detector)
            except Exception as exc:
                logger.warning(f"Skipping detector {detector.name}: {exc}")

        if not fitted:
            raise RuntimeError("No base detectors could be fitted")

        self.detectors_ = fitted

        scoring_set = X_val if X_val is not None else X_train

        scores_list = []
        for det in self.detectors_:
            s = np.asarray(det._score(scoring_set), dtype=float)
            scores_list.append(s)

        scores = np.column_stack(scores_list)

        self.score_min_ = np.min(scores, axis=0)
        self.score_max_ = np.max(scores, axis=0)

        norm_scores = self._normalize_scores(scores)

        if X_val is not None and y_val is not None and np.sum(np.asarray(y_val)) > 0:
            yv = np.asarray(y_val).astype(int)

            self.meta_learner = LogisticRegression(max_iter=1000)
            self.meta_learner.fit(norm_scores, yv)

            coef = np.abs(self.meta_learner.coef_).ravel()
            total = coef.sum() + 1e-12
            self.weights_ = coef / total
        else:
            self.meta_learner = None

            if self.weights is not None and len(self.weights) == len(self.detectors_):
                w = np.asarray(self.weights, dtype=float)
                self.weights_ = w / (w.sum() + 1e-12)
            else:
                self.weights_ = np.ones(len(self.detectors_)) / len(self.detectors_)

        calibration_set = X_val if X_val is not None else X_train
        calibration_labels = y_val if y_val is not None else y_train

        if self.threshold is None:
            calibration_scores = self._score(calibration_set)
            self.threshold = self._select_threshold(
                calibration_scores,
                calibration_labels,
            )

        self.is_fitted = True
        return self

    @staticmethod
    def _select_threshold(
        scores: np.ndarray,
        labels: Optional[np.ndarray],
    ) -> float:
        """Choose a usable default threshold on the detector's own scale."""
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        if len(scores) == 0:
            return 0.5

        if labels is not None:
            y = np.asarray(labels).astype(int)
            if len(y) == len(scores) and np.any(y == 1):
                precision, recall, thresholds = precision_recall_curve(y, scores)
                if len(thresholds) > 0:
                    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
                    return float(thresholds[int(np.argmax(f1))])

        # For unlabeled data, retain the highest-scoring 0.5% by default. This
        # is a sensible fraud-screening starting point and avoids a no-alert
        # system merely because all learned scores fall below 0.5.
        return float(np.quantile(scores, 0.995))

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if self.score_min_ is None or self.score_max_ is None:
            return scores

        return (scores - self.score_min_) / (self.score_max_ - self.score_min_ + 1e-12)

    def _collect_scores(self, X: Any) -> np.ndarray:
        scores_list = []

        for det in self.detectors_:
            s = np.asarray(det._score(X), dtype=float)
            scores_list.append(s)

        return np.column_stack(scores_list)

    def _score(self, X: Any) -> np.ndarray:
        scores = self._collect_scores(X)
        norm_scores = self._normalize_scores(scores)

        if self.meta_learner is not None:
            return self.meta_learner.predict_proba(norm_scores)[:, 1]

        return norm_scores @ self.weights_

    def get_anomaly_explanation(self, sample: Any) -> Dict[str, Any]:
        scores = self._collect_scores(sample).ravel()
        norm_scores = self._normalize_scores(scores.reshape(1, -1)).ravel()
        score = float(self._score(sample)[0])

        return {
            "detector": self.name,
            "score": score,
            "threshold": self.threshold,
            "is_anomaly": bool(score >= (self.threshold or 0.5)),
            "base_detector_scores": {
                det.name: float(scores[i]) for i, det in enumerate(self.detectors_)
            },
            "normalized_scores": {
                det.name: float(norm_scores[i]) for i, det in enumerate(self.detectors_)
            },
            "weights": {
                det.name: float(self.weights_[i]) for i, det in enumerate(self.detectors_)
            }
            if self.weights_ is not None
            else {},
        }

    def save_model(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {"detectors": []}

        for det in self.detectors_:
            det_path = path / det.name
            det.save_model(det_path)
            meta["detectors"].append(
                {
                    "name": det.name,
                    "class": det.__class__.__name__,
                }
            )

        joblib.dump(
            {
                "weights": self.weights_,
                "score_min": self.score_min_,
                "score_max": self.score_max_,
                "meta_learner": self.meta_learner,
                "threshold": self.threshold,
            },
            path / "state.joblib",
        )

        (path / "meta.json").write_text(
            json.dumps(meta, indent=2, default=str),
            encoding="utf-8",
        )

    def load_model(self, path: str | Path) -> "EnsembleDetector":
        path = Path(path)

        state = joblib.load(path / "state.joblib")

        self.weights_ = state["weights"]
        self.score_min_ = state["score_min"]
        self.score_max_ = state["score_max"]
        self.meta_learner = state["meta_learner"]
        self.threshold = state["threshold"]

        self.detectors_ = []

        meta_path = path / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

            for entry in meta.get("detectors", []):
                class_name = entry.get("class")
                detector_name = entry.get("name")

                cls = DETECTOR_REGISTRY.get(class_name)
                if cls is None:
                    logger.warning(f"Unknown detector class: {class_name}")
                    continue

                try:
                    detector = cls()
                    detector.load_model(path / detector_name)
                    self.detectors_.append(detector)
                except Exception as exc:
                    logger.warning(f"Could not load detector {detector_name}: {exc}")

        self.is_fitted = True
        return self


DETECTOR_REGISTRY = {
    "StatisticalDetector": StatisticalDetector,
    "IsolationForestDetector": IsolationForestDetector,
    "LSTMAutoencoderDetector": LSTMAutoencoderDetector,
    "DBSCANDetector": DBSCANDetector,
    "EnsembleDetector": EnsembleDetector,
}
