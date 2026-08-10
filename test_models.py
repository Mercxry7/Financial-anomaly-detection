from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models import (
    EnsembleDetector,
    IsolationForestDetector,
    StatisticalDetector,
)


def make_tabular_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    normal = rng.normal(loc=0.0, scale=1.0, size=(n, 5))

    anomalies = rng.normal(loc=6.0, scale=1.0, size=(5, 5))

    X = np.vstack([normal, anomalies])
    y = np.array([0] * n + [1] * 5)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["label"] = y

    return df


def test_statistical_detector_fit_predict():
    df = make_tabular_data()
    X = df.drop(columns=["label"])

    detector = StatisticalDetector(method="modified_zscore")
    detector.fit(X)

    scores, labels = detector.predict(X)

    assert len(scores) == len(X)
    assert len(labels) == len(X)
    assert detector.threshold is not None


def test_isolation_forest_serialization(tmp_path):
    df = make_tabular_data()
    X = df.drop(columns=["label"])

    detector = IsolationForestDetector(n_estimators=50, contamination=0.01)
    detector.fit(X)

    save_path = tmp_path / "iforest"
    detector.save_model(save_path)

    loaded = IsolationForestDetector().load_model(save_path)

    scores_original, _ = detector.predict(X)
    scores_loaded, _ = loaded.predict(X)

    assert np.allclose(scores_original, scores_loaded, atol=1e-8)


def test_dataframe_prediction_reorders_features_and_preserves_boolean_flags():
    X = pd.DataFrame(
        {
            "amount": [10.0, 12.0, 11.0, 100.0],
            "impossible_travel_flag": [False, False, False, True],
            "tx_count_1h": [1.0, 1.0, 2.0, 10.0],
        }
    )
    detector = IsolationForestDetector(n_estimators=25, contamination=0.25)
    detector.fit(X)

    reordered = X[["tx_count_1h", "amount", "impossible_travel_flag"]].copy()
    reordered["unused_numeric_column"] = [99.0] * len(reordered)

    original_scores, _ = detector.predict(X)
    reordered_scores, _ = detector.predict(reordered)

    assert detector.feature_names == [
        "amount",
        "impossible_travel_flag",
        "tx_count_1h",
    ]
    assert np.allclose(original_scores, reordered_scores)

    with pytest.raises(ValueError, match="Missing required model features"):
        detector.predict(X.drop(columns=["amount"]))


def test_ensemble_weights_sum_to_one():
    df = make_tabular_data()
    X = df.drop(columns=["label"])
    y = df["label"].to_numpy()

    train_split = 400

    X_train = X.iloc[:train_split]
    X_val = X.iloc[train_split:]

    y_train = y[:train_split]
    y_val = y[train_split:]

    statistical = StatisticalDetector()
    iforest = IsolationForestDetector(n_estimators=50, contamination=0.01)

    ensemble = EnsembleDetector(detectors=[statistical, iforest])

    ensemble.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    assert ensemble.weights_ is not None
    assert np.isclose(ensemble.weights_.sum(), 1.0)


def test_ensemble_derives_an_operating_threshold_without_validation_data():
    df = make_tabular_data(n=150)
    X = df.drop(columns=["label"])
    ensemble = EnsembleDetector(
        detectors=[
            StatisticalDetector(),
            IsolationForestDetector(n_estimators=25, contamination=0.02),
        ]
    )

    ensemble.fit(X_train=X)
    scores, labels = ensemble.predict(X)

    assert ensemble.threshold is not None
    assert np.isclose(ensemble.threshold, np.quantile(scores, 0.995))
    assert labels.sum() >= 1


@pytest.mark.skipif(
    True,
    reason="LSTM autoencoder test requires TensorFlow and can be slow in CI",
)
def test_lstm_autoencoder_reconstruction_improves():
    # Placeholder for CI stability.
    # In a full environment, you would:
    # 1. create sequence data
    # 2. fit LSTMAutoencoderDetector
    # 3. assert reconstruction error decreases versus untrained baseline
    pass
