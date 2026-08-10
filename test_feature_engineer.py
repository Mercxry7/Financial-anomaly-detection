from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineer import FinancialFeatureEngineer, haversine_km


def make_transaction_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["T1", "T2", "T3"],
            "user_id": ["U1", "U1", "U1"],
            "timestamp": pd.to_datetime(
                [
                    "2023-01-01 10:00:00",
                    "2023-01-01 10:30:00",
                    "2023-01-01 12:00:00",
                ],
                utc=True,
            ),
            "amount": [100.0, 200.0, 300.0],
            "merchant_id": ["M1", "M2", "M3"],
            "merchant_category": ["grocery", "casino", "electronics"],
            "latitude": [40.0, 40.01, 41.0],
            "longitude": [-74.0, -74.01, -75.0],
        }
    )


def test_rolling_windows_do_not_leak_future():
    df = make_transaction_df()
    engineer = FinancialFeatureEngineer(windows={"1h": "1h"})

    features = engineer.build_transaction_features(df)

    # At first transaction, only one transaction should exist in past/current 1h window
    first_row = features.loc[features["transaction_id"] == "T1"].iloc[0]

    assert first_row["tx_count_1h"] == 1.0

    # Second transaction at 10:30 should see T1 and T2, but not T3 at 12:00
    second_row = features.loc[features["transaction_id"] == "T2"].iloc[0]
    assert second_row["tx_count_1h"] == 2.0


def test_rolling_features_handle_duplicate_timestamps_and_merchant_ids():
    """Rolling values must be assigned by row position, not timestamp labels."""
    df = pd.DataFrame(
        {
            "transaction_id": ["T1", "T2", "T3", "T4"],
            "user_id": ["U1", "U1", "U2", "U2"],
            "timestamp": pd.to_datetime(
                [
                    "2023-01-01 10:00:00",
                    "2023-01-01 10:00:00",
                    "2023-01-01 10:00:00",
                    "2023-01-01 10:00:00",
                ],
                utc=True,
            ),
            "amount": [10.0, 20.0, 30.0, 40.0],
            "merchant_id": ["M1", "M2", "M3", "M3"],
        }
    )

    features = FinancialFeatureEngineer(windows={"1h": "1h"}).build_transaction_features(df)

    assert features["tx_count_1h"].tolist() == [1.0, 2.0, 1.0, 2.0]
    assert features["unique_merchants_1h"].tolist() == [1.0, 2.0, 1.0, 1.0]
    assert features[["tx_count_1h", "amount_mean_1h"]].notna().all().all()


def test_impossible_travel_detection():
    df = pd.DataFrame(
        {
            "transaction_id": ["T1", "T2"],
            "user_id": ["U1", "U1"],
            "timestamp": pd.to_datetime(
                [
                    "2023-01-01 10:00:00",
                    "2023-01-01 11:00:00",
                ],
                utc=True,
            ),
            "amount": [100.0, 200.0],
            "merchant_id": ["M1", "M2"],
            "merchant_category": ["grocery", "gas_station"],
            "latitude": [40.7128, 41.8781],   # NYC -> Chicago approx
            "longitude": [-74.0060, -87.6298],
        }
    )

    engineer = FinancialFeatureEngineer(impossible_travel_speed_kmh=900.0)
    features = engineer.build_transaction_features(df)

    second_row = features.loc[features["transaction_id"] == "T2"].iloc[0]

    assert second_row["impossible_travel_flag"] == True
    assert second_row["geo_velocity_kmh"] > 500.0


def test_high_risk_merchant_flag():
    df = make_transaction_df()
    engineer = FinancialFeatureEngineer()

    features = engineer.build_transaction_features(df)

    casino_row = features.loc[features["merchant_category"] == "casino"].iloc[0]
    grocery_row = features.loc[features["merchant_category"] == "grocery"].iloc[0]

    assert casino_row["high_risk_merchant_flag"] == True
    assert grocery_row["high_risk_merchant_flag"] == False


def test_haversine_positive():
    dist = haversine_km(
        np.array([40.7128]),
        np.array([-74.0060]),
        np.array([41.8781]),
        np.array([-87.6298]),
    )

    assert dist[0] > 1000.0
