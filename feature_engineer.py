from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Vectorized haversine distance in kilometers.
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_km = 6371.0
    return earth_radius_km * c


class FinancialFeatureEngineer:
    """
    Builds transaction and market features with strict temporal behavior:
    - rolling computations use only current and past observations
    - no random shuffling
    - no future information
    """

    def __init__(
        self,
        windows: Optional[Dict[str, str]] = None,
        high_risk_categories: Optional[List[str]] = None,
        impossible_travel_speed_kmh: float = 900.0,
        night_start_hour: int = 0,
        night_end_hour: int = 6,
    ):
        self.windows = windows or {
            "5min": "5min",
            "1h": "1h",
            "1d": "1D",
            "7d": "7D",
            "30d": "30D",
        }

        self.high_risk_categories = {
            (x or "").lower()
            for x in (
                high_risk_categories
                or [
                    "casino",
                    "gambling",
                    "crypto_exchange",
                    "money_transfer",
                    "wire_transfer",
                    "jewelry",
                    "electronics",
                    "adult_content",
                ]
            )
        }

        self.impossible_travel_speed_kmh = impossible_travel_speed_kmh
        self.night_start_hour = night_start_hour
        self.night_end_hour = night_end_hour

    def build_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expected columns:
        - transaction_id
        - user_id
        - timestamp
        - amount
        - merchant_id
        - merchant_category
        - latitude
        - longitude
        """
        required = ["user_id", "timestamp", "amount"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required transaction columns: {missing}")

        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        # Stable ordering makes same-timestamp transactions deterministic.  The
        # rolling results below are assigned by position, so the input must stay
        # in exactly the same group/time order used by GroupBy.rolling.
        out = out.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)

        # Basic temporal risk features
        out["is_weekend"] = out["timestamp"].dt.weekday >= 5
        out["is_night_time"] = out["timestamp"].dt.hour.between(
            self.night_start_hour, self.night_end_hour
        )

        if "merchant_category" in out.columns:
            out["high_risk_merchant_flag"] = (
                out["merchant_category"].fillna("").str.lower().isin(self.high_risk_categories)
            )
        else:
            out["high_risk_merchant_flag"] = False

        # Rolling / velocity features
        #
        # `rolling(..., on="timestamp")` returns a MultiIndex ending in the
        # timestamp, not the original row index.  Therefore its result must be
        # assigned positionally; assigning the Series directly aligns timestamp
        # labels against the DataFrame's RangeIndex and either produces all NaN
        # values or raises when timestamps repeat.
        has_merchant_id = "merchant_id" in out.columns
        if has_merchant_id:
            # RollingGroupby does not implement nunique consistently across
            # supported pandas releases and cannot aggregate object values.  A
            # factorized numeric representation works with rolling.apply.
            out["_merchant_id_code"] = pd.factorize(
                out["merchant_id"].fillna("__MISSING__"), sort=False
            )[0]

        for name, window in self.windows.items():
            safe_name = name.replace(" ", "")

            grouped = out.groupby("user_id", sort=False).rolling(window, on="timestamp")

            out[f"tx_count_{safe_name}"] = grouped["amount"].count().to_numpy()

            out[f"amount_mean_{safe_name}"] = grouped["amount"].mean().to_numpy()

            out[f"amount_std_{safe_name}"] = grouped["amount"].std().fillna(0.0).to_numpy()

            if has_merchant_id:
                out[f"unique_merchants_{safe_name}"] = (
                    grouped["_merchant_id_code"]
                    .apply(lambda values: values.nunique(), raw=False)
                    .to_numpy()
                )
            else:
                out[f"unique_merchants_{safe_name}"] = 0

        if has_merchant_id:
            out = out.drop(columns=["_merchant_id_code"])

        # Ensure common named features exist
        if "amount_std_1d" in out.columns and "amount_std_24h" not in out.columns:
            out["amount_std_24h"] = out["amount_std_1d"]

        if "amount_mean_7d" in out.columns and "rolling_mean_amount_7d" not in out.columns:
            out["rolling_mean_amount_7d"] = out["amount_mean_7d"]

        # Behavioral features
        out["time_since_last_tx_hours"] = (
            out.groupby("user_id")["timestamp"]
            .diff()
            .dt.total_seconds()
            .fillna(0.0)
            / 3600.0
        )

        mean_30d_col = "amount_mean_30d"
        std_30d_col = "amount_std_30d"

        if mean_30d_col in out.columns and std_30d_col in out.columns:
            out["amount_zscore_per_user"] = (
                (out["amount"] - out[mean_30d_col])
                / (out[std_30d_col] + 1e-9)
            ).fillna(0.0)
        else:
            out["amount_zscore_per_user"] = 0.0

        # Geographic velocity / impossible travel
        if {"latitude", "longitude"}.issubset(out.columns):
            out["prev_lat"] = out.groupby("user_id")["latitude"].shift(1)
            out["prev_lon"] = out.groupby("user_id")["longitude"].shift(1)
            out["prev_timestamp"] = out.groupby("user_id")["timestamp"].shift(1)

            out["prev_lat"] = out["prev_lat"].fillna(out["latitude"])
            out["prev_lon"] = out["prev_lon"].fillna(out["longitude"])
            out["prev_timestamp"] = out["prev_timestamp"].fillna(out["timestamp"])

            dist_km = haversine_km(
                out["prev_lat"].to_numpy(dtype=float),
                out["prev_lon"].to_numpy(dtype=float),
                out["latitude"].to_numpy(dtype=float),
                out["longitude"].to_numpy(dtype=float),
            )

            time_delta_hours = (
                (out["timestamp"] - out["prev_timestamp"]).dt.total_seconds() / 3600.0
            ).replace(0.0, np.nan)

            out["geo_distance_km"] = dist_km
            out["geo_velocity_kmh"] = (out["geo_distance_km"] / time_delta_hours).fillna(0.0)

            out["impossible_travel_flag"] = (
                out["geo_velocity_kmh"] > self.impossible_travel_speed_kmh
            ) | (
                (out["geo_distance_km"] > 500.0) & (time_delta_hours < 2.0)
            )

            out = out.drop(columns=["prev_lat", "prev_lon", "prev_timestamp"])
        else:
            out["geo_distance_km"] = 0.0
            out["geo_velocity_kmh"] = 0.0
            out["impossible_travel_flag"] = False

        # Aggregation features
        first_seen = out.groupby("user_id")["timestamp"].transform("min")
        out["days_since_first_seen"] = (
            (out["timestamp"] - first_seen).dt.total_seconds() / 86400.0
        ).fillna(0.0)

        return out

    def build_market_features(
        self,
        bars: pd.DataFrame,
        rsi_period: int = 14,
        atr_period: int = 14,
        bollinger_period: int = 20,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ) -> pd.DataFrame:
        """
        Expected columns:
        - symbol
        - timestamp
        - open
        - high
        - low
        - close
        - volume
        """
        required = ["symbol", "timestamp", "close"]
        missing = [c for c in required if c not in bars.columns]
        if missing:
            raise ValueError(f"Missing required market columns: {missing}")

        frames = []

        for symbol, group in bars.groupby("symbol"):
            g = group.sort_values("timestamp").copy()
            g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)

            close = g["close"].astype(float)
            high = g["high"].astype(float) if "high" in g.columns else close
            low = g["low"].astype(float) if "low" in g.columns else close
            volume = (
                g["volume"].astype(float)
                if "volume" in g.columns
                else pd.Series(0.0, index=g.index)
            )

            g["return_1"] = close.pct_change().fillna(0.0)

            # RSI
            delta = close.diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)

            avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
            avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()

            rs = avg_gain / (avg_loss + 1e-12)
            g["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

            # MACD
            ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
            ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
            g["macd"] = ema_fast - ema_slow
            g["macd_signal"] = g["macd"].ewm(span=macd_signal, adjust=False).mean()
            g["macd_hist"] = g["macd"] - g["macd_signal"]

            # Bollinger Bands
            mid = close.rolling(bollinger_period).mean()
            std = close.rolling(bollinger_period).std()
            g["bollinger_mid"] = mid
            g["bollinger_upper"] = mid + 2 * std
            g["bollinger_lower"] = mid - 2 * std
            g["bollinger_width"] = (
                (g["bollinger_upper"] - g["bollinger_lower"]) / (g["bollinger_mid"] + 1e-12)
            ).fillna(0.0)

            # ATR
            prev_close = close.shift(1)
            tr = pd.concat(
                [
                    high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            g["atr_14"] = tr.rolling(atr_period).mean().fillna(0.0)

            # OBV
            direction = np.sign(close.diff()).fillna(0.0)
            g["obv"] = (direction * volume).cumsum()

            # Microstructure features
            g["bid_ask_spread_proxy"] = ((high - low) / (close + 1e-12)).fillna(0.0)

            # Offset windows require a DatetimeIndex.  Keeping the original
            # RangeIndex here causes pandas to reject the market pipeline.
            time_index = pd.DatetimeIndex(g["timestamp"])
            volume_by_time = pd.Series(volume.to_numpy(), index=time_index)
            returns_by_time = pd.Series(g["return_1"].to_numpy(), index=time_index)

            volume_mean_1d = volume_by_time.rolling("1D", min_periods=1).mean()
            g["volume_anomaly_ratio"] = (
                volume.to_numpy() / (volume_mean_1d.to_numpy() + 1e-12)
            )

            vol_1d = returns_by_time.rolling("1D", min_periods=1).std()
            vol_7d = returns_by_time.rolling("7D", min_periods=1).std()
            g["volatility_regime"] = (
                vol_1d.to_numpy() / (vol_7d.to_numpy() + 1e-12)
            )

            numeric_columns = g.select_dtypes(include=[np.number]).columns
            g[numeric_columns] = (
                g[numeric_columns]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            frames.append(g)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def build_cross_asset_features(
        self,
        bars: pd.DataFrame,
        benchmark_symbol: str = "SPY",
        window: str = "7D",
    ) -> pd.DataFrame:
        """
        Adds correlation breakdown and beta deviation relative to benchmark.
        """
        df = bars.copy()
        df["return_1"] = df.groupby("symbol")["close"].pct_change().fillna(0.0)

        returns = df.pivot(index="timestamp", columns="symbol", values="return_1")

        if benchmark_symbol not in returns.columns:
            df["correlation_breakdown"] = 0.0
            df["beta_deviation"] = 0.0
            return df

        bench = returns[benchmark_symbol]

        frames = []

        for symbol in returns.columns:
            if symbol == benchmark_symbol:
                continue

            asset = returns[symbol]

            corr = asset.rolling(window).corr(bench)
            cov = asset.rolling(window).cov(bench)
            var = bench.rolling(window).var()

            beta = cov / (var + 1e-12)

            corr_breakdown = (corr < 0.2).astype(float)
            beta_deviation = (beta - beta.rolling(window).mean()).abs()

            tmp = pd.DataFrame(
                {
                    "timestamp": returns.index,
                    "symbol": symbol,
                    "correlation_breakdown": corr_breakdown.values,
                    "beta_deviation": beta_deviation.values,
                }
            )
            frames.append(tmp)

        if not frames:
            df["correlation_breakdown"] = 0.0
            df["beta_deviation"] = 0.0
            return df

        cross = pd.concat(frames, ignore_index=True)
        out = df.merge(cross, on=["timestamp", "symbol"], how="left")

        out["correlation_breakdown"] = out["correlation_breakdown"].fillna(0.0)
        out["beta_deviation"] = out["beta_deviation"].fillna(0.0)

        return out


class FeatureStore:
    """
    Parquet-backed feature cache with simple versioning and manifest metadata.
    """

    def __init__(self, root: str | Path = "feature_store", version: str = "v1"):
        self.root = Path(root)
        self.version = version
        self.version_dir = self.root / self.version
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.version_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {"entries": []}

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _hash_dataframe(df: pd.DataFrame) -> str:
        payload = {
            "shape": df.shape,
            "columns": list(df.columns),
        }

        if "timestamp" in df.columns and len(df) > 0:
            payload["min_timestamp"] = str(pd.Timestamp(df["timestamp"].min()))
            payload["max_timestamp"] = str(pd.Timestamp(df["timestamp"].max()))

        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def save(self, name: str, df: pd.DataFrame) -> Path:
        df_hash = self._hash_dataframe(df)
        file_path = self.version_dir / f"{name}_{df_hash}.parquet"

        df.to_parquet(file_path, index=False)

        entry = {
            "name": name,
            "version": self.version,
            "path": str(file_path),
            "hash": df_hash,
            "rows": len(df),
            "columns": list(df.columns),
            "created_at": pd.Timestamp.utcnow().isoformat(),
        }

        self.manifest["entries"].append(entry)
        self._save_manifest()

        logger.info(f"Saved feature set {name} to {file_path}")
        return file_path

    def load_latest(self, name: str) -> Optional[pd.DataFrame]:
        entries = [
            e
            for e in self.manifest.get("entries", [])
            if e.get("name") == name and e.get("version") == self.version
        ]

        if not entries:
            return None

        latest = sorted(entries, key=lambda e: e.get("created_at", ""))[-1]
        path = Path(latest["path"])

        if not path.exists():
            logger.warning(f"Manifest entry exists but file missing: {path}")
            return None

        return pd.read_parquet(path)

    def load_by_hash(self, name: str, df_hash: str) -> Optional[pd.DataFrame]:
        path = self.version_dir / f"{name}_{df_hash}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None
