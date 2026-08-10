from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _read_any_source(
    source: str,
    query: Optional[str] = None,
) -> pd.DataFrame:
    """
    Reads from:
    - CSV
    - Parquet
    - SQLAlchemy-compatible SQL source
    """
    source_str = str(source)

    if source_str.startswith(
        (
            "postgresql",
            "postgres",
            "clickhouse",
            "mysql",
            "sqlite",
        )
    ):
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise ImportError(
                "SQLAlchemy is required for SQL sources. Install with pip install sqlalchemy"
            ) from exc

        engine = create_engine(source_str)
        sql = query or "SELECT * FROM transactions"
        return pd.read_sql(sql, engine)

    path = Path(source_str)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


@dataclass
class DataQualityReport:
    row_count: int = 0
    missing_values: Dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    duplicate_ids: int = 0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_count": self.row_count,
            "missing_values": self.missing_values,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_ids": self.duplicate_ids,
            "issues": self.issues,
            "warnings": self.warnings,
        }


class DataLoader(ABC):
    """Abstract data loader with required interface."""

    @abstractmethod
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        raise NotImplementedError


class TransactionLoader(DataLoader):
    """
    Loads transactional data from CSV/Parquet/SQL.

    Responsibilities:
    - timezone normalization to UTC
    - temporal integrity checks
    - leakage detection
    - time-based train/validation/test splitting
    """

    REQUIRED_COLUMNS = [
        "transaction_id",
        "user_id",
        "timestamp",
        "amount",
    ]

    def __init__(
        self,
        timestamp_col: str = "timestamp",
        id_col: str = "transaction_id",
        user_col: str = "user_id",
    ):
        self.timestamp_col = timestamp_col
        self.id_col = id_col
        self.user_col = user_col

    def load(self, source: str, query: Optional[str] = None, **kwargs) -> pd.DataFrame:
        df = _read_any_source(source, query=query)
        df = self.normalize_timezone(df)
        df = df.sort_values([self.user_col, self.timestamp_col]).reset_index(drop=True)
        return df

    def normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.timestamp_col not in df.columns:
            raise ValueError(f"Missing required timestamp column: {self.timestamp_col}")

        ts = pd.to_datetime(df[self.timestamp_col], utc=True, errors="coerce")

        if ts.isna().any():
            bad_count = int(ts.isna().sum())
            logger.warning(f"Dropping {bad_count} rows with unparseable timestamps")
            df = df.loc[~ts.isna()].copy()
            ts = ts.loc[~ts.isna()]

        df[self.timestamp_col] = ts.dt.tz_convert("UTC")
        return df

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        report = DataQualityReport(row_count=len(df))

        missing_required = [
            col for col in self.REQUIRED_COLUMNS if col not in df.columns
        ]
        if missing_required:
            report.issues.append(f"Missing required columns: {missing_required}")

        report.missing_values = df.isna().sum().to_dict()

        report.duplicate_rows = int(df.duplicated().sum())
        if self.id_col in df.columns:
            report.duplicate_ids = int(df[self.id_col].duplicated().sum())

        if "amount" in df.columns:
            negative_amount_count = int((df["amount"] < 0).sum())
            if negative_amount_count > 0:
                report.warnings.append(
                    f"Found {negative_amount_count} negative transaction amounts"
                )

        if self.timestamp_col in df.columns:
            future_rows = int((df[self.timestamp_col] > pd.Timestamp.now(tz="UTC")).sum())
            if future_rows > 0:
                report.warnings.append(f"Found {future_rows} future-dated transactions")

        return report

    def detect_leakage(
        self,
        df: pd.DataFrame,
        cutoff: pd.Timestamp,
        future_date_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Detects obvious temporal leakage:
        - rows after cutoff appearing in training
        - future-dated helper columns before cutoff
        """
        cutoff = (
            pd.Timestamp(cutoff).tz_localize("UTC")
            if pd.Timestamp(cutoff).tzinfo is None
            else pd.Timestamp(cutoff).tz_convert("UTC")
        )

        leakage = {
            "rows_after_cutoff": 0,
            "future_dated_columns": {},
        }

        if self.timestamp_col in df.columns:
            leakage["rows_after_cutoff"] = int((df[self.timestamp_col] > cutoff).sum())

        future_date_columns = future_date_columns or []
        for col in future_date_columns:
            if col not in df.columns:
                continue

            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
            future_count = int((ts > cutoff).sum())
            if future_count > 0:
                leakage["future_dated_columns"][col] = future_count

        leakage["has_leakage"] = (
            leakage["rows_after_cutoff"] > 0
            or any(v > 0 for v in leakage["future_dated_columns"].values())
        )

        return leakage

    def time_based_split(
        self,
        df: pd.DataFrame,
        train_end: pd.Timestamp,
        val_end: Optional[pd.Timestamp] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Strict temporal split.
        No shuffling. No random splitting.
        """
        df = df.sort_values(self.timestamp_col).copy()

        train_end = pd.Timestamp(train_end)
        if train_end.tzinfo is None:
            train_end = train_end.tz_localize("UTC")
        else:
            train_end = train_end.tz_convert("UTC")

        train = df[df[self.timestamp_col] < train_end].copy()

        if val_end is None:
            val = df[df[self.timestamp_col] >= train_end].copy()
            test = pd.DataFrame(columns=df.columns)
            return train, val, test

        val_end = pd.Timestamp(val_end)
        if val_end.tzinfo is None:
            val_end = val_end.tz_localize("UTC")
        else:
            val_end = val_end.tz_convert("UTC")

        val = df[
            (df[self.timestamp_col] >= train_end)
            & (df[self.timestamp_col] < val_end)
        ].copy()

        test = df[df[self.timestamp_col] >= val_end].copy()

        return train, val, test


class MarketDataLoader(DataLoader):
    """
    Loads and preprocesses market tick / bar data.

    Responsibilities:
    - resampling tick data to multiple frequencies
    - market hours filtering
    - missing bar interpolation
    - price gap detection
    """

    REQUIRED_COLUMNS = [
        "symbol",
        "timestamp",
        "price",
    ]

    def __init__(
        self,
        timestamp_col: str = "timestamp",
        symbol_col: str = "symbol",
        price_col: str = "price",
        volume_col: str = "volume",
    ):
        self.timestamp_col = timestamp_col
        self.symbol_col = symbol_col
        self.price_col = price_col
        self.volume_col = volume_col

    def load(self, source: str, query: Optional[str] = None, **kwargs) -> pd.DataFrame:
        df = _read_any_source(source, query=query)
        df = self.normalize_timezone(df)
        df = df.sort_values([self.symbol_col, self.timestamp_col]).reset_index(drop=True)
        return df

    def normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.timestamp_col not in df.columns:
            raise ValueError(f"Missing required timestamp column: {self.timestamp_col}")

        df[self.timestamp_col] = pd.to_datetime(
            df[self.timestamp_col], utc=True, errors="coerce"
        )
        df = df.dropna(subset=[self.timestamp_col])
        return df

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        report = DataQualityReport(row_count=len(df))

        missing_required = [
            col for col in self.REQUIRED_COLUMNS if col not in df.columns
        ]
        if missing_required:
            report.issues.append(f"Missing required columns: {missing_required}")

        report.missing_values = df.isna().sum().to_dict()
        report.duplicate_rows = int(df.duplicated().sum())

        return report

    def filter_market_hours(
        self,
        df: pd.DataFrame,
        start_time: str = "09:30",
        end_time: str = "16:00",
        weekdays_only: bool = True,
    ) -> pd.DataFrame:
        df = df.copy()
        ts = df[self.timestamp_col]

        start = pd.to_timedelta(start_time)
        end = pd.to_timedelta(end_time)

        time_of_day = ts.dt.hour * pd.Timedelta(hours=1) + ts.dt.minute * pd.Timedelta(minutes=1)

        mask = (time_of_day >= start) & (time_of_day <= end)

        if weekdays_only:
            mask &= ts.dt.weekday < 5

        return df.loc[mask].reset_index(drop=True)

    def resample_tick_data(
        self,
        df: pd.DataFrame,
        freq: str = "1min",
    ) -> pd.DataFrame:
        """
        Resamples tick data into OHLCV bars.

        Missing-bar handling:
        - prices: linear interpolation
        - volume: forward fill, per requirement
        """
        records = []

        for symbol, group in df.groupby(self.symbol_col):
            g = group.set_index(self.timestamp_col).sort_index()

            ohlc = g[self.price_col].resample(freq).ohlc()

            if self.volume_col in g.columns:
                volume = g[self.volume_col].resample(freq).sum()
            else:
                volume = pd.Series(0.0, index=ohlc.index)

            out = ohlc.copy()
            out["volume"] = volume
            out["symbol"] = symbol

            out = out.asfreq(freq)

            price_cols = ["open", "high", "low", "close"]
            out[price_cols] = out[price_cols].interpolate(method="linear")
            out[price_cols] = out[price_cols].ffill().bfill()

            out["volume"] = out["volume"].ffill().fillna(0.0)

            records.append(out.reset_index())

        if not records:
            return pd.DataFrame()

        result = pd.concat(records, ignore_index=True)
        result = result.rename(columns={self.timestamp_col: "timestamp"})
        return result

    def detect_price_gaps(
        self,
        bars: pd.DataFrame,
        threshold_pct: float = 0.05,
    ) -> pd.DataFrame:
        """
        Flags suspicious single-bar moves > threshold_pct.
        """
        bars = bars.copy()

        if "close" not in bars.columns:
            raise ValueError("detect_price_gaps requires a close column")

        bars = bars.sort_values([self.symbol_col, self.timestamp_col])

        bars["prev_close"] = bars.groupby(self.symbol_col)["close"].shift(1)
        bars["price_gap_pct"] = (
            (bars["close"] - bars["prev_close"]) / bars["prev_close"]
        ).abs()

        bars["price_gap_flag"] = bars["price_gap_pct"] > threshold_pct

        return bars.drop(columns=["prev_close"])

    def interpolate_missing_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Explicit interpolation helper if bars are already resampled.
        """
        bars = bars.copy()
        price_cols = ["open", "high", "low", "close"]

        bars[price_cols] = bars.groupby(self.symbol_col)[price_cols].transform(
            lambda s: s.interpolate(method="linear").ffill().bfill()
        )

        if "volume" in bars.columns:
            bars["volume"] = bars.groupby(self.symbol_col)["volume"].transform(
                lambda s: s.ffill().fillna(0.0)
            )

        return bars