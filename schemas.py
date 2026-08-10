from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


class AnomalyType(str, Enum):
    FRAUD = "FRAUD"
    FLASH_CRASH = "FLASH_CRASH"
    MONEY_LAUNDERING = "MONEY_LAUNDERING"
    PUMP_DUMP = "PUMP_DUMP"
    ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"


@dataclass
class TransactionRecord:
    transaction_id: str
    user_id: str
    timestamp: datetime
    amount: float
    currency: str = "USD"

    merchant_id: Optional[str] = None
    merchant_category: Optional[str] = None
    channel: Optional[str] = None

    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    is_fraud: bool = False
    anomaly_type: Optional[AnomalyType] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MarketDataPoint:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    bid: Optional[float] = None
    ask: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnomalyResult:
    id: str
    detector_name: str
    entity_id: str
    timestamp: datetime
    score: float
    is_anomaly: bool
    anomaly_type: Optional[AnomalyType] = None
    explanations: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetrics:
    model_name: str

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    roc_auc: float = 0.0
    pr_auc: float = 0.0

    precision_at_k: float = 0.0
    alert_fatigue_score: float = 0.0

    false_positive_rate: float = 0.0
    avg_detection_delay_periods: Optional[float] = None

    cost_sensitive_loss: Optional[float] = None
    var_reduction: Optional[float] = None
    cascade_detection_rate: Optional[float] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TransactionFeatureDict(TypedDict, total=False):
    transaction_id: str
    user_id: str
    timestamp: datetime

    amount: float
    currency: str
    merchant_id: str
    merchant_category: str
    channel: str

    tx_count_5min: float
    tx_count_1h: float
    tx_count_1d: float
    tx_count_7d: float
    tx_count_30d: float

    amount_mean_5min: float
    amount_mean_1h: float
    amount_mean_1d: float
    amount_mean_7d: float
    amount_mean_30d: float

    amount_std_5min: float
    amount_std_1h: float
    amount_std_24h: float
    amount_std_7d: float
    amount_std_30d: float

    unique_merchants_5min: int
    unique_merchants_1h: int
    unique_merchants_1d: int
    unique_merchants_7d: int
    unique_merchants_30d: int

    amount_zscore_per_user: float
    time_since_last_tx_hours: float
    geo_velocity_kmh: float
    impossible_travel_flag: bool

    rolling_mean_amount_7d: float
    days_since_first_seen: float

    is_weekend: bool
    is_night_time: bool
    high_risk_merchant_flag: bool


class MarketFeatureDict(TypedDict, total=False):
    symbol: str
    timestamp: datetime

    return_1: float
    rsi_14: float
    macd: float
    macd_signal: float
    macd_hist: float

    bollinger_mid: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_width: float

    atr_14: float
    obv: float

    bid_ask_spread_proxy: float
    volume_anomaly_ratio: float
    volatility_regime: float

    correlation_breakdown: float
    beta_deviation: float