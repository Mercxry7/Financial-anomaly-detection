from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresConfig(BaseModel):
    """PostgreSQL configuration for transactional data."""

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "postgres"
    dbname: str = "transactions"
    schema_name: str = "public"

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.dbname}"
        )


class ClickHouseConfig(BaseModel):
    """ClickHouse configuration for tick / market data."""

    host: str = "localhost"
    port: int = 8123
    database: str = "market_data"
    user: str = "default"
    password: str = ""

    @property
    def http_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class IsolationForestParams(BaseModel):
    n_estimators: int = 200
    max_samples: str | float | int = "auto"
    contamination: float = 0.001
    max_features: float = 1.0
    bootstrap: bool = False
    random_state: int = 42


class LSTMAutoencoderParams(BaseModel):
    sequence_length: int = 24
    hidden_units: List[int] = Field(default_factory=lambda: [64, 32])
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    reconstruction_error_percentile: float = 99.0
    masking_value: float = 0.0


class DBSCANParams(BaseModel):
    eps_km: float = 5.0
    min_samples: int = 5
    metric: str = "haversine"


class FeatureWindowConfig(BaseModel):
    """
    Feature engineering window sizes.
    Keys are semantic names, values are pandas-compatible offset aliases.
    """

    windows: Dict[str, str] = Field(
        default_factory=lambda: {
            "5min": "5min",
            "1h": "1h",
            "1d": "1D",
            "7d": "7D",
            "30d": "30D",
        }
    )

    high_risk_merchant_categories: List[str] = Field(
        default_factory=lambda: [
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

    impossible_travel_speed_kmh: float = 900.0
    night_start_hour: int = 0
    night_end_hour: int = 6


class ThresholdCalibrationConfig(BaseModel):
    objective: str = "f1"
    min_precision: float = 0.80
    target_recall: float = 0.60
    max_alerts_per_day: int = 100
    search_grid: List[float] = Field(
        default_factory=lambda: [
            0.50,
            0.60,
            0.70,
            0.80,
            0.85,
            0.90,
            0.95,
            0.975,
            0.99,
            0.995,
            0.999,
        ]
    )


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logfile: Optional[str] = "logs/anomaly_detection.log"


class TrainingConfig(BaseModel):
    n_jobs: int = 1
    random_state: int = 42
    use_lstm: bool = True
    use_dbscan: bool = True
    use_isolation_forest: bool = True
    use_statistical: bool = True
    use_ensemble: bool = True


class Settings(BaseSettings):
    """
    Root application settings.

    Environment variables example:
    ANOMALY_POSTGRES__HOST=pg.internal
    ANOMALY_POSTGRES__PASSWORD=secret
    ANOMALY_IFOREST__CONTAMINATION=0.0005
    """

    model_config = SettingsConfigDict(
        env_prefix="ANOMALY_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "financial-anomaly-detection"
    environment: str = "dev"

    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    clickhouse: ClickHouseConfig = Field(default_factory=ClickHouseConfig)

    iforest: IsolationForestParams = Field(default_factory=IsolationForestParams)
    lstm: LSTMAutoencoderParams = Field(default_factory=LSTMAutoencoderParams)
    dbscan: DBSCANParams = Field(default_factory=DBSCANParams)

    features: FeatureWindowConfig = Field(default_factory=FeatureWindowConfig)
    calibration: ThresholdCalibrationConfig = Field(
        default_factory=ThresholdCalibrationConfig
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    feature_store_dir: Path = Path("feature_store")
    report_dir: Path = Path("reports")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_settings(config_path: Optional[str] = None) -> Settings:
    """
    Load settings from environment first, optionally overlaying a YAML file.
    """
    if config_path and Path(config_path).exists():
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return Settings(**data)

    return get_settings()
