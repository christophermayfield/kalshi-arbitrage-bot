from pydantic import BaseModel, Field, validator
from typing import Any, Dict, List, Optional
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class KalshiConfig(BaseModel):
    api_key_id: str = Field(..., description="Kalshi API key ID")
    private_key_path: str = Field(..., description="Path to RSA private key")
    base_url: str = Field("https://demo-api.kalshi.co", description="API base URL")
    demo_mode: bool = Field(True, description="Use demo environment")


class TradingConfig(BaseModel):
    paper_mode: bool = Field(True, description="Paper trading mode")
    arbitrage_threshold: float = Field(0.99, ge=0, le=1, description="Min confidence threshold")
    min_profit_cents: int = Field(10, ge=0, description="Minimum profit in cents")
    max_position_contracts: int = Field(1000, ge=1, description="Max contracts per position")
    max_order_value_cents: int = Field(10000, ge=1, description="Max order value")
    order_timeout_seconds: int = Field(30, ge=1, description="Order timeout")
    retry_attempts: int = Field(3, ge=0, description="Retry attempts")
    retry_delay_seconds: float = Field(1.0, ge=0, description="Retry delay")


class OrderbookConfig(BaseModel):
    min_liquidity_score: int = Field(50, ge=0, le=100)
    max_slippage_percent: float = Field(2.0, ge=0)
    min_fill_probability: float = Field(0.8, ge=0, le=1)
    max_spread_percent: float = Field(5.0, ge=0)


class MonitoringConfig(BaseModel):
    scan_interval_seconds: float = Field(1.0, ge=0.1)
    log_level: LogLevel = Field(LogLevel.INFO)
    notification_enabled: bool = Field(False)
    notification_webhook: Optional[str] = None
    metrics_port: int = Field(8000, ge=1, le=65535)
    environment: str = Field("development")


class RiskConfig(BaseModel):
    max_daily_loss_cents: int = Field(10000, ge=0)
    max_open_positions: int = Field(50, ge=1)
    circuit_breaker_threshold: int = Field(5, ge=1)
    circuit_breaker_window_seconds: int = Field(300, ge=1)


class MarketFiltersConfig(BaseModel):
    min_volume: int = Field(100, ge=0)
    min_liquidity: int = Field(50, ge=0)
    status: str = Field("open")
    exclude_closely_predicted: bool = Field(False)


class APIConfig(BaseModel):
    rate_limit_rps: float = Field(5.0, ge=0)
    rate_limit_rpm: float = Field(300.0, ge=0)
    rate_limit_burst: int = Field(10, ge=1)


class DatabaseConfig(BaseModel):
    path: str = Field("data/arbitrage.db")
    cleanup_days: int = Field(30, ge=1)


class RedisConfig(BaseModel):
    enabled: bool = Field(False)
    host: str = Field("localhost")
    port: int = Field(6379, ge=1, le=65535)
    db: int = Field(0, ge=0)


class WebSocketConfig(BaseModel):
    enabled: bool = Field(False)
    reconnect_delay_seconds: float = Field(5.0, ge=1)
    heartbeat_interval_seconds: float = Field(30.0, ge=1)


class NotificationConfig(BaseModel):
    enabled: bool = Field(False)
    slack_webhook_url: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None


class Config(BaseModel):
    kalshi: KalshiConfig
    trading: TradingConfig
    orderbook: OrderbookConfig = Field(default_factory=OrderbookConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    market_filters: MarketFiltersConfig = Field(default_factory=MarketFiltersConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)

    class Config:
        arbitrary_types_allowed = True

    @validator('trading')
    def validate_trading_config(cls, v, values):
        if v.paper_mode is False:
            if not values.get('kalshi', {}).api_key_id:
                raise ValueError("API key required for live trading")
        return v

    @validator('redis')
    def validate_redis_config(cls, v):
        if v.enabled:
            if not v.host:
                raise ValueError("Redis host required when enabled")
        return v

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            elif hasattr(value, k):
                value = getattr(value, k)
            else:
                return default
            if value is None:
                return default
        return value


def load_config(config_dict: Dict[str, Any]) -> Config:
    return Config(**config_dict)


def validate_config(config_dict: Dict[str, Any]) -> tuple[bool, List[str]]:
    try:
        config = load_config(config_dict)
        return True, []
    except Exception as e:
        return False, [str(e)]
