import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.environ.get(
                "CONFIG_PATH", str(Path(__file__).parent.parent.parent / "config.yaml")
            )
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = {}

    def reload(self) -> None:
        self._load_config()

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        env_key = key.upper().replace(".", "_")
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value
        return value

    @property
    def kalshi(self) -> Dict[str, Any]:
        return self._config.get("kalshi", {})

    @property
    def trading(self) -> Dict[str, Any]:
        return self._config.get("trading", {})

    @property
    def orderbook(self) -> Dict[str, Any]:
        return self._config.get("orderbook", {})

    @property
    def monitoring(self) -> Dict[str, Any]:
        return self._config.get("monitoring", {})

    @property
    def risk(self) -> Dict[str, Any]:
        return self._config.get("risk", {})

    @property
    def database(self) -> Dict[str, Any]:
        return self._config.get("database", {})

    @property
    def redis(self) -> Dict[str, Any]:
        return self._config.get("redis", {})

    @property
    def paper_mode(self) -> bool:
        return self.get("trading.paper_mode", True)

    @property
    def arbitrage_threshold(self) -> float:
        return float(self.get("trading.arbitrage_threshold", 0.99))

    @property
    def min_profit_cents(self) -> int:
        return int(self.get("trading.min_profit_cents", 10))

    @property
    def max_position_contracts(self) -> int:
        return int(self.get("trading.max_position_contracts", 1000))

    @property
    def scan_interval_seconds(self) -> float:
        return float(self.get("monitoring.scan_interval_seconds", 1.0))

    @property
    def limited_risk(self) -> Dict[str, Any]:
        return self._config.get("limited_risk", {})

    @property
    def limited_risk_enabled(self) -> bool:
        return self.get("limited_risk.enabled", False)

    @property
    def limited_risk_min_cents(self) -> int:
        return int(self.get("limited_risk.min_trade_cents", 1000))

    @property
    def limited_risk_max_cents(self) -> int:
        return int(self.get("limited_risk.max_trade_cents", 1500))

    @property
    def limited_risk_auto_enable_balance_cents(self) -> int:
        return int(self.get("limited_risk.auto_enable_balance_cents", 100000))
