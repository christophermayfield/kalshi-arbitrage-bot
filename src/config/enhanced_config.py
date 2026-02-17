"""Enhanced configuration for high-frequency trading system."""

from typing import Dict, Any, Optional

# Enhanced configuration for high-frequency trading
CONFIG_TEMPLATES = {
    "conservative_hf": {
        "kalshi": {
            "api_key_id": "YOUR_API_KEY_ID",
            "private_key_path": "~/.kalshi/private-key.pem",
            "base_url": "https://demo-api.kalshi.co",
            "demo_mode": True,
        },
        "trading": {
            "paper_mode": True,
            "arbitrage_threshold": 0.99,
            "min_profit_cents": 5,  # Lower for HF
            "max_position_contracts": 100,  # Smaller for risk management
            "max_order_value_cents": 5000,
            "order_timeout_seconds": 10,  # Faster timeout
            "retry_attempts": 3,
            "retry_delay_seconds": 0.5,  # Faster retry
            "high_frequency": {
                "enabled": True,
                "max_concurrent_orders": 5,  # Concurrent order limit
                "order_timeout_ms": 3000,  # 3 second timeout
                "max_retries": 2,  # Fewer retries
                "retry_delay_ms": 50,
            },
        },
        "orderbook": {
            "min_liquidity_score": 70,  # Higher liquidity requirement
            "max_slippage_percent": 1.0,  # Lower slippage tolerance
            "min_fill_probability": 0.9,  # Higher fill probability
            "max_spread_percent": 3.0,
        },
        "risk": {
            "max_daily_loss_cents": 5000,  # Lower daily loss limit
            "max_open_positions": 10,  # Fewer positions
            "circuit_breaker_threshold": 3,
            "position_sizing": {
                "strategy": "kelly_fraction",
                "kelly_fraction": 0.15,  # Conservative Kelly
                "max_position_percent": 0.05,  # 5% max per position
            },
        },
        "monitoring": {
            "scan_interval_seconds": 0.1,  # 100ms for HF
            "log_level": "INFO",
            "notification_enabled": False,
            "metrics_port": 8000,
            "environment": "development",
            "performance_tracking": {
                "enabled": True,
                "latency_threshold_ms": 100,
                "success_rate_threshold": 0.8,
            },
        },
        "statistical": {
            "enabled": True,
            "strategies": ["mean_reversion"],
            "mean_reversion": {
                "z_threshold": 1.5,  # Lower threshold for more signals
                "min_profit_cents": 3,
                "max_volatility": 0.3,
                "lookback_period_days": 7,
            },
            "pairs_trading": {
                "enabled": False  # Conservative: no pairs trading
            },
        },
        "scanning": {
            "scan_interval_ms": 100,  # 100ms scanning
            "max_concurrent_scans": 10,  # Concurrent scans
            "cache_ttl_seconds": 3,  # Shorter cache for HF
            "opportunity_threshold": 4.0,  # Lower threshold
        },
        "scoring": {
            "profit_weight": 0.25,  # Lower profit weight for risk management
            "confidence_weight": 0.35,  # Higher confidence weight
            "liquidity_weight": 0.25,
            "volatility_weight": 0.1,
            "speed_weight": 0.05,  # Lower speed for conservative approach
            "risk_weight": 0.0,  # Conservative approach: risk not penalized
            "min_confidence": 0.7,
            "max_risk_score": 0.5,
            "min_liquidity_score": 70,
        },
        "auto_mode": False,  # Conservative: start in manual mode
        "cache": {
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 0,
            "essential_markets": [],
            "connection_pool_size": 10,
            "cache_ttl": {
                "orderbook": 30,
                "opportunities": 10,
                "portfolio": 60,
                "metrics": 5,
                "rates": 3600,
            },
        },
    },
    "moderate_hf": {
        "kalshi": {
            "api_key_id": "YOUR_API_KEY_ID",
            "private_key_path": "~/.kalshi/private-key.pem",
            "base_url": "https://demo-api.kalshi.co",
            "demo_mode": True,
        },
        "trading": {
            "paper_mode": True,
            "arbitrage_threshold": 0.95,
            "min_profit_cents": 10,
            "max_position_contracts": 500,
            "max_order_value_cents": 10000,
            "order_timeout_seconds": 15,
            "retry_attempts": 3,
            "retry_delay_seconds": 1,
            "high_frequency": {
                "enabled": True,
                "max_concurrent_orders": 10,  # Moderate concurrent orders
                "order_timeout_ms": 5000,
                "max_retries": 3,
                "retry_delay_ms": 100,
            },
        },
        "orderbook": {
            "min_liquidity_score": 50,
            "max_slippage_percent": 2.0,
            "min_fill_probability": 0.8,
            "max_spread_percent": 5.0,
        },
        "risk": {
            "max_daily_loss_cents": 10000,
            "max_open_positions": 30,
            "circuit_breaker_threshold": 5,
            "position_sizing": {
                "strategy": "volatility_target",
                "base_fraction": 0.1,
                "volatility_target": 0.02,
            },
        },
        "monitoring": {
            "scan_interval_seconds": 0.5,  # 500ms
            "log_level": "INFO",
            "notification_enabled": False,
            "metrics_port": 8000,
            "environment": "development",
            "performance_tracking": {
                "enabled": True,
                "latency_threshold_ms": 200,
                "success_rate_threshold": 0.7,
            },
        },
        "statistical": {
            "enabled": True,
            "strategies": ["mean_reversion", "pairs_trading"],
            "mean_reversion": {
                "z_threshold": 2.0,
                "min_profit_cents": 10,
                "max_volatility": 0.5,
                "lookback_period_days": 14,
            },
            "pairs_trading": {
                "enabled": True,
                "min_correlation": 0.7,
                "min_profit_cents": 15,
                "hedge_ratio_lookback": 30,
            },
        },
        "scanning": {
            "scan_interval_ms": 500,  # 500ms scanning
            "max_concurrent_scans": 15,
            "cache_ttl_seconds": 5,
            "opportunity_threshold": 5.0,
        },
        "scoring": {
            "profit_weight": 0.3,
            "confidence_weight": 0.25,
            "liquidity_weight": 0.2,
            "volatileity_weight": 0.1,
            "speed_weight": 0.1,
            "risk_weight": 0.05,
            "min_confidence": 0.6,
            "max_risk_score": 0.8,
            "min_liquidity_score": 50,
        },
        "auto_mode": False,  # Start in manual mode
        "cache": {
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 0,
            "essential_markets": [],
            "connection_pool_size": 15,
            "cache_ttl": {
                "orderbook": 60,
                "opportunities": 30,
                "portfolio": 120,
                "metrics": 10,
                "rates": 7200,
            },
        },
    },
    "aggressive_hf": {
        "kalshi": {
            "api_key_id": "YOUR_API_KEY_ID",
            "private_key_path": "~/.kalshi/private-key.pem",
            "base_url": "https://demo-api.kalshi.co",
            "demo_mode": False,  # More realistic setting
        },
        "trading": {
            "paper_mode": True,  # Still paper for safety
            "arbitrage_threshold": 0.9,  # Lower threshold
            "min_profit_cents": 2,  # Very low profit threshold
            "max_position_contracts": 2000,  # Higher position size
            "max_order_value_cents": 25000,
            "order_timeout_seconds": 10,  # Faster timeout
            "retry_attempts": 2,  # Fewer retries for speed
            "retry_delay_seconds": 0.2,  # Very fast retry
            "high_frequency": {
                "enabled": True,
                "max_concurrent_orders": 20,  # High concurrency
                "order_timeout_ms": 2000,  # 2 second timeout
                "max_retries": 2,
                "retry_delay_ms": 25,  # Very fast retry
            },
        },
        "orderbook": {
            "min_liquidity_score": 30,  # Lower liquidity requirement
            "max_slippage_percent": 5.0,  # Higher slippage tolerance
            "min_fill_probability": 0.6,  # Lower fill probability
            "max_spread_percent": 10.0,
        },
        "risk": {
            "max_daily_loss_cents": 50000,  # Higher risk tolerance
            "max_open_positions": 100,  # More positions
            "circuit_breaker_threshold": 10,
            "position_sizing": {
                "strategy": "fixed_fraction",
                "fraction": 0.2,  # 20% position sizing
                "max_size": 500,
            },
        },
        "monitoring": {
            "scan_interval_seconds": 0.05,  # 50ms scanning
            "log_level": "DEBUG",  # More verbose logging
            "notification_enabled": True,
            "metrics_port": 8000,
            "environment": "development",
            "performance_tracking": {
                "enabled": True,
                "latency_threshold_ms": 50,  # Very low latency threshold
                "success_rate_threshold": 0.6,
            },
        },
        "statistical": {
            "enabled": True,
            "strategies": [
                "mean_reversion",
                "pairs_trading",
                "triangular",
            ],  # Add triangular
            "mean_reversion": {
                "z_threshold": 1.0,  # Very low threshold
                "min_profit_cents": 2,
                "max_volatility": 0.8,
                "lookback_period_days": 5,
            },
            "pairs_trading": {
                "enabled": True,
                "min_correlation": 0.5,  # Lower correlation threshold
                "min_profit_cents": 10,
                "hedge_ratio_lookback": 20,
            },
        },
        "scanning": {
            "scan_interval_ms": 50,  # 50ms ultra-fast scanning
            "max_concurrent_scans": 25,
            "cache_ttl_seconds": 2,  # Very fast cache
            "opportunity_threshold": 3.0,  # Very low threshold
        },
        "scoring": {
            "profit_weight": 0.4,  # Higher profit weight
            "confidence_weight": 0.2,  # Lower confidence weight
            "liquidity_weight": 0.15,
            "volatileity_weight": 0.05,  # Lower volatility weight
            "speed_weight": 0.15,  # Higher speed weight
            "risk_weight": 0.05,
            "min_confidence": 0.5,  # Lower confidence threshold
            "max_risk_score": 1.2,  # Higher risk tolerance
            "min_liquidity_score": 30,
        },
        "auto_mode": True,  # Start in auto mode for HF
        "cache": {
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 0,
            "essential_markets": [],
            "connection_pool_size": 25,
            "cache_ttl": {
                "orderbook": 30,
                "opportunities": 15,
                "portfolio": 60,
                "metrics": 3,
                "rates": 1800,
            },
        },
    },
}


def create_enhanced_config(strategy_type: str = "moderate_hf") -> Dict[str, Any]:
    """Create enhanced configuration for high-frequency trading."""
    return CONFIG_TEMPLATES[strategy_type]


def get_available_strategies() -> list:
    """Get list of available strategy templates."""
    return list(CONFIG_TEMPLATES.keys())


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration and return status."""
    issues = []

    # Check required fields
    if "kalshi" not in config:
        issues.append("Missing kalshi configuration")

    if "trading" not in config:
        issues.append("Missing trading configuration")

    if "monitoring" not in config:
        issues.append("Missing monitoring configuration")

    # Validate trading parameters
    if "trading" in config:
        trading = config["trading"]

        if trading.get("min_profit_cents", 0) <= 0:
            issues.append("min_profit_cents must be > 0")

        if trading.get("max_position_contracts", 0) <= 0:
            issues.append("max_position_contracts must be > 0")

        if "high_frequency" in trading:
            hf = trading["high_frequency"]

            if hf.get("max_concurrent_orders", 0) <= 0:
                issues.append("max_concurrent_orders must be > 0")

            if hf.get("order_timeout_ms", 0) <= 100:
                issues.append("order_timeout_ms should be > 100ms for safety")

    # Validate scanning parameters
    if "scanning" in config:
        scanning = config["scanning"]

        if scanning.get("scan_interval_ms", 0) < 50:
            issues.append("scan_interval_ms should be >= 50ms for API rate limits")

        if scanning.get("max_concurrent_scans", 0) <= 0:
            issues.append("max_concurrent_scans must be > 0")

    # Validate cache parameters
    if "cache" in config:
        cache = config["cache"]

        if "redis_host" not in cache:
            issues.append("Missing redis_host in cache configuration")

        if "redis_port" not in cache:
            issues.append("Missing redis_port in cache configuration")

    return {"valid": len(issues) == 0, "issues": issues}


def get_cost_estimate(config: Dict[str, Any]) -> Dict[str, float]:
    """Get estimated monthly cost for configuration."""
    # Base costs (low-end cloud providers)
    base_costs = {
        "bot_instance": 20.0,  # $20/month for small instance
        "storage": 10.0,  # $10/month for database
        "cache": 15.0,  # $15/month for Redis
        "monitoring": 10.0,  # $10/month for metrics
        "bandwidth": 20.0,  # $20/month for API calls
    }

    total_cost = sum(base_costs.values())

    # Adjustments based on configuration
    multipliers = {
        "high_frequency_trading": 1.5,  # 50% more for HF trading
        "statistical_arbitrage": 1.3,  # 30% more for statistical strategies
        "enhanced_monitoring": 1.2,  # 20% more for advanced monitoring
        "auto_mode": 1.1,  # 10% more for auto trading
    }

    if "trading" in config:
        trading = config["trading"]
        if trading.get("high_frequency", {}).get("enabled", False):
            total_cost *= multipliers["high_frequency_trading"]

    if "statistical" in config:
        statistical = config["statistical"]
        if statistical.get("enabled", False):
            total_cost *= multipliers["statistical_arbitrage"]

    if "monitoring" in config:
        monitoring = config["monitoring"]
        if monitoring.get("performance_tracking", {}).get("enabled", False):
            total_cost *= multipliers["enhanced_monitoring"]

    if config.get("auto_mode", False):
        total_cost *= multipliers["auto_mode"]

    return {
        "monthly_cost": total_cost,
        "base_costs": base_costs,
        "multipliers": multipliers,
        "within_budget": total_cost <= 300,
        "monthly_savings": max(0, 300 - total_cost),
    }


def create_custom_config(
    scan_interval_ms: int = 500,
    min_profit_cents: int = 10,
    max_concurrent_orders: int = 10,
    statistical_enabled: bool = True,
    auto_mode: bool = False,
) -> Dict[str, Any]:
    """Create custom configuration for specific requirements."""
    base_config = CONFIG_TEMPLATES["moderate_hf"].copy()

    # Override custom settings
    base_config["monitoring"]["scan_interval_seconds"] = scan_interval_ms / 1000
    base_config["trading"]["min_profit_cents"] = min_profit_cents
    base_config["trading"]["high_frequency"]["max_concurrent_orders"] = (
        max_concurrent_orders
    )
    base_config["statistical"]["enabled"] = statistical_enabled
    base_config["auto_mode"] = auto_mode

    # Adjust other settings for better HF performance
    base_config["trading"]["order_timeout_seconds"] = 12
    base_config["scanning"]["scan_interval_ms"] = scan_interval_ms
    base_config["cache"]["connection_pool_size"] = max_concurrent_orders + 5

    return base_config


def optimize_for_budget(max_budget: float = 300) -> str:
    """Optimize configuration for given budget."""
    costs = {}

    for strategy_name, template in CONFIG_TEMPLATES.items():
        cost_est = get_cost_estimate(template)
        if cost_est["within_budget"]:
            costs[strategy_name] = cost_est

    if costs:
        # Find the best performance/price ratio
        best_strategy = max(
            costs.keys(),
            key=lambda k: (
                costs[k]["monthly_cost"] <= max_budget,
                costs[k]["monthly_savings"],
            ),
        )

        return best_strategy
    else:
        return "conservative_hf"  # Fallback to conservative


# Export configuration utilities
__all__ = [
    "create_enhanced_config",
    "get_available_strategies",
    "validate_config",
    "get_cost_estimate",
    "create_custom_config",
    "optimize_for_budget",
    "CONFIG_TEMPLATES",
]
