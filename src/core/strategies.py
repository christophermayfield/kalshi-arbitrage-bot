"""Example strategy presets for different trading approaches."""

from typing import Any, Dict

STRATEGIES = {
    "conservative": {
        "name": "Conservative",
        "description": "Low-risk strategy with small position sizes and high confidence requirements",
        "trading": {
            "paper_mode": True,
            "arbitrage_threshold": 0.99,
            "min_profit_cents": 50,
            "max_position_contracts": 100,
            "max_order_value_cents": 1000,
        },
        "orderbook": {
            "min_liquidity_score": 75,
            "max_slippage_percent": 1.0,
            "min_fill_probability": 0.9,
            "max_spread_percent": 3.0,
        },
        "risk": {
            "max_daily_loss_cents": 5000,
            "max_open_positions": 10,
            "circuit_breaker_threshold": 3,
        },
        "position_sizing": {
            "strategy": "kelly",
            "kelly_fraction": 0.25,
            "max_position_percent": 0.1,
        },
        "api": {
            "rate_limit_rps": 2.0,
            "rate_limit_rpm": 100.0,
        }
    },

    "moderate": {
        "name": "Moderate",
        "description": "Balanced strategy with medium risk/reward tradeoff",
        "trading": {
            "paper_mode": True,
            "arbitrage_threshold": 0.95,
            "min_profit_cents": 20,
            "max_position_contracts": 500,
            "max_order_value_cents": 5000,
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
        },
        "position_sizing": {
            "strategy": "volatility",
            "base_fraction": 0.1,
            "volatility_target": 0.02,
        },
        "api": {
            "rate_limit_rps": 5.0,
            "rate_limit_rpm": 300.0,
        }
    },

    "aggressive": {
        "name": "Aggressive",
        "description": "High-risk strategy seeking maximum profits",
        "trading": {
            "paper_mode": False,
            "arbitrage_threshold": 0.8,
            "min_profit_cents": 5,
            "max_position_contracts": 2000,
            "max_order_value_cents": 20000,
        },
        "orderbook": {
            "min_liquidity_score": 30,
            "max_slippage_percent": 5.0,
            "min_fill_probability": 0.6,
            "max_spread_percent": 10.0,
        },
        "risk": {
            "max_daily_loss_cents": 50000,
            "max_open_positions": 100,
            "circuit_breaker_threshold": 10,
        },
        "position_sizing": {
            "strategy": "fixed_fraction",
            "fraction": 0.2,
            "max_size": 200,
        },
        "api": {
            "rate_limit_rps": 10.0,
            "rate_limit_rpm": 600.0,
        }
    },

    "scalper": {
        "name": "Scalper",
        "description": "High-frequency strategy for small, frequent profits",
        "trading": {
            "paper_mode": True,
            "arbitrage_threshold": 0.85,
            "min_profit_cents": 2,
            "max_position_contracts": 50,
            "max_order_value_cents": 500,
        },
        "orderbook": {
            "min_liquidity_score": 80,
            "max_slippage_percent": 0.5,
            "min_fill_probability": 0.95,
            "max_spread_percent": 2.0,
        },
        "risk": {
            "max_daily_loss_cents": 2000,
            "max_open_positions": 5,
            "circuit_breaker_threshold": 2,
        },
        "position_sizing": {
            "strategy": "fixed_fraction",
            "fraction": 0.05,
            "max_size": 50,
        },
        "api": {
            "rate_limit_rps": 20.0,
            "rate_limit_rpm": 1200,
        },
        "monitoring": {
            "scan_interval_seconds": 0.1,
        }
    },

    "swing": {
        "name": "Swing Trader",
        "description": "Lower frequency strategy for larger moves",
        "trading": {
            "paper_mode": True,
            "arbitrage_threshold": 0.98,
            "min_profit_cents": 200,
            "max_position_contracts": 500,
            "max_order_value_cents": 50000,
        },
        "orderbook": {
            "min_liquidity_score": 60,
            "max_slippage_percent": 1.5,
            "min_fill_probability": 0.85,
            "max_spread_percent": 4.0,
        },
        "risk": {
            "max_daily_loss_cents": 20000,
            "max_open_positions": 5,
            "circuit_breaker_threshold": 3,
        },
        "position_sizing": {
            "strategy": "kelly",
            "kelly_fraction": 0.5,
            "max_position_percent": 0.25,
        },
        "api": {
            "rate_limit_rps": 1.0,
            "rate_limit_rpm": 60.0,
        },
        "monitoring": {
            "scan_interval_seconds": 5.0,
        }
    },

    "ml_enhanced": {
        "name": "ML Enhanced",
        "description": "Strategy using machine learning for position sizing",
        "trading": {
            "paper_mode": True,
            "arbitrage_threshold": 0.9,
            "min_profit_cents": 10,
            "max_position_contracts": 1000,
            "max_order_value_cents": 10000,
        },
        "orderbook": {
            "min_liquidity_score": 50,
            "max_slippage_percent": 2.0,
            "min_fill_probability": 0.8,
            "max_spread_percent": 5.0,
        },
        "risk": {
            "max_daily_loss_cents": 15000,
            "max_open_positions": 50,
            "circuit_breaker_threshold": 5,
        },
        "position_sizing": {
            "strategy": "ml",
            "model_path": "models/position_sizer.joblib",
        },
        "api": {
            "rate_limit_rps": 5.0,
            "rate_limit_rpm": 300.0,
        }
    }
}


def get_strategy_names() -> list:
    return list(STRATEGIES.keys())


def get_strategy(name: str) -> Dict[str, Any]:
    return STRATEGIES.get(name, {})


def apply_strategy(config: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    strategy = get_strategy(strategy_name)
    if not strategy:
        return config

    merged = config.copy()

    for section, values in strategy.items():
        if section in ['name', 'description']:
            continue
        if section in merged:
            merged[section].update(values)
        else:
            merged[section] = values

    return merged


def create_config_from_strategy(strategy_name: str) -> Dict[str, Any]:
    template = {
        'kalshi': {
            'api_key_id': 'YOUR_API_KEY_ID',
            'private_key_path': '~/.kalshi/private-key.pem',
            'base_url': 'https://demo-api.kalshi.co',
            'demo_mode': True
        },
        'trading': {},
        'orderbook': {},
        'risk': {},
        'api': {},
        'monitoring': {
            'scan_interval_seconds': 1.0,
            'log_level': 'INFO'
        }
    }

    return apply_strategy(template, strategy_name)
