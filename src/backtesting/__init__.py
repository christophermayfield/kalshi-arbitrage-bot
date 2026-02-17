"""
Backtesting module for advanced strategy validation and parameter optimization
"""

from .advanced_backtesting import (
    BacktestConfig,
    BacktestResults,
    BacktestMode,
    OptimizationMethod,
    Trade,
    create_backtest_config,
    create_parameter_grid,
    run_comprehensive_backtest,
)
from src.utils.logging_utils import get_logger
from src.utils.config import Config

__all__ = [
    "BacktestConfig",
    "BacktestResults",
    "BacktestMode",
    "OptimizationMethod",
    "Trade",
    "create_backtest_config",
    "create_parameter_grid",
    "run_comprehensive_backtest",
]
