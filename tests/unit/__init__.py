import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

pytest_plugins = ['pytest_asyncio']


def test_orderbook_module():
    from src.core.orderbook import OrderBook, OrderBookLevel, OrderSide
    assert OrderBook is not None
    assert OrderBookLevel is not None
    assert OrderSide is not None


def test_arbitrage_module():
    from src.core.arbitrage import ArbitrageDetector, ArbitrageOpportunity, ArbitrageType
    assert ArbitrageDetector is not None
    assert ArbitrageOpportunity is not None
    assert ArbitrageType is not None


def test_portfolio_module():
    from src.core.portfolio import PortfolioManager, Position, PositionSide
    assert PortfolioManager is not None
    assert Position is not None
    assert PositionSide is not None


def test_trading_module():
    from src.execution.trading import TradingExecutor, TradeResult, OrderStatus
    assert TradingExecutor is not None
    assert TradeResult is not None
    assert OrderStatus is not None


def test_kalshi_client():
    from src.clients.kalshi_client import KalshiClient
    assert KalshiClient is not None


def test_monitoring_module():
    from src.monitoring.monitoring import MonitoringSystem, MetricsCollector
    assert MonitoringSystem is not None
    assert MetricsCollector is not None


def test_utils_module():
    from src.utils.config import Config
    from src.utils.logging_utils import setup_logging
    assert Config is not None
    assert setup_logging is not None
