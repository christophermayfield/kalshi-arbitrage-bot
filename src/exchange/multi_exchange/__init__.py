"""
Multi-exchange arbitrage infrastructure
Support for multiple prediction markets with unified arbitrage detection and execution
"""

from .multi_exchange import (
    ExchangeType,
    ExchangeConfig,
    MarketMapping,
    CrossExchangeOpportunity,
    ExchangeClient,
    KalshiExchangeClient,
    PolymarketExchangeClient,
    ExchangeManager,
    CrossExchangeArbitrageDetector,
    MultiExchangeArbitrageEngine,
    create_multi_exchange_engine
    execute_cross_exchange_arbitrage
)

__all__ = [
    'ExchangeType',
    'ExchangeConfig',
    'MarketMapping', 
    'CrossExchangeOpportunity',
    'ExchangeClient',
    'KalshiExchangeClient',
    'PolymarketExchangeClient',
    'ExchangeManager',
    'CrossExchangeArbitrageDetector',
    'MultiExchangeArbitrageEngine',
    'create_multi_exchange_engine',
    'execute_cross_exchange_arbitrage'
]