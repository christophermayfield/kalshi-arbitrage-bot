"""
Multi-Exchange Client - Unified interface for multiple prediction markets.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from src.utils.logging_utils import get_logger

logger = get_logger("multi_exchange")


@dataclass
class MarketData:
    """Unified market data format."""

    exchange: str
    market_id: str
    event_id: str
    title: str
    status: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    volume: int
    liquidity: int
    last_update: datetime


@dataclass
class OrderBook:
    """Unified order book format."""

    market_id: str
    bids: List[tuple[float, int]]  # (price, size)
    asks: List[tuple[float, int]]
    timestamp: datetime


class ExchangeClient(ABC):
    """Base class for exchange clients."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)

    @abstractmethod
    async def connect(self) -> bool:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def get_markets(self, status: str = "open") -> List[MarketData]:
        pass

    @abstractmethod
    async def get_orderbook(self, market_id: str) -> Optional[OrderBook]:
        pass

    @abstractmethod
    async def create_order(
        self, market_id: str, side: str, order_type: str, price: float, count: int
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    async def get_balance(self) -> float:
        pass

    @abstractmethod
    async def get_positions(self) -> Dict[str, Any]:
        pass


class PolymarketClient(ExchangeClient):
    """Polymarket API client."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("api_base_url", "https://api.polymarket.com")
        self._session = None

    async def connect(self) -> bool:
        if not self.enabled:
            return False
        logger.info("Connecting to Polymarket...")
        self._session = asyncio.Session()
        return True

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()

    async def get_markets(self, status: str = "open") -> List[MarketData]:
        # Implementation would call Polymarket API
        return []

    async def get_orderbook(self, market_id: str) -> Optional[OrderBook]:
        return None

    async def create_order(
        self, market_id: str, side: str, order_type: str, price: float, count: int
    ) -> Dict[str, Any]:
        return {"success": False, "error": "Not implemented"}

    async def cancel_order(self, order_id: str) -> bool:
        return False

    async def get_balance(self) -> float:
        return 0.0

    async def get_positions(self) -> Dict[str, Any]:
        return {}


class PredictItClient(ExchangeClient):
    """PredictIt API client."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("api_base_url", "https://www.predictit.com/api")

    async def connect(self) -> bool:
        if not self.enabled:
            return False
        logger.info("Connecting to PredictIt...")
        return True

    async def disconnect(self) -> None:
        pass

    async def get_markets(self, status: str = "open") -> List[MarketData]:
        # PredictIt has limited API - mostly market data
        return []

    async def get_orderbook(self, market_id: str) -> Optional[OrderBook]:
        return None

    async def create_order(
        self, market_id: str, side: str, order_type: str, price: float, count: int
    ) -> Dict[str, Any]:
        return {"success": False, "error": "PredictIt trading not supported via API"}

    async def cancel_order(self, order_id: str) -> bool:
        return False

    async def get_balance(self) -> float:
        return 0.0

    async def get_positions(self) -> Dict[str, Any]:
        return {}


class MultiExchangeManager:
    """Manages multiple exchange clients."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchanges: Dict[str, ExchangeClient] = {}
        self._initialize_exchanges()

    def _initialize_exchanges(self) -> None:
        exchange_configs = self.config.get("exchanges", {})

        for name, exch_config in exchange_configs.items():
            if not exch_config.get("enabled", False):
                continue

            if name == "polymarket":
                self.exchanges[name] = PolymarketClient(exch_config)
            elif name == "predictit":
                self.exchanges[name] = PredictItClient(exch_config)
            else:
                logger.warning(f"Unknown exchange: {name}")

    async def connect_all(self) -> None:
        for name, client in self.exchanges.items():
            try:
                await client.connect()
            except Exception as e:
                logger.error(f"Failed to connect to {name}: {e}")

    async def disconnect_all(self) -> None:
        for client in self.exchanges.values():
            try:
                await client.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")

    async def get_all_markets(self) -> List[MarketData]:
        """Get markets from all enabled exchanges."""
        all_markets = []

        for name, client in self.exchanges.items():
            if client.enabled:
                try:
                    markets = await client.get_markets()
                    all_markets.extend(markets)
                except Exception as e:
                    logger.error(f"Error getting markets from {name}: {e}")

        return all_markets

    def get_enabled_exchanges(self) -> List[str]:
        return [name for name, c in self.exchanges.items() if c.enabled]
