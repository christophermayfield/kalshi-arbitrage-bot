"""Mock Kalshi API for integration testing."""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
from unittest.mock import AsyncMock, MagicMock


@dataclass
class MockOrderBookLevel:
    """Mock order book level."""
    price: int
    count: int
    total: int


@dataclass
class MockOrderBook:
    """Mock order book response."""
    market_id: str
    bids: List[MockOrderBookLevel]
    asks: List[MockOrderBookLevel]
    timestamp: str


@dataclass
class MockOrder:
    """Mock order response."""
    order_id: str
    market_id: str
    side: str  # "BUY" or "SELL"
    price: int
    quantity: int
    status: str  # "PENDING", "FILLED", "PARTIAL", "CANCELLED"
    filled_quantity: int = 0
    created_at: str = None
    filled_at: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class MockMarket:
    """Mock market response."""
    market_id: str
    title: str
    category: str
    status: str  # "OPEN", "CLOSED", "HALTED"
    bid_price: int
    ask_price: int
    last_price: int
    open_time: str
    close_time: str
    liquidity: int


class MockKalshiAPI:
    """Mock Kalshi API for testing."""

    def __init__(self):
        self.orders: Dict[str, MockOrder] = {}
        self.positions: Dict[str, int] = {}  # market_id -> quantity
        self.balance_cents = 1000000  # $10,000 starting balance
        self.markets: Dict[str, MockMarket] = {}
        self.orderbooks: Dict[str, MockOrderBook] = {}
        self._setup_default_markets()
        self._order_counter = 0

    def _setup_default_markets(self):
        """Set up default test markets."""
        self.markets = {
            "market_001": MockMarket(
                market_id="market_001",
                title="Will BTC reach $100K by end of 2026?",
                category="cryptocurrency",
                status="OPEN",
                bid_price=7500,
                ask_price=7600,
                last_price=7550,
                open_time=datetime.now(timezone.utc).isoformat(),
                close_time="2026-12-31T23:59:59Z",
                liquidity=500000,
            ),
            "market_002": MockMarket(
                market_id="market_002",
                title="Will SPX close above 6000 on 2026-12-31?",
                category="stocks",
                status="OPEN",
                bid_price=6200,
                ask_price=6300,
                last_price=6250,
                open_time=datetime.now(timezone.utc).isoformat(),
                close_time="2026-12-31T23:59:59Z",
                liquidity=800000,
            ),
        }

        # Set up default orderbooks
        for market_id in self.markets.keys():
            self.orderbooks[market_id] = self._create_orderbook(market_id)

    def _create_orderbook(self, market_id: str) -> MockOrderBook:
        """Create a mock orderbook for a market."""
        market = self.markets.get(market_id)
        if not market:
            return None

        bid_price = market.bid_price
        ask_price = market.ask_price

        return MockOrderBook(
            market_id=market_id,
            bids=[
                MockOrderBookLevel(price=bid_price, count=100, total=bid_price * 100),
                MockOrderBookLevel(price=bid_price - 10, count=50, total=(bid_price - 10) * 50),
            ],
            asks=[
                MockOrderBookLevel(price=ask_price, count=100, total=ask_price * 100),
                MockOrderBookLevel(price=ask_price + 10, count=50, total=(ask_price + 10) * 50),
            ],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def get_markets(self, status: str = "open") -> List[Dict[str, Any]]:
        """Get markets by status."""
        markets = [
            m for m in self.markets.values()
            if status.lower() == "all" or m.status.lower() == status.lower()
        ]
        return [asdict(m) for m in markets]

    async def get_market_orderbook(self, market_id: str) -> Dict[str, Any]:
        """Get orderbook for a market."""
        orderbook = self.orderbooks.get(market_id)
        if not orderbook:
            raise ValueError(f"Market {market_id} not found")

        return {
            "market_id": orderbook.market_id,
            "bids": [asdict(b) for b in orderbook.bids],
            "asks": [asdict(b) for b in orderbook.asks],
            "timestamp": orderbook.timestamp,
        }

    async def create_order(
        self,
        market_id: str,
        side: str,
        order_type: str,
        price: int,
        count: int,
    ) -> Dict[str, Any]:
        """Create an order."""
        # Validation
        if market_id not in self.markets:
            raise ValueError(f"Market {market_id} not found")

        if side not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid side: {side}")

        if count <= 0:
            raise ValueError("Count must be positive")

        # Check balance for BUY orders
        if side == "BUY":
            cost = price * count
            if cost > self.balance_cents:
                raise ValueError("Insufficient balance")

        # Create order
        self._order_counter += 1
        order_id = f"order_{self._order_counter}"

        order = MockOrder(
            order_id=order_id,
            market_id=market_id,
            side=side,
            price=price,
            quantity=count,
            status="PENDING",
            filled_quantity=0,
        )

        self.orders[order_id] = order

        # Simulate immediate fill for market orders
        if order_type == "MARKET":
            await self._fill_order(order_id, count)

        return asdict(self.orders[order_id])

    async def _fill_order(self, order_id: str, filled_quantity: int):
        """Simulate order fill."""
        order = self.orders[order_id]

        order.filled_quantity = filled_quantity
        order.status = "FILLED" if filled_quantity == order.quantity else "PARTIAL"
        order.filled_at = datetime.now(timezone.utc).isoformat()

        # Update balance and positions
        cost = order.price * filled_quantity
        if order.side == "BUY":
            self.balance_cents -= cost
            self.positions[order.market_id] = self.positions.get(order.market_id, 0) + filled_quantity
        else:  # SELL
            self.balance_cents += cost
            self.positions[order.market_id] = self.positions.get(order.market_id, 0) - filled_quantity

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")

        order = self.orders[order_id]
        if order.status in ["FILLED", "CANCELLED"]:
            raise ValueError(f"Cannot cancel order with status {order.status}")

        order.status = "CANCELLED"
        return asdict(order)

    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order details."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")

        return asdict(self.orders[order_id])

    async def get_portfolio(self) -> Dict[str, Any]:
        """Get portfolio (balance and positions)."""
        return {
            "balance_cents": self.balance_cents,
            "positions": self.positions,
            "total_value_cents": self.balance_cents + sum(
                self.markets[mid].last_price * qty
                for mid, qty in self.positions.items()
                if mid in self.markets
            ),
        }

    # Simulation methods for testing
    def simulate_price_movement(self, market_id: str, price_change: int):
        """Simulate price movement in a market."""
        if market_id not in self.markets:
            return

        market = self.markets[market_id]
        market.last_price += price_change
        market.bid_price += price_change
        market.ask_price += price_change

        self.orderbooks[market_id] = self._create_orderbook(market_id)

    def simulate_order_fill(self, order_id: str, filled_quantity: int):
        """Simulate order fill in background."""
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        order.filled_quantity = min(filled_quantity, order.quantity)
        order.status = "FILLED" if order.filled_quantity == order.quantity else "PARTIAL"
        order.filled_at = datetime.now(timezone.utc).isoformat()

    def reset(self):
        """Reset mock state for next test."""
        self.orders.clear()
        self.positions.clear()
        self.balance_cents = 1000000
        self._order_counter = 0
        self._setup_default_markets()


# Pytest fixtures
import pytest


@pytest.fixture
def mock_kalshi_api():
    """Provide mock Kalshi API for tests."""
    api = MockKalshiAPI()
    yield api
    api.reset()


@pytest.fixture
async def mock_kalshi_client(mock_kalshi_api):
    """Provide async mock Kalshi client."""
    from unittest.mock import AsyncMock

    client = AsyncMock()
    client.get_markets = mock_kalshi_api.get_markets
    client.get_market_orderbook = mock_kalshi_api.get_market_orderbook
    client.create_order = mock_kalshi_api.create_order
    client.cancel_order = mock_kalshi_api.cancel_order
    client.get_order = mock_kalshi_api.get_order
    client.get_portfolio = mock_kalshi_api.get_portfolio

    return client
