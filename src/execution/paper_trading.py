"""
Paper Trading Simulator - Simulates real market execution for testing strategies.
"""

import asyncio
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

from src.core.orderbook import OrderBook, OrderBookLevel, OrderSide
from src.utils.logging_utils import get_logger

logger = get_logger("paper_trading")


@dataclass
class PaperOrder:
    order_id: str
    market_id: str
    side: str  # "buy" or "sell"
    order_type: str  # "limit" or "market"
    price: int
    quantity: int
    filled_quantity: int = 0
    status: str = "pending"  # pending, submitted, partial, filled, cancelled, rejected
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None
    fill_price: Optional[int] = None


@dataclass
class PaperTrade:
    trade_id: str
    order_id: str
    market_id: str
    side: str
    price: int
    quantity: int
    timestamp: datetime


@dataclass
class PaperPosition:
    market_id: str
    quantity: int  # positive = long, negative = short
    avg_price: int
    realized_pnl: int = 0
    unrealized_pnl: int = 0


class PaperTradingSimulator:
    def __init__(
        self,
        initial_balance: int = 100000,  # $1000.00 in cents
        slippage_model: str = "fixed",  # fixed, random, volume_based
        slippage_rate: float = 0.001,  # 0.1% slippage
        fill_probability: float = 0.95,  # 95% fill rate
        commission_rate: float = 0.01,  # 1% commission
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.slippage_model = slippage_model
        self.slippage_rate = slippage_rate
        self.fill_probability = fill_probability
        self.commission_rate = commission_rate

        self.orders: Dict[str, PaperOrder] = {}
        self.trades: List[PaperTrade] = []
        self.positions: Dict[str, PaperPosition] = {}
        self.orderbooks: Dict[str, OrderBook] = {}

        self._order_counter = 0
        self._trade_counter = 0
        self._pending_fills: Dict[str, asyncio.Task] = {}

        self.stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "rejected_orders": 0,
            "total_volume": 0,
            "total_commission": 0,
            "total_slippage": 0,
        }

    def _generate_order_id(self) -> str:
        self._order_counter += 1
        return f"paper_{self._order_counter}_{int(datetime.utcnow().timestamp())}"

    def _generate_trade_id(self) -> str:
        self._trade_counter += 1
        return f"trade_{self._trade_counter}_{int(datetime.utcnow().timestamp())}"

    def update_orderbook(self, market_id: str, orderbook: OrderBook) -> None:
        """Update the simulated orderbook for a market."""
        self.orderbooks[market_id] = orderbook

    def calculate_slippage(self, price: int, side: str) -> int:
        """Calculate slippage based on configured model."""
        if self.slippage_model == "fixed":
            slippage = int(price * self.slippage_rate)
        elif self.slippage_model == "random":
            slippage = int(price * self.slippage_rate * random.uniform(0.5, 1.5))
        elif self.slippage_model == "volume_based":
            # Larger orders get more slippage
            slippage = int(price * self.slippage_rate * 1.5)
        else:
            slippage = 0

        if side == "buy":
            return price + slippage
        return price - slippage

    def calculate_fill_price(self, order: PaperOrder) -> Optional[int]:
        """Calculate the fill price based on orderbook state."""
        if order.market_id not in self.orderbooks:
            return None

        ob = self.orderbooks[order.market_id]

        if order.order_type == "market":
            # Market orders fill at best available price
            if order.side == "buy" and ob.asks:
                return ob.asks[0].price
            elif order.side == "sell" and ob.bids:
                return ob.bids[0].price
            return None

        # Limit orders: check if price is available
        if order.side == "buy" and ob.asks:
            best_ask = ob.asks[0].price
            if order.price >= best_ask:
                return best_ask
        elif order.side == "sell" and ob.bids:
            best_bid = ob.bids[0].price
            if order.price <= best_bid:
                return best_bid

        return None

    async def create_order(
        self,
        market_id: str,
        side: str,
        order_type: str,
        price: int,
        quantity: int,
    ) -> PaperOrder:
        """Create a new paper trading order."""
        order_id = self._generate_order_id()

        order = PaperOrder(
            order_id=order_id,
            market_id=market_id,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            status="submitted",
        )

        self.orders[order_id] = order
        self.stats["total_orders"] += 1

        # Try to fill immediately
        asyncio.create_task(self._process_order_fill(order))

        return order

    async def _process_order_fill(self, order: PaperOrder) -> None:
        """Process order fill asynchronously."""
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate network delay

        if order.status != "submitted":
            return

        # Check if order fills based on probability
        if random.random() > self.fill_probability:
            order.status = "rejected"
            self.stats["rejected_orders"] += 1
            logger.debug(f"Paper order rejected: {order.order_id}")
            return

        fill_price = self.calculate_fill_price(order)

        if fill_price is None:
            # Price not available - keep as pending
            logger.debug(f"Paper order {order.order_id} pending - price not available")
            return

        # Apply slippage
        fill_price = self.calculate_slippage(fill_price, order.side)

        # Calculate cost
        cost = fill_price * order.quantity
        commission = int(cost * self.commission_rate)

        # Check balance for buy orders
        if order.side == "buy" and self.balance < cost + commission:
            order.status = "rejected"
            self.stats["rejected_orders"] += 1
            logger.warning(f"Paper order rejected: insufficient balance")
            return

        # Execute fill
        order.status = "filled"
        order.filled_quantity = order.quantity
        order.fill_price = fill_price
        order.filled_at = datetime.utcnow()

        # Deduct balance for buys, add for sells
        if order.side == "buy":
            self.balance -= cost + commission
        else:
            self.balance += cost - commission

        # Record trade
        trade = PaperTrade(
            trade_id=self._generate_trade_id(),
            order_id=order.order_id,
            market_id=order.market_id,
            side=order.side,
            price=fill_price,
            quantity=order.quantity,
            timestamp=datetime.utcnow(),
        )
        self.trades.append(trade)

        # Update position
        self._update_position(order.market_id, order.side, order.quantity, fill_price)

        # Update stats
        self.stats["filled_orders"] += 1
        self.stats["total_volume"] += cost
        self.stats["total_commission"] += commission
        self.stats["total_slippage"] += abs(fill_price - order.price) * order.quantity

        logger.debug(f"Paper order filled: {order.order_id} @ {fill_price}")

    def _update_position(
        self, market_id: str, side: str, quantity: int, price: int
    ) -> None:
        """Update position after a trade."""
        if market_id not in self.positions:
            self.positions[market_id] = PaperPosition(
                market_id=market_id,
                quantity=0,
                avg_price=0,
            )

        pos = self.positions[market_id]

        if side == "buy":
            if pos.quantity >= 0:
                # Adding to long position
                total_cost = pos.quantity * pos.avg_price + quantity * price
                pos.quantity += quantity
                pos.avg_price = total_cost // pos.quantity if pos.quantity > 0 else 0
            else:
                # Closing short position
                if quantity >= abs(pos.quantity):
                    pos.realized_pnl += (pos.avg_price - price) * abs(pos.quantity)
                    pos.quantity = quantity - abs(pos.quantity)
                    pos.avg_price = price if pos.quantity > 0 else 0
                else:
                    pos.realized_pnl += (pos.avg_price - price) * quantity
                    pos.quantity -= quantity
        else:  # sell
            if pos.quantity <= 0:
                # Adding to short position
                total_cost = abs(pos.quantity) * pos.avg_price + quantity * price
                pos.quantity -= quantity
                pos.avg_price = (
                    total_cost // abs(pos.quantity) if pos.quantity < 0 else 0
                )
            else:
                # Closing long position
                if quantity >= pos.quantity:
                    pos.realized_pnl += (price - pos.avg_price) * pos.quantity
                    pos.quantity = -(quantity - pos.quantity)
                    pos.avg_price = price if pos.quantity < 0 else 0
                else:
                    pos.realized_pnl += (price - pos.avg_price) * quantity
                    pos.quantity -= quantity

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending paper order."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]

        if order.status in ["filled", "cancelled", "rejected"]:
            return False

        order.status = "cancelled"
        self.stats["cancelled_orders"] += 1

        return True

    def get_order(self, order_id: str) -> Optional[PaperOrder]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_orders(
        self, market_id: Optional[str] = None, status: Optional[str] = None
    ) -> List[PaperOrder]:
        """Get orders with optional filtering."""
        orders = list(self.orders.values())

        if market_id:
            orders = [o for o in orders if o.market_id == market_id]
        if status:
            orders = [o for o in orders if o.status == status]

        return orders

    def get_balance(self) -> int:
        """Get current paper trading balance."""
        return self.balance

    def get_available_balance(self) -> int:
        """Get available balance (扣除持仓保证金)."""
        return self.balance

    def get_positions(self) -> Dict[str, PaperPosition]:
        """Get all open positions."""
        return self.positions.copy()

    def calculate_unrealized_pnl(self) -> int:
        """Calculate unrealized P&L from open positions."""
        total_unrealized = 0

        for market_id, pos in self.positions.items():
            if pos.quantity == 0:
                continue

            if market_id in self.orderbooks:
                ob = self.orderbooks[market_id]
                current_price = ob.get_mid_price()

                if current_price and pos.quantity > 0:
                    pos.unrealized_pnl = int(
                        (current_price - pos.avg_price) * pos.quantity
                    )
                elif current_price and pos.quantity < 0:
                    pos.unrealized_pnl = int(
                        (pos.avg_price - current_price) * abs(pos.quantity)
                    )

            total_unrealized += pos.unrealized_pnl

        return total_unrealized

    def get_total_pnl(self) -> int:
        """Get total P&L (realized + unrealized)."""
        realized = sum(pos.realized_pnl for pos in self.positions.values())
        unrealized = self.calculate_unrealized_pnl()
        return realized + unrealized

    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        return {
            **self.stats,
            "balance": self.balance,
            "total_pnl": self.get_total_pnl(),
            "realized_pnl": sum(pos.realized_pnl for pos in self.positions.values()),
            "unrealized_pnl": self.calculate_unrealized_pnl(),
            "open_positions": sum(
                1 for p in self.positions.values() if p.quantity != 0
            ),
        }

    def reset(self) -> None:
        """Reset the simulator to initial state."""
        self.balance = self.initial_balance
        self.orders.clear()
        self.trades.clear()
        self.positions.clear()
        self.orderbooks.clear()
        self._order_counter = 0
        self._trade_counter = 0
        self.stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "rejected_orders": 0,
            "total_volume": 0,
            "total_commission": 0,
            "total_slippage": 0,
        }
        logger.info("Paper trading simulator reset")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        filled_orders = [o for o in self.orders.values() if o.status == "filled"]

        winning_trades = []
        losing_trades = []
        for market_id, pos in self.positions.items():
            if pos.realized_pnl > 0:
                winning_trades.append(pos)
            elif pos.realized_pnl < 0:
                losing_trades.append(pos)

        total_trades = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        avg_win = (
            sum(p.realized_pnl for p in winning_trades) / len(winning_trades)
            if winning_trades
            else 0
        )
        avg_loss = (
            sum(p.realized_pnl for p in losing_trades) / len(losing_trades)
            if losing_trades
            else 0
        )

        return {
            "summary": {
                "initial_balance": self.initial_balance,
                "current_balance": self.balance,
                "total_pnl": self.get_total_pnl(),
                "realized_pnl": sum(p.realized_pnl for p in self.positions.values()),
                "unrealized_pnl": self.calculate_unrealized_pnl(),
                "total_return_pct": (
                    (self.balance - self.initial_balance) / self.initial_balance * 100
                )
                if self.initial_balance > 0
                else 0,
            },
            "trading_stats": {
                "total_orders": self.stats["total_orders"],
                "filled_orders": self.stats["filled_orders"],
                "cancelled_orders": self.stats["cancelled_orders"],
                "rejected_orders": self.stats["rejected_orders"],
                "fill_rate": self.stats["filled_orders"] / self.stats["total_orders"]
                if self.stats["total_orders"] > 0
                else 0,
            },
            "performance": {
                "win_rate": win_rate * 100,
                "total_trades": total_trades,
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                "total_volume": self.stats["total_volume"],
                "total_commission": self.stats["total_commission"],
            },
            "positions": {
                "open_positions": sum(
                    1 for p in self.positions.values() if p.quantity != 0
                ),
                "total_realized_pnl": sum(
                    p.realized_pnl for p in self.positions.values()
                ),
            },
        }
