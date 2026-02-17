"""
Automated Stop-Loss and Take-Profit Mechanisms
Advanced position management with dynamic adjustments
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import redis.asyncio as redis

from ..utils.performance_cache import PerformanceCache

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types of orders"""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    DYNAMIC_STOP = "dynamic_stop"


class OrderStatus(Enum):
    """Order status"""

    PENDING = "pending"
    ACTIVE = "active"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class StopLossMethod(Enum):
    """Stop loss calculation methods"""

    FIXED_PERCENTAGE = "fixed_percentage"
    AVERAGE_TRUE_RANGE = "average_true_range"
    BOLLINGER_BANDS = "bollinger_bands"
    SUPPORT_RESISTANCE = "support_resistance"
    VOLATILITY_BASED = "volatility_based"
    TRAILING_AVERAGE = "trailing_average"


@dataclass
class StopLossConfig:
    """Configuration for stop loss"""

    method: StopLossMethod = StopLossMethod.FIXED_PERCENTAGE
    fixed_percentage: float = 0.05  # 5% stop loss
    atr_multiplier: float = 2.0  # 2x ATR
    atr_period: int = 14  # 14-period ATR
    bollinger_period: int = 20  # 20-period Bollinger Bands
    bollinger_std: float = 2.0  # 2 standard deviations
    trailing_percentage: float = 0.03  # 3% trailing stop
    trailing_activation: float = 0.02  # Activate after 2% profit
    volatility_threshold: float = 0.15  # 15% volatility threshold

    # Dynamic adjustments
    enable_dynamic: bool = True
    volatility_adjustment: bool = True
    time_decay: bool = True
    momentum_adjustment: bool = True


@dataclass
class TakeProfitConfig:
    """Configuration for take profit"""

    method: StopLossMethod = StopLossMethod.FIXED_PERCENTAGE
    fixed_percentage: float = 0.10  # 10% take profit
    risk_reward_ratio: float = 2.0  # 2:1 risk/reward
    partial_levels: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    partial_sizes: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])
    trailing_profit: bool = False
    trailing_activation: float = 0.05  # 5% profit before trailing

    # Time-based exits
    max_holding_time: Optional[int] = None  # Maximum holding time in minutes
    time_exit_profit: float = 0.02  # Exit with 2% profit if time exceeded


@dataclass
class StopOrder:
    """Stop loss/take profit order"""

    order_id: str
    symbol: str
    position_size: float
    entry_price: float
    order_type: OrderType
    status: OrderStatus
    trigger_price: float
    current_price: float
    trailing_amount: float = 0.0
    created_time: datetime = field(default_factory=datetime.now)
    triggered_time: Optional[datetime] = None
    updated_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionTracker:
    """Track position with stop loss/take profit"""

    symbol: str
    position_size: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0

    # Stop loss configuration
    stop_loss_config: Optional[StopLossConfig] = None
    take_profit_config: Optional[TakeProfitConfig] = None

    # Current orders
    stop_loss_order: Optional[StopOrder] = None
    take_profit_orders: List[StopOrder] = field(default_factory=list)
    trailing_stop_order: Optional[StopOrder] = None

    # Price history
    price_history: deque = field(default_factory=lambda: deque(maxlen=200))

    last_updated: datetime = field(default_factory=datetime.now)


class AutomatedStopManager:
    """
    Automated stop loss and take profit management
    with dynamic adjustments and multiple strategies
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stop_config = StopLossConfig(**config.get("stop_loss", {}))
        self.take_config = TakeProfitConfig(**config.get("take_profit", {}))

        # Position tracking
        self.positions: Dict[str, PositionTracker] = {}
        self.active_orders: Dict[str, StopOrder] = {}  # order_id -> StopOrder

        # Performance tracking
        self.execution_history: deque = deque(maxlen=1000)
        self.pnl_history: deque = deque(maxlen=1000)

        # Configuration for execution
        self.execution_delay: float = config.get("execution_delay", 0.1)  # 100ms delay
        self.max_slippage: float = config.get(
            "max_slippage", 0.001
        )  # 0.1% max slippage

        # Performance cache
        self.cache = PerformanceCache(
            redis_url=config.get("redis_url", "redis://localhost:6379"), default_ttl=30
        )

        # Technical indicators cache
        self.atr_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.sma_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.std_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

        logger.info("Automated Stop Manager initialized")

    async def initialize(self) -> None:
        """Initialize stop manager with existing data"""
        try:
            # Load existing positions from cache
            await self._load_positions()

            logger.info("Automated Stop Manager initialized successfully")

        except Exception as e:
            logger.error(f"Stop Manager initialization failed: {e}")
            raise

    async def open_position(
        self,
        symbol: str,
        position_size: float,
        entry_price: float,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ) -> str:
        """
        Open a new position with automatic stop loss/take profit setup

        Returns:
            Position ID
        """
        try:
            position_id = f"{symbol}_{datetime.now().timestamp()}"

            # Create position tracker
            position = PositionTracker(
                symbol=symbol,
                position_size=position_size,
                entry_price=entry_price,
                entry_time=datetime.now(),
                current_price=entry_price,
                unrealized_pnl=0.0,
                stop_loss_config=self.stop_config,
                take_profit_config=self.take_config,
            )

            # Add price history
            position.price_history.append(entry_price)

            # Setup stop loss
            if position.stop_loss_config:
                stop_loss_price = await self._calculate_stop_loss(
                    symbol, entry_price, position_size, "stop_loss", stop_loss_pct
                )
                if stop_loss_price:
                    stop_order = StopOrder(
                        order_id=f"sl_{position_id}",
                        symbol=symbol,
                        position_size=position_size,
                        entry_price=entry_price,
                        order_type=OrderType.STOP_LOSS,
                        status=OrderStatus.ACTIVE,
                        trigger_price=stop_loss_price,
                        current_price=entry_price,
                        metadata={"position_id": position_id},
                    )
                    position.stop_loss_order = stop_order
                    self.active_orders[stop_order.order_id] = stop_order

            # Setup take profit
            if position.take_profit_config:
                take_profit_prices = await self._calculate_take_profit_levels(
                    symbol, entry_price, position_size, take_profit_pct
                )

                for i, tp_price in enumerate(take_profit_prices):
                    tp_size = position_size * self.take_config.partial_sizes[i]
                    tp_order = StopOrder(
                        order_id=f"tp_{i}_{position_id}",
                        symbol=symbol,
                        position_size=tp_size,
                        entry_price=entry_price,
                        order_type=OrderType.TAKE_PROFIT,
                        status=OrderStatus.ACTIVE,
                        trigger_price=tp_price,
                        current_price=entry_price,
                        metadata={"position_id": position_id, "level": i + 1},
                    )
                    position.take_profit_orders.append(tp_order)
                    self.active_orders[tp_order.order_id] = tp_order

            # Store position
            self.positions[position_id] = position

            # Cache position
            await self.cache.set(
                f"position:{position_id}",
                {
                    "symbol": symbol,
                    "position_size": position_size,
                    "entry_price": entry_price,
                    "entry_time": position.entry_time.isoformat(),
                    "unrealized_pnl": 0.0,
                },
                ttl=86400,  # 24 hours
            )

            logger.info(f"Position opened: {position_id} - {symbol} @ {entry_price}")

            return position_id

        except Exception as e:
            logger.error(f"Position opening failed for {symbol}: {e}")
            raise

    async def update_price(
        self, symbol: str, price: float, timestamp: Optional[datetime] = None
    ) -> None:
        """Update price for all positions with the given symbol"""
        try:
            if timestamp is None:
                timestamp = datetime.now()

            updated_positions = []

            for position_id, position in self.positions.items():
                if position.symbol == symbol and position.position_size != 0:
                    # Update position
                    old_price = position.current_price
                    position.current_price = price
                    position.last_updated = timestamp

                    # Update price history
                    position.price_history.append(price)

                    # Calculate unrealized P&L
                    if position.position_size > 0:  # Long position
                        position.unrealized_pnl = (
                            price - position.entry_price
                        ) * position.position_size
                    else:  # Short position
                        position.unrealized_pnl = (position.entry_price - price) * abs(
                            position.position_size
                        )

                    # Update technical indicators
                    await self._update_technical_indicators(symbol, price)

                    # Check stop loss adjustments
                    if position.stop_loss_order:
                        await self._update_stop_loss(position, price)

                    # Check take profit adjustments
                    await self._update_take_profit(position, price)

                    # Check for order triggers
                    await self._check_order_triggers(position, price)

                    updated_positions.append(position_id)

            # Cache price update
            if updated_positions:
                await self.cache.set(
                    f"price_update:{symbol}:{timestamp.isoformat()}",
                    {"price": price, "updated_positions": updated_positions},
                    ttl=3600,  # 1 hour
                )

        except Exception as e:
            logger.error(f"Price update failed for {symbol}: {e}")

    async def close_position(
        self, position_id: str, close_price: float, reason: str = "manual"
    ) -> Tuple[float, bool]:
        """
        Close a position and calculate realized P&L

        Returns:
            Tuple of (realized_pnl, success)
        """
        try:
            position = self.positions.get(position_id)
            if not position:
                logger.error(f"Position {position_id} not found")
                return 0.0, False

            # Calculate realized P&L
            if position.position_size > 0:  # Long position
                realized_pnl = (
                    close_price - position.entry_price
                ) * position.position_size
            else:  # Short position
                realized_pnl = (position.entry_price - close_price) * abs(
                    position.position_size
                )

            # Update position
            position.realized_pnl = realized_pnl
            position.position_size = 0.0
            position.unrealized_pnl = 0.0

            # Cancel all orders
            await self._cancel_all_orders(position_id)

            # Record execution
            execution = {
                "position_id": position_id,
                "symbol": position.symbol,
                "close_price": close_price,
                "realized_pnl": realized_pnl,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "holding_time": (datetime.now() - position.entry_time).total_seconds(),
            }

            self.execution_history.append(execution)
            self.pnl_history.append(realized_pnl)

            # Cache execution
            await self.cache.set(
                f"execution:{position_id}:{datetime.now().isoformat()}",
                execution,
                ttl=86400,
            )

            # Remove from active positions
            del self.positions[position_id]

            logger.info(
                f"Position closed: {position_id} - P&L: ${realized_pnl:.2f} ({reason})"
            )

            return realized_pnl, True

        except Exception as e:
            logger.error(f"Position closing failed for {position_id}: {e}")
            return 0.0, False

    async def _calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
        order_type: str,
        custom_percentage: Optional[float] = None,
    ) -> Optional[float]:
        """Calculate stop loss price"""
        try:
            config = self.stop_config

            if custom_percentage and order_type == "stop_loss":
                # Use custom percentage if provided
                if position_size > 0:  # Long
                    return entry_price * (1 - custom_percentage)
                else:  # Short
                    return entry_price * (1 + custom_percentage)

            if config.method == StopLossMethod.FIXED_PERCENTAGE:
                if position_size > 0:  # Long
                    return entry_price * (1 - config.fixed_percentage)
                else:  # Short
                    return entry_price * (1 + config.fixed_percentage)

            elif config.method == StopLossMethod.AVERAGE_TRUE_RANGE:
                atr = await self._calculate_atr(symbol)
                if atr:
                    if position_size > 0:  # Long
                        return entry_price - (atr * config.atr_multiplier)
                    else:  # Short
                        return entry_price + (atr * config.atr_multiplier)

            elif config.method == StopLossMethod.BOLLINGER_BANDS:
                bb_lower, bb_upper = await self._calculate_bollinger_bands(symbol)
                if bb_lower and bb_upper:
                    if position_size > 0:  # Long
                        return bb_lower
                    else:  # Short
                        return bb_upper

            elif config.method == StopLossMethod.VOLATILITY_BASED:
                # Calculate volatility-based stop
                prices = list(self.positions.values())
                if prices:
                    current_prices = [
                        p.current_price for p in prices if p.symbol == symbol
                    ]
                    if len(current_prices) >= 20:
                        returns = np.diff(np.log(current_prices[-20:]))
                        volatility = np.std(returns) * np.sqrt(252)

                        # Adjust stop based on volatility
                        if position_size > 0:  # Long
                            return entry_price * (
                                1 - max(config.fixed_percentage, volatility * 2)
                            )
                        else:  # Short
                            return entry_price * (
                                1 + max(config.fixed_percentage, volatility * 2)
                            )

            # Fallback to fixed percentage
            if position_size > 0:  # Long
                return entry_price * (1 - config.fixed_percentage)
            else:  # Short
                return entry_price * (1 + config.fixed_percentage)

        except Exception as e:
            logger.error(f"Stop loss calculation failed for {symbol}: {e}")
            return None

    async def _calculate_take_profit_levels(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
        custom_percentage: Optional[float] = None,
    ) -> List[float]:
        """Calculate take profit levels"""
        try:
            config = self.take_config
            levels = []

            if custom_percentage:
                # Single custom level
                if position_size > 0:  # Long
                    levels.append(entry_price * (1 + custom_percentage))
                else:  # Short
                    levels.append(entry_price * (1 - custom_percentage))
                return levels

            # Risk/reward based levels
            stop_loss_price = await self._calculate_stop_loss(
                symbol, entry_price, position_size, "stop_loss"
            )
            if stop_loss_price:
                if position_size > 0:  # Long
                    risk = entry_price - stop_loss_price
                    for level in config.partial_levels:
                        profit_price = entry_price + (
                            risk * config.risk_reward_ratio * level
                        )
                        levels.append(profit_price)
                else:  # Short
                    risk = stop_loss_price - entry_price
                    for level in config.partial_levels:
                        profit_price = entry_price - (
                            risk * config.risk_reward_ratio * level
                        )
                        levels.append(profit_price)

            # Fallback to fixed percentage
            if not levels:
                if position_size > 0:  # Long
                    levels.append(entry_price * (1 + config.fixed_percentage))
                else:  # Short
                    levels.append(entry_price * (1 - config.fixed_percentage))

            return levels

        except Exception as e:
            logger.error(f"Take profit calculation failed for {symbol}: {e}")
            return []

    async def _update_technical_indicators(self, symbol: str, price: float) -> None:
        """Update technical indicators for stop loss calculations"""
        try:
            # Update price history
            if symbol not in self.atr_cache:
                self.atr_cache[symbol] = deque(maxlen=50)
                self.sma_cache[symbol] = deque(maxlen=50)
                self.std_cache[symbol] = deque(maxlen=50)

            self.atr_cache[symbol].append(price)
            self.sma_cache[symbol].append(price)

        except Exception as e:
            logger.error(f"Technical indicators update failed for {symbol}: {e}")

    async def _calculate_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate Average True Range"""
        try:
            prices = list(self.atr_cache[symbol])
            if len(prices) < period + 1:
                return None

            # Calculate true ranges
            true_ranges = []
            for i in range(1, len(prices)):
                high = prices[i]
                low = prices[i]
                prev_close = prices[i - 1]

                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)

                true_ranges.append(max(tr1, tr2, tr3))

            # Calculate ATR
            atr = np.mean(true_ranges[-period:])
            return atr

        except Exception as e:
            logger.error(f"ATR calculation failed for {symbol}: {e}")
            return None

    async def _calculate_bollinger_bands(
        self, symbol: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate Bollinger Bands"""
        try:
            prices = list(self.sma_cache[symbol])
            if len(prices) < self.stop_config.bollinger_period:
                return None, None

            # Calculate SMA and standard deviation
            recent_prices = prices[-self.stop_config.bollinger_period :]
            sma = np.mean(recent_prices)
            std = np.std(recent_prices)

            # Calculate bands
            upper_band = sma + (std * self.stop_config.bollinger_std)
            lower_band = sma - (std * self.stop_config.bollinger_std)

            return lower_band, upper_band

        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed for {symbol}: {e}")
            return None, None

    async def _update_stop_loss(
        self, position: PositionTracker, current_price: float
    ) -> None:
        """Update stop loss order based on market conditions"""
        try:
            if (
                not position.stop_loss_order
                or position.stop_loss_order.status != OrderStatus.ACTIVE
            ):
                return

            config = position.stop_loss_config

            # Trailing stop logic
            if (
                config.trailing_percentage > 0
                and position.position_size > 0
                and current_price
                > position.entry_price * (1 + config.trailing_activation)
            ):
                # Calculate new trailing stop
                new_stop = current_price * (1 - config.trailing_percentage)

                # Only move stop up, never down (for long positions)
                if new_stop > position.stop_loss_order.trigger_price:
                    position.stop_loss_order.trigger_price = new_stop
                    position.stop_loss_order.updated_time = datetime.now()

            # Dynamic volatility adjustment
            if config.enable_dynamic and config.volatility_adjustment:
                # Check if volatility has changed significantly
                prices = list(position.price_history)
                if len(prices) >= 20:
                    returns = np.diff(np.log(prices[-20:]))
                    current_vol = np.std(returns)

                    if current_vol > config.volatility_threshold:
                        # Tighten stop in high volatility
                        adjustment_factor = 0.5
                        new_stop = position.entry_price * (
                            1 - config.fixed_percentage * adjustment_factor
                        )

                        if position.position_size > 0:  # Long
                            if new_stop > position.stop_loss_order.trigger_price:
                                position.stop_loss_order.trigger_price = new_stop
                        else:  # Short
                            if new_stop < position.stop_loss_order.trigger_price:
                                position.stop_loss_order.trigger_price = new_stop

        except Exception as e:
            logger.error(f"Stop loss update failed for {position.symbol}: {e}")

    async def _update_take_profit(
        self, position: PositionTracker, current_price: float
    ) -> None:
        """Update take profit orders"""
        try:
            if not position.take_profit_orders:
                return

            config = position.take_profit_config

            # Time-based exit check
            if config.max_holding_time:
                holding_time = (
                    datetime.now() - position.entry_time
                ).total_seconds() / 60  # minutes
                if (
                    holding_time > config.max_holding_time
                    and position.unrealized_pnl > 0
                ):
                    # Exit with minimum profit
                    profit_pct = position.unrealized_pnl / abs(
                        position.position_size * position.entry_price
                    )
                    if profit_pct > config.time_exit_profit:
                        await self.close_position(
                            next(
                                pid
                                for pid, p in self.positions.items()
                                if p == position
                            ),
                            current_price,
                            "time_exit",
                        )
                        return

            # Trailing profit logic
            if config.trailing_profit and position.unrealized_pnl > 0:
                profit_pct = position.unrealized_pnl / abs(
                    position.position_size * position.entry_price
                )

                if profit_pct > config.trailing_activation:
                    # Move take profit to trail
                    for tp_order in position.take_profit_orders:
                        if tp_order.status == OrderStatus.ACTIVE:
                            # Move take profit closer to current price
                            trail_distance = current_price * 0.01  # 1% trail
                            if position.position_size > 0:  # Long
                                new_tp = current_price - trail_distance
                                if new_tp > tp_order.trigger_price:
                                    tp_order.trigger_price = new_tp
                                    tp_order.updated_time = datetime.now()
                            else:  # Short
                                new_tp = current_price + trail_distance
                                if new_tp < tp_order.trigger_price:
                                    tp_order.trigger_price = new_tp
                                    tp_order.updated_time = datetime.now()

        except Exception as e:
            logger.error(f"Take profit update failed for {position.symbol}: {e}")

    async def _check_order_triggers(
        self, position: PositionTracker, current_price: float
    ) -> None:
        """Check if any orders should be triggered"""
        try:
            position_id = next(
                pid for pid, p in self.positions.items() if p == position
            )

            # Check stop loss
            if (
                position.stop_loss_order
                and position.stop_loss_order.status == OrderStatus.ACTIVE
            ):
                sl_order = position.stop_loss_order
                should_trigger = False

                if position.position_size > 0:  # Long position
                    should_trigger = current_price <= sl_order.trigger_price
                else:  # Short position
                    should_trigger = current_price >= sl_order.trigger_price

                if should_trigger:
                    # Trigger stop loss
                    await self._trigger_order(sl_order, current_price, "stop_loss")

            # Check take profit levels
            for tp_order in position.take_profit_orders:
                if tp_order.status == OrderStatus.ACTIVE:
                    should_trigger = False

                    if position.position_size > 0:  # Long position
                        should_trigger = current_price >= tp_order.trigger_price
                    else:  # Short position
                        should_trigger = current_price <= tp_order.trigger_price

                    if should_trigger:
                        # Partial close
                        await self._trigger_order(
                            tp_order, current_price, "take_profit"
                        )

            # Check if position should be fully closed
            if position.position_size == 0:
                await self.close_position(
                    position_id, current_price, "orders_completed"
                )

        except Exception as e:
            logger.error(f"Order trigger check failed for {position.symbol}: {e}")

    async def _trigger_order(
        self, order: StopOrder, trigger_price: float, reason: str
    ) -> None:
        """Trigger a stop loss or take profit order"""
        try:
            order.status = OrderStatus.TRIGGERED
            order.triggered_time = datetime.now()
            order.current_price = trigger_price

            position_id = order.metadata["position_id"]
            position = self.positions.get(position_id)

            if not position:
                logger.error(
                    f"Position {position_id} not found for order {order.order_id}"
                )
                return

            # Execute partial or full close
            if order.order_type == OrderType.STOP_LOSS:
                # Full position close
                await self.close_position(position_id, trigger_price, reason)

            elif order.order_type == OrderType.TAKE_PROFIT:
                # Partial close
                partial_pnl = 0.0
                if position.position_size > 0:  # Long
                    partial_pnl = (
                        trigger_price - position.entry_price
                    ) * order.position_size
                else:  # Short
                    partial_pnl = (
                        position.entry_price - trigger_price
                    ) * order.position_size

                # Update position size
                position.position_size -= order.position_size
                position.realized_pnl += partial_pnl

                logger.info(
                    f"Take profit triggered: {order.order_id} - P&L: ${partial_pnl:.2f}"
                )

            # Record execution
            execution = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "trigger_price": trigger_price,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            }

            self.execution_history.append(execution)

        except Exception as e:
            logger.error(f"Order triggering failed for {order.order_id}: {e}")

    async def _cancel_all_orders(self, position_id: str) -> None:
        """Cancel all orders for a position"""
        try:
            position = self.positions.get(position_id)
            if not position:
                return

            # Cancel stop loss
            if position.stop_loss_order:
                position.stop_loss_order.status = OrderStatus.CANCELLED
                if position.stop_loss_order.order_id in self.active_orders:
                    del self.active_orders[position.stop_loss_order.order_id]

            # Cancel take profit orders
            for tp_order in position.take_profit_orders:
                tp_order.status = OrderStatus.CANCELLED
                if tp_order.order_id in self.active_orders:
                    del self.active_orders[tp_order.order_id]

            # Cancel trailing stop
            if position.trailing_stop_order:
                position.trailing_stop_order.status = OrderStatus.CANCELLED
                if position.trailing_stop_order.order_id in self.active_orders:
                    del self.active_orders[position.trailing_stop_order.order_id]

        except Exception as e:
            logger.error(f"Order cancellation failed for position {position_id}: {e}")

    async def _load_positions(self) -> None:
        """Load existing positions from cache"""
        try:
            # This would load from persistent storage in production
            # For now, initialize empty
            logger.info("Positions loaded from cache")

        except Exception as e:
            logger.error(f"Position loading failed: {e}")

    async def get_positions_status(self) -> Dict[str, Any]:
        """Get status of all managed positions"""
        try:
            positions_data = {}

            for position_id, position in self.positions.items():
                position_data = {
                    "symbol": position.symbol,
                    "position_size": position.position_size,
                    "entry_price": position.entry_price,
                    "current_price": position.current_price,
                    "unrealized_pnl": position.unrealized_pnl,
                    "realized_pnl": position.realized_pnl,
                    "entry_time": position.entry_time.isoformat(),
                    "holding_time_minutes": (
                        datetime.now() - position.entry_time
                    ).total_seconds()
                    / 60,
                    "stop_loss": {
                        "active": position.stop_loss_order is not None,
                        "trigger_price": position.stop_loss_order.trigger_price
                        if position.stop_loss_order
                        else None,
                        "distance_pct": (
                            (
                                position.stop_loss_order.trigger_price
                                - position.current_price
                            )
                            / position.current_price
                            * 100
                        )
                        if position.stop_loss_order
                        else None,
                    },
                    "take_profit_levels": [
                        {
                            "level": i + 1,
                            "trigger_price": tp.trigger_price,
                            "size": tp.position_size,
                            "active": tp.status == OrderStatus.ACTIVE,
                        }
                        for i, tp in enumerate(position.take_profit_orders)
                    ],
                }

                positions_data[position_id] = position_data

            return {
                "total_positions": len(self.positions),
                "active_positions": len(
                    [p for p in self.positions.values() if p.position_size != 0]
                ),
                "total_unrealized_pnl": sum(
                    p.unrealized_pnl for p in self.positions.values()
                ),
                "total_realized_pnl": sum(
                    p.realized_pnl for p in self.positions.values()
                ),
                "positions": positions_data,
                "active_orders_count": len(self.active_orders),
            }

        except Exception as e:
            logger.error(f"Positions status retrieval failed: {e}")
            return {}

    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            # Cancel all active orders
            for position_id in list(self.positions.keys()):
                await self._cancel_all_orders(position_id)

            # Save execution history
            if self.execution_history:
                await self.cache.set(
                    "execution_history",
                    list(self.execution_history)[-100:],  # Last 100 executions
                    ttl=86400 * 7,  # 7 days
                )

            await self.cache.close()
            logger.info("Automated Stop Manager cleaned up")

        except Exception as e:
            logger.error(f"Stop Manager cleanup failed: {e}")


# Utility functions
async def create_stop_manager(config: Dict[str, Any]) -> AutomatedStopManager:
    """Create and initialize stop manager"""
    manager = AutomatedStopManager(config)
    await manager.initialize()
    return manager


def calculate_risk_reward_ratio(
    entry_price: float, stop_loss: float, take_profit: float
) -> float:
    """Calculate risk/reward ratio"""
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    return reward / risk if risk > 0 else 0


def calculate_position_heatmap(positions: Dict[str, PositionTracker]) -> str:
    """Generate position P&L heatmap"""
    if not positions:
        return "No active positions"

    total_pnl = sum(p.unrealized_pnl + p.realized_pnl for p in positions.values())

    if total_pnl > 0:
        return f"🟢 Profitable: ${total_pnl:.2f}"
    elif total_pnl < 0:
        return f"🔴 Loss: ${total_pnl:.2f}"
    else:
        return "🟡 Breakeven"
