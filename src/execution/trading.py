import time
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime

from src.clients.kalshi_client import KalshiClient
from src.core.orderbook import OrderSide
from src.core.arbitrage import ArbitrageOpportunity
from src.utils.logging_utils import get_logger

logger = get_logger("trading")


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class TradeResult:
    success: bool
    order_id: Optional[str] = None
    filled_quantity: int = 0
    filled_price: int = 0
    total_cost: int = 0
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ExecutionPlan:
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    buy_filled: bool = False
    sell_filled: bool = False
    emergency_cancelled: bool = False
    sell_filled: bool = False
    quantity: int = 0
    expected_profit: int = 0


class TradingExecutor:
    def __init__(
        self,
        client: KalshiClient,
        paper_mode: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        order_timeout: int = 30,
        max_position_size: int = 1000,
        max_order_value: int = 10000,
        max_daily_loss: int = 10000,
    ):
        self.client = client
        self.paper_mode = paper_mode
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.order_timeout = order_timeout
        self.pending_orders: Dict[str, Dict] = {}

        # Risk controls
        self.max_position_size = max_position_size
        self.max_order_value = max_order_value
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0
        self.consecutive_failures = 0
        self.circuit_breaker_threshold = 5

    def check_risk_limits(self, quantity: int, price: int) -> Tuple[bool, str]:
        """Check if order passes risk checks."""
        order_value = quantity * price

        if quantity > self.max_position_size:
            return (
                False,
                f"Quantity {quantity} exceeds max position size {self.max_position_size}",
            )

        if order_value > self.max_order_value:
            return (
                False,
                f"Order value {order_value} exceeds max order value {self.max_order_value}",
            )

        if self.daily_pnl <= -self.max_daily_loss:
            return False, f"Daily loss limit reached: {self.daily_pnl}"

        if self.consecutive_failures >= self.circuit_breaker_threshold:
            return (
                False,
                f"Circuit breaker open: {self.consecutive_failures} consecutive failures",
            )

        return True, "OK"

    def record_trade_result(self, success: bool, profit: int = 0) -> None:
        """Record trade result for risk tracking."""
        if success:
            self.consecutive_failures = 0
            self.daily_pnl += profit
        else:
            self.consecutive_failures += 1

    def reset_daily_limits(self) -> None:
        """Reset daily limits (call at start of trading day)."""
        self.daily_pnl = 0
        self.consecutive_failures = 0

    async def execute_arbitrage(
        self, opportunity: ArbitrageOpportunity
    ) -> Tuple[bool, int]:
        if self.paper_mode:
            logger.info(f"[PAPER] Would execute arbitrage: {opportunity.id}")
            return True, opportunity.net_profit_cents

        buy_order_id = None
        sell_order_id = None

        try:
            # Execute both orders simultaneously for atomic execution
            buy_task = self._execute_buy(
                opportunity.buy_market_id, opportunity.buy_price, opportunity.quantity
            )
            sell_task = self._execute_sell(
                opportunity.sell_market_id,
                opportunity.sell_price,
                opportunity.quantity,
            )

            # Wait for both orders to be placed
            buy_result, sell_result = await asyncio.gather(buy_task, sell_task)

            if not buy_result.success:
                logger.error(f"Buy order failed: {buy_result.error_message}")
                await self._emergency_cancel_order(buy_result.order_id)
                return False, 0

            if not sell_result.success:
                logger.error(f"Sell order failed: {sell_result.error_message}")
                await self._emergency_cancel_order(sell_result.order_id)
                return False, 0

            buy_order_id = buy_result.order_id
            sell_order_id = sell_result.order_id

            # Wait for both fills with timeout
            try:
                buy_filled, sell_filled = await asyncio.wait_for(
                    asyncio.gather(
                        self._wait_for_fill(buy_order_id),
                        self._wait_for_fill(sell_order_id),
                    ),
                    timeout=self.order_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Arbitrage timeout - canceling both orders: {buy_order_id}, {sell_order_id}"
                )
                await asyncio.gather(
                    self._emergency_cancel_order(buy_order_id),
                    self._emergency_cancel_order(sell_order_id),
                )
                return False, 0

            # Both orders filled - calculate profit
            actual_profit = sell_result.total_cost - buy_result.total_cost
            logger.info(
                f"Arbitrage executed successfully. Profit: {actual_profit} cents"
            )
            return True, actual_profit

        except Exception as e:
            logger.error(f"Arbitrage execution error: {e}")
            # Emergency rollback
            if buy_order_id:
                await self._emergency_cancel_order(buy_order_id)
            if sell_order_id:
                await self._emergency_cancel_order(sell_order_id)
            return False, 0

    async def _execute_buy(
        self, market_id: str, price: int, quantity: int
    ) -> TradeResult:
        for attempt in range(self.max_retries):
            try:
                order = await self.client.create_order(
                    market_id=market_id,
                    side="yes",
                    order_type="limit",
                    price=price,
                    count=quantity,
                )

                order_id = order.get("order", {}).get("id")
                if not order_id:
                    return TradeResult(
                        success=False, error_message="No order ID returned"
                    )

                logger.info(f"Buy order submitted: {order_id}")

                filled = await self._wait_for_fill(order_id)

                if filled:
                    return TradeResult(
                        success=True,
                        order_id=order_id,
                        filled_quantity=filled.get("count", 0),
                        filled_price=price,
                        total_cost=filled.get("count", 0) * price,
                    )
                else:
                    self.client.cancel_order(order_id)
                    return TradeResult(
                        success=False,
                        order_id=order_id,
                        error_message="Order timed out",
                    )

            except Exception as e:
                logger.warning(f"Buy attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

        return TradeResult(success=False, error_message="Max retries exceeded")

    async def _execute_sell(
        self, market_id: str, price: int, quantity: int
    ) -> TradeResult:
        for attempt in range(self.max_retries):
            try:
                order = await self.client.create_order(
                    market_id=market_id,
                    side="no",
                    order_type="limit",
                    price=price,
                    count=quantity,
                )

                order_id = order.get("order", {}).get("id")
                if not order_id:
                    return TradeResult(
                        success=False, error_message="No order ID returned"
                    )

                logger.info(f"Sell order submitted: {order_id}")

                filled = await self._wait_for_fill(order_id)

                if filled:
                    return TradeResult(
                        success=True,
                        order_id=order_id,
                        filled_quantity=filled.get("count", 0),
                        filled_price=price,
                        total_cost=filled.get("count", 0) * price,
                    )
                else:
                    self.client.cancel_order(order_id)
                    return TradeResult(
                        success=False,
                        order_id=order_id,
                        error_message="Order timed out",
                    )

            except Exception as e:
                logger.warning(f"Sell attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

        return TradeResult(success=False, error_message="Max retries exceeded")

    async def _wait_for_fill(
        self, order_id: str, timeout: Optional[int] = None
    ) -> Optional[Dict]:
        timeout = timeout or self.order_timeout
        start_time = time.time()
        retry_count = 0
        base_delay = 0.5  # Start with 0.5 second polling
        max_delay = 5.0  # Cap at 5 seconds

        while time.time() - start_time < timeout:
            try:
                order_status = await self.client.get_order(order_id)
                status = order_status.get("order", {}).get("status")

                if status == "filled":
                    return order_status

                # Exponential backoff
                delay = min(base_delay * (2**retry_count), max_delay)
                await asyncio.sleep(delay)
                retry_count += 1

            except Exception as e:
                logger.warning(f"Error checking order status: {e}")
                delay = min(base_delay * (2**retry_count), max_delay)
                await asyncio.sleep(delay)
                retry_count += 1

        return None

    async def _emergency_cancel_order(self, order_id: str) -> bool:
        """Emergency order cancellation with retry logic"""
        try:
            logger.warning(f"Emergency canceling order: {order_id}")
            result = await self.client.cancel_order(order_id)
            return result.success
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def batch_execute(
        self, opportunities: List[ArbitrageOpportunity], max_concurrent: int = 3
    ) -> List[Tuple[ArbitrageOpportunity, bool, int]]:
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def execute_with_limit(opp: ArbitrageOpportunity):
            async with semaphore:
                success, profit = await self.execute_arbitrage(opp)
                return (opp, success, profit)

        tasks = [execute_with_limit(opp) for opp in opportunities]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        executed = []
        for result in results:
            if isinstance(result, tuple):
                executed.append(result)

        return executed

    async def _emergency_cancel_order(self, order_id: str) -> bool:
        """Emergency order cancellation with retry logic"""
        try:
            logger.warning(f"Emergency canceling order: {order_id}")
            result = await self.client.cancel_order(order_id)
            return result.success
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def validate_execution(
        self, opportunity: ArbitrageOpportunity, current_orderbook: Any
    ) -> Tuple[bool, str]:
        if opportunity.risk_level == "high":
            return False, "Risk level too high"

        buy_fill_prob = current_orderbook.get_fill_probability(
            OrderSide.BUY, opportunity.quantity, opportunity.buy_price
        )
        sell_fill_prob = current_orderbook.get_fill_probability(
            OrderSide.SELL, opportunity.quantity, opportunity.sell_price
        )

        if buy_fill_prob < 0.5 or sell_fill_prob < 0.5:
            return False, "Low fill probability"

        return True, "Valid"
