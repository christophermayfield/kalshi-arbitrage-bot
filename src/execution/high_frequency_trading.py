"""High-frequency async trading engine with connection pooling and optimized execution."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import httpx
import aiohttp
import json

from src.core.arbitrage import ArbitrageOpportunity, ArbitrageType
from src.core.orderbook import OrderBook, OrderSide
from src.core.portfolio import PortfolioManager
from src.clients.kalshi_client import KalshiClient
from src.utils.logging_utils import get_logger
from src.utils.performance_cache import get_cache_manager

logger = get_logger("trading_engine")


@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""

    execution_id: str
    strategy: str
    market_id: str
    side: str
    price: int
    quantity: int
    start_time: datetime
    end_time: Optional[datetime] = None
    latency_ms: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    fill_price: Optional[int] = None
    fill_quantity: Optional[int] = None
    slippage_cents: int = 0


class HighFrequencyExecutor:
    """High-frequency trading executor with connection pooling."""

    def __init__(
        self,
        client: KalshiClient,
        portfolio: PortfolioManager,
        max_concurrent_orders: int = 10,
        order_timeout_ms: int = 5000,
        max_retries: int = 3,
        retry_delay_ms: int = 100,
    ):
        """Initialize high-frequency executor."""
        self.client = client
        self.portfolio = portfolio
        self.max_concurrent_orders = max_concurrent_orders
        self.order_timeout_ms = order_timeout_ms
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms

        # Connection pool settings
        self.pool_size = 20
        self.connect_timeout_ms = 1000
        self.read_timeout_ms = 2000
        self.write_timeout_ms = 2000

        # Execution tracking
        self.active_orders: Dict[str, ExecutionMetrics] = {}
        self.execution_history: List[ExecutionMetrics] = []
        self.semaphore = asyncio.Semaphore(max_concurrent_orders)

        # Performance tracking
        self.execution_times: List[float] = []
        self.success_rate = 0.0
        self.avg_slippage = 0.0

        # Initialize connection pools
        self._initialize_pools()

        # Cache manager
        self.cache_manager = get_cache_manager()

    def _initialize_pools(self):
        """Initialize HTTP connection pools."""
        # aiohttp session pool
        self.aiohttp_session = None
        self.httpx_client = None

        try:
            # aiohttp session with optimized settings
            connector = aiohttp.TCPConnector(
                limit=self.pool_size,
                limit_per_host=self.pool_size,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=60,
                enable_cleanup_closed=True,
            )

            timeout = aiohttp.ClientTimeout(
                total=self.order_timeout_ms / 1000,
                connect=self.connect_timeout_ms / 1000,
                sock_read=self.read_timeout_ms / 1000,
                sock_connect=self.connect_timeout_ms / 1000,
            )

            self.aiohttp_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": "KalshiArbitrageBot/1.0-HF"},
            )

            # httpx client as backup
            limits = httpx.Limits(
                max_keepalive_connections=self.pool_size, max_connections=self.pool_size
            )

            timeout = httpx.Timeout(
                connect=self.connect_timeout_ms / 1000,
                read=self.read_timeout_ms / 1000,
                write=self.write_timeout_ms / 1000,
                pool=self.order_timeout_ms / 1000,
            )

            self.httpx_client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
                http2=True,
                headers={"User-Agent": "KalshiArbitrageBot/1.0-HF"},
            )

            logger.info("Connection pools initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize connection pools: {e}")

    async def execute_opportunity_fast(
        self, opportunity: ArbitrageOpportunity
    ) -> ExecutionMetrics:
        """Execute arbitrage opportunity with ultra-low latency."""
        execution_id = f"exec_{int(time.time() * 1000000)}"

        start_time = datetime.utcnow()

        metrics = ExecutionMetrics(
            execution_id=execution_id,
            strategy=self._determine_strategy(opportunity),
            market_id=opportunity.market_id_1 or opportunity.market_id_2 or "unknown",
            side="buy",
            price=opportunity.buy_price,
            quantity=opportunity.quantity,
            start_time=start_time,
        )

        self.active_orders[execution_id] = metrics

        try:
            # Execute with semaphore for concurrency control
            async with self.semaphore:
                success, result = await self._execute_with_retry(opportunity, metrics)

                end_time = datetime.utcnow()
                metrics.end_time = end_time
                metrics.latency_ms = (end_time - start_time).total_seconds() * 1000
                metrics.success = success

                if success:
                    metrics.fill_price = result.get("price")
                    metrics.fill_quantity = result.get("quantity")

                    # Calculate slippage
                    if metrics.fill_price and metrics.price:
                        metrics.slippage_cents = abs(metrics.fill_price - metrics.price)

                    # Update portfolio
                    self.portfolio.cash_balance += result.get("profit", 0)
                else:
                    metrics.error_message = result.get("error", "Execution failed")

                # Cache execution result
                await self.cache_manager.cache.set_orderbook(metrics.market_id, result)

                # Update metrics
                self.execution_times.append(metrics.latency_ms)
                self.execution_history.append(metrics)

                # Keep only recent history
                if len(self.execution_history) > 1000:
                    self.execution_history = self.execution_history[-500:]

                return metrics

        except Exception as e:
            logger.error(f"Fast execution failed for {execution_id}: {e}")
            metrics.end_time = datetime.utcnow()
            metrics.error_message = str(e)
            metrics.success = False
            return metrics

        finally:
            # Clean up active orders
            self.active_orders.pop(execution_id, None)

    def _determine_strategy(self, opportunity: ArbitrageOpportunity) -> str:
        """Determine strategy type for tracking."""
        if hasattr(opportunity, "type"):
            if opportunity.type == ArbitrageType.CROSS_MARKET:
                return "cross_market"
            elif opportunity.type == ArbitrageType.INTERNAL:
                return "internal"
            elif opportunity.type == ArbitrageType.TRIANGULAR:
                return "triangular"

        return "arbitrage"

    async def _execute_with_retry(
        self, opportunity: ArbitrageOpportunity, metrics: ExecutionMetrics
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute opportunity with intelligent retry logic."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # Choose optimal execution method
                if attempt == 0:
                    # First attempt: Use fastest method
                    result = await self._execute_ultra_fast(opportunity)
                else:
                    # Retry: Use more reliable method with backoff
                    await asyncio.sleep(
                        self.retry_delay_ms * (2 ** (attempt - 1)) / 1000
                    )
                    result = await self._execute_reliable(opportunity)

                if result and result.get("success", False):
                    return True, result

                last_error = result.get("error", "Unknown error")
                logger.warning(f"Execution attempt {attempt + 1} failed: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.error(f"Execution attempt {attempt + 1} exception: {e}")

        return False, {"error": last_error, "attempts": self.max_retries + 1}

    async def _execute_ultra_fast(
        self, opportunity: ArbitrageOpportunity
    ) -> Dict[str, Any]:
        """Ultra-fast execution using optimized HTTP client."""
        try:
            # Try aiohttp first (usually faster)
            if self.aiohttp_session:
                result = await self._execute_with_aiohttp(opportunity)
                if result.get("success", False):
                    return result

            # Fallback to httpx
            if self.httpx_client:
                result = await self._execute_with_httpx(opportunity)
                return result

            # Fallback to original client
            return await self._execute_with_original_client(opportunity)

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_reliable(
        self, opportunity: ArbitrageOpportunity
    ) -> Dict[str, Any]:
        """More reliable execution with additional validation."""
        try:
            # Use original client with enhanced error handling
            success, result = await self.client.execute_arbitrage(opportunity)

            if success:
                return {
                    "success": True,
                    "price": opportunity.buy_price,
                    "quantity": opportunity.quantity,
                    "profit": result,
                }
            else:
                return {"success": False, "error": "Execution failed"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_with_aiohttp(
        self, opportunity: ArbitrageOpportunity
    ) -> Dict[str, Any]:
        """Execute using aiohttp with direct API calls."""
        try:
            # Construct direct API request
            if opportunity.buy_market_id and opportunity.sell_market_id:
                # Cross-market arbitrage
                tasks = [
                    self._submit_order_aiohttp(
                        opportunity.buy_market_id,
                        "buy",
                        opportunity.buy_price,
                        opportunity.quantity,
                    ),
                    self._submit_order_aiohttp(
                        opportunity.sell_market_id,
                        "sell",
                        opportunity.sell_price,
                        opportunity.quantity,
                    ),
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                buy_success = isinstance(results[0], dict) and results[0].get("success")
                sell_success = isinstance(results[1], dict) and results[1].get(
                    "success"
                )

                if buy_success and sell_success:
                    return {
                        "success": True,
                        "price": opportunity.buy_price,
                        "quantity": opportunity.quantity,
                        "profit": opportunity.net_profit_cents,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Buy: {buy_success}, Sell: {sell_success}",
                    }
            else:
                # Single market arbitrage
                return await self._submit_order_aiohttp(
                    opportunity.market_id_1,
                    "buy" if opportunity.quantity > 0 else "sell",
                    opportunity.buy_price,
                    abs(opportunity.quantity),
                )

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_with_httpx(
        self, opportunity: ArbitrageOpportunity
    ) -> Dict[str, Any]:
        """Execute using httpx client."""
        try:
            # Similar to aiohttp but using httpx
            if opportunity.buy_market_id and opportunity.sell_market_id:
                tasks = [
                    self._submit_order_httpx(
                        opportunity.buy_market_id,
                        "buy",
                        opportunity.buy_price,
                        opportunity.quantity,
                    ),
                    self._submit_order_httpx(
                        opportunity.sell_market_id,
                        "sell",
                        opportunity.sell_price,
                        opportunity.quantity,
                    ),
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                buy_success = isinstance(results[0], dict) and results[0].get("success")
                sell_success = isinstance(results[1], dict) and results[1].get(
                    "success"
                )

                if buy_success and sell_success:
                    return {
                        "success": True,
                        "price": opportunity.buy_price,
                        "quantity": opportunity.quantity,
                        "profit": opportunity.net_profit_cents,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Buy: {buy_success}, Sell: {sell_success}",
                    }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_with_original_client(
        self, opportunity: ArbitrageOpportunity
    ) -> Dict[str, Any]:
        """Execute using original client as fallback."""
        try:
            success, result = await self.client.execute_arbitrage(opportunity)

            return {
                "success": success,
                "price": opportunity.buy_price,
                "quantity": opportunity.quantity,
                "profit": result if success else None,
                "error": None if success else "Execution failed",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _submit_order_aiohttp(
        self, market_id: str, side: str, price: int, quantity: int
    ) -> Dict[str, Any]:
        """Submit order using aiohttp client."""
        # This would implement direct API call to Kalshi
        # For now, return mock success
        await asyncio.sleep(0.01)  # Simulate API call
        return {"success": True, "order_id": f"aio_{market_id}_{side}"}

    async def _submit_order_httpx(
        self, market_id: str, side: str, price: int, quantity: int
    ) -> Dict[str, Any]:
        """Submit order using httpx client."""
        # This would implement direct API call to Kalshi
        await asyncio.sleep(0.01)  # Simulate API call
        return {"success": True, "order_id": f"httpx_{market_id}_{side}"}

    async def execute_batch(
        self, opportunities: List[ArbitrageOpportunity]
    ) -> List[ExecutionMetrics]:
        """Execute multiple opportunities concurrently."""
        logger.info(f"Executing batch of {len(opportunities)} opportunities")

        start_time = time.time()

        # Execute all opportunities concurrently
        tasks = [self.execute_opportunity_fast(opp) for opp in opportunities]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        execution_metrics = []
        for i, result in enumerate(results):
            if isinstance(result, ExecutionMetrics):
                execution_metrics.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Opportunity {i} execution failed: {result}")
            else:
                logger.error(f"Unknown result type for opportunity {i}: {type(result)}")

        total_time = time.time() - start_time
        logger.info(f"Batch execution completed in {total_time:.3f}s")

        return execution_metrics

    async def get_order_status(
        self, order_id: str, max_attempts: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Get order status with optimized calls."""
        for attempt in range(max_attempts):
            try:
                # Use cached status first
                status_key = f"order_status:{order_id}"
                cached_status = await self.cache_manager.cache.get_orderbook(status_key)

                if cached_status:
                    return cached_status

                # If not cached, fetch from exchange
                status = await self._fetch_order_status_fast(order_id)

                if status:
                    # Cache the status
                    await self.cache_manager.cache.set_orderbook(status_key, status)
                    return status

                await asyncio.sleep(0.1 * (2**attempt))  # Exponential backoff

            except Exception as e:
                logger.error(f"Status fetch error for {order_id}: {e}")

        return None

    async def _fetch_order_status_fast(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Fast order status fetch using optimized client."""
        try:
            # This would implement optimized status fetch
            # For now, return mock status
            await asyncio.sleep(0.005)  # 5ms latency
            return {
                "order_id": order_id,
                "status": "filled",
                "filled_quantity": 100,
                "filled_price": 5000,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Fast status fetch failed: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.execution_times:
            return {
                "total_executions": 0,
                "avg_latency_ms": 0,
                "success_rate": 0,
                "avg_slippage_cents": 0,
                "active_orders": len(self.active_orders),
            }

        # Calculate statistics
        avg_latency = sum(self.execution_times) / len(self.execution_times)
        successful_executions = [m for m in self.execution_history if m.success]

        success_rate = (
            len(successful_executions) / len(self.execution_history)
            if self.execution_history
            else 0
        )

        slippages = [m.slippage_cents for m in self.execution_history if m.success]
        avg_slippage = sum(slippages) / len(slippages) if slippages else 0

        return {
            "total_executions": len(self.execution_history),
            "avg_latency_ms": avg_latency,
            "success_rate": success_rate * 100,
            "avg_slippage_cents": avg_slippage,
            "active_orders": len(self.active_orders),
            "recent_latency": self.execution_times[-10:]
            if len(self.execution_times) >= 10
            else self.execution_times,
        }

    async def cleanup_old_orders(self):
        """Clean up old order data from tracking."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)

        # Clean up active orders that are too old
        old_orders = [
            order_id
            for order_id, metrics in self.active_orders.items()
            if metrics.start_time < cutoff_time
        ]

        for order_id in old_orders:
            logger.warning(f"Removing old order from tracking: {order_id}")
            self.active_orders.pop(order_id, None)

        # Clean up execution history
        self.execution_history = [
            m for m in self.execution_history if m.start_time > cutoff_time
        ]

        logger.info(f"Cleaned up {len(old_orders)} old orders")

    async def shutdown(self):
        """Gracefully shutdown the trading engine."""
        logger.info("Shutting down high-frequency executor...")

        # Cancel active orders if needed
        if self.active_orders:
            logger.info(f"Cancelling {len(self.active_orders)} active orders")
            for order_id in list(self.active_orders.keys()):
                try:
                    # Cancel order logic here
                    pass
                except Exception as e:
                    logger.error(f"Failed to cancel order {order_id}: {e}")

        # Close connection pools
        if self.aiohttp_session:
            await self.aiohttp_session.close()

        if self.httpx_client:
            await self.httpx_client.aclose()

        logger.info("High-frequency executor shutdown complete")


class TradingEngineManager:
    """Manager for multiple trading strategies with load balancing."""

    def __init__(self, config, portfolio: PortfolioManager):
        """Initialize trading engine manager."""
        self.config = config
        self.portfolio = portfolio
        self.executors: Dict[str, HighFrequencyExecutor] = {}
        self.active_strategies = {}
        self.load_balancer_index = 0

        # Initialize executors for different strategies
        self._initialize_executors()

    def _initialize_executors(self):
        """Initialize multiple executors for different strategies."""
        strategies = [
            (
                "high_frequency",
                {
                    "max_concurrent_orders": 15,
                    "order_timeout_ms": 3000,
                    "max_retries": 2,
                    "retry_delay_ms": 50,
                },
            ),
            (
                "balanced",
                {
                    "max_concurrent_orders": 10,
                    "order_timeout_ms": 5000,
                    "max_retries": 3,
                    "retry_delay_ms": 100,
                },
            ),
            (
                "conservative",
                {
                    "max_concurrent_orders": 5,
                    "order_timeout_ms": 8000,
                    "max_retries": 5,
                    "retry_delay_ms": 200,
                },
            ),
        ]

        for strategy_name, config in strategies:
            from src.clients.kalshi_client import KalshiClient

            client = KalshiClient(self.config)

            executor = HighFrequencyExecutor(
                client=client, portfolio=self.portfolio, **config
            )

            self.executors[strategy_name] = executor
            self.active_strategies[strategy_name] = False

        logger.info(f"Initialized {len(self.executors)} trading executors")

    async def execute_opportunity(
        self, opportunity: ArbitrageOpportunity, strategy: str = "balanced"
    ) -> ExecutionMetrics:
        """Execute opportunity using specified strategy."""
        if strategy not in self.executors:
            logger.warning(f"Unknown strategy: {strategy}, using balanced")
            strategy = "balanced"

        executor = self.executors[strategy]
        return await executor.execute_opportunity_fast(opportunity)

    async def execute_with_load_balancing(
        self, opportunity: ArbitrageOpportunity
    ) -> ExecutionMetrics:
        """Execute opportunity using load balancing."""
        # Choose executor based on current load and performance
        best_executor = self._choose_best_executor()

        if best_executor:
            return await best_executor.execute_opportunity_fast(opportunity)
        else:
            # Fallback to balanced executor
            return await self.executors["balanced"].execute_opportunity_fast(
                opportunity
            )

    def _choose_best_executor(self) -> Optional[HighFrequencyExecutor]:
        """Choose best executor based on performance and load."""
        available_executors = [
            (name, executor)
            for name, executor in self.executors.items()
            if len(executor.active_orders) < executor.max_concurrent_orders * 0.8
        ]

        if not available_executors:
            return None

        # Sort by performance (success rate * inverse latency)
        def score_executor(name_executor):
            name, executor = name_executor
            stats = executor.get_performance_stats()

            # Higher success rate is better
            success_weight = stats.get("success_rate", 0) / 100

            # Lower latency is better
            avg_latency = stats.get("avg_latency_ms", 1000)
            latency_weight = 1 / max(1, avg_latency / 1000)

            # Load factor
            load_factor = 1 - (
                len(executor.active_orders) / executor.max_concurrent_orders
            )

            return success_weight * latency_weight * load_factor

        available_executors.sort(key=score_executor, reverse=True)
        return available_executors[0][1]

    async def get_all_performance_stats(self) -> Dict[str, Any]:
        """Get performance stats for all executors."""
        stats = {}
        for name, executor in self.executors.items():
            stats[name] = executor.get_performance_stats()
            stats[name]["active"] = self.active_strategies.get(name, False)

        return stats

    async def shutdown_all(self):
        """Shutdown all trading executors."""
        logger.info("Shutting down all trading executors...")

        tasks = [executor.shutdown() for executor in self.executors.values()]

        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("All trading executors shutdown complete")
