"""
Smart Order Routing & Execution Algorithms
Professional-grade execution system with multiple algorithms and venue optimization
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import asyncio
from collections import defaultdict, deque
import uuid
import math

from src.utils.logging_utils import get_logger
from src.utils.config import Config
from src.core.orderbook import OrderBook, OrderBookLevel
from src.core.arbitrage import ArbitrageOpportunity

logger = get_logger("smart_order_routing")


class ExecutionAlgorithm(Enum):
    """Execution algorithm types"""

    SIMPLE = "simple"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    POV = "pov"  # Percentage of Volume
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ADAPTIVE = "adaptive"
    LIQUIDITY_SEEKING = "liquidity_seeking"
    DARK_POOL = "dark_pool"


class OrderSide(Enum):
    """Order sides"""

    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


class OrderType(Enum):
    """Order types"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    HIDDEN = "hidden"


@dataclass
class LiquidityVenue:
    """Represents a trading venue with liquidity characteristics"""

    venue_id: str
    venue_name: str
    venue_type: str  # exchange, dark_pool, internal

    # Liquidity metrics
    avg_daily_volume: float = 0.0
    bid_ask_spread: float = 0.0
    depth_score: float = 0.0
    fill_rate: float = 0.0

    # Cost metrics
    commission_rate: float = 0.0
    slippage_estimate: float = 0.0
    market_impact_factor: float = 0.0

    # Timing metrics
    avg_execution_time_ms: float = 0.0
    latency_ms: float = 0.0

    # Constraints
    min_order_size: float = 0.0
    max_order_size: float = float("inf")
    allowed_order_types: List[OrderType] = field(default_factory=list)

    # Quality metrics
    reliability_score: float = 1.0
    data_quality_score: float = 1.0


@dataclass
class OrderRequest:
    """Order request for execution"""

    request_id: str
    market_id: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None

    # Execution parameters
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.SIMPLE
    time_limit_seconds: int = 30
    max_slippage_percent: float = 2.0

    # Routing preferences
    preferred_venues: List[str] = field(default_factory=list)
    excluded_venues: List[str] = field(default_factory=list)
    allow_dark_pools: bool = True

    # Constraints
    min_fill_quantity: int = 1
    max_venue_exposure: float = 0.5  # Max 50% of order size per venue

    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionResult:
    """Result of order execution"""

    execution_id: str
    request_id: str
    venue_id: str

    # Execution details
    filled_quantity: int
    average_price: float
    total_cost: float

    # Timing
    created_at: datetime
    started_at: datetime
    completed_at: datetime

    # Quality metrics
    slippage: float = 0.0
    market_impact: float = 0.0
    execution_time_ms: float = 0.0

    # Status
    status: str = "completed"  # completed, partial, failed, cancelled

    # Breakdown
    venue_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class LiquidityAnalyzer:
    """Analyzes liquidity across venues"""

    def __init__(self, config: Config):
        self.config = config
        self.venues: Dict[str, LiquidityVenue] = {}
        self.market_liquidity: Dict[str, Dict[str, float]] = {}

    def register_venue(self, venue: LiquidityVenue) -> None:
        """Register a trading venue"""
        self.venues[venue.venue_id] = venue
        logger.info(f"Registered venue: {venue.venue_name}")

    def analyze_market_liquidity(
        self, market_id: str, orderbooks: Dict[str, OrderBook]
    ) -> Dict[str, float]:
        """Analyze liquidity for a specific market"""
        try:
            liquidity_metrics = {}

            for venue_id, orderbook in orderbooks.items():
                if venue_id not in self.venues:
                    continue

                venue = self.venues[venue_id]

                # Calculate depth metrics
                bid_depth = sum(level.total for level in orderbook.bids[:5])
                ask_depth = sum(level.total for level in orderbook.asks[:5])
                total_depth = bid_depth + ask_depth

                # Calculate spread metrics
                if orderbook.bids and orderbook.asks:
                    spread = orderbook.asks[0].price - orderbook.bids[0].price
                    mid_price = (orderbook.bids[0].price + orderbook.asks[0].price) / 2
                    spread_percent = (spread / mid_price) * 100 if mid_price > 0 else 0
                else:
                    spread_percent = 0

                # Calculate quality score
                depth_score = min(1.0, total_depth / 10000)  # Normalize to 10k
                spread_score = max(0, 1 - spread_percent / 5)  # 5% spread is bad
                quality_score = depth_score * 0.6 + spread_score * 0.4

                liquidity_metrics[venue_id] = {
                    "depth_score": depth_score,
                    "spread_percent": spread_percent,
                    "quality_score": quality_score,
                    "bid_depth": bid_depth,
                    "ask_depth": ask_depth,
                    "total_depth": total_depth,
                }

            self.market_liquidity[market_id] = liquidity_metrics
            return liquidity_metrics

        except Exception as e:
            logger.error(f"Failed to analyze market liquidity: {e}")
            return {}

    def get_best_venues(
        self,
        market_id: str,
        order_size: int,
        side: OrderSide,
        exclude_venues: List[str] = None,
    ) -> List[Tuple[str, float]]:
        """Get best venues for an order"""
        try:
            if market_id not in self.market_liquidity:
                return []

            liquidity_metrics = self.market_liquidity[market_id]
            venue_scores = []

            for venue_id, metrics in liquidity_metrics.items():
                venue = self.venues.get(venue_id)
                if not venue:
                    continue

                # Skip excluded venues
                if exclude_venues and venue_id in exclude_venues:
                    continue

                # Check size constraints
                if (
                    order_size < venue.min_order_size
                    or order_size > venue.max_order_size
                ):
                    continue

                # Calculate venue score
                score = metrics["quality_score"]

                # Adjust for venue-specific factors
                score *= venue.reliability_score
                score *= venue.data_quality_score

                # Adjust for cost
                cost_factor = 1 - (venue.commission_rate + venue.slippage_estimate)
                score *= cost_factor

                # Adjust for order size (larger orders prefer venues with more depth)
                size_factor = min(1.0, metrics["total_depth"] / max(1, order_size))
                score *= 0.7 + 0.3 * size_factor

                venue_scores.append((venue_id, score))

            # Sort by score (descending)
            venue_scores.sort(key=lambda x: x[1], reverse=True)

            return venue_scores

        except Exception as e:
            logger.error(f"Failed to get best venues: {e}")
            return []


class TWAPAlgorithm:
    """Time-Weighted Average Price execution algorithm"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_slices = config.get("num_slices", 10)
        self.slice_interval_seconds = config.get("slice_interval_seconds", 30)
        self.participation_rate = config.get("participation_rate", 0.1)

    async def execute(
        self,
        request: OrderRequest,
        venues: List[LiquidityVenue],
        orderbooks: Dict[str, OrderBook],
    ) -> ExecutionResult:
        """Execute TWAP algorithm"""
        try:
            execution_id = str(uuid.uuid4())
            start_time = datetime.now()

            # Calculate slice size
            slice_quantity = request.quantity // self.num_slices
            remaining_quantity = request.quantity

            filled_quantity = 0
            total_cost = 0.0
            venue_breakdown = {}

            # Execute slices
            for slice_num in range(self.num_slices):
                if remaining_quantity <= 0:
                    break

                # Calculate slice size (handle remainder)
                current_slice_size = min(slice_quantity, remaining_quantity)

                # Get best venues for this slice
                best_venues = self._get_slice_venues(request, venues, orderbooks)

                if not best_venues:
                    logger.warning(f"No venues available for slice {slice_num + 1}")
                    break

                # Execute slice
                slice_result = await self._execute_slice(
                    request, current_slice_size, best_venues, orderbooks
                )

                # Update totals
                filled_quantity += slice_result["filled_quantity"]
                total_cost += slice_result["total_cost"]
                remaining_quantity -= slice_result["filled_quantity"]

                # Update venue breakdown
                for venue_id, venue_data in slice_result["venue_breakdown"].items():
                    if venue_id not in venue_breakdown:
                        venue_breakdown[venue_id] = {
                            "filled_quantity": 0,
                            "total_cost": 0.0,
                            "avg_price": 0.0,
                        }

                    venue_breakdown[venue_id]["filled_quantity"] += venue_data[
                        "filled_quantity"
                    ]
                    venue_breakdown[venue_id]["total_cost"] += venue_data["total_cost"]

                # Wait between slices
                if slice_num < self.num_slices - 1 and remaining_quantity > 0:
                    await asyncio.sleep(self.slice_interval_seconds)

            # Calculate average price
            avg_price = (
                total_cost / max(1, filled_quantity) if filled_quantity > 0 else 0
            )

            # Create result
            result = ExecutionResult(
                execution_id=execution_id,
                request_id=request.request_id,
                venue_id="TWAP",
                filled_quantity=filled_quantity,
                average_price=avg_price,
                total_cost=total_cost,
                created_at=request.created_at,
                started_at=start_time,
                completed_at=datetime.now(),
                venue_breakdown=venue_breakdown,
            )

            return result

        except Exception as e:
            logger.error(f"TWAP execution failed: {e}")
            raise

    def _get_slice_venues(
        self,
        request: OrderRequest,
        venues: List[LiquidityVenue],
        orderbooks: Dict[str, OrderBook],
    ) -> List[Tuple[str, float]]:
        """Get venues for current slice"""
        # For TWAP, we prefer venues with good fill rates and reasonable spreads
        venue_scores = []

        for venue in venues:
            if venue.venue_id not in orderbooks:
                continue

            orderbook = orderbooks[venue.venue_id]

            # Calculate score based on fill probability and spread
            fill_prob = self._estimate_fill_probability(
                orderbook, request.side, request.quantity
            )
            spread_score = self._calculate_spread_score(orderbook)

            score = fill_prob * 0.6 + spread_score * 0.4
            score *= venue.fill_rate

            venue_scores.append((venue.venue_id, score))

        venue_scores.sort(key=lambda x: x[1], reverse=True)
        return venue_scores

    def _estimate_fill_probability(
        self, orderbook: OrderBook, side: OrderSide, quantity: int
    ) -> float:
        """Estimate fill probability for an order"""
        try:
            if side in [OrderSide.BUY, OrderSide.COVER]:
                # Buy order - check ask side
                available_volume = sum(
                    level.total
                    for level in orderbook.asks
                    if level.price <= orderbook.asks[0].price * 1.02
                )
            else:
                # Sell order - check bid side
                available_volume = sum(
                    level.total
                    for level in orderbook.bids
                    if level.price >= orderbook.bids[0].price * 0.98
                )

            return min(1.0, available_volume / max(1, quantity))

        except Exception:
            return 0.5  # Default probability

    def _calculate_spread_score(self, orderbook: OrderBook) -> float:
        """Calculate spread quality score"""
        try:
            if not orderbook.bids or not orderbook.asks:
                return 0.5

            spread = orderbook.asks[0].price - orderbook.bids[0].price
            mid_price = (orderbook.bids[0].price + orderbook.asks[0].price) / 2
            spread_percent = (spread / mid_price) * 100 if mid_price > 0 else 0

            # Lower spread is better
            return max(0, 1 - spread_percent / 5)  # 5% spread is terrible

        except Exception:
            return 0.5

    async def _execute_slice(
        self,
        request: OrderRequest,
        slice_quantity: int,
        venues: List[Tuple[str, float]],
        orderbooks: Dict[str, OrderBook],
    ) -> Dict[str, Any]:
        """Execute a single slice"""
        filled_quantity = 0
        total_cost = 0.0
        venue_breakdown = {}

        for venue_id, score in venues:
            if filled_quantity >= slice_quantity:
                break

            venue = next(v for v in self.venues if v.venue_id == venue_id)
            orderbook = orderbooks[venue_id]

            # Calculate fill quantity for this venue
            max_fill = min(
                slice_quantity - filled_quantity,
                slice_quantity * self.participation_rate,
            )
            actual_fill = min(
                max_fill, self._get_available_liquidity(orderbook, request.side)
            )

            if actual_fill <= 0:
                continue

            # Calculate execution price
            execution_price = self._get_execution_price(
                orderbook, request.side, actual_fill
            )

            # Calculate cost
            commission = execution_price * actual_fill * venue.commission_rate
            slippage = self._estimate_slippage(orderbook, request.side, actual_fill)
            slice_cost = execution_price * actual_fill + commission + slippage

            # Update totals
            filled_quantity += actual_fill
            total_cost += slice_cost

            venue_breakdown[venue_id] = {
                "filled_quantity": actual_fill,
                "total_cost": slice_cost,
                "avg_price": execution_price,
                "commission": commission,
                "slippage": slippage,
            }

        return {
            "filled_quantity": filled_quantity,
            "total_cost": total_cost,
            "venue_breakdown": venue_breakdown,
        }

    def _get_available_liquidity(self, orderbook: OrderBook, side: OrderSide) -> int:
        """Get available liquidity for an order"""
        try:
            if side in [OrderSide.BUY, OrderSide.COVER]:
                return sum(level.total for level in orderbook.asks[:5])
            else:
                return sum(level.total for level in orderbook.bids[:5])
        except Exception:
            return 0

    def _get_execution_price(
        self, orderbook: OrderBook, side: OrderSide, quantity: int
    ) -> float:
        """Get estimated execution price"""
        try:
            if side in [OrderSide.BUY, OrderSide.COVER]:
                # Buy - walk up the ask side
                remaining_qty = quantity
                total_cost = 0.0

                for level in orderbook.asks:
                    if remaining_qty <= 0:
                        break

                    fill_qty = min(remaining_qty, level.total)
                    total_cost += fill_qty * level.price
                    remaining_qty -= fill_qty

                return total_cost / max(1, quantity - remaining_qty)
            else:
                # Sell - walk down the bid side
                remaining_qty = quantity
                total_cost = 0.0

                for level in orderbook.bids:
                    if remaining_qty <= 0:
                        break

                    fill_qty = min(remaining_qty, level.total)
                    total_cost += fill_qty * level.price
                    remaining_qty -= fill_qty

                return total_cost / max(1, quantity - remaining_qty)

        except Exception:
            # Fallback to best bid/ask
            if side in [OrderSide.BUY, OrderSide.COVER] and orderbook.asks:
                return orderbook.asks[0].price
            elif orderbook.bids:
                return orderbook.bids[0].price
            return 0.0

    def _estimate_slippage(
        self, orderbook: OrderBook, side: OrderSide, quantity: int
    ) -> float:
        """Estimate slippage for an order"""
        try:
            # Simple slippage model based on order size vs depth
            if side in [OrderSide.BUY, OrderSide.COVER]:
                depth = sum(level.total for level in orderbook.asks[:5])
            else:
                depth = sum(level.total for level in orderbook.bids[:5])

            if depth == 0:
                return 0.0

            # Larger orders have more slippage
            size_ratio = quantity / depth
            base_slippage = 0.001  # 0.1% base slippage

            return base_slippage * (1 + size_ratio * 2)

        except Exception:
            return 0.0


class VWAPAlgorithm:
    """Volume-Weighted Average Price execution algorithm"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.participation_rate = config.get("participation_rate", 0.15)
        self.lookback_periods = config.get("lookback_periods", 20)

    async def execute(
        self,
        request: OrderRequest,
        venues: List[LiquidityVenue],
        orderbooks: Dict[str, OrderBook],
    ) -> ExecutionResult:
        """Execute VWAP algorithm"""
        try:
            execution_id = str(uuid.uuid4())
            start_time = datetime.now()

            # For simplicity, implement a basic VWAP that targets volume participation
            filled_quantity = 0
            total_cost = 0.0
            venue_breakdown = {}

            # Get historical volume profile (simplified)
            volume_profile = self._calculate_volume_profile(orderbooks)

            # Execute based on volume profile
            total_volume = sum(volume_profile.values())
            target_quantity = min(
                request.quantity, total_volume * self.participation_rate
            )

            for venue in venues:
                if (
                    venue.venue_id not in orderbooks
                    or filled_quantity >= target_quantity
                ):
                    continue

                orderbook = orderbooks[venue.venue_id]
                venue_volume = volume_profile.get(venue.venue_id, 0)

                if venue_volume == 0:
                    continue

                # Calculate venue allocation based on volume share
                venue_share = venue_volume / total_volume
                venue_quantity = min(
                    target_quantity * venue_share, request.quantity - filled_quantity
                )

                if venue_quantity <= 0:
                    continue

                # Execute on this venue
                execution_price = self._get_vwap_price(
                    orderbook, request.side, venue_quantity
                )
                commission = execution_price * venue_quantity * venue.commission_rate
                slippage = self._estimate_vwap_slippage(
                    orderbook, request.side, venue_quantity
                )
                venue_cost = execution_price * venue_quantity + commission + slippage

                # Update totals
                filled_quantity += venue_quantity
                total_cost += venue_cost

                venue_breakdown[venue.venue_id] = {
                    "filled_quantity": venue_quantity,
                    "total_cost": venue_cost,
                    "avg_price": execution_price,
                    "commission": commission,
                    "slippage": slippage,
                }

            # Calculate average price
            avg_price = (
                total_cost / max(1, filled_quantity) if filled_quantity > 0 else 0
            )

            # Create result
            result = ExecutionResult(
                execution_id=execution_id,
                request_id=request.request_id,
                venue_id="VWAP",
                filled_quantity=filled_quantity,
                average_price=avg_price,
                total_cost=total_cost,
                created_at=request.created_at,
                started_at=start_time,
                completed_at=datetime.now(),
                venue_breakdown=venue_breakdown,
            )

            return result

        except Exception as e:
            logger.error(f"VWAP execution failed: {e}")
            raise

    def _calculate_volume_profile(
        self, orderbooks: Dict[str, OrderBook]
    ) -> Dict[str, float]:
        """Calculate volume profile across venues"""
        try:
            volume_profile = {}

            for venue_id, orderbook in orderbooks.items():
                # Simplified volume calculation based on orderbook depth
                bid_volume = sum(level.total for level in orderbook.bids[:10])
                ask_volume = sum(level.total for level in orderbook.asks[:10])
                total_volume = bid_volume + ask_volume

                volume_profile[venue_id] = total_volume

            return volume_profile

        except Exception:
            return {}

    def _get_vwap_price(
        self, orderbook: OrderBook, side: OrderSide, quantity: int
    ) -> float:
        """Get VWAP execution price"""
        try:
            if side in [OrderSide.BUY, OrderSide.COVER]:
                # Buy - calculate VWAP across ask levels
                remaining_qty = quantity
                total_cost = 0.0

                for level in orderbook.asks:
                    if remaining_qty <= 0:
                        break

                    fill_qty = min(remaining_qty, level.total)
                    total_cost += fill_qty * level.price
                    remaining_qty -= fill_qty

                return total_cost / max(1, quantity - remaining_qty)
            else:
                # Sell - calculate VWAP across bid levels
                remaining_qty = quantity
                total_cost = 0.0

                for level in orderbook.bids:
                    if remaining_qty <= 0:
                        break

                    fill_qty = min(remaining_qty, level.total)
                    total_cost += fill_qty * level.price
                    remaining_qty -= fill_qty

                return total_cost / max(1, quantity - remaining_qty)

        except Exception:
            # Fallback to best bid/ask
            if side in [OrderSide.BUY, OrderSide.COVER] and orderbook.asks:
                return orderbook.asks[0].price
            elif orderbook.bids:
                return orderbook.bids[0].price
            return 0.0

    def _estimate_vwap_slippage(
        self, orderbook: OrderBook, side: OrderSide, quantity: int
    ) -> float:
        """Estimate slippage for VWAP execution"""
        # Similar to TWAP but adjusted for volume participation
        return 0.001  # Simplified


class SmartOrderRouter:
    """Smart order routing system with multiple execution algorithms"""

    def __init__(self, config: Config):
        self.config = config
        self.routing_config = config.get("smart_order_routing", {})

        # Components
        self.liquidity_analyzer = LiquidityAnalyzer(config)
        self.algorithms: Dict[ExecutionAlgorithm, Any] = {}

        # Initialize algorithms
        self._initialize_algorithms()

        # Routing state
        self.active_executions: Dict[str, asyncio.Task] = {}
        self.execution_history: deque = deque(maxlen=1000)

        # Configuration
        self.default_algorithm = ExecutionAlgorithm(
            self.routing_config.get("default_algorithm", "simple")
        )
        self.max_concurrent_executions = self.routing_config.get("max_concurrent", 10)
        self.enable_venue_optimization = self.routing_config.get(
            "enable_venue_optimization", True
        )

        logger.info("Smart order router initialized")

    def _initialize_algorithms(self) -> None:
        """Initialize execution algorithms"""
        try:
            # TWAP algorithm
            twap_config = self.routing_config.get("algorithms", {}).get("twap", {})
            self.algorithms[ExecutionAlgorithm.TWAP] = TWAPAlgorithm(twap_config)

            # VWAP algorithm
            vwap_config = self.routing_config.get("algorithms", {}).get("vwap", {})
            self.algorithms[ExecutionAlgorithm.VWAP] = VWAPAlgorithm(vwap_config)

            logger.info(f"Initialized {len(self.algorithms)} execution algorithms")

        except Exception as e:
            logger.error(f"Failed to initialize algorithms: {e}")

    def register_venue(self, venue: LiquidityVenue) -> None:
        """Register a trading venue"""
        self.liquidity_analyzer.register_venue(venue)

    async def execute_order(self, request: OrderRequest) -> ExecutionResult:
        """Execute an order with smart routing"""
        try:
            # Check concurrent execution limit
            if len(self.active_executions) >= self.max_concurrent_executions:
                raise Exception("Maximum concurrent executions reached")

            # Get available venues and orderbooks
            venues = list(self.liquidity_analyzer.venues.values())
            orderbooks = self._get_current_orderbooks(request.market_id)

            if not venues or not orderbooks:
                raise Exception("No venues or orderbooks available")

            # Analyze liquidity
            self.liquidity_analyzer.analyze_market_liquidity(
                request.market_id, orderbooks
            )

            # Get best venues for this order
            best_venues = self.liquidity_analyzer.get_best_venues(
                request.market_id,
                request.quantity,
                request.side,
                request.excluded_venues,
            )

            if not best_venues:
                raise Exception("No suitable venues available")

            # Filter venues by preferences
            selected_venues = self._filter_venues_by_preferences(best_venues, request)

            # Select execution algorithm
            algorithm = self._select_algorithm(request, selected_venues)

            # Execute order
            result = await algorithm.execute(request, selected_venues, orderbooks)

            # Store execution history
            self.execution_history.append(
                {"request": request, "result": result, "timestamp": datetime.now()}
            )

            logger.info(
                f"Order executed: {request.request_id}, filled: {result.filled_quantity}/{request.quantity}"
            )

            return result

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            raise

    def _get_current_orderbooks(self, market_id: str) -> Dict[str, OrderBook]:
        """Get current orderbooks for all venues"""
        # This would integrate with the real-time market data system
        # For now, return empty dict
        return {}

    def _filter_venues_by_preferences(
        self, best_venues: List[Tuple[str, float]], request: OrderRequest
    ) -> List[LiquidityVenue]:
        """Filter venues based on request preferences"""
        try:
            selected_venues = []

            for venue_id, score in best_venues:
                venue = self.liquidity_analyzer.venues[venue_id]

                # Check preferred venues
                if (
                    request.preferred_venues
                    and venue_id not in request.preferred_venues
                ):
                    continue

                # Check excluded venues
                if venue_id in request.excluded_venues:
                    continue

                # Check dark pool preference
                if venue.venue_type == "dark_pool" and not request.allow_dark_pools:
                    continue

                # Check order type compatibility
                if request.order_type not in venue.allowed_order_types:
                    continue

                selected_venues.append(venue)

            return selected_venues

        except Exception as e:
            logger.error(f"Failed to filter venues: {e}")
            return []

    def _select_algorithm(
        self, request: OrderRequest, venues: List[LiquidityVenue]
    ) -> Any:
        """Select execution algorithm"""
        try:
            # Use requested algorithm if available
            if request.algorithm in self.algorithms:
                return self.algorithms[request.algorithm]

            # Algorithm selection logic based on order characteristics
            if request.quantity > 1000:  # Large order
                return self.algorithms.get(
                    ExecutionAlgorithm.TWAP, self.algorithms[ExecutionAlgorithm.SIMPLE]
                )
            elif len(venues) > 3:  # Multiple venues
                return self.algorithms.get(
                    ExecutionAlgorithm.VWAP, self.algorithms[ExecutionAlgorithm.SIMPLE]
                )
            else:
                return self.algorithms.get(
                    self.default_algorithm, self.algorithms[ExecutionAlgorithm.SIMPLE]
                )

        except Exception as e:
            logger.error(f"Failed to select algorithm: {e}")
            return self.algorithms[ExecutionAlgorithm.SIMPLE]

    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an execution"""
        try:
            if execution_id in self.active_executions:
                task = self.active_executions[execution_id]
                if task.done():
                    return {"status": "completed", "result": task.result()}
                else:
                    return {"status": "running"}
            else:
                # Check execution history
                for record in self.execution_history:
                    if record["request"].request_id == execution_id:
                        return {"status": "completed", "result": record["result"]}

            return None

        except Exception as e:
            logger.error(f"Failed to get execution status: {e}")
            return None

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        try:
            stats = {
                "total_executions": len(self.execution_history),
                "active_executions": len(self.active_executions),
                "registered_venues": len(self.liquidity_analyzer.venues),
                "available_algorithms": list(self.algorithms.keys()),
                "algorithm_usage": defaultdict(int),
                "venue_usage": defaultdict(int),
                "avg_execution_time_ms": 0.0,
                "avg_fill_rate": 0.0,
            }

            # Calculate statistics from history
            if self.execution_history:
                execution_times = []
                fill_rates = []

                for record in self.execution_history:
                    result = record["result"]
                    request = record["request"]

                    # Execution time
                    exec_time = (
                        result.completed_at - result.started_at
                    ).total_seconds() * 1000
                    execution_times.append(exec_time)

                    # Fill rate
                    fill_rate = result.filled_quantity / max(1, request.quantity)
                    fill_rates.append(fill_rate)

                    # Algorithm usage
                    stats["algorithm_usage"][request.algorithm.value] += 1

                    # Venue usage
                    for venue_id in result.venue_breakdown.keys():
                        stats["venue_usage"][venue_id] += 1

                if execution_times:
                    stats["avg_execution_time_ms"] = np.mean(execution_times)

                if fill_rates:
                    stats["avg_fill_rate"] = np.mean(fill_rates)

            return dict(stats)

        except Exception as e:
            logger.error(f"Failed to get routing statistics: {e}")
            return {}


# Utility functions
def create_smart_order_router(config: Config) -> SmartOrderRouter:
    """Create and return smart order router instance"""
    return SmartOrderRouter(config)


def create_liquidity_venue(
    venue_id: str, venue_name: str, venue_type: str = "exchange", **kwargs
) -> LiquidityVenue:
    """Create a liquidity venue"""
    return LiquidityVenue(
        venue_id=venue_id, venue_name=venue_name, venue_type=venue_type, **kwargs
    )


async def execute_arbitrage_with_smart_routing(
    router: SmartOrderRouter,
    opportunity: ArbitrageOpportunity,
    execution_params: Dict[str, Any] = None,
) -> Dict[str, ExecutionResult]:
    """Execute arbitrage opportunity using smart order routing"""
    try:
        execution_params = execution_params or {}

        # Create buy order
        buy_request = OrderRequest(
            request_id=str(uuid.uuid4()),
            market_id=opportunity.buy_market_id,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=opportunity.quantity,
            price=opportunity.buy_price,
            algorithm=ExecutionAlgorithm(execution_params.get("algorithm", "simple")),
            time_limit_seconds=execution_params.get("timeout", 30),
        )

        # Create sell order
        sell_request = OrderRequest(
            request_id=str(uuid.uuid4()),
            market_id=opportunity.sell_market_id,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=opportunity.quantity,
            price=opportunity.sell_price,
            algorithm=ExecutionAlgorithm(execution_params.get("algorithm", "simple")),
            time_limit_seconds=execution_params.get("timeout", 30),
        )

        # Execute orders concurrently
        buy_result, sell_result = await asyncio.gather(
            router.execute_order(buy_request), router.execute_order(sell_request)
        )

        return {"buy": buy_result, "sell": sell_result}

    except Exception as e:
        logger.error(f"Smart routing arbitrage execution failed: {e}")
        raise
