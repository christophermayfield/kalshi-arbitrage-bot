"""
Triangular Arbitrage Strategy Implementation
Advanced three-market arbitrage opportunities detection and execution
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import networkx as nx
from decimal import Decimal, getcontext

from ..utils.performance_cache import PerformanceCache

# Set high precision for arbitrage calculations
getcontext().prec = 28

logger = logging.getLogger(__name__)


class ArbitrageType(Enum):
    """Types of triangular arbitrage"""

    FORWARD = "forward"  # A->B->C->A
    REVERSE = "reverse"  # A->C->B->A
    LOOP = "loop"  # Complex multi-step loops


class OpportunityStatus(Enum):
    """Status of arbitrage opportunity"""

    DETECTED = "detected"
    VALIDATING = "validating"
    VALIDATED = "validated"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class TriangularPath:
    """Represents a triangular arbitrage path"""

    path: List[str]  # e.g., ['BTC', 'ETH', 'USDT']
    exchanges: List[str]  # Exchange for each leg
    rates: List[float]  # Exchange rate for each leg
    fees: List[float]  # Trading fee for each leg
    profit_pct: float  # Net profit percentage
    net_rate: float  # Net exchange rate
    volume: float  # Maximum tradable volume
    confidence: float  # Confidence in the opportunity
    timestamp: datetime = field(default_factory=datetime.now)
    liquidity_score: float = 0.0  # Liquidity assessment
    execution_time_estimate: float = 0.0  # Estimated execution time


@dataclass
class TriangularExecution:
    """Execution details for triangular arbitrage"""

    path: TriangularPath
    trade_amounts: List[float]  # Amount for each leg
    expected_profit: float  # Expected profit
    actual_profit: float = 0.0  # Actual profit
    execution_times: List[float] = field(default_factory=list)
    status: OpportunityStatus = OpportunityStatus.DETECTED
    errors: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None


class TriangularArbitrageEngine:
    """
    Advanced triangular arbitrage detection and execution engine
    with multi-exchange support and real-time opportunity monitoring
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.triangular_config = config.get("triangular_arbitrage", {})

        # Trading parameters
        self.min_profit_threshold = self.triangular_config.get(
            "min_profit_threshold", 0.001
        )  # 0.1%
        self.max_path_length = self.triangular_config.get("max_path_length", 3)
        self.min_liquidity = self.triangular_config.get("min_liquidity", 10000)
        self.max_execution_time = self.triangular_config.get(
            "max_execution_time", 5.0
        )  # seconds

        # Fee structures
        self.default_fee = self.triangular_config.get("default_fee", 0.001)  # 0.1%
        self.exchange_fees = self.triangular_config.get("exchange_fees", {})

        # Market data
        self.order_books: Dict[str, Dict] = defaultdict(
            dict
        )  # exchange -> symbol -> orderbook
        self.tickers: Dict[str, Dict] = defaultdict(
            dict
        )  # exchange -> symbol -> ticker
        self.exchange_rates: Dict[str, Dict] = defaultdict(dict)

        # Opportunity tracking
        self.active_opportunities: Dict[str, TriangularPath] = {}
        self.execution_history: deque = deque(maxlen=1000)
        self.performance_metrics: Dict[str, Any] = defaultdict(float)

        # Execution tracking
        self.pending_executions: Dict[str, TriangularExecution] = {}
        self.execution_locks: Set[str] = set()

        # Performance cache
        self.cache = PerformanceCache(
            redis_url=config.get("redis_url", "redis://localhost:6379"),
            default_ttl=10,  # 10 second TTL for arbitrage data
        )

        # Supported exchanges and symbols
        self.supported_exchanges = self.triangular_config.get(
            "supported_exchanges", ["binance", "coinbase", "kraken"]
        )
        self.supported_symbols = self.triangular_config.get("supported_symbols", [])

        # Network graph for path finding
        self.exchange_graphs: Dict[str, nx.DiGraph] = {}

        logger.info("Triangular Arbitrage Engine initialized")

    async def initialize(self) -> None:
        """Initialize the triangular arbitrage engine"""
        try:
            # Initialize exchange graphs
            await self._initialize_exchange_graphs()

            # Load historical data
            await self._load_historical_data()

            # Start opportunity monitoring
            asyncio.create_task(self._opportunity_monitoring_loop())

            logger.info("Triangular Arbitrage Engine initialized successfully")

        except Exception as e:
            logger.error(f"Triangular Arbitrage Engine initialization failed: {e}")
            raise

    async def scan_triangular_opportunities(self) -> List[TriangularPath]:
        """Scan for triangular arbitrage opportunities across all exchanges"""
        opportunities = []

        try:
            # Update market data
            await self._update_market_data()

            # Scan each exchange
            for exchange in self.supported_exchanges:
                exchange_opps = await self._scan_exchange_opportunities(exchange)
                opportunities.extend(exchange_opps)

                # Scan cross-exchange opportunities
                cross_exchange_opps = await self._scan_cross_exchange_opportunities(
                    exchange
                )
                opportunities.extend(cross_exchange_opps)

            # Filter and rank opportunities
            filtered_opps = await self._filter_and_rank_opportunities(opportunities)

            # Cache opportunities
            await self.cache.set(
                "triangular_opportunities",
                [self._serialize_path(opp) for opp in filtered_opps],
                ttl=5,
            )

            return filtered_opps

        except Exception as e:
            logger.error(f"Triangular opportunity scanning failed: {e}")
            return []

    async def _scan_exchange_opportunities(self, exchange: str) -> List[TriangularPath]:
        """Scan for triangular opportunities within a single exchange"""
        opportunities = []

        try:
            graph = self.exchange_graphs.get(exchange)
            if not graph:
                return opportunities

            # Find all simple cycles of length 3
            symbols = list(graph.nodes())

            for i, base in enumerate(symbols):
                for j, quote in enumerate(symbols[i + 1 :], i + 1):
                    for k, bridge in enumerate(symbols[j + 1 :], j + 1):
                        # Check all possible triangular paths
                        paths_to_check = [
                            [base, quote, bridge, base],
                            [base, bridge, quote, base],
                            [quote, base, bridge, quote],
                            [quote, bridge, base, quote],
                            [bridge, base, quote, bridge],
                            [bridge, quote, base, bridge],
                        ]

                        for path in paths_to_check:
                            opportunity = await self._calculate_triangular_opportunity(
                                exchange, path
                            )
                            if (
                                opportunity
                                and opportunity.profit_pct > self.min_profit_threshold
                            ):
                                opportunities.append(opportunity)

        except Exception as e:
            logger.error(f"Exchange opportunity scanning failed for {exchange}: {e}")

        return opportunities

    async def _scan_cross_exchange_opportunities(
        self, exchange: str
    ) -> List[TriangularPath]:
        """Scan for cross-exchange triangular opportunities"""
        opportunities = []

        try:
            # Get supported symbols for this exchange
            exchange_symbols = set(self.tickers[exchange].keys())

            # Check combinations with other exchanges
            for other_exchange in self.supported_exchanges:
                if other_exchange == exchange:
                    continue

                other_symbols = set(self.tickers[other_exchange].keys())

                # Find common symbols for cross-exchange arbitrage
                common_symbols = exchange_symbols.intersection(other_symbols)

                if len(common_symbols) >= 3:
                    # Build cross-exchange graph
                    cross_opp = await self._calculate_cross_exchange_opportunity(
                        exchange, other_exchange, list(common_symbols)[:3]
                    )
                    if cross_opp and cross_opp.profit_pct > self.min_profit_threshold:
                        opportunities.append(cross_opp)

        except Exception as e:
            logger.error(f"Cross-exchange opportunity scanning failed: {e}")

        return opportunities

    async def _calculate_triangular_opportunity(
        self, exchange: str, path: List[str]
    ) -> Optional[TriangularPath]:
        """Calculate triangular arbitrage opportunity for a specific path"""
        try:
            if len(path) != 4 or path[0] != path[3]:
                return None

            rates = []
            fees = []
            volumes = []

            # Calculate each leg of the triangle
            for i in range(3):
                from_symbol = path[i]
                to_symbol = path[i + 1]

                # Get exchange rate and fee
                rate, fee, volume = await self._get_exchange_rate_and_fee(
                    exchange, from_symbol, to_symbol
                )

                if rate is None or fee is None or volume is None:
                    return None

                rates.append(rate)
                fees.append(fee)
                volumes.append(volume)

            # Calculate net rate and profit
            net_rate = (
                rates[0]
                * (1 - fees[0])
                * rates[1]
                * (1 - fees[1])
                * rates[2]
                * (1 - fees[2])
            )
            profit_pct = (net_rate - 1.0) * 100

            # Calculate maximum tradable volume
            max_volume = min(volumes)

            # Assess liquidity
            liquidity_score = await self._assess_liquidity(exchange, path, max_volume)

            # Estimate execution time
            execution_time = await self._estimate_execution_time(exchange, path)

            # Calculate confidence based on market conditions
            confidence = await self._calculate_confidence(exchange, path, profit_pct)

            return TriangularPath(
                path=path[:3],  # Remove duplicate end symbol
                exchanges=[exchange] * 3,
                rates=rates,
                fees=fees,
                profit_pct=profit_pct,
                net_rate=net_rate,
                volume=max_volume,
                confidence=confidence,
                liquidity_score=liquidity_score,
                execution_time_estimate=execution_time,
            )

        except Exception as e:
            logger.error(f"Triangular opportunity calculation failed: {e}")
            return None

    async def _calculate_cross_exchange_opportunity(
        self, exchange1: str, exchange2: str, symbols: List[str]
    ) -> Optional[TriangularPath]:
        """Calculate cross-exchange triangular opportunity"""
        try:
            if len(symbols) != 3:
                return None

            # Create path that leverages price differences between exchanges
            path = symbols
            exchanges = [exchange1, exchange2, exchange1]  # Alternate exchanges

            rates = []
            fees = []
            volumes = []

            for i in range(3):
                from_symbol = path[i]
                to_symbol = path[(i + 1) % 3]
                exchange = exchanges[i]

                rate, fee, volume = await self._get_exchange_rate_and_fee(
                    exchange, from_symbol, to_symbol
                )

                if rate is None:
                    return None

                rates.append(rate)
                fees.append(fee)
                volumes.append(volume)

            # Calculate cross-exchange opportunity
            net_rate = (
                rates[0]
                * (1 - fees[0])
                * rates[1]
                * (1 - fees[1])
                * rates[2]
                * (1 - fees[2])
            )
            profit_pct = (net_rate - 1.0) * 100

            if profit_pct < self.min_profit_threshold:
                return None

            # Account for transfer fees between exchanges
            transfer_fee = self.triangular_config.get(
                "cross_exchange_transfer_fee", 0.001
            )
            net_rate *= 1 - transfer_fee
            profit_pct = (net_rate - 1.0) * 100

            return TriangularPath(
                path=path,
                exchanges=exchanges,
                rates=rates,
                fees=fees,
                profit_pct=profit_pct,
                net_rate=net_rate,
                volume=min(volumes),
                confidence=0.8,  # Lower confidence for cross-exchange
                liquidity_score=0.7,
                execution_time_estimate=8.0,  # Longer for cross-exchange
            )

        except Exception as e:
            logger.error(f"Cross-exchange opportunity calculation failed: {e}")
            return None

    async def _get_exchange_rate_and_fee(
        self, exchange: str, from_symbol: str, to_symbol: str
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get exchange rate, fee, and volume for a trading pair"""
        try:
            # Form trading pair
            pair = f"{from_symbol}/{to_symbol}"
            reverse_pair = f"{to_symbol}/{from_symbol}"

            # Check order book
            if exchange in self.order_books and pair in self.order_books[exchange]:
                orderbook = self.order_books[exchange][pair]

                # Get best bid/ask
                best_bid = (
                    orderbook.get("bids", [])[0] if orderbook.get("bids") else None
                )
                best_ask = (
                    orderbook.get("asks", [])[0] if orderbook.get("asks") else None
                )

                if best_bid and best_ask:
                    rate = (best_bid[0] + best_ask[0]) / 2  # Mid price
                    volume = min(best_bid[1], best_ask[1])  # Minimum liquidity

                    # Get exchange fee
                    fee = self.exchange_fees.get(exchange, self.default_fee)

                    return rate, fee, volume

            # Check reverse pair
            if (
                exchange in self.order_books
                and reverse_pair in self.order_books[exchange]
            ):
                orderbook = self.order_books[exchange][reverse_pair]

                best_bid = (
                    orderbook.get("bids", [])[0] if orderbook.get("bids") else None
                )
                best_ask = (
                    orderbook.get("asks", [])[0] if orderbook.get("asks") else None
                )

                if best_bid and best_ask:
                    rate = 1.0 / ((best_bid[0] + best_ask[0]) / 2)  # Inverse rate
                    volume = min(best_bid[1], best_ask[1])

                    fee = self.exchange_fees.get(exchange, self.default_fee)

                    return rate, fee, volume

            return None, None, None

        except Exception as e:
            logger.error(
                f"Exchange rate retrieval failed for {exchange} {from_symbol}/{to_symbol}: {e}"
            )
            return None, None, None

    async def _assess_liquidity(
        self, exchange: str, path: List[str], volume: float
    ) -> float:
        """Assess liquidity score for the opportunity"""
        try:
            # Get average 24h volume for the pairs
            total_volume = 0
            pair_count = 0

            for i in range(3):
                from_symbol = path[i]
                to_symbol = path[(i + 1) % 3]
                pair = f"{from_symbol}/{to_symbol}"

                if exchange in self.tickers and pair in self.tickers[exchange]:
                    ticker = self.tickers[exchange][pair]
                    volume_24h = ticker.get("volume_24h", 0)
                    total_volume += volume_24h
                    pair_count += 1

            if pair_count == 0:
                return 0.0

            avg_volume = total_volume / pair_count

            # Normalize liquidity score (0-1)
            if avg_volume >= 1000000:  # $1M+ daily volume
                return 1.0
            elif avg_volume >= 100000:  # $100K+ daily volume
                return 0.8
            elif avg_volume >= 10000:  # $10K+ daily volume
                return 0.6
            else:
                return 0.4

        except Exception as e:
            logger.error(f"Liquidity assessment failed: {e}")
            return 0.5

    async def _estimate_execution_time(self, exchange: str, path: List[str]) -> float:
        """Estimate execution time for the triangular arbitrage"""
        try:
            # Base time per leg
            base_time = {
                "binance": 0.1,  # 100ms
                "coinbase": 0.15,  # 150ms
                "kraken": 0.2,  # 200ms
                "bitfinex": 0.12,  # 120ms
                "huobi": 0.18,  # 180ms
            }

            leg_time = base_time.get(exchange, 0.2)

            # Add network latency and processing overhead
            network_overhead = 0.05  # 50ms
            processing_overhead = 0.1  # 100ms

            total_time = (leg_time * 3) + network_overhead + processing_overhead

            # Add delay for cross-exchange transfers if applicable
            cross_exchange_delay = 0
            for i in range(3):
                from_symbol = path[i]
                to_symbol = path[(i + 1) % 3]

                # Check if this requires cross-exchange transfer
                if await self._requires_cross_exchange_transfer(from_symbol, to_symbol):
                    cross_exchange_delay += 2.0  # 2 seconds per transfer

            return total_time + cross_exchange_delay

        except Exception as e:
            logger.error(f"Execution time estimation failed: {e}")
            return 5.0  # Conservative estimate

    async def _calculate_confidence(
        self, exchange: str, path: List[str], profit_pct: float
    ) -> float:
        """Calculate confidence score for the opportunity"""
        try:
            base_confidence = 0.7

            # Adjust based on profit margin (higher profit = higher confidence up to a point)
            if profit_pct < 0.1:
                profit_factor = profit_pct * 10  # Scale 0.1% -> 1.0
            elif profit_pct < 0.5:
                profit_factor = 1.0
            elif profit_pct < 1.0:
                profit_factor = (
                    1.0 - (profit_pct - 0.5) * 0.5
                )  # Very high profit might be stale
            else:
                profit_factor = 0.75

            # Adjust based on liquidity
            liquidity_score = await self._assess_liquidity(exchange, path, 0)
            liquidity_factor = liquidity_score

            # Adjust based on market volatility (lower volatility = higher confidence)
            volatility_factor = 0.8  # Default

            # Adjust based on exchange reliability
            exchange_reliability = {
                "binance": 0.95,
                "coinbase": 0.90,
                "kraken": 0.85,
                "bitfinex": 0.80,
                "huobi": 0.85,
            }
            reliability_factor = exchange_reliability.get(exchange, 0.8)

            # Combine factors
            confidence = (
                base_confidence
                * profit_factor
                * liquidity_factor
                * volatility_factor
                * reliability_factor
            )

            return min(1.0, max(0.0, confidence))

        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    async def _requires_cross_exchange_transfer(
        self, from_symbol: str, to_symbol: str
    ) -> bool:
        """Check if transfer requires cross-exchange movement"""
        # Simplified logic - in production, this would check wallet locations
        return False

    async def _filter_and_rank_opportunities(
        self, opportunities: List[TriangularPath]
    ) -> List[TriangularPath]:
        """Filter and rank opportunities by quality"""
        try:
            # Filter by minimum profit and liquidity
            filtered = []
            for opp in opportunities:
                if (
                    opp.profit_pct >= self.min_profit_threshold
                    and opp.liquidity_score >= 0.3
                    and opp.confidence >= 0.5
                    and opp.execution_time_estimate <= self.max_execution_time
                ):
                    filtered.append(opp)

            # Sort by综合 score (profit, confidence, liquidity, speed)
            def score_opportunity(opp: TriangularPath) -> float:
                profit_score = min(opp.profit_pct / 1.0, 1.0)  # Normalize to 1% max
                confidence_score = opp.confidence
                liquidity_score = opp.liquidity_score
                speed_score = max(
                    0, 1.0 - opp.execution_time_estimate / 5.0
                )  # 5s as baseline

                # Weighted average
                return (
                    profit_score * 0.4
                    + confidence_score * 0.3
                    + liquidity_score * 0.2
                    + speed_score * 0.1
                )

            # Sort and return top opportunities
            filtered.sort(key=score_opportunity, reverse=True)

            # Limit to top 10 opportunities
            return filtered[:10]

        except Exception as e:
            logger.error(f"Opportunity filtering and ranking failed: {e}")
            return []

    async def execute_triangular_arbitrage(
        self, opportunity: TriangularPath, initial_amount: float
    ) -> Optional[TriangularExecution]:
        """Execute a triangular arbitrage opportunity"""
        try:
            execution_id = f"triangular_{datetime.now().timestamp()}"

            # Create execution object
            execution = TriangularExecution(
                path=opportunity,
                expected_profit=initial_amount * (opportunity.profit_pct / 100),
                status=OpportunityStatus.EXECUTING,
                trade_amounts=[initial_amount],
            )

            # Check if opportunity is still valid
            current_opp = await self._validate_opportunity(opportunity)
            if not current_opp or current_opp.profit_pct < self.min_profit_threshold:
                execution.status = OpportunityStatus.EXPIRED
                execution.errors.append("Opportunity expired during execution")
                return execution

            # Execute each leg
            current_amount = initial_amount

            for i in range(3):
                from_symbol = opportunity.path[i]
                to_symbol = (
                    opportunity.path[(i + 1) % 3] if i < 2 else opportunity.path[0]
                )
                exchange = opportunity.exchanges[i]

                # Execute trade
                leg_result = await self._execute_leg(
                    exchange, from_symbol, to_symbol, current_amount
                )

                if not leg_result["success"]:
                    execution.status = OpportunityStatus.FAILED
                    execution.errors.append(
                        f"Leg {i + 1} failed: {leg_result.get('error', 'Unknown error')}"
                    )
                    return execution

                # Update amount for next leg
                current_amount = leg_result["output_amount"]
                execution.trade_amounts.append(current_amount)
                execution.execution_times.append(leg_result["execution_time"])

            # Calculate actual profit
            execution.actual_profit = current_amount - initial_amount
            execution.status = OpportunityStatus.COMPLETED
            execution.end_time = datetime.now()

            # Update performance metrics
            await self._update_performance_metrics(execution)

            # Cache execution result
            await self.cache.set(
                f"triangular_execution:{execution_id}",
                self._serialize_execution(execution),
                ttl=3600,
            )

            return execution

        except Exception as e:
            logger.error(f"Triangular arbitrage execution failed: {e}")
            return None

    async def _execute_leg(
        self, exchange: str, from_symbol: str, to_symbol: str, amount: float
    ) -> Dict[str, Any]:
        """Execute a single leg of the triangular arbitrage"""
        try:
            start_time = datetime.now()

            # In production, this would make actual API calls
            # For now, simulate execution

            # Simulate execution time
            await asyncio.sleep(0.1)  # 100ms execution time

            # Get exchange rate
            rate, _, _ = await self._get_exchange_rate_and_fee(
                exchange, from_symbol, to_symbol
            )

            if rate is None:
                return {"success": False, "error": "No exchange rate available"}

            # Calculate output amount (accounting for fees)
            fee = self.exchange_fees.get(exchange, self.default_fee)
            output_amount = amount * rate * (1 - fee)

            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": True,
                "input_amount": amount,
                "output_amount": output_amount,
                "exchange_rate": rate,
                "fee": fee,
                "execution_time": execution_time,
            }

        except Exception as e:
            logger.error(f"Leg execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _validate_opportunity(
        self, opportunity: TriangularPath
    ) -> Optional[TriangularPath]:
        """Validate that opportunity still exists"""
        try:
            # Recalculate with current market data
            exchange = opportunity.exchanges[0]
            current_opp = await self._calculate_triangular_opportunity(
                exchange, opportunity.path + [opportunity.path[0]]
            )

            return current_opp

        except Exception as e:
            logger.error(f"Opportunity validation failed: {e}")
            return None

    async def _update_performance_metrics(self, execution: TriangularExecution) -> None:
        """Update performance metrics after execution"""
        try:
            self.execution_history.append(execution)

            # Update counters
            self.performance_metrics["total_executions"] += 1

            if execution.status == OpportunityStatus.COMPLETED:
                self.performance_metrics["successful_executions"] += 1
                self.performance_metrics["total_profit"] += execution.actual_profit

                # Calculate execution time
                if execution.start_time and execution.end_time:
                    execution_time = (
                        execution.end_time - execution.start_time
                    ).total_seconds()
                    self.performance_metrics["avg_execution_time"] = (
                        self.performance_metrics.get("avg_execution_time", 0)
                        * (self.performance_metrics["total_executions"] - 1)
                        + execution_time
                    ) / self.performance_metrics["total_executions"]
            else:
                self.performance_metrics["failed_executions"] += 1

            # Calculate success rate
            if self.performance_metrics["total_executions"] > 0:
                self.performance_metrics["success_rate"] = (
                    self.performance_metrics["successful_executions"]
                    / self.performance_metrics["total_executions"]
                )

        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")

    async def _initialize_exchange_graphs(self) -> None:
        """Initialize network graphs for each exchange"""
        try:
            for exchange in self.supported_exchanges:
                graph = nx.DiGraph()

                # Add nodes (symbols)
                for symbol in self.supported_symbols:
                    graph.add_node(symbol)

                # Add edges (trading pairs)
                # This would be populated with actual trading pairs
                # For now, add sample edges
                graph.add_edge("BTC", "USDT", weight=1.0)
                graph.add_edge("USDT", "BTC", weight=1.0)
                graph.add_edge("ETH", "BTC", weight=1.0)
                graph.add_edge("BTC", "ETH", weight=1.0)
                graph.add_edge("ETH", "USDT", weight=1.0)
                graph.add_edge("USDT", "ETH", weight=1.0)

                self.exchange_graphs[exchange] = graph

            logger.info(
                f"Initialized graphs for {len(self.supported_exchanges)} exchanges"
            )

        except Exception as e:
            logger.error(f"Exchange graph initialization failed: {e}")

    async def _load_historical_data(self) -> None:
        """Load historical market data"""
        try:
            # This would load actual historical data from database or files
            # For now, initialize empty structures
            logger.info("Historical data loaded")

        except Exception as e:
            logger.error(f"Historical data loading failed: {e}")

    async def _update_market_data(self) -> None:
        """Update market data from exchanges"""
        try:
            # This would fetch real-time data from exchanges
            # For now, simulate market data

            for exchange in self.supported_exchanges:
                # Simulate order book updates
                for symbol in ["BTC/USDT", "ETH/BTC", "ETH/USDT"]:
                    if exchange not in self.order_books:
                        self.order_books[exchange] = {}

                    # Generate mock orderbook
                    mid_price = np.random.uniform(40000, 60000)  # Mock BTC price
                    spread = mid_price * 0.001  # 0.1% spread

                    self.order_books[exchange][symbol] = {
                        "bids": [[mid_price - spread / 2, np.random.uniform(10, 100)]],
                        "asks": [[mid_price + spread / 2, np.random.uniform(10, 100)]],
                    }

                # Simulate ticker updates
                self.tickers[exchange] = {
                    symbol: {
                        "last": mid_price,
                        "volume_24h": np.random.uniform(1000000, 10000000),
                    }
                    for symbol, mid_price in [
                        ("BTC/USDT", 50000),
                        ("ETH/BTC", 0.07),
                        ("ETH/USDT", 3500),
                    ]
                }

        except Exception as e:
            logger.error(f"Market data update failed: {e}")

    async def _opportunity_monitoring_loop(self) -> None:
        """Background loop for continuous opportunity monitoring"""
        while True:
            try:
                # Scan for opportunities
                opportunities = await self.scan_triangular_opportunities()

                # Update active opportunities
                self.active_opportunities = {
                    f"{'-'.join(opp.path)}_{datetime.now().isoformat()}": opp
                    for opp in opportunities
                }

                # Log high-quality opportunities
                high_quality_opps = [
                    opp for opp in opportunities if opp.profit_pct > 0.5
                ]
                if high_quality_opps:
                    logger.info(
                        f"Found {len(high_quality_opps)} high-quality triangular opportunities"
                    )
                    for opp in high_quality_opps[:3]:  # Log top 3
                        logger.info(
                            f"  {opp.path}: {opp.profit_pct:.3f}% profit, {opp.confidence:.2f} confidence"
                        )

                await asyncio.sleep(1)  # Scan every second

            except Exception as e:
                logger.error(f"Opportunity monitoring loop error: {e}")
                await asyncio.sleep(5)

    def _serialize_path(self, path: TriangularPath) -> Dict[str, Any]:
        """Serialize TriangularPath for caching"""
        return {
            "path": path.path,
            "exchanges": path.exchanges,
            "profit_pct": path.profit_pct,
            "net_rate": path.net_rate,
            "volume": path.volume,
            "confidence": path.confidence,
            "liquidity_score": path.liquidity_score,
            "timestamp": path.timestamp.isoformat(),
        }

    def _serialize_execution(self, execution: TriangularExecution) -> Dict[str, Any]:
        """Serialize TriangularExecution for caching"""
        return {
            "path": self._serialize_path(execution.path),
            "expected_profit": execution.expected_profit,
            "actual_profit": execution.actual_profit,
            "status": execution.status.value,
            "start_time": execution.start_time.isoformat(),
            "end_time": execution.end_time.isoformat() if execution.end_time else None,
        }

    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            return {
                "opportunities": {
                    "current_active": len(self.active_opportunities),
                    "total_detected": self.performance_metrics.get(
                        "total_opportunities", 0
                    ),
                    "avg_profit_pct": np.mean(
                        [opp.profit_pct for opp in self.active_opportunities.values()]
                    )
                    if self.active_opportunities
                    else 0,
                },
                "executions": {
                    "total": self.performance_metrics.get("total_executions", 0),
                    "successful": self.performance_metrics.get(
                        "successful_executions", 0
                    ),
                    "failed": self.performance_metrics.get("failed_executions", 0),
                    "success_rate": self.performance_metrics.get("success_rate", 0),
                    "avg_execution_time": self.performance_metrics.get(
                        "avg_execution_time", 0
                    ),
                },
                "profitability": {
                    "total_profit": self.performance_metrics.get("total_profit", 0),
                    "avg_profit_per_execution": (
                        self.performance_metrics.get("total_profit", 0)
                        / max(
                            1, self.performance_metrics.get("successful_executions", 1)
                        )
                    ),
                    "best_trade": max(
                        [e.actual_profit for e in self.execution_history], default=0
                    ),
                },
                "performance": {
                    "opportunities_per_hour": len(self.execution_history)
                    / max(1, len(self.execution_history) / 3600),
                    "profit_per_hour": (
                        sum(e.actual_profit for e in self.execution_history)
                        / max(1, len(self.execution_history) / 3600)
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {}


# Utility functions
async def create_triangular_engine(config: Dict[str, Any]) -> TriangularArbitrageEngine:
    """Create and initialize triangular arbitrage engine"""
    engine = TriangularArbitrageEngine(config)
    await engine.initialize()
    return engine


def calculate_triangular_profit(rates: List[float], fees: List[float]) -> float:
    """Calculate profit from triangular rates"""
    net_rate = 1.0
    for rate, fee in zip(rates, fees):
        net_rate *= rate * (1 - fee)
    return (net_rate - 1.0) * 100


def assess_opportunity_quality(opportunity: TriangularPath) -> float:
    """Assess overall quality of triangular opportunity"""
    weights = {
        "profit": 0.3,
        "confidence": 0.25,
        "liquidity": 0.2,
        "speed": 0.15,
        "stability": 0.1,
    }

    profit_score = min(opportunity.profit_pct / 1.0, 1.0)
    confidence_score = opportunity.confidence
    liquidity_score = opportunity.liquidity_score
    speed_score = max(0, 1.0 - opportunity.execution_time_estimate / 5.0)
    stability_score = 0.8  # Would be calculated from historical volatility

    return (
        profit_score * weights["profit"]
        + confidence_score * weights["confidence"]
        + liquidity_score * weights["liquidity"]
        + speed_score * weights["speed"]
        + stability_score * weights["stability"]
    )
