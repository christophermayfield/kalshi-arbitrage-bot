"""High-frequency scanning loop with ultra-low latency opportunity detection."""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from src.core.orderbook import OrderBook
from src.core.arbitrage import ArbitrageDetector, ArbitrageOpportunity
from src.core.statistical_arbitrage import StatisticalArbitrageDetector
from src.clients.kalshi_client import KalshiClient
from src.utils.logging_utils import get_logger
from src.utils.performance_cache import get_cache_manager
from src.core.opportunity_scoring import get_scoring_service

logger = get_logger("high_frequency_scanner")


class HighFrequencyScanner:
    """Ultra-low latency market scanner for high-frequency arbitrage."""

    def __init__(
        self,
        client: KalshiClient,
        detector: ArbitrageDetector,
        statistical_detector: Optional[StatisticalArbitrageDetector] = None,
        scan_interval_ms: int = 100,  # 100ms default
        max_concurrent_scans: int = 20,
        cache_ttl_seconds: int = 5,
    ):
        """Initialize high-frequency scanner."""
        self.client = client
        self.detector = detector
        self.statistical_detector = statistical_detector

        # Scanning configuration
        self.scan_interval_ms = scan_interval_ms
        self.max_concurrent_scans = max_concurrent_scans
        self.cache_ttl_seconds = cache_ttl_seconds

        # Performance tracking
        self.scan_times: List[float] = []
        self.opportunity_counts: List[int] = []
        self.latency_stats: Dict[str, float] = {}

        # State management
        self.running = False
        self.active_scans: Set[str] = set()
        self.last_scan_time = 0.0
        self.market_priorities: Dict[str, int] = {}

        # Caching
        self.cache_manager = get_cache_manager()
        self.orderbook_cache: Dict[str, Dict[str, Any]] = {}
        self.opportunity_cache: Dict[str, Any] = {}

        # Thread pool for blocking operations
        self.thread_pool = ThreadPoolExecutor(max_workers=10)

        # Services
        self.scoring_service = get_scoring_service()

        # Market tracking
        self.watched_markets: Set[str] = set()
        self.market_priorities = {}
        self.last_market_updates: Dict[str, float] = {}

        logger.info(
            f"High-frequency scanner initialized with {scan_interval_ms}ms interval"
        )

    async def start_scanning(self):
        """Start the high-frequency scanning loop."""
        logger.info("Starting high-frequency scanning...")
        self.running = True

        # Initialize watched markets
        await self._initialize_watched_markets()

        # Start scanning loops
        tasks = [
            asyncio.create_task(self._opportunity_scan_loop()),
            asyncio.create_task(self._market_data_scan_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
        ]

        if self.statistical_detector:
            tasks.append(asyncio.create_task(self._statistical_scan_loop()))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("High-frequency scanning cancelled")
        finally:
            self.running = False

    async def stop_scanning(self):
        """Stop the high-frequency scanning."""
        logger.info("Stopping high-frequency scanning...")
        self.running = False

        # Cancel all active scans
        for scan_id in list(self.active_scans):
            logger.debug(f"Cancelling scan: {scan_id}")
            # Cancelation logic here

        self.active_scans.clear()

    async def _opportunity_scan_loop(self):
        """Main opportunity detection loop."""
        while self.running:
            scan_start = time.time()

            try:
                # Get current opportunities
                opportunities = await self._scan_for_opportunities()

                # Score and prioritize opportunities
                if opportunities:
                    scored_opps = (
                        await self.scoring_service.score_and_rank_opportunities(
                            opportunities
                        )
                    )

                    # Update opportunity cache
                    await self._update_opportunity_cache(scored_opps)

                    # Log top opportunities
                    top_3 = scored_opps[:3]
                    logger.info(
                        f"Top opportunities: {[(opp.opportunity_id, opp.total_score) for opp in top_3]}"
                    )

                # Update performance stats
                scan_time = time.time() - scan_start
                self.scan_times.append(scan_time)
                self.opportunity_counts.append(len(opportunities))

                # Keep only recent stats
                if len(self.scan_times) > 1000:
                    self.scan_times = self.scan_times[-500:]
                    self.opportunity_counts = self.opportunity_counts[-500:]

                # Cache results
                await self._cache_scan_results(len(opportunities), scan_time)

            except Exception as e:
                logger.error(f"Opportunity scan error: {e}")

            # Sleep until next scan
            await asyncio.sleep(self.scan_interval_ms / 1000)

    async def _market_data_scan_loop(self):
        """Separate loop for market data updates."""
        while self.running:
            try:
                await self._update_market_data()
                await asyncio.sleep(0.5)  # Update market data every 500ms

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Market data scan error: {e}")

    async def _statistical_scan_loop(self):
        """Statistical arbitrage scanning loop."""
        if not self.statistical_detector:
            return

        while self.running:
            try:
                # Get current orderbooks
                orderbooks = await self._get_cached_orderbooks()

                if orderbooks:
                    # Scan for statistical opportunities
                    stat_opps = self.statistical_detector.find_opportunities(
                        orderbooks,
                        min_profit_cents=5,  # Lower threshold for statistical
                    )

                    if stat_opps:
                        # Score statistical opportunities
                        scored_stat_opps = (
                            await self.scoring_service.score_and_rank_opportunities(
                                stat_opps
                            )
                        )

                        # Update cache
                        await self._update_statistical_cache(scored_stat_opps)

                        logger.info(f"Found {len(stat_opps)} statistical opportunities")

                await asyncio.sleep(2.0)  # Statistical scans every 2 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Statistical scan error: {e}")

    async def _performance_monitoring_loop(self):
        """Performance monitoring and optimization loop."""
        while self.running:
            try:
                await self._update_performance_metrics()
                await asyncio.sleep(10.0)  # Update every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")

    async def _initialize_watched_markets(self):
        """Initialize list of markets to watch."""
        try:
            # Get initial market list
            markets_response = self.client.get_markets(status="open", limit=200)
            markets = markets_response.get("markets", [])

            # Filter and prioritize markets
            self.watched_markets = set()
            for market in markets:
                market_id = market.get("market_id", "")
                if market_id:
                    self.watched_markets.add(market_id)

                    # Calculate market priority based on volume and activity
                    priority = self._calculate_market_priority(market)
                    self.market_priorities[market_id] = priority

            logger.info(f"Initialized {len(self.watched_markets)} watched markets")

            # Cache market list
            await self.cache_manager.cache.set_orderbook(
                "watched_markets", list(self.watched_markets)
            )

        except Exception as e:
            logger.error(f"Failed to initialize watched markets: {e}")

    def _calculate_market_priority(self, market: Dict[str, Any]) -> int:
        """Calculate priority score for a market."""
        priority = 5  # Default priority

        # Volume-based priority
        volume = market.get("volume_24h", 0)
        if volume > 100000:
            priority = 1  # Highest
        elif volume > 50000:
            priority = 2
        elif volume > 10000:
            priority = 3
        elif volume > 1000:
            priority = 4

        # Liquidity-based priority
        liquidity = market.get("liquidity_score", 0)
        if liquidity > 80:
            priority = max(1, priority - 1)
        elif liquidity < 20:
            priority = min(10, priority + 2)

        return priority

    async def _scan_for_opportunities(self) -> List[ArbitrageOpportunity]:
        """Scan for arbitrage opportunities with ultra-low latency."""
        scan_id = f"opp_scan_{int(time.time() * 1000)}"
        self.active_scans.add(scan_id)

        try:
            # Get current orderbooks
            orderbooks = await self._get_cached_orderbooks()

            if not orderbooks:
                return []

            # Use thread pool for CPU-intensive arbitrage detection
            loop = asyncio.get_running_loop()

            # Run arbitrage detection in thread pool
            opportunities = await loop.run_in_executor(
                self.thread_pool, self.detector.scan_for_opportunities, orderbooks
            )

            return opportunities or []

        finally:
            self.active_scans.discard(scan_id)

    async def _get_cached_orderbooks(self) -> Dict[str, OrderBook]:
        """Get cached orderbooks with fallback to API."""
        orderbooks = {}
        missing_markets = []

        # Try cache first
        for market_id in self.watched_markets:
            cached_orderbook = await self.cache_manager.cache.get_orderbook(market_id)
            if cached_orderbook and cached_orderbook:
                try:
                    orderbooks[market_id] = OrderBook.from_api_response(
                        cached_orderbook
                    )
                except Exception as e:
                    logger.debug(
                        f"Failed to deserialize cached orderbook for {market_id}: {e}"
                    )
                    missing_markets.append(market_id)
            else:
                missing_markets.append(market_id)

        # Fetch missing orderbooks concurrently
        if missing_markets:
            logger.debug(f"Fetching {len(missing_markets)} missing orderbooks")

            # Batch API calls for missing orderbooks
            fetch_tasks = []
            for market_id in missing_markets:
                task = asyncio.create_task(self._fetch_orderbook_fast(market_id))
                fetch_tasks.append((market_id, task))

            # Wait for all fetches
            results = await asyncio.gather(
                *[task for _, task in fetch_tasks], return_exceptions=True
            )

            # Process results
            for i, (market_id, _) in enumerate(fetch_tasks):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"Failed to fetch orderbook for {market_id}: {result}")
                else:
                    orderbooks[market_id] = result

                    # Update cache
                    if result:
                        orderbook_data = {
                            "bids": [bid.__dict__ for bid in result.bids],
                            "asks": [ask.__dict__ for ask in result.asks],
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                        await self.cache_manager.cache.set_orderbook(
                            market_id, orderbook_data
                        )

                        self.last_market_updates[market_id] = time.time()

        return orderbooks

    async def _fetch_orderbook_fast(self, market_id: str) -> Optional[OrderBook]:
        """Fetch orderbook with optimized timing."""
        try:
            # Use fast API call
            orderbook_data = self.client.get_market_orderbook(market_id)
            if orderbook_data:
                return OrderBook.from_api_response(orderbook_data)
            return None

        except Exception as e:
            logger.debug(f"Fast orderbook fetch failed for {market_id}: {e}")
            return None

    async def _update_market_data(self):
        """Update market data for priority calculations."""
        try:
            # Get current market status
            market_status = await self.cache_manager.cache.get_market_status()

            if market_status:
                # Update market priorities based on real-time conditions
                self._update_market_priorities(market_status)

        except Exception as e:
            logger.error(f"Market data update error: {e}")

    def _update_market_priorities(self, market_status: Dict[str, Any]):
        """Update market priorities based on current conditions."""
        # This would update market priorities based on real-time conditions
        # For now, keep current priorities
        pass

    async def _update_opportunity_cache(self, opportunities: List[Any]):
        """Update opportunity cache with latest results."""
        current_time = time.time()

        # Remove old opportunities
        cutoff_time = current_time - self.cache_ttl_seconds
        old_opps = [
            opp_id
            for opp_id, data in self.opportunity_cache.items()
            if isinstance(data, dict) and data.get("timestamp", 0) < cutoff_time
        ]

        for opp_id in old_opps:
            del self.opportunity_cache[opp_id]

        # Add new opportunities
        for opp in opportunities:
            self.opportunity_cache[opp.opportunity_id] = {
                "opportunity": opp.__dict__ if hasattr(opp, "__dict__") else opp,
                "timestamp": current_time,
                "score": getattr(opp, "total_score", 0),
            }

        # Cache updated opportunity list
        await self.cache_manager.cache.set_opportunities(
            [data.get("opportunity", {}) for data in self.opportunity_cache.values()],
            "high_frequency_cache",
        )

        logger.debug(
            f"Opportunity cache updated: {len(opportunities)} new, {len(old_opps)} removed"
        )

    async def _update_statistical_cache(self, opportunities: List[Any]):
        """Update statistical opportunity cache."""
        current_time = time.time()

        # Separate statistical opportunities
        stat_opps = {
            f"stat_{opp.opportunity_id}": {
                "opportunity": opp.__dict__ if hasattr(opp, "__dict__") else opp,
                "timestamp": current_time,
                "score": getattr(opp, "total_score", 0),
            }
            for opp in opportunities
        }

        # Merge with existing cache
        self.opportunity_cache.update(stat_opps)

        logger.debug(
            f"Statistical cache updated: {len(opportunities)} new opportunities"
        )

    async def _cache_scan_results(self, opportunity_count: int, scan_time: float):
        """Cache scan performance results."""
        self.last_scan_time = time.time()

        # Update performance metrics
        await self.cache_manager.cache.increment_metrics(
            "opportunities_found", opportunity_count
        )
        await self.cache_manager.cache.increment_metrics(
            "scan_time_ms", int(scan_time * 1000)
        )

        # Cache latest scan summary
        scan_summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "opportunity_count": opportunity_count,
            "scan_time_ms": scan_time * 1000,
            "scans_per_second": 1000 / self.scan_interval_ms,
            "active_markets": len(self.watched_markets),
        }

        await self.cache_manager.cache.set_orderbook(
            "latest_scan_summary", scan_summary
        )

    async def _update_performance_metrics(self):
        """Update and analyze performance metrics."""
        if len(self.scan_times) < 10:
            return

        # Calculate recent performance
        recent_times = self.scan_times[-100:]
        recent_counts = self.opportunity_counts[-100:]

        avg_scan_time = sum(recent_times) / len(recent_times)
        avg_opportunity_count = sum(recent_counts) / len(recent_counts)

        # Calculate success rate (scans that found opportunities)
        successful_scans = len([c for c in recent_counts if c > 0])
        success_rate = successful_scans / len(recent_counts) if recent_counts else 0

        # Performance classification
        if avg_scan_time < 0.05:  # < 50ms
            performance_level = "excellent"
        elif avg_scan_time < 0.1:  # < 100ms
            performance_level = "good"
        elif avg_scan_time < 0.2:  # < 200ms
            performance_level = "fair"
        else:
            performance_level = "poor"

        # Update metrics
        self.latency_stats.update(
            {
                "avg_scan_time_ms": avg_scan_time * 1000,
                "avg_opportunities_per_scan": avg_opportunity_count,
                "success_rate_percent": success_rate * 100,
                "performance_level": performance_level,
                "scans_per_second": 1000 / self.scan_interval_ms,
            }
        )

        # Cache performance metrics
        await self.cache_manager.cache.set_orderbook(
            "scanner_performance", self.latency_stats
        )

        # Log performance summary
        if int(time.time()) % 60 == 0:  # Every minute
            logger.info(
                f"Scanner performance: {performance_level}, "
                f"avg time: {avg_scan_time * 1000:.1f}ms, "
                f"avg opportunities: {avg_opportunity_count:.1f}, "
                f"success rate: {success_rate * 100:.1f}%"
            )

    def get_scan_stats(self) -> Dict[str, Any]:
        """Get comprehensive scanning statistics."""
        return {
            "running": self.running,
            "scan_interval_ms": self.scan_interval_ms,
            "watched_markets": len(self.watched_markets),
            "active_scans": len(self.active_scans),
            "performance": self.latency_stats,
            "cache_stats": {
                "opportunity_cache_size": len(self.opportunity_cache),
                "last_scan_time": self.last_scan_time,
            },
        }

    async def optimize_scan_interval(self):
        """Dynamically optimize scan interval based on performance."""
        if len(self.scan_times) < 50:
            return

        # Calculate optimal interval based on opportunity frequency
        recent_counts = self.opportunity_counts[-50:]
        total_opportunities = sum(recent_counts)
        opportunities_per_second = total_opportunities / (
            len(recent_counts) * self.scan_interval_ms / 1000
        )

        # Optimize for higher opportunity discovery rate
        if opportunities_per_second < 0.1:  # Less than 0.1 opportunities per second
            # Decrease interval to scan more frequently
            new_interval = max(50, self.scan_interval_ms - 25)
        elif opportunities_per_second > 1.0:  # More than 1 opportunity per second
            # Increase interval to reduce API pressure
            new_interval = min(500, self.scan_interval_ms + 25)
        else:
            return  # No change needed

        if new_interval != self.scan_interval_ms:
            logger.info(
                f"Optimizing scan interval: {self.scan_interval_ms}ms -> {new_interval}ms"
            )
            self.scan_interval_ms = new_interval


class ScanningService:
    """Service for managing high-frequency scanning operations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize scanning service."""
        self.config = config
        self.scanner = None
        self.background_tasks = []

    async def initialize(
        self,
        client: KalshiClient,
        detector: ArbitrageDetector,
        statistical_detector: Optional[StatisticalArbitrageDetector] = None,
    ):
        """Initialize scanning service with components."""
        logger.info("Initializing scanning service...")

        scan_config = {
            "scan_interval_ms": self.config.get("scanning.scan_interval_ms", 100),
            "max_concurrent_scans": self.config.get(
                "scanning.max_concurrent_scans", 20
            ),
            "cache_ttl_seconds": self.config.get("scanning.cache_ttl_seconds", 5),
        }

        self.scanner = HighFrequencyScanner(
            client=client,
            detector=detector,
            statistical_detector=statistical_detector,
            **scan_config,
        )

        logger.info("Scanning service initialized")

    async def start(self):
        """Start scanning service."""
        if not self.scanner:
            raise RuntimeError("Scanner not initialized")

        # Start scanner
        scan_task = asyncio.create_task(self.scanner.start_scanning())
        self.background_tasks.append(scan_task)

        # Start optimization loop
        optimization_task = asyncio.create_task(self._optimization_loop())
        self.background_tasks.append(optimization_task)

        logger.info("Scanning service started")

    async def stop(self):
        """Stop scanning service."""
        logger.info("Stopping scanning service...")

        # Stop scanner
        if self.scanner:
            await self.scanner.stop_scanning()

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.background_tasks.clear()
        logger.info("Scanning service stopped")

    async def _optimization_loop(self):
        """Background optimization loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute

                if self.scanner:
                    await self.scanner.optimize_scan_interval()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")

    def get_scanner(self) -> Optional[HighFrequencyScanner]:
        """Get the scanner instance."""
        return self.scanner


# Global scanning service instance
_scanning_service: Optional[ScanningService] = None


def get_scanning_service(config: Optional[Dict[str, Any]] = None) -> ScanningService:
    """Get global scanning service instance."""
    global _scanning_service

    if _scanning_service is None and config is not None:
        _scanning_service = ScanningService(config)

    return _scanning_service


async def initialize_scanning(
    config: Dict[str, Any],
    client: KalshiClient,
    detector: ArbitrageDetector,
    statistical_detector: Optional[StatisticalArbitrageDetector] = None,
) -> ScanningService:
    """Initialize and return scanning service."""
    service = get_scanning_service(config)
    await service.initialize(client, detector, statistical_detector)
    await service.start()
    return service


async def shutdown_scanning():
    """Shutdown global scanning service."""
    global _scanning_service

    if _scanning_service is not None:
        await _scanning_service.stop()
        _scanning_service = None
