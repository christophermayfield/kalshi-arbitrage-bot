"""Enhanced main bot with high-frequency trading capabilities."""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Optional, Any

from src.utils.config import Config
from src.utils.logging_utils import setup_logging, get_logger
from src.clients.kalshi_client import KalshiClient
from src.core.orderbook import OrderBook
from src.core.arbitrage import ArbitrageDetector
from src.core.portfolio import PortfolioManager
from src.core.statistical_arbitrage import StatisticalArbitrageDetector
from src.core.realtime_market_data import RealTimeMarketDataManager, MarketUpdate
from src.strategies.kalshi_correlation import KalshiCorrelationStrategy
from src.core.auto_tuning_engine import AutoTuningEngine, PerformanceMetrics
from src.execution.high_frequency_trading import TradingEngineManager
from src.core.opportunity_scoring import get_scoring_service, ScoringService
from src.core.high_frequency_scanner import get_scanning_service, ScanningService
from src.utils.performance_cache import initialize_cache, shutdown_cache
from src.backtesting.advanced_backtesting import (
    run_comprehensive_backtest,
    create_backtest_config,
)
from src.analytics.performance_attribution import (
    create_attribution_engine,
    record_arbitrage_execution,
)

logger = get_logger("main")


class EnhancedArbitrageBot:
    """Enhanced arbitrage bot with high-frequency trading and statistical strategies."""

    def __init__(self, config: Config):
        """Initialize enhanced arbitrage bot."""
        self.config = config
        self.running = False

        # Core components
        self.client = KalshiClient(config)
        self.detector = ArbitrageDetector(
            min_profit_cents=config.get("trading.min_profit_cents", 10),
            fee_rate=0.01,
            enable_statistical_arbitrage=config.get("statistical.enabled", False),
            statistical_config=config.get("statistical", {}),
        )
        self.portfolio = PortfolioManager(
            max_daily_loss=config.get("risk.max_daily_loss_cents", 10000),
            max_open_positions=config.get("risk.max_open_positions", 50),
        )

        # Real-time market data manager
        self.market_data_manager = RealTimeMarketDataManager(config, self.client)
        self.market_data_manager.add_update_callback(self._on_market_update)
        self.market_data_manager.add_error_callback(self._on_market_data_error)

        # High-frequency components
        self.trading_engine = None
        self.scanner = None
        self.scoring_service = get_scoring_service()

        # Performance tracking for WebSocket
        self.websocket_updates_received = 0

        # Statistical arbitrage
        self.statistical_detector = None
        if config.get("statistical.enabled", False):
            self.statistical_detector = StatisticalArbitrageDetector(
                strategies=["mean_reversion", "pairs_trading"],
                config=config.get("statistical", {}),
            )

        # Kalshi correlation strategy
        self.correlation_strategy = None
        if config.get("correlation_strategy.enabled", False):
            self.correlation_strategy = KalshiCorrelationStrategy(config)

        # Auto-tuning engine
        self.auto_tuning_engine = None
        if config.get("auto_tuning.enabled", False):
            self.auto_tuning_engine = AutoTuningEngine(config)

        # Performance tracking
        self.start_time = None
        self.total_opportunities = 0
        self.executed_opportunities = 0
        self.daily_pnl = 0.0

        # Backtesting components
        self.backtesting_enabled = config.get("backtesting.enabled", False)
        self.attribution_engine = (
            create_attribution_engine(config)
            if config.get("performance_attribution.enabled", False)
            else None
        )

        # Configuration
        self.scan_interval_ms = (
            config.get("monitoring.scan_interval_seconds", 1.0) * 1000
        )  # Convert to ms
        self.enable_hf_trading = config.get("trading.high_frequency.enabled", False)
        self.auto_mode = config.get("trading.auto_mode", False)

        logger.info("Enhanced arbitrage bot initialized")

    async def start(self) -> None:
        """Start the enhanced arbitrage bot."""
        logger.info("Starting enhanced arbitrage bot...")
        self.running = True
        self.start_time = datetime.utcnow()

        # Initialize correlation strategy
        if self.correlation_strategy:
            logger.info("Kalshi correlation strategy enabled")

        # Initialize backtesting if enabled
        if self.backtesting_enabled:
            logger.info(
                "Backtesting mode enabled - opportunity detection will run in backtest mode"
            )

        # Initialize auto-tuning engine
        if self.auto_tuning_engine:
            logger.info("Auto-tuning engine enabled")

        # Initialize scoring service
        await self.scoring_service.initialize()
        logger.info("Scoring service initialized")

        # Initialize scanner
        from src.core.high_frequency_scanner import initialize_scanning

        self.scanner = await initialize_scanning(
            self.config, self.client, self.detector, self.statistical_detector
        )
        logger.info("High-frequency scanner initialized")

    async def _check_exchange_status(self) -> None:
        """Check exchange and market status."""
        try:
            status = self.client.get_exchange_status()
            exchange_active = status.get("exchange_active", False)
            trading_active = status.get("trading_active", False)

            if not exchange_active:
                logger.warning("Exchange is not active")
            if not trading_active:
                logger.warning("Trading is not active")

            logger.info(
                f"Exchange status - Active: {exchange_active}, Trading: {trading_active}"
            )

        except Exception as e:
            logger.error(f"Failed to check exchange status: {e}")

    async def _main_trading_loop(self) -> None:
        """Main trading loop with high-frequency capabilities."""
        logger.info("Starting main trading loop...")

        # Concurrent monitoring and trading tasks
        tasks = [
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._opportunity_monitoring_loop()),
        ]

        if self.auto_mode and self.trading_engine:
            tasks.append(asyncio.create_task(self._auto_trading_loop()))
        else:
            tasks.append(asyncio.create_task(self._manual_trading_loop()))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Main trading loop cancelled")
        except Exception as e:
            logger.error(f"Main trading loop error: {e}")
        finally:
            self.running = False

    async def _auto_trading_loop(self) -> None:
        """Fully automated trading loop."""
        logger.info("Starting automated trading loop...")

        while self.running:
            try:
                all_opportunities = []

                # Get opportunities from scanner
                if self.scanner:
                    top_opportunities = (
                        await self.scanner.get_scanner().get_top_opportunities(
                            max_count=5, min_score=5.0
                        )
                    )
                    all_opportunities.extend(top_opportunities)

                # Get correlation opportunities
                if self.correlation_strategy:
                    orderbooks = self.market_data_manager.get_all_orderbooks()
                    correlation_opps = await self.correlation_strategy.analyze_markets(
                        orderbooks
                    )
                    # Convert to arbitrage opportunities
                    for corr_opp in correlation_opps:
                        arbitrage_opp = corr_opp.to_arbitrage_opportunity()
                        all_opportunities.append(arbitrage_opp)

                if all_opportunities:
                    logger.info(
                        f"Found {len(all_opportunities)} opportunities for auto-execution"
                    )

                    # Execute top opportunities concurrently
                    for opp in all_opportunities[:5]:  # Limit to top 5
                        try:
                            if self.trading_engine:
                                # Execute with load balancing
                                result = await self.trading_engine.execute_with_load_balancing(
                                    opp
                                )

                                if result.success:
                                    self.executed_opportunities += 1
                                    logger.info(
                                        f"Auto-executed {opp.id}: profit={getattr(result, 'profit', 'N/A')}"
                                    )
                                else:
                                    logger.warning(
                                        f"Auto-execution failed for {opp.id}: {getattr(result, 'error_message', 'Unknown error')}"
                                    )

                        except Exception as e:
                            logger.error(f"Auto-execution error for {opp.id}: {e}")

                    # Update daily P&L
                    self._update_daily_pnl()
                else:
                    logger.debug("No opportunities found for auto-execution")

                # Wait before next cycle
                await asyncio.sleep(0.2)  # 200ms cycle time

            except Exception as e:
                logger.error(f"Auto trading loop error: {e}")
                await asyncio.sleep(1)  # Wait on error

    async def _manual_trading_loop(self) -> None:
        """Manual trading loop with opportunity detection only."""
        logger.info("Starting manual trading loop (opportunity detection only)...")

        while self.running:
            try:
                # Get opportunities from scanner
                opportunities = (
                    await self.scanner.get_scanner()._scan_for_opportunities()
                )

                if opportunities:
                    self.total_opportunities += len(opportunities)
                    logger.info(f"Found {len(opportunities)} opportunities")

                    # Log top opportunities for manual review
                    top_3 = opportunities[:3]
                    for i, opp in enumerate(top_3, 1):
                        logger.info(
                            f"  {i}. {opp.id}: {opp.net_profit_cents} cents profit"
                        )
                else:
                    logger.debug("No opportunities found")

                # Wait before next scan
                await asyncio.sleep(self.scan_interval_ms / 1000)

            except Exception as e:
                logger.error(f"Manual trading loop error: {e}")
                await asyncio.sleep(1)

    def _convert_to_arbitrage_opportunity(self, scored_opportunity):
        """Convert scored opportunity to arbitrage opportunity."""
        from src.core.arbitrage import ArbitrageOpportunity, ArbitrageType

        # This would need to convert from RealTimeScore to ArbitrageOpportunity
        # For now, create a mock opportunity
        return ArbitrageOpportunity(
            id=scored_opportunity.opportunity_id,
            type=ArbitrageType.CROSS_MARKET,
            market_id_1=scored_opportunity.market_id_1,
            market_id_2=scored_opportunity.market_id_2,
            buy_market_id=scored_opportunity.market_id_2,
            sell_market_id=scored_opportunity.market_id_1,
            buy_price=5000,  # Mock price
            sell_price=5500,  # Mock price
            quantity=100,
            profit_cents=int(scored_opportunity.total_score * 10),
            net_profit_cents=int(scored_opportunity.total_score * 10),
            confidence=scored_opportunity.total_score / 10,
        )

    async def _opportunity_monitoring_loop(self) -> None:
        """Monitor opportunities and update statistics."""
        while self.running:
            try:
                # Get current stats from all services
                stats = {}

                if self.scanner:
                    scanner_stats = self.scanner.get_scanner().get_scan_stats()
                    stats["scanner"] = scanner_stats

                if self.trading_engine:
                    trading_stats = (
                        await self.trading_engine.get_all_performance_stats()
                    )
                    stats["trading_engine"] = trading_stats

                if self.scoring_service:
                    scoring_stats = (
                        self.scoring_service.get_scorer().get_scoring_stats()
                    )
                    stats["scoring"] = scoring_stats

                # Log comprehensive status every minute
                if int(asyncio.get_event_loop().time()) % 60 == 0:
                    self._log_comprehensive_status(stats)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Opportunity monitoring error: {e}")
                await asyncio.sleep(5)

    async def _performance_monitoring_loop(self) -> None:
        """Monitor overall system performance."""
        while self.running:
            try:
                # Calculate uptime
                if self.start_time:
                    uptime = datetime.utcnow() - self.start_time
                    uptime_seconds = uptime.total_seconds()
                    uptime_minutes = uptime_seconds / 60
                    uptime_hours = uptime_minutes / 60

                    logger.info(f"Uptime: {uptime_hours:.1f} hours")
                    logger.info(f"Opportunities found: {self.total_opportunities}")
                    logger.info(
                        f"Opportunities executed: {self.executed_opportunities}"
                    )
                    logger.info(f"Daily P&L: ${self.daily_pnl:.2f}")

                # Update auto-tuning engine with performance metrics
                if self.auto_tuning_engine and self.executed_opportunities > 0:
                    performance_metrics = PerformanceMetrics(
                        total_profit_cents=self.daily_pnl * 100,  # Convert to cents
                        win_rate=0.7
                        if self.executed_opportunities > 0
                        else 0.0,  # Simplified
                        avg_profit_per_trade=(self.daily_pnl * 100)
                        / max(1, self.executed_opportunities),
                        opportunities_per_hour=self.total_opportunities
                        / max(1, uptime_hours),
                        market_volatility=0.1,  # Would be calculated from market data
                        avg_spread_cents=5.0,  # Would be calculated from orderbooks
                        avg_liquidity_score=50.0,  # Would be calculated from orderbooks
                    )
                    await self.auto_tuning_engine.update_performance(
                        performance_metrics
                    )

                await asyncio.sleep(300)  # Log every 5 minutes

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(30)

    def _update_daily_pnl(self):
        """Update daily P&L calculation."""
        # This would integrate with actual portfolio P&L
        # For now, use simple calculation
        self.daily_pnl += 10  # Mock daily profit

    def _log_comprehensive_status(self, stats: Dict[str, Any]):
        """Log comprehensive system status."""
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE SYSTEM STATUS")
        logger.info("=" * 60)

        # Scanner status
        if "scanner" in stats:
            scanner_stats = stats["scanner"]
            logger.info(f"Scanner Status:")
            logger.info(f"  Running: {scanner_stats.get('running', False)}")
            logger.info(f"  Watched Markets: {scanner_stats.get('watched_markets', 0)}")
            logger.info(f"  Active Scans: {scanner_stats.get('active_scans', 0)}")
            if "performance" in scanner_stats:
                perf = scanner_stats["performance"]
                logger.info(f"  Avg Scan Time: {perf.get('avg_scan_time_ms', 0):.1f}ms")
                logger.info(
                    f"  Success Rate: {perf.get('success_rate_percent', 0):.1f}%"
                )
                logger.info(
                    f"  Performance: {perf.get('performance_level', 'unknown')}"
                )

        # Trading engine status
        if "trading_engine" in stats:
            trading_stats = stats["trading_engine"]
            logger.info(f"Trading Engine Status:")
            for strategy, perf in trading_stats.items():
                logger.info(f"  {strategy}:")
                if isinstance(perf, dict):
                    logger.info(
                        f"    Total Executions: {perf.get('total_executions', 0)}"
                    )
                    logger.info(
                        f"    Avg Latency: {perf.get('avg_latency_ms', 0):.1f}ms"
                    )
                    logger.info(f"    Success Rate: {perf.get('success_rate', 0):.1f}%")

        # Scoring status
        if "scoring" in stats:
            scoring_stats = stats["scoring"]
            logger.info(f"Scoring Status:")
            logger.info(f"  Total Scored: {scoring_stats.get('total_scored', 0)}")
            logger.info(f"  Avg Score: {scoring_stats.get('avg_score', 0):.2f}")
            if "score_distribution" in scoring_stats:
                dist = scoring_stats["score_distribution"]
                logger.info(
                    f"  Distribution: Excellent={dist.get('excellent', 0)}, "
                    f"Very Good={dist.get('very_good', 0)}, "
                    f"Good={dist.get('good', 0)}, "
                    f"Fair={dist.get('fair', 0)}, "
                    f"Poor={dist.get('poor', 0)}"
                )

        logger.info("=" * 60)

    async def stop(self) -> None:
        """Stop the enhanced arbitrage bot."""
        logger.info("Stopping enhanced arbitrage bot...")
        self.running = False

        # Cancel all background tasks
        if self.trading_loop_task:
            self.trading_loop_task.cancel()
            try:
                await self.trading_loop_task
            except asyncio.CancelledError:
                pass

        if self.manual_trading_task:
            self.manual_trading_loop_task.cancel()
            try:
                await self.manual_trading_loop_task
            except asyncio.CancelledError:
                pass

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Bot stopped. Total executed: {self.executed_opportunities}")
        logger.info(f"Final daily P&L: ${self.daily_pnl:.2f}")

    async def _run_backtest_mode(self) -> None:
        """Run in backtesting mode - evaluate strategies without live trading."""
        logger.info("Running in backtesting mode...")

        while self.running:
            try:
                # Simulate market data for backtesting
                # In production, this would use real historical data
                await self._run_backtest_cycle()

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Backtest cycle error: {e}")
                await asyncio.sleep(5)

    async def _run_backtest_cycle(self) -> None:
        """Run one backtesting cycle."""
        try:
            # Generate sample opportunities (in production, this would scan real markets)
            sample_opportunities = await self._generate_sample_opportunities()

            if sample_opportunities:
                logger.info(f"Backtest found {len(sample_opportunities)} opportunities")

                # Score and filter opportunities
                filtered_opps = await self.detector.filter_by_threshold(
                    sample_opportunities
                )

                if filtered_opps:
                    for opp in filtered_opps:
                        confidence = getattr(opp, "confidence", 0.5)
                        expected_profit = getattr(opp, "total_score", 0.0)
                        logger.info(
                            f"Backtest opp: {opp.id}, confidence: {confidence:.2f}, expected profit: {expected_profit:.0f}"
                        )

                # Update backtesting metrics
                # This would calculate comprehensive backtest performance metrics
                logger.info(
                    f"Backtest cycle completed with {len(filtered_opps)} filtered opportunities"
                )

        except Exception as e:
            logger.error(f"Backtest cycle error: {e}")

    async def _generate_sample_opportunities(self) -> List[Any]:
        """Generate sample opportunities for backtesting demonstration."""
        # This would generate realistic market scenarios
        # For demonstration, create a few sample opportunities
        sample_opps = []

        # Sample opportunity 1
        sample_opps.append(
            {
                "id": "backtest_sample_1",
                "market_id_1": "PRESIDENTIAL-2024-DEM",
                "market_id_2": "PRESIDENTIAL-2024-REP",
                "buy_market_id": "PRESIDENTIAL-2024-REP",
                "sell_market_id": "PRESIDENTIAL-2024-DEM",
                "buy_price": 45,  # YES at 45 cents
                "sell_price": 55,  # NO at 55 cents
                "confidence": 0.8,
                "total_score": 85,
                "quantity": 1000,
            }
        )

        # Sample opportunity 2 (cross-market)
        sample_opps.append(
            {
                "id": "backtest_sample_2",
                "market_id_1": "BTC-PRICE-2024",
                "market_id_2": "ETH-PRICE-2024",
                "buy_market_id": "BTC-PRICE-2024",  # Buy BTC
                "sell_market_id": "ETH-PRICE-2024",  # Sell ETH
                "buy_price": 45000,  # BTC price
                "sell_price": 3000,  # ETH price
                "confidence": 0.7,
                "total_score": 75,
                "quantity": 10,
            }
        )

        return sample_opps
        logger.info(f"WebSocket updates received: {self.websocket_updates_received}")

        # Save auto-tuning parameters
        if self.auto_tuning_engine:
            await self.auto_tuning_engine.save_parameters()

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        status = {
            "running": self.running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
            if self.start_time
            else 0,
            "total_opportunities": self.total_opportunities,
            "executed_opportunities": self.executed_opportunities,
            "daily_pnl": self.daily_pnl,
            "auto_mode": self.auto_mode,
            "hf_trading_enabled": self.enable_hf_trading,
            "statistical_arbitrage_enabled": self.statistical_detector is not None,
            "correlation_strategy_enabled": self.correlation_strategy is not None,
            "auto_tuning_enabled": self.auto_tuning_engine is not None,
            "scan_interval_ms": self.scan_interval_ms,
            "websocket_updates_received": self.websocket_updates_received,
        }

        # Add auto-tuning status if enabled
        if self.auto_tuning_engine:
            status["auto_tuning"] = self.auto_tuning_engine.get_optimization_report()

        return status


async def main():
    """Main entry point for enhanced arbitrage bot."""
    config = Config()

    log_level = config.get("monitoring.log_level", "INFO")
    logger = setup_logging(level=log_level)

    bot = EnhancedArbitrageBot(config)

    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(bot.stop())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        await bot.stop()
    except Exception as e:
        logger.error(f"Bot error: {e}")
        await bot.stop()
        raise


if __name__ == "__main__":
    asyncio.run(main())
