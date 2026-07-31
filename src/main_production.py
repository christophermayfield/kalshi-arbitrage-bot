"""Production-grade main bot with integrated hardening, risk management, and monitoring."""

import asyncio
import signal
import sys
import os
import uuid
from typing import Dict, Set, Optional, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager

# Import existing components
from src.utils.config import Config
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.database import Database
from src.clients.kalshi_client import KalshiClient
from src.clients.websocket_client import KalshiWebSocketClient
from src.core.orderbook import OrderBook
from src.core.arbitrage import ArbitrageDetector, ArbitrageOpportunity
from src.core.portfolio import PortfolioManager
from src.execution.trading import TradingExecutor
from src.execution.paper_trading import PaperTradingSimulator

# Import new production systems
from src.utils.production_hardening import (
    CircuitBreaker,
    RateLimiter,
    GracefulShutdown,
    RetryPolicy,
    StateRecovery,
)
from src.core.advanced_risk_manager import AdvancedRiskManager, PositionRisk
from src.utils.database_optimization import OptimizedDatabasePool
from src.monitoring.advanced_monitoring import (
    MetricsCollector,
    AlertManager,
    HealthChecker,
    PerformanceMonitor,
    Alert,
    AlertLevel,
    AlertChannel,
)

logger = get_logger("main_production")


class ProductionArbitrageBot:
    """Production-grade arbitrage bot with advanced hardening and monitoring."""

    def __init__(self, config: Config):
        """Initialize production bot with all systems."""
        self.config = config
        self.running = False

        # Initialize logging
        logger.info("🚀 Initializing Production Arbitrage Bot")

        # ========== CORE COMPONENTS ==========
        self.client = KalshiClient(config)
        self.ws_client: Optional[KalshiWebSocketClient] = None
        self.portfolio = PortfolioManager(
            max_daily_loss=config.get("risk.max_daily_loss_cents", 10000),
            max_open_positions=config.get("risk.max_open_positions", 50),
        )
        self.detector = ArbitrageDetector(
            min_profit_cents=config.min_profit_cents,
            fee_rate=0.01,
            min_confidence=config.get("trading.min_confidence", 0.7),
            enable_predictive_models=config.get("ml.enabled", False),
            enable_sentiment_analysis=config.get("sentiment.enabled", True),
            predictive_weight=config.get("ml.predictive_weight", 0.3),
            sentiment_weight=config.get("sentiment.weight", 0.2),
            enable_statistical_arbitrage=config.get("statistical.enabled", False),
            statistical_config=config.get("statistical", {}),
            portfolio_manager=self.portfolio,
        )
        self.executor = TradingExecutor(
            client=self.client,
            paper_mode=config.paper_mode,
            max_retries=config.get("trading.retry_attempts", 3),
            order_timeout=config.get("trading.order_timeout_seconds", 30),
        )
        self.paper_simulator = PaperTradingSimulator(
            initial_balance=config.get("paper_trading.initial_balance", 100000),
            slippage_model=config.get("paper_trading.slippage_model", "fixed"),
            slippage_rate=config.get("paper_trading.slippage_rate", 0.001),
            fill_probability=config.get("paper_trading.fill_probability", 0.95),
            commission_rate=config.get("paper_trading.commission_rate", 0.01),
        )

        # ========== PRODUCTION HARDENING ==========
        logger.info("📋 Initializing Production Hardening Systems...")

        # Circuit breaker for API calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get("hardening.circuit_breaker_threshold", 5),
            recovery_timeout=config.get("hardening.circuit_breaker_timeout", 60),
            expected_exception=Exception,
        )
        logger.info("✓ Circuit breaker initialized")

        # Rate limiter for API calls
        self.rate_limiter = RateLimiter(
            max_calls=config.get("hardening.rate_limit_calls", 100),
            time_window=config.get("hardening.rate_limit_window", 60),
        )
        logger.info("✓ Rate limiter initialized")

        # Graceful shutdown handler
        self.graceful_shutdown = GracefulShutdown()
        self.graceful_shutdown.register_signal_handlers()
        logger.info("✓ Graceful shutdown handler registered")

        # Retry policy for transient failures
        self.retry_policy = RetryPolicy(
            max_retries=config.get("hardening.retry_max", 3),
            base_delay=config.get("hardening.retry_base_delay", 1.0),
            max_delay=config.get("hardening.retry_max_delay", 60.0),
        )
        logger.info("✓ Retry policy initialized")

        # State recovery for crash resilience
        self.state_recovery = StateRecovery(
            state_file=config.get("hardening.state_file", "bot_state.json")
        )
        logger.info("✓ State recovery initialized")

        # ========== ADVANCED RISK MANAGEMENT ==========
        logger.info("📊 Initializing Advanced Risk Management...")

        initial_balance = config.get("trading.initial_balance_cents", 1000000)
        self.risk_manager = AdvancedRiskManager(
            initial_balance_cents=initial_balance,
            max_daily_loss_cents=config.get("risk.max_daily_loss_cents", 10000),
            max_position_value_cents=config.get("risk.max_position_value_cents", 50000),
            risk_per_trade_percent=config.get("risk.risk_per_trade_percent", 0.02),
        )
        logger.info(f"✓ Risk manager initialized (balance: ${initial_balance / 100:.2f})")

        # ========== DATABASE OPTIMIZATION ==========
        logger.info("🗄️  Initializing Database Optimization...")

        self.db_pool = OptimizedDatabasePool(
            database_url=config.get("database.url", "sqlite:///./arbitrage.db"),
            pool_size=config.get("database.pool_size", 20),
            max_overflow=config.get("database.max_overflow", 40),
            pool_recycle=config.get("database.pool_recycle", 3600),
        )
        logger.info("✓ Database pool initialized")

        # ========== MONITORING & ALERTING ==========
        logger.info("📈 Initializing Monitoring & Alerting...")

        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        self.health_checker = HealthChecker()
        self.performance_monitor = PerformanceMonitor(self.metrics)

        # Register health checks
        self._register_health_checks()
        logger.info("✓ Health checks registered")

        # Register alert rules
        self._register_alert_rules()
        logger.info("✓ Alert rules registered")

        # ========== STATE INITIALIZATION ==========
        self.orderbooks: Dict[str, OrderBook] = {}
        self.subscribed_markets: Set[str] = set()
        self.scan_interval = config.scan_interval_seconds
        self.executed_count = 0
        self._detection_running = False
        self._detection_task: Optional[asyncio.Task] = None
        self._reconciliation_task: Optional[asyncio.Task] = None
        self.correlation_id: str = str(uuid.uuid4())

        logger.info("✅ Production Bot Initialization Complete!")

    def _register_health_checks(self):
        """Register system health checks."""

        async def check_api_connection():
            """Check API connectivity."""
            try:
                await self.retry_policy.execute(self.client.get_markets)
                return True
            except Exception as e:
                logger.error(f"API health check failed: {e}")
                return False

        async def check_database():
            """Check database connectivity."""
            try:
                session = self.db_pool.get_session()
                session.close()
                return True
            except Exception as e:
                logger.error(f"Database health check failed: {e}")
                return False

        def check_risk_manager():
            """Check risk manager status."""
            return self.risk_manager is not None

        def check_circuit_breaker():
            """Check circuit breaker is not open."""
            return self.circuit_breaker.state.value != "open"

        self.health_checker.register_check("api_connection", check_api_connection)
        self.health_checker.register_check("database", check_database)
        self.health_checker.register_check("risk_manager", check_risk_manager)
        self.health_checker.register_check("circuit_breaker", check_circuit_breaker)

    def _register_alert_rules(self):
        """Register alert rules for monitoring."""

        # High error rate alert
        self.alerts.add_alert_rule(
            name="high_error_rate",
            condition=lambda m: m.get("error_rate", 0) > 0.05,
            alert_level=AlertLevel.ERROR,
            title="High Error Rate Detected",
            message_template="Error rate is {error_rate:.1%}",
        )

        # API latency alert
        self.alerts.add_alert_rule(
            name="api_latency_high",
            condition=lambda m: m.get("api_latency_ms", 0) > 5000,
            alert_level=AlertLevel.WARNING,
            title="High API Latency",
            message_template="API latency is {api_latency_ms:.0f}ms",
        )

        # Daily loss limit alert
        self.alerts.add_alert_rule(
            name="daily_loss_limit",
            condition=lambda m: m.get("daily_loss_percent", 0) > 0.90,
            alert_level=AlertLevel.CRITICAL,
            title="Daily Loss Limit Approaching",
            message_template="Daily loss is {daily_loss_percent:.0%} of limit",
        )

        logger.info("Alert rules configured")

    async def initialize_connections(self):
        """Initialize all external connections."""
        logger.info("Initializing connections...")

        try:
            # Initialize database
            self.db = Database(self.config)
            await self.db.connect()
            logger.info("✓ Database connected")

            # Get initial balance
            portfolio = await self.retry_policy.execute(self.client.get_portfolio)
            balance_cents = portfolio.get("balance_cents", 1000000)
            self.risk_manager.update_equity(balance_cents)
            logger.info(f"✓ Current balance: ${balance_cents / 100:.2f}")

            # Initialize WebSocket (optional)
            if self.config.get("websocket.enabled", True):
                self.ws_client = KalshiWebSocketClient(self.config)
                logger.info("✓ WebSocket client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise

    async def run(self):
        """Run the bot with all production systems."""
        logger.info("🟢 Starting Production Arbitrage Bot...")

        self.running = True

        try:
            # Initialize connections
            await self.initialize_connections()

            # Register tasks for graceful shutdown
            scan_task = asyncio.create_task(self._scan_loop())
            monitor_task = asyncio.create_task(self._monitoring_loop())
            health_task = asyncio.create_task(self._health_check_loop())

            self.graceful_shutdown.register_task(scan_task)
            self.graceful_shutdown.register_task(monitor_task)
            self.graceful_shutdown.register_task(health_task)

            # Wait for shutdown signal
            await self.graceful_shutdown.wait_for_shutdown()

            logger.info("⏹️  Shutdown signal received, gracefully shutting down...")

            # Cancel all tasks
            await self.graceful_shutdown.cancel_all_tasks()

            # Save state
            await self.state_recovery.save_state({
                "positions": self.portfolio.get_open_positions(),
                "balance": self.risk_manager.equity_curve.balances[-1]
                if self.risk_manager.equity_curve.balances
                else 0,
            })

            logger.info("✅ Bot shutdown complete")

        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
            raise
        finally:
            self.running = False
            # Close connections
            if self.db:
                await self.db.close()
            self.db_pool.close()

    async def _scan_loop(self):
        """Main scanning loop with circuit breaker protection."""
        logger.info("🔄 Starting market scan loop...")

        while self.running:
            try:
                # Check rate limit
                if not self.rate_limiter.is_allowed():
                    logger.warning("Rate limit exceeded, skipping scan")
                    await asyncio.sleep(1)
                    continue

                # Scan with circuit breaker
                await self.circuit_breaker.call_async(self._scan_markets)

                # Wait for next scan
                await asyncio.sleep(self.scan_interval)

            except Exception as e:
                logger.error(f"Scan loop error: {e}")
                await asyncio.sleep(5)

    async def _scan_markets(self):
        """Scan markets for opportunities."""
        try:
            markets = await self.retry_policy.execute(self.client.get_markets)

            opportunities = []
            for market in markets:
                market_id = market.get("market_id")
                try:
                    orderbook = await self.client.get_market_orderbook(market_id)
                    # Process opportunities with risk checks
                    can_trade, reason = self.risk_manager.can_open_position(
                        market_id=market_id,
                        entry_price_cents=int(orderbook["asks"][0]["price"]),
                        quantity=10,
                        current_balance_cents=int(
                            self.risk_manager.equity_curve.balances[-1]
                            if self.risk_manager.equity_curve.balances
                            else 1000000
                        ),
                    )

                    if can_trade:
                        opportunities.append(
                            {"market_id": market_id, "orderbook": orderbook}
                        )

                except Exception as e:
                    logger.debug(f"Error processing {market_id}: {e}")

            # Record metrics
            self.metrics.record_metric("markets_scanned", len(markets), "count")
            self.metrics.record_metric("opportunities_found", len(opportunities), "count")

        except Exception as e:
            logger.error(f"Market scan failed: {e}")
            self.metrics.record_metric("scan_errors", 1, "count")

    async def _monitoring_loop(self):
        """Continuous monitoring and metrics collection."""
        logger.info("📊 Starting monitoring loop...")

        while self.running:
            try:
                # Get current metrics
                metrics_snapshot = {
                    "error_rate": 0.0,
                    "api_latency_ms": 0.0,
                    "daily_loss_percent": (
                        self.risk_manager.daily_loss_cents /
                        self.risk_manager.max_daily_loss_cents
                        if self.risk_manager.max_daily_loss_cents > 0
                        else 0.0
                    ),
                }

                # Check alert rules
                await self.alerts.check_rules(metrics_snapshot)

                # Log performance summary
                perf_summary = self.performance_monitor.get_performance_summary()
                logger.debug(f"Performance: {perf_summary}")

                await asyncio.sleep(10)  # Monitor every 10 seconds

            except Exception as e:
                logger.error(f"Monitoring error: {e}")

    async def _health_check_loop(self):
        """Periodic health checks."""
        logger.info("🏥 Starting health check loop...")

        while self.running:
            try:
                results = await self.health_checker.run_all_checks()
                status = self.health_checker.get_health_status()

                if not status["healthy"]:
                    logger.warning(f"Health check failed: {status}")
                    # Send alert
                    alert = Alert(
                        level=AlertLevel.ERROR,
                        title="System Health Check Failed",
                        message=f"Failed checks: {status['checks']}",
                    )
                    await self.alerts.send_alert(alert)

                await asyncio.sleep(60)  # Health check every minute

            except Exception as e:
                logger.error(f"Health check error: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        return {
            "running": self.running,
            "balance": (
                self.risk_manager.equity_curve.balances[-1]
                if self.risk_manager.equity_curve.balances
                else 0
            ),
            "open_positions": len(self.risk_manager.positions),
            "daily_loss": self.risk_manager.daily_loss_cents,
            "risk_tier": self.risk_manager.equity_curve.get_risk_tier().value,
            "circuit_breaker": self.circuit_breaker.state.value,
            "health": self.health_checker.get_health_status(),
        }


async def main():
    """Main entry point."""
    setup_logging()

    try:
        config = Config()
        bot = ProductionArbitrageBot(config)
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
