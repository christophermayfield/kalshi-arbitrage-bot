"""
Phase 2 Enhanced Risk Management Integration Script
Brings together all risk management components for production deployment
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import signal
import sys

from src.core.real_time_risk import create_risk_manager
from src.core.position_sizing_enhanced import create_position_sizer
from src.core.circuit_breaker_enhanced import create_circuit_breaker
from src.core.stop_loss_manager import create_stop_manager
from src.monitoring.risk_dashboard import RealTimeRiskDashboard
from src.core.high_frequency_scanner import HighFrequencyScanner
from src.core.opportunity_scoring import OpportunityScorer
from src.execution.high_frequency_trading import HighFrequencyExecutor
from src.config.enhanced_config import get_enhanced_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/phase2_risk_management.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


class Phase2RiskManager:
    """
    Integrated Phase 2 Risk Management System
    Coordinates all risk management components
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = None
        self.running = False

        # Core components
        self.risk_manager = None
        self.position_sizer = None
        self.circuit_breaker = None
        self.stop_manager = None
        self.risk_dashboard = None

        # Trading components (existing from Phase 1)
        self.scanner = None
        self.scorer = None
        self.executor = None

        # Task handles
        self.monitoring_task = None
        self.trading_task = None

        logger.info("Phase 2 Risk Manager initialized")

    async def initialize(self) -> None:
        """Initialize all risk management components"""
        try:
            logger.info("=== Starting Phase 2 Risk Management Initialization ===")

            # Load configuration
            self.config = await self._load_config()

            # Initialize core risk components
            await self._initialize_risk_components()

            # Initialize dashboard
            await self._initialize_dashboard()

            # Initialize existing trading components
            await self._initialize_trading_components()

            # Set up signal handlers
            self._setup_signal_handlers()

            logger.info(
                "✅ Phase 2 Risk Management initialization completed successfully"
            )

        except Exception as e:
            logger.error(f"❌ Phase 2 initialization failed: {e}")
            raise

    async def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            config = get_enhanced_config()
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise

    async def _initialize_risk_components(self) -> None:
        """Initialize core risk management components"""
        try:
            logger.info("🔧 Initializing risk management components...")

            # Real-time risk manager
            self.risk_manager = await create_risk_manager(self.config)
            logger.info("✅ Real-time Risk Manager initialized")

            # Dynamic position sizer
            self.position_sizer = await create_position_sizer(
                self.config, self.risk_manager
            )
            logger.info("✅ Dynamic Position Sizer initialized")

            # Enhanced circuit breaker
            self.circuit_breaker = await create_circuit_breaker(self.config)
            logger.info("✅ Enhanced Circuit Breaker initialized")

            # Automated stop-loss manager
            self.stop_manager = await create_stop_manager(self.config)
            logger.info("✅ Automated Stop Manager initialized")

        except Exception as e:
            logger.error(f"Risk components initialization failed: {e}")
            raise

    async def _initialize_dashboard(self) -> None:
        """Initialize risk dashboard"""
        try:
            logger.info("📊 Initializing risk dashboard...")

            self.risk_dashboard = RealTimeRiskDashboard(
                config=self.config,
                risk_manager=self.risk_manager,
                position_sizer=self.position_sizer,
                circuit_breaker=self.circuit_breaker,
                stop_manager=self.stop_manager,
            )

            # Start monitoring
            await self.risk_dashboard.start_monitoring()
            logger.info("✅ Risk Dashboard initialized and monitoring started")

        except Exception as e:
            logger.error(f"Dashboard initialization failed: {e}")
            raise

    async def _initialize_trading_components(self) -> None:
        """Initialize existing trading components from Phase 1"""
        try:
            logger.info("🚀 Initializing trading components...")

            # High-frequency scanner
            self.scanner = HighFrequencyScanner(self.config)
            await self.scanner.initialize()
            logger.info("✅ High-Frequency Scanner initialized")

            # Opportunity scorer
            self.scorer = OpportunityScorer(self.config)
            await self.scorer.initialize()
            logger.info("✅ Opportunity Scorer initialized")

            # High-frequency executor
            self.executor = HighFrequencyExecutor(self.config)
            await self.executor.initialize()
            logger.info("✅ High-Frequency Executor initialized")

        except Exception as e:
            logger.error(f"Trading components initialization failed: {e}")
            raise

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def start(self) -> None:
        """Start the integrated risk management system"""
        try:
            if self.running:
                logger.warning("System is already running")
                return

            logger.info("🚀 Starting Phase 2 Risk Management System...")
            self.running = True

            # Start trading loop
            self.trading_task = asyncio.create_task(self._trading_loop())

            # Print startup summary
            await self._print_startup_summary()

            logger.info("✅ Phase 2 Risk Management System started successfully")

        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.shutdown()
            raise

    async def _trading_loop(self) -> None:
        """Main trading loop with integrated risk management"""
        try:
            while self.running:
                try:
                    # Check circuit breaker state
                    circuit_status = await self.circuit_breaker.get_status()
                    if circuit_status.get("state") == "open":
                        logger.warning("Circuit breaker is OPEN - pausing trading")
                        await asyncio.sleep(30)  # Wait 30 seconds before retrying
                        continue

                    # Scan for opportunities
                    opportunities = await self.scanner.scan_markets()

                    if not opportunities:
                        await asyncio.sleep(1)  # No opportunities, short wait
                        continue

                    # Score opportunities
                    scored_opportunities = []
                    for opportunity in opportunities[:5]:  # Limit to top 5
                        score = await self.scorer.score_opportunity(opportunity)
                        opportunity["score"] = score
                        if score.confidence > 0.6:  # Minimum confidence threshold
                            scored_opportunities.append(opportunity)

                    # Process high-quality opportunities
                    for opportunity in scored_opportunities[:2]:  # Limit to top 2
                        await self._process_opportunity(opportunity)

                    # Short pause between cycles
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Trading loop error: {e}")
                    await asyncio.sleep(1)  # Pause on error

        except Exception as e:
            logger.error(f"Trading loop failed: {e}")

    async def _process_opportunity(self, opportunity: Dict[str, Any]) -> None:
        """Process a single opportunity with full risk management"""
        try:
            symbol = opportunity.get("symbol", "UNKNOWN")
            price = opportunity.get("price", 0)
            confidence = opportunity.get("score", {}).get("confidence", 0)

            # Calculate position size with risk manager
            sizing_decision = await self.position_sizer.calculate_position_size(
                symbol=symbol,
                opportunity_confidence=confidence,
                expected_return=opportunity.get("expected_return", 0),
            )

            # Check risk limits
            is_allowed, risk_alerts = await self.risk_manager.check_position_risk(
                symbol=symbol,
                new_position_size=sizing_decision.adjusted_size,
                price=price,
                portfolio_value=self.position_sizer.portfolio_value,
            )

            if not is_allowed:
                logger.warning(f"Position rejected due to risk limits: {symbol}")
                return

            # Check circuit breaker
            triggers = await self.circuit_breaker.check_triggers(
                symbol=symbol, price=price, volume=opportunity.get("volume", 0)
            )

            if triggers:
                logger.warning(
                    f"Circuit breaker triggered for {symbol}: {[t.message for t in triggers]}"
                )
                return

            # Execute trade
            execution_result = await self.executor.execute_opportunity(
                opportunity, sizing_decision.adjusted_size
            )

            if execution_result.get("success"):
                # Open position with stop loss/take profit
                position_id = await self.stop_manager.open_position(
                    symbol=symbol,
                    position_size=sizing_decision.adjusted_size,
                    entry_price=price,
                )

                # Update risk manager
                await self.risk_manager.update_position(
                    symbol=symbol,
                    size_change=sizing_decision.adjusted_size,
                    price=price,
                )

                logger.info(
                    f"✅ Trade executed: {symbol} @ {price} (size: {sizing_decision.adjusted_size})"
                )

                # Update position sizer with trade result (will be updated on close)
                # This tracks the opportunity for performance analysis
                await self.position_sizer.update_trade_result(
                    symbol=symbol,
                    pnl=0,  # Will be updated on close
                    holding_time=asyncio.timedelta(0),
                    confidence=confidence,
                )
            else:
                logger.error(
                    f"❌ Trade execution failed: {execution_result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(
                f"Opportunity processing failed for {opportunity.get('symbol', 'UNKNOWN')}: {e}"
            )

    async def _print_startup_summary(self) -> None:
        """Print comprehensive startup summary"""
        try:
            print("\n" + "=" * 80)
            print("🎯 PHASE 2 ENHANCED RISK MANAGEMENT SYSTEM - STARTUP COMPLETE")
            print("=" * 80)

            print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()

            # Risk Management Components
            print("🛡️  RISK MANAGEMENT COMPONENTS:")
            print("  ✅ Real-Time Risk Manager - Dynamic position limits")
            print("  ✅ Dynamic Position Sizer - 6 strategies available")
            print("  ✅ Enhanced Circuit Breaker - Predictive triggers")
            print("  ✅ Automated Stop Manager - Multiple stop methods")
            print("  ✅ Real-Time Risk Dashboard - Live monitoring")
            print()

            # Configuration Summary
            risk_config = self.config.get("risk_limits", {})
            print("⚙️  RISK CONFIGURATION:")
            print(
                f"  📊 Max Position Size: {risk_config.get('max_position_size', 0.05):.1%}"
            )
            print(
                f"  💰 Max Portfolio Exposure: {risk_config.get('max_portfolio_exposure', 1.0):.1%}"
            )
            print(
                f"  🔄 Max Correlation: {risk_config.get('max_correlation', 0.7):.1%}"
            )
            print(f"  📈 Max Drawdown: {risk_config.get('max_drawdown', 0.1):.1%}")
            print()

            # Performance Settings
            perf_config = self.config.get("performance", {})
            print("⚡ PERFORMANCE SETTINGS:")
            print(f"  🔍 Scan Interval: {perf_config.get('scan_interval', 0.1):.1f}s")
            print(
                f"  🧵 Max Concurrent Scans: {perf_config.get('max_concurrent_scans', 25)}"
            )
            print(f"  💾 Cache TTL: {perf_config.get('cache_ttl', 30)}s")
            print()

            # System Health
            print("🏥 SYSTEM HEALTH:")
            risk_metrics = await self.risk_manager.get_risk_metrics()
            portfolio_risk = risk_metrics.get("portfolio_risk", {})

            print(f"  💎 Portfolio Value: ${portfolio_risk.get('total_value', 0):,.2f}")
            print(
                f"  📊 Total Exposure: ${portfolio_risk.get('total_exposure', 0):,.2f}"
            )
            print(
                f"  📉 Current Drawdown: {portfolio_risk.get('current_drawdown', 0):.2%}"
            )
            print(f"  ⚠️  Active Alerts: {len(self.risk_dashboard.active_alerts)}")
            print()

            # Circuit Breaker Status
            breaker_status = await self.circuit_breaker.get_status()
            print("🔌 CIRCUIT BREAKER:")
            print(f"  🔧 State: {breaker_status.get('state', 'unknown').upper()}")
            print(
                f"  📊 Total Triggers: {breaker_status.get('metrics', {}).get('total_triggers', 0)}"
            )
            print(
                f"  ✅ Successful Recoveries: {breaker_status.get('metrics', {}).get('successful_recoveries', 0)}"
            )
            print()

            print("🎯 SYSTEM IS READY FOR HIGH-FREQUENCY TRADING")
            print("📊 Dashboard available at: http://localhost:8000/dashboard")
            print("🔍 Real-time monitoring active")
            print("=" * 80)

        except Exception as e:
            logger.error(f"Startup summary generation failed: {e}")

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get status from all components
            risk_metrics = await self.risk_manager.get_risk_metrics()
            sizing_report = await self.position_sizer.get_sizing_report()
            breaker_status = await self.circuit_breaker.get_status()
            positions_status = await self.stop_manager.get_positions_status()
            dashboard_data = await self.risk_dashboard.get_dashboard_data()

            return {
                "timestamp": datetime.now().isoformat(),
                "running": self.running,
                "components": {
                    "risk_manager": "active" if self.risk_manager else "inactive",
                    "position_sizer": "active" if self.position_sizer else "inactive",
                    "circuit_breaker": "active" if self.circuit_breaker else "inactive",
                    "stop_manager": "active" if self.stop_manager else "inactive",
                    "risk_dashboard": "active" if self.risk_dashboard else "inactive",
                    "scanner": "active" if self.scanner else "inactive",
                    "scorer": "active" if self.scorer else "inactive",
                    "executor": "active" if self.executor else "inactive",
                },
                "risk_metrics": risk_metrics,
                "position_sizing": sizing_report,
                "circuit_breaker": breaker_status,
                "positions": positions_status,
                "dashboard": dashboard_data,
                "performance": {
                    "active_positions": positions_status.get("active_positions", 0),
                    "total_unrealized_pnl": positions_status.get(
                        "total_unrealized_pnl", 0
                    ),
                    "total_realized_pnl": positions_status.get("total_realized_pnl", 0),
                    "active_alerts": len(self.risk_dashboard.active_alerts),
                },
            }

        except Exception as e:
            logger.error(f"System status retrieval failed: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    async def shutdown(self) -> None:
        """Graceful shutdown of all components"""
        try:
            if not self.running:
                return

            logger.info("🛑 Initiating graceful shutdown...")
            self.running = False

            # Cancel tasks
            if self.trading_task:
                self.trading_task.cancel()
                try:
                    await self.trading_task
                except asyncio.CancelledError:
                    pass

            # Shutdown components in reverse order
            if self.scanner:
                await self.scanner.cleanup()
                logger.info("✅ Scanner cleaned up")

            if self.executor:
                await self.executor.cleanup()
                logger.info("✅ Executor cleaned up")

            if self.scorer:
                await self.scorer.cleanup()
                logger.info("✅ Scorer cleaned up")

            if self.risk_dashboard:
                await self.risk_dashboard.cleanup()
                logger.info("✅ Risk Dashboard cleaned up")

            if self.stop_manager:
                await self.stop_manager.cleanup()
                logger.info("✅ Stop Manager cleaned up")

            if self.circuit_breaker:
                await self.circuit_breaker.cleanup()
                logger.info("✅ Circuit Breaker cleaned up")

            if self.position_sizer:
                await self.position_sizer.cleanup()
                logger.info("✅ Position Sizer cleaned up")

            if self.risk_manager:
                await self.risk_manager.cleanup()
                logger.info("✅ Risk Manager cleaned up")

            logger.info("✅ Phase 2 Risk Management System shutdown complete")

        except Exception as e:
            logger.error(f"Shutdown error: {e}")


async def main():
    """Main function to run the Phase 2 Risk Management System"""
    try:
        # Create and initialize the system
        risk_system = Phase2RiskManager()
        await risk_system.initialize()

        # Start the system
        await risk_system.start()

        # Keep running until shutdown
        while risk_system.running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        if "risk_system" in locals():
            await risk_system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
