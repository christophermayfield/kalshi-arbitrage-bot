import asyncio
import signal
import sys
import os
import uuid
from typing import Dict, Set, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from src.utils.config import Config
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.database import Database
from src.clients.kalshi_client import KalshiClient
from src.clients.websocket_client import KalshiWebSocketClient
from src.core.orderbook import OrderBook
from src.core.arbitrage import ArbitrageDetector, ArbitrageOpportunity
from src.core.portfolio import PortfolioManager
from src.core.circuit_breaker import (
    CircuitBreakerManager,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreaker,
)
from src.execution.trading import TradingExecutor
from src.execution.paper_trading import PaperTradingSimulator
from src.execution.limited_risk_executor import LimitedRiskExecutor
from src.core.limited_risk_manager import LimitedRiskManager, LimitedRiskConfig
from src.analytics.limited_position_sizing import LimitedRiskPositionSizer
from src.monitoring.limited_risk_metrics import LimitedRiskMetrics
from src.utils.rate_limiter import RateLimiter, RateLimitConfig
from src.utils.redis_client import RedisClient
from src.monitoring.webhooks import WebhookHandler, WebhookMessage, WebhookProvider

logger = get_logger("main")


class CorrelationContext:
    """Thread-local context for correlation IDs."""

    _current_id: Optional[str] = None

    @classmethod
    def get(cls) -> Optional[str]:
        return cls._current_id

    @classmethod
    def set(cls, value: str) -> None:
        cls._current_id = value

    @classmethod
    def clear(cls) -> None:
        cls._current_id = None


class CorrelationLogger:
    """Logger wrapper that includes correlation IDs."""

    def __init__(self, logger):
        self.logger = logger

    def _format_message(self, message: str) -> str:
        corr_id = CorrelationContext.get()
        if corr_id:
            return f"[{corr_id[:8]}] {message}"
        return message

    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(self._format_message(msg), *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(self._format_message(msg), *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(self._format_message(msg), *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(self._format_message(msg), *args, **kwargs)


correlated_logger = CorrelationLogger(logger)


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


def with_correlation(func):
    """Decorator to add correlation ID to a function call."""

    async def wrapper(*args, **kwargs):
        corr_id = str(uuid.uuid4())
        CorrelationContext.set(corr_id)
        try:
            return await func(*args, **kwargs)
        finally:
            CorrelationContext.clear()

    return wrapper


class ArbitrageBot:
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        self.client = KalshiClient(config)
        self.ws_client: Optional[KalshiWebSocketClient] = None
        self.db: Optional[Database] = None
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
        )
        self.portfolio = PortfolioManager(
            max_daily_loss=config.get("risk.max_daily_loss_cents", 10000),
            max_open_positions=config.get("risk.max_open_positions", 50),
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
        self.orderbooks: Dict[str, OrderBook] = {}
        self.subscribed_markets: Set[str] = set()
        self.scan_interval = config.scan_interval_seconds
        self.executed_count = 0
        self._detection_running = False
        self._detection_task: Optional[asyncio.Task] = None
        self._reconciliation_task: Optional[asyncio.Task] = None
        self._circuit_breaker_manager = CircuitBreakerManager()
        self._trade_circuit: Optional[CircuitBreaker] = None
        self.correlation_id: str = str(uuid.uuid4())

        rl_config = RateLimitConfig(
            requests_per_second=config.get("api.rate_limit_per_second", 5.0),
            requests_per_minute=config.get("api.rate_limit_per_minute", 300.0),
            burst_limit=config.get("api.rate_limit_burst", 10),
        )
        self.rate_limiter = RateLimiter(rl_config)

        self.redis: Optional[RedisClient] = None
        self._cache_ttl = config.get("redis.cache_ttl_seconds", 60)

        self.webhook_handler: Optional[WebhookHandler] = None

        # Initialize Limited Risk Trading Mode
        self.limited_risk_manager: Optional[LimitedRiskManager] = None
        self.limited_risk_executor: Optional[LimitedRiskExecutor] = None
        self.limited_risk_metrics: Optional[LimitedRiskMetrics] = None
        self.limited_risk_sizer: Optional[LimitedRiskPositionSizer] = None
        self._init_limited_risk_mode()
        self._market_refresh_interval = self.config.get(
            "monitoring.scan_interval_seconds", 5.0
        )
        self._market_refresh_task: Optional[asyncio.Task] = None

    def _init_limited_risk_mode(self) -> None:
        """Initialize limited risk trading mode if enabled in config."""
        if not self.config.limited_risk_enabled:
            logger.info("Limited Risk Mode: Disabled in configuration")
            return

        try:
            # Create limited risk configuration from config file
            lr_config = LimitedRiskConfig(
                enabled=True,
                auto_enable_balance_cents=self.config.limited_risk_auto_enable_balance_cents,
                min_trade_cents=self.config.limited_risk_min_cents,
                max_trade_cents=self.config.limited_risk_max_cents,
                max_daily_trades=self.config.get("limited_risk.max_daily_trades", 10),
                max_daily_loss_cents=self.config.get(
                    "limited_risk.max_daily_loss_cents", 5000
                ),
                cooldown_seconds=self.config.get("limited_risk.cooldown_seconds", 60),
                require_confirmation=self.config.get(
                    "limited_risk.require_confirmation", False
                ),
                min_profit_after_fees_cents=self.config.get(
                    "limited_risk.min_profit_after_fees_cents", 50
                ),
                max_fee_percent_of_profit=self.config.get(
                    "limited_risk.max_fee_percent_of_profit", 50.0
                ),
                illiquid_volume_threshold_cents=self.config.get(
                    "limited_risk.illiquid_volume_threshold_cents", 100000
                ),
                progressive_scaling_enabled=self.config.get(
                    "limited_risk.progressive_scaling.enabled", True
                ),
                scaling_threshold_trades=self.config.get(
                    "limited_risk.progressive_scaling.threshold_trades", 10
                ),
                scaling_profitable_streak_required=self.config.get(
                    "limited_risk.progressive_scaling.profitable_streak_required", True
                ),
                scaling_increment_cents=self.config.get(
                    "limited_risk.progressive_scaling.increment_cents", 500
                ),
                max_trade_cents_scaled=self.config.get(
                    "limited_risk.progressive_scaling.max_trade_cents", 3000
                ),
                excluded_markets=self.config.get("limited_risk.excluded_markets", []),
            )

            # Initialize components
            self.limited_risk_manager = LimitedRiskManager(lr_config)
            self.limited_risk_executor = LimitedRiskExecutor(
                base_executor=self.executor, risk_manager=self.limited_risk_manager
            )
            self.limited_risk_metrics = LimitedRiskMetrics(self.limited_risk_manager)
            self.limited_risk_sizer = LimitedRiskPositionSizer(
                self.limited_risk_manager
            )

            logger.info("=" * 60)
            logger.info("LIMITED RISK TRADING MODE INITIALIZED")
            logger.info("=" * 60)
            logger.info(
                f"Trade size range: ${lr_config.min_trade_cents / 100:.2f} - ${lr_config.max_trade_cents / 100:.2f}"
            )
            logger.info(
                f"Auto-enable threshold: ${lr_config.auto_enable_balance_cents / 100:.2f}"
            )
            logger.info(
                f"Daily limits: {lr_config.max_daily_trades} trades, ${lr_config.max_daily_loss_cents / 100:.2f} loss"
            )
            logger.info(f"Cooldown: {lr_config.cooldown_seconds}s between trades")
            logger.info(
                f"Progressive scaling: {'Enabled' if lr_config.progressive_scaling_enabled else 'Disabled'}"
            )
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Failed to initialize limited risk mode: {e}")
            self.limited_risk_manager = None
            self.limited_risk_executor = None

    async def start(self) -> None:
        logger.info("Starting arbitrage bot...")
        self.running = True

        self._initialize_circuit_breakers()
        await self._initialize_database()
        await self._initialize_redis()
        await self._initialize_webhooks()
        await self._check_exchange_status()

        try:
            real_balance = await self.client.get_balance()
            self.portfolio.set_balance(real_balance)
            logger.info(f"Real balance synced: ${real_balance:.2f}")

            # Check if limited risk mode should auto-enable
            if self.limited_risk_manager:
                self.limited_risk_manager.check_auto_enable(int(real_balance * 100))
                if self.limited_risk_manager.enabled:
                    logger.info(
                        f"Limited Risk Mode: ACTIVE (Balance: ${real_balance:.2f})"
                    )
                else:
                    logger.info(
                        f"Limited Risk Mode: Inactive (Balance: ${real_balance:.2f} above threshold)"
                    )
        except Exception as e:
            logger.error(f"Failed to sync balance, using default: {e}")
            self.portfolio.set_balance(10000)

        await self._initialize_websocket()
        await self._fetch_initial_markets()
        await self._start_detection_loop()
        await self._start_reconciliation_loop()
        await self._start_market_data_loop()
        await self._start_market_refresh_loop()

    def _initialize_circuit_breakers(self) -> None:
        cb_config = CircuitBreakerConfig(
            failure_threshold=self.config.get("risk.circuit_breaker_threshold", 5),
            success_threshold=3,
            timeout_seconds=self.config.get(
                "risk.circuit_breaker_window_seconds", 60.0
            ),
            half_open_max_calls=3,
        )
        self._trade_circuit = self._circuit_breaker_manager.get_or_create(
            "trade_execution", cb_config
        )
        logger.info("Circuit breakers initialized")

    async def _initialize_websocket(self) -> None:
        try:
            base_url = self.config.get("kalshi.base_url", "https://api.kalshi.com")
            demo_mode = self.config.get("kalshi.demo_mode", True)

            self.ws_client = KalshiWebSocketClient(
                base_url=base_url,
                demo=demo_mode,
                reconnect_delay=5.0,
                heartbeat_interval=30.0,
            )

            api_key_id = self.config.get("kalshi.api_key_id")
            private_key_path = self.config.get("kalshi.private_key_path")

            if api_key_id and private_key_path:
                import os

                private_key_path = os.path.expanduser(private_key_path)
                if os.path.exists(private_key_path):
                    with open(private_key_path, "r") as f:
                        private_key = f.read()
                    self.ws_client.set_credentials(api_key_id, private_key)
                    await self.ws_client.connect()
                    await self.ws_client.authenticate()
                    logger.info("WebSocket authenticated successfully")
                else:
                    logger.warning(
                        f"Private key not found at {private_key_path}, using unauthenticated mode"
                    )
                    await self.ws_client.connect()
            else:
                logger.info(
                    "No API credentials configured, connecting in unauthenticated mode"
                )
                await self.ws_client.connect()

            self.ws_client.on("orderbook_delta")(self._handle_orderbook_update)
            self.ws_client.on("orderbook_snapshot")(self._handle_orderbook_snapshot)
            self.ws_client.on("trade")(self._handle_trade)

            logger.info("WebSocket initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize WebSocket: {e}")
            logger.warning("Falling back to REST polling mode")

    async def _fetch_initial_markets(self) -> None:
        try:
            markets_response = self.client.get_markets(status="open", limit=100)
            markets = markets_response.get("markets", [])

            market_ids = [m.get("market_id") for m in markets if m.get("market_id")]
            logger.info(f"Found {len(market_ids)} active markets")

            for market_id in market_ids[:50]:
                await self._fetch_orderbook(market_id)

            if self.ws_client and self.ws_client.is_connected:
                for market_id in market_ids[:50]:
                    await self.ws_client.subscribe_orderbook(market_id)
                    self.subscribed_markets.add(market_id)

            logger.info(
                f"Subscribed to {len(self.subscribed_markets)} market orderbooks"
            )

        except Exception as e:
            logger.error(f"Failed to fetch initial markets: {e}")

    async def _handle_orderbook_update(self, message) -> None:
        try:
            data = message.data
            market_id = data.get("market_id")
            if not market_id:
                return

            delta = data.get("delta", {})
            bids = delta.get("bids", [])
            asks = delta.get("asks", [])

            if market_id not in self.orderbooks:
                return

            orderbook = self.orderbooks[market_id]

            for price, size in bids:
                orderbook.update_bid(price, size)
            for price, size in asks:
                orderbook.update_ask(price, size)

            orderbook.last_update = datetime.utcnow()

        except Exception as e:
            logger.debug(f"Error handling orderbook update: {e}")

    async def _handle_orderbook_snapshot(self, message) -> None:
        try:
            data = message.data
            market_id = data.get("market_id")
            if not market_id:
                return

            bids = data.get("bids", [])
            asks = data.get("asks", [])

            orderbook = OrderBook(market_id=market_id)
            for price, size in bids:
                orderbook.update_bid(price, size)
            for price, size in asks:
                orderbook.update_ask(price, size)

            orderbook.last_update = datetime.utcnow()
            self.orderbooks[market_id] = orderbook

            if market_id not in self.subscribed_markets:
                self.subscribed_markets.add(market_id)

        except Exception as e:
            logger.debug(f"Error handling orderbook snapshot: {e}")

    async def _handle_trade(self, message) -> None:
        try:
            data = message.data
            market_id = data.get("market_id")
            if market_id and market_id in self.orderbooks:
                self.orderbooks[market_id].last_update = datetime.utcnow()
        except Exception as e:
            logger.debug(f"Error handling trade: {e}")

    async def _start_detection_loop(self) -> None:
        async def detection_worker():
            while self.running:
                try:
                    if self._detection_running and len(self.orderbooks) >= 2:
                        opportunities = await self.detector.scan_for_opportunities(
                            self.orderbooks
                        )
                        opportunities = self.detector.filter_by_threshold(opportunities)

                        if opportunities:
                            logger.info(
                                f"Found {len(opportunities)} arbitrage opportunities"
                            )

                            for opp in opportunities[:5]:
                                asyncio.create_task(self._execute_opportunity(opp))

                    await asyncio.sleep(self.scan_interval)

                except Exception as e:
                    logger.error(f"Detection loop error: {e}")
                    await asyncio.sleep(5)

        self._detection_task = asyncio.create_task(detection_worker())
        self._detection_running = self.config.get("arbitrage.enabled", False)
        logger.info(f"Detection loop started (running: {self._detection_running})")

    async def start_detection(self) -> None:
        """Start the arbitrage detection loop."""
        self._detection_running = True
        logger.info("Arbitrage detection started")

    async def stop_detection(self) -> None:
        """Stop the arbitrage detection loop."""
        self._detection_running = False
        logger.info("Arbitrage detection stopped")

    def get_detection_status(self) -> dict:
        """Get current detection status."""
        return {
            "running": self._detection_running,
            "markets_tracked": len(self.orderbooks),
            "opportunities_found": self.executed_count,
        }

    async def _start_reconciliation_loop(self) -> None:
        async def reconciliation_worker():
            while self.running:
                try:
                    await asyncio.sleep(60)
                    await self._reconcile_positions()

                except Exception as e:
                    logger.error(f"Reconciliation loop error: {e}")
                    await asyncio.sleep(60)

        self._reconciliation_task = asyncio.create_task(reconciliation_worker())
        logger.info("Reconciliation loop started")

    async def _start_market_data_loop(self) -> None:
        """Background job to record market data snapshots."""
        if not self.db:
            logger.info("Database not initialized, skipping market data recording")
            return

        async def market_data_worker():
            while self.running:
                try:
                    await asyncio.sleep(60)  # Record every minute
                    await self._record_market_data()
                except Exception as e:
                    logger.debug(f"Market data recording error: {e}")

        asyncio.create_task(market_data_worker())
        logger.info("Market data recording loop started")

    async def _start_market_refresh_loop(self) -> None:
        """Background job to refresh orderbook data via REST API."""

        async def market_refresh_worker():
            while self.running:
                try:
                    await asyncio.sleep(self._market_refresh_interval)

                    if not self.subscribed_markets:
                        continue

                    for market_id in list(self.subscribed_markets)[:20]:
                        try:
                            await self._fetch_orderbook(market_id)
                        except Exception:
                            pass

                except Exception as e:
                    logger.debug(f"Market refresh error: {e}")

        self._market_refresh_task = asyncio.create_task(market_refresh_worker())
        logger.info("Market refresh loop started")

    async def refresh_markets(self) -> dict:
        """Manually trigger a market data refresh."""
        try:
            await self._fetch_initial_markets()
            return {"status": "refreshed", "markets": len(self.orderbooks)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _record_market_data(self) -> None:
        """Record current market data to database."""
        if not self.db:
            return

        try:
            for market_id, orderbook in self.orderbooks.items():
                snapshot_data = {
                    "market_id": market_id,
                    "best_bid": orderbook.bids[0].price if orderbook.bids else None,
                    "best_ask": orderbook.asks[0].price if orderbook.asks else None,
                    "mid_price": orderbook.get_mid_price(),
                    "spread": orderbook.get_spread(),
                    "spread_percent": orderbook.get_spread_percent(),
                    "bid_depth_5": sum(l.count for l in orderbook.bids[:5]),
                    "ask_depth_5": sum(l.count for l in orderbook.asks[:5]),
                    "total_volume": sum(l.total for l in orderbook.bids)
                    + sum(l.total for l in orderbook.asks),
                    "liquidity_score": orderbook.get_liquidity_score(),
                }
                await self.db.save_market_data_snapshot(snapshot_data)

        except Exception as e:
            logger.debug(f"Failed to record market data: {e}")

    async def _reconcile_positions(self) -> None:
        try:
            exchange_positions = self.client.get_positions()
            if exchange_positions:
                for market_id, position in exchange_positions.get(
                    "positions", {}
                ).items():
                    self.portfolio.positions[market_id] = position
                logger.debug(
                    f"Reconciled {len(exchange_positions.get('positions', {}))} positions"
                )

            exchange_balance = self.client.get_balance()
            if exchange_balance:
                self.portfolio.set_balance(exchange_balance)

        except Exception as e:
            logger.debug(f"Position reconciliation failed: {e}")

    async def _check_exchange_status(self) -> None:
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

    async def _initialize_database(self) -> None:
        try:
            db_path = self.config.get("database.path", "data/arbitrage.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            db_url = f"sqlite:///{db_path}"
            self.db = Database(database_url=db_url)
            self.db.create_tables()
            logger.info(f"Database initialized at {db_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize database: {e}")

    async def _initialize_redis(self) -> None:
        if not self.config.get("redis.enabled", False):
            logger.info("Redis caching is disabled")
            return

        try:
            self.redis = RedisClient(
                host=self.config.get("redis.host", "localhost"),
                port=self.config.get("redis.port", 6379),
                db=self.config.get("redis.db", 0),
                password=self.config.get("redis.password"),
                prefix="arbitrage",
            )
            await self.redis.connect()
            logger.info("Redis caching initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis: {e}")
            self.redis = None

    async def _initialize_webhooks(self) -> None:
        if not self.config.get("notifications.enabled", False):
            logger.info("Notifications are disabled")
            return

        try:
            self.webhook_handler = WebhookHandler()

            slack_url = self.config.get("notifications.slack.webhook_url")
            if slack_url:
                self.webhook_handler.register_slack(slack_url)
                logger.info("Slack webhook registered")

            discord_url = self.config.get("notifications.discord.webhook_url")
            if discord_url:
                self.webhook_handler.register_discord(discord_url)
                logger.info("Discord webhook registered")

            telegram_token = self.config.get("notifications.telegram.token")
            telegram_chat = self.config.get("notifications.telegram.chat_id")
            if telegram_token and telegram_chat:
                self.webhook_handler.register_telegram(telegram_token, telegram_chat)
                logger.info("Telegram webhook registered")

        except Exception as e:
            logger.warning(f"Failed to initialize webhooks: {e}")
            self.webhook_handler = None

    async def send_alert(self, title: str, message: str, level: str = "info") -> None:
        """Send an alert notification."""
        if not self.webhook_handler:
            return

        try:
            msg = WebhookMessage(title=title, message=message, level=level)
            await self.webhook_handler.send(msg)
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    async def _cache_get(self, key: str) -> Optional[str]:
        if not self.redis:
            return None
        try:
            return await self.redis.get(key)
        except Exception:
            return None

    async def _cache_set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        if not self.redis:
            return
        try:
            await self.redis.set(key, value, expire_seconds=ttl or self._cache_ttl)
        except Exception:
            pass

    async def _fetch_orderbook(self, market_id: str) -> None:
        try:
            orderbook_data = self.client.get_market_orderbook(market_id)
            orderbook = OrderBook.from_api_response(orderbook_data)
            self.orderbooks[market_id] = orderbook

        except Exception as e:
            logger.debug(f"Failed to fetch orderbook for {market_id}: {e}")

    async def _execute_opportunity(self, opportunity: ArbitrageOpportunity) -> None:
        corr_id = str(uuid.uuid4())
        CorrelationContext.set(corr_id)

        if self._trade_circuit and self._trade_circuit.state == CircuitState.OPEN:
            correlated_logger.warning(
                f"Circuit breaker is OPEN, skipping execution of {opportunity.id}"
            )
            CorrelationContext.clear()
            return

        allowed, wait_time = await self.rate_limiter.acquire("trade")
        if not allowed:
            correlated_logger.warning(
                f"Rate limit reached, waiting {wait_time:.2f}s before executing {opportunity.id}"
            )
            await asyncio.sleep(wait_time)

        # Check if limited risk mode is active and adjust opportunity
        if self.limited_risk_manager and self.limited_risk_manager.enabled:
            # Check daily limits first
            can_trade, status, reason = self.limited_risk_manager.can_execute_trade()
            if not can_trade:
                correlated_logger.warning(f"Limited risk mode: Cannot trade - {reason}")
                CorrelationContext.clear()
                return

            # Adjust opportunity for limited risk constraints
            adjusted_opp = self.limited_risk_sizer.adjust_opportunity_for_limited_risk(
                opportunity
            )
            if adjusted_opp is None:
                correlated_logger.debug(
                    f"Opportunity {opportunity.id} cannot fit in $10-$15 range"
                )
                CorrelationContext.clear()
                return

            # Check market eligibility
            is_eligible, eligibility_reason = (
                self.limited_risk_manager.is_market_eligible(opportunity.market_id_1)
            )
            if not is_eligible:
                correlated_logger.debug(
                    f"Market {opportunity.market_id_1} not eligible: {eligibility_reason}"
                )
                CorrelationContext.clear()
                return

            # Use adjusted opportunity
            opportunity = adjusted_opp
            correlated_logger.info(
                f"Limited Risk Mode: Adjusted opportunity to {opportunity.quantity} contracts, "
                f"net profit ${opportunity.net_profit_cents / 100:.2f}"
            )

        if self.config.paper_mode:
            await self._execute_paper_trade(opportunity)
            CorrelationContext.clear()
            return

        can_trade, reason = self.portfolio.can_open_position(
            opportunity.quantity, opportunity.buy_price
        )
        if not can_trade:
            correlated_logger.debug(f"Cannot execute {opportunity.id}: {reason}")
            CorrelationContext.clear()
            return

        try:
            # Use limited risk executor if active
            if (
                self.limited_risk_executor
                and self.limited_risk_manager
                and self.limited_risk_manager.enabled
            ):
                success, profit = await self.limited_risk_executor.execute_arbitrage(
                    opportunity
                )
            elif self._trade_circuit:
                success, profit = await self._trade_circuit.call(
                    self.executor.execute_arbitrage, opportunity
                )
            else:
                success, profit = await self.executor.execute_arbitrage(opportunity)

            if success:
                self.executed_count += 1
                self.portfolio.cash_balance += profit
                correlated_logger.info(
                    f"Executed {opportunity.id}, profit: {profit} cents, total executed: {self.executed_count}"
                )
            else:
                correlated_logger.warning(f"Failed to execute {opportunity.id}")

        except Exception as e:
            correlated_logger.error(f"Trade execution error: {e}")
        finally:
            CorrelationContext.clear()

    async def _execute_paper_trade(self, opportunity: ArbitrageOpportunity) -> None:
        """Execute a trade using the paper trading simulator."""
        try:
            market_id = opportunity.buy_market_id or opportunity.market_id_1

            if market_id in self.orderbooks:
                self.paper_simulator.update_orderbook(
                    market_id, self.orderbooks[market_id]
                )

            buy_order = await self.paper_simulator.create_order(
                market_id=market_id,
                side="buy",
                order_type="limit",
                price=opportunity.buy_price,
                quantity=opportunity.quantity,
            )

            await asyncio.sleep(0.1)

            if buy_order.status == "filled":
                sell_market_id = opportunity.sell_market_id or opportunity.market_id_2

                if sell_market_id and sell_market_id in self.orderbooks:
                    self.paper_simulator.update_orderbook(
                        sell_market_id, self.orderbooks[sell_market_id]
                    )

                sell_order = await self.paper_simulator.create_order(
                    market_id=sell_market_id or market_id,
                    side="sell",
                    order_type="limit",
                    price=opportunity.sell_price,
                    quantity=opportunity.quantity,
                )

                await asyncio.sleep(0.1)

                stats = self.paper_simulator.get_stats()
                correlated_logger.info(
                    f"[PAPER] Executed: {opportunity.id} - "
                    f"Buy: {buy_order.status} @ {buy_order.fill_price}, "
                    f"Sell: {sell_order.status} @ {sell_order.fill_price}, "
                    f"Balance: ${stats['balance'] / 100:.2f}"
                )

                self.executed_count += 1
            else:
                correlated_logger.info(
                    f"[PAPER] Order not filled: {opportunity.id} - {buy_order.status}"
                )

        except Exception as e:
            correlated_logger.error(f"Paper trade execution error: {e}")

    async def get_paper_trading_stats(self) -> Dict[str, Any]:
        """Get paper trading statistics."""
        return self.paper_simulator.get_stats()

    async def stop(self) -> None:
        logger.info("Stopping arbitrage bot...")
        self.running = False

        if self._detection_task:
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass

        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass

        if self.ws_client:
            await self.ws_client.disconnect()

        for order_id in list(self.executor.pending_orders.keys()):
            try:
                self.client.cancel_order(order_id)
                correlated_logger.info(f"Cancelled pending order: {order_id}")
            except Exception as e:
                correlated_logger.warning(f"Failed to cancel order {order_id}: {e}")

        if self.db:
            correlated_logger.info("Closing database connections...")

        correlated_logger.info(f"Bot stopped. Total executed: {self.executed_count}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the bot and its dependencies."""
        status = {
            "status": "healthy",
            "running": self.running,
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": self.correlation_id,
            "components": {},
        }

        status["components"]["exchange"] = self._check_exchange_health()
        status["components"]["websocket"] = self._check_websocket_health()
        status["components"]["database"] = self._check_database_health()
        status["components"]["circuit_breaker"] = self._check_circuit_breaker_health()
        status["components"]["rate_limiter"] = self._check_rate_limiter_health()
        status["components"]["limited_risk"] = self._check_limited_risk_health()

        if not all(c["healthy"] for c in status["components"].values()):
            status["status"] = "degraded"

        return status

    def _check_exchange_health(self) -> Dict[str, Any]:
        try:
            exchange_status = self.client.get_exchange_status()
            return {
                "healthy": exchange_status.get("exchange_active", False),
                "details": exchange_status,
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def _check_websocket_health(self) -> Dict[str, Any]:
        if self.ws_client is None:
            return {"healthy": False, "reason": "WebSocket not initialized"}
        return {
            "healthy": self.ws_client.is_connected,
            "subscribed_markets": len(self.subscribed_markets),
        }

    def _check_database_health(self) -> Dict[str, Any]:
        if self.db is None:
            return {"healthy": False, "reason": "Database not initialized"}
        return {"healthy": True}

    def _check_circuit_breaker_health(self) -> Dict[str, Any]:
        if self._trade_circuit is None:
            return {"healthy": False, "reason": "Circuit breaker not initialized"}
        return {
            "healthy": self._trade_circuit.state != CircuitState.OPEN,
            "state": self._trade_circuit.state.value,
            "stats": {
                "total_calls": self._trade_circuit.stats.total_calls,
                "failed_calls": self._trade_circuit.stats.failed_calls,
            },
        }

    def _check_rate_limiter_health(self) -> Dict[str, Any]:
        return {"healthy": True}

    def _check_limited_risk_health(self) -> Dict[str, Any]:
        """Check limited risk mode health status."""
        if self.limited_risk_manager is None:
            return {"healthy": True, "active": False, "reason": "Not configured"}

        if not self.limited_risk_manager.enabled:
            return {"healthy": True, "active": False, "reason": "Disabled"}

        can_trade, status, reason = self.limited_risk_manager.can_execute_trade()

        return {
            "healthy": can_trade,
            "active": True,
            "status": status.value,
            "reason": reason,
            "daily_stats": {
                "trades_executed": self.limited_risk_manager.daily_stats.trades_executed,
                "trades_won": self.limited_risk_manager.daily_stats.trades_won,
                "daily_pnl_cents": self.limited_risk_manager.daily_stats.daily_pnl_cents,
                "win_rate": round(self.limited_risk_manager.daily_stats.win_rate, 1),
            },
            "limits": {
                "max_daily_trades": self.limited_risk_manager.config.max_daily_trades,
                "max_daily_loss_cents": self.limited_risk_manager.config.max_daily_loss_cents,
            },
        }

    def get_limited_risk_status(self) -> Dict[str, Any]:
        """Get detailed limited risk mode status and metrics."""
        if self.limited_risk_manager is None or not self.limited_risk_manager.enabled:
            return {"enabled": False}

        status = self.limited_risk_manager.get_status()

        if self.limited_risk_metrics:
            status["today_summary"] = self.limited_risk_metrics.get_today_summary()
            status["recent_trades"] = self.limited_risk_metrics.get_recent_trades(5)
            status["alerts"] = self.limited_risk_metrics.get_alert_status()

        return status


async def main():
    config = Config()

    log_level = config.get("monitoring.log_level", "INFO")
    logger = setup_logging(level=log_level)

    bot = ArbitrageBot(config)

    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(bot.stop())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()
    except Exception as e:
        logger.error(f"Bot error: {e}")
        await bot.stop()
        raise


if __name__ == "__main__":
    asyncio.run(main())
