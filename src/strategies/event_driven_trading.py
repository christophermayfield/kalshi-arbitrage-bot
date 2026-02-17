"""
Event-Driven Trading System with News Sentiment Analysis
Advanced event detection and automated trading based on market events and news
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import aiohttp
import json
from collections import defaultdict, deque
import redis.asyncio as redis

from .sentiment_analyzer import SentimentAnalyzer
from ..utils.performance_cache import PerformanceCache

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of market events"""

    NEWS_SENTIMENT = "news_sentiment"
    PRICE_ANOMALY = "price_anomaly"
    VOLUME_SPIKE = "volume_spike"
    REGULATORY_ANNOUNCEMENT = "regulatory_announcement"
    EARNINGS_REPORT = "earnings_report"
    TECHNICAL_BREAKOUT = "technical_breakout"
    MARKET_REGIME_CHANGE = "market_regime_change"
    SOCIAL_MEDIA_TREND = "social_media_trend"


class EventSeverity(Enum):
    """Severity levels for events"""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class TradingAction(Enum):
    """Trading actions triggered by events"""

    BUY = "buy"
    SELL = "sell"
    CLOSE_ALL = "close_all"
    REDUCE_POSITION = "reduce_position"
    INCREASE_POSITION = "increase_position"
    ADJUST_STOPS = "adjust_stops"
    NO_ACTION = "no_action"


@dataclass
class MarketEvent:
    """Market event data structure"""

    event_id: str
    event_type: EventType
    severity: EventSeverity
    assets_affected: List[str]
    sentiment_score: float
    confidence: float
    source: str
    headline: str
    description: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    trading_signals: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EventRule:
    """Event-driven trading rule"""

    rule_id: str
    event_types: List[EventType]
    severity_filter: List[EventSeverity]
    sentiment_filter: Tuple[float, float]  # (min, max)
    confidence_threshold: float
    assets: List[str]
    action: TradingAction
    position_size_multiplier: float
    riskMultiplier: float
    cooldown_period: timedelta
    max_daily_trades: int
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


class EventDrivenTradingSystem:
    """
    Advanced event-driven trading system with news sentiment analysis
    and automated rule-based trading
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.event_config = config.get("event_driven_trading", {})

        # Event monitoring
        self.enabled_event_types = [
            EventType(t)
            for t in self.event_config.get(
                "enabled_event_types",
                [
                    "news_sentiment",
                    "price_anomaly",
                    "volume_spike",
                    "social_media_trend",
                ],
            )
        ]

        # Trading parameters
        self.max_events_per_hour = self.event_config.get("max_events_per_hour", 10)
        self.event_confirmation_time = self.event_config.get(
            "event_confirmation_time", 60
        )  # seconds
        self.position_size_base = self.event_config.get("position_size_base", 1000)

        # Sentiment thresholds
        self.bullish_threshold = self.event_config.get("bullish_threshold", 0.6)
        self.bearish_threshold = self.event_config.get("bearish_threshold", 0.4)

        # Event tracking
        self.active_events: Dict[str, MarketEvent] = {}
        self.event_history: deque = deque(maxlen=1000)
        self.pending_events: Dict[str, MarketEvent] = {}
        self.triggered_rules: Dict[str, datetime] = {}

        # Trading rules
        self.trading_rules: Dict[str, EventRule] = {}

        # Performance metrics
        self.performance_metrics: Dict[str, Any] = defaultdict(float)

        # Sentiment analyzer integration
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None

        # Market data subscriptions
        self.price_subscriptions: Set[str] = set()
        self.volume_subscriptions: Set[str] = set()

        # News sources
        self.news_sources = self.event_config.get(
            "news_sources", ["crypto_news", "twitter", "reddit", "discord"]
        )

        # Performance cache
        self.cache = PerformanceCache(
            redis_url=config.get("redis_url", "redis://localhost:6379"), default_ttl=60
        )

        # HTTP session for API calls
        self.http_session: Optional[aiohttp.ClientSession] = None

        # Event handlers
        self.event_handlers: Dict[EventType, callable] = {}

        logger.info("Event-Driven Trading System initialized")

    async def initialize(self) -> None:
        """Initialize the event-driven trading system"""
        try:
            # Initialize HTTP session
            timeout = aiohttp.ClientTimeout(total=30)
            self.http_session = aiohttp.ClientSession(timeout=timeout)

            # Initialize sentiment analyzer
            try:
                self.sentiment_analyzer = SentimentAnalyzer(self.config)
                await self.sentiment_analyzer.initialize()
                logger.info("Sentiment Analyzer initialized")
            except Exception as e:
                logger.warning(f"Sentiment Analyzer initialization failed: {e}")

            # Load trading rules
            await self._load_trading_rules()

            # Initialize event handlers
            await self._initialize_event_handlers()

            # Start monitoring loops
            asyncio.create_task(self._news_monitoring_loop())
            asyncio.create_task(self._event_processing_loop())
            asyncio.create_task(self._rule_evaluation_loop())

            logger.info("Event-Driven Trading System initialized successfully")

        except Exception as e:
            logger.error(f"Event-Driven Trading System initialization failed: {e}")
            raise

    async def ingest_market_event(self, event_data: Dict[str, Any]) -> MarketEvent:
        """Ingest and process a market event"""
        try:
            event_id = f"event_{datetime.now().timestamp()}"

            # Parse event data
            event_type = EventType(event_data.get("type", "news_sentiment"))
            severity = EventSeverity(event_data.get("severity", "moderate"))
            assets = event_data.get("assets", [])
            sentiment_score = event_data.get("sentiment", 0.5)
            confidence = event_data.get("confidence", 0.5)
            source = event_data.get("source", "unknown")
            headline = event_data.get("headline", "")
            description = event_data.get("description", "")

            # Create event
            event = MarketEvent(
                event_id=event_id,
                event_type=event_type,
                severity=severity,
                assets_affected=assets,
                sentiment_score=sentiment_score,
                confidence=confidence,
                source=source,
                headline=headline,
                description=description,
                timestamp=datetime.now(),
                metadata=event_data.get("metadata", {}),
            )

            # Analyze sentiment if needed
            if event_type == EventType.NEWS_SENTIMENT and headline:
                await self._analyze_event_sentiment(event)

            # Store event
            self.active_events[event_id] = event
            self.event_history.append(event)

            # Cache event
            await self.cache.set(
                f"event:{event_id}",
                {
                    "event_type": event.event_type.value,
                    "severity": event.severity.value,
                    "assets": event.assets_affected,
                    "sentiment": event.sentiment_score,
                    "confidence": event.confidence,
                    "headline": event.headline,
                    "timestamp": event.timestamp.isoformat(),
                },
                ttl=3600,
            )

            # Log event
            self._log_event(event)

            return event

        except Exception as e:
            logger.error(f"Event ingestion failed: {e}")
            raise

    async def _analyze_event_sentiment(self, event: MarketEvent) -> None:
        """Analyze sentiment of news event"""
        try:
            if not self.sentiment_analyzer:
                return

            # Analyze headline sentiment
            headline_sentiment = await self.sentiment_analyzer.analyze_sentiment(
                event.headline
            )

            # Analyze description sentiment
            description_sentiment = await self.sentiment_analyzer.analyze_sentiment(
                event.description
            )

            # Combine sentiments
            combined_sentiment = weighted_average(
                [
                    headline_sentiment.get("score", 0.5),
                    description_sentiment.get("score", 0.5),
                ],
                weights=[0.7, 0.3],
            )

            # Update event sentiment
            event.sentiment_score = combined_sentiment
            event.metadata["headline_sentiment"] = headline_sentiment
            event.metadata["description_sentiment"] = description_sentiment

            # Calculate confidence based on sentiment strength
            sentiment_deviation = abs(combined_sentiment - 0.5)
            event.confidence = max(
                0.5, min(1.0, event.confidence + sentiment_deviation)
            )

        except Exception as e:
            logger.error(f"Event sentiment analysis failed: {e}")

    async def _news_monitoring_loop(self) -> None:
        """Background loop for monitoring news sources"""
        while True:
            try:
                # Check each news source
                for source in self.news_sources:
                    await self._fetch_news_from_source(source)

                # Check event rate limits
                await self._check_event_rate_limits()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"News monitoring loop error: {e}")
                await asyncio.sleep(30)

    async def _fetch_news_from_source(self, source: str) -> None:
        """Fetch news from a specific source"""
        try:
            if source == "crypto_news":
                await self._fetch_crypto_news()
            elif source == "twitter":
                await self._fetch_twitter_news()
            elif source == "reddit":
                await self._fetch_reddit_news()
            elif source == "discord":
                await self._fetch_discord_news()

        except Exception as e:
            logger.error(f"News fetch failed for {source}: {e}")

    async def _fetch_crypto_news(self) -> None:
        """Fetch crypto news from API"""
        try:
            # Simulate news fetching
            # In production, this would make actual API calls to crypto news sites

            news_items = []

            # Generate sample news
            headlines = [
                "BTC breaks $50k resistance level",
                "Ethereum network upgrade scheduled",
                "Coinbase announces new listing",
                "Crypto market sentiment turns bullish",
                "SEC releases new crypto regulations",
                "Bitcoin mining difficulty adjusts upward",
            ]

            for headline in headlines:
                sentiment = (
                    "positive"
                    if "bullish" in headline.lower()
                    or "breaks" in headline.lower()
                    or "upgrade" in headline.lower()
                    else "negative"
                    if "regulations" in headline.lower()
                    or "difficulty" in headline.lower()
                    else "neutral"
                )

                sentiment_score = {
                    "positive": np.random.uniform(0.6, 0.9),
                    "negative": np.random.uniform(0.1, 0.4),
                    "neutral": np.random.uniform(0.4, 0.6),
                }.get(sentiment, 0.5)

                news_items.append(
                    {
                        "type": "news_sentiment",
                        "severity": "moderate" if sentiment == "neutral" else "high",
                        "assets": ["BTC/USDT"] if "BTC" in headline else ["ETH/USDT"],
                        "sentiment": sentiment_score,
                        "confidence": np.random.uniform(0.6, 0.9),
                        "source": "crypto_news",
                        "headline": headline,
                        "description": f"Breaking news: {headline}",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # Ingest news events
            for news in news_items:
                await self.ingest_market_event(news)

        except Exception as e:
            logger.error(f"Crypto news fetch failed: {e}")

    async def _fetch_twitter_news(self) -> None:
        """Fetch crypto-related tweets"""
        try:
            # Simulate Twitter monitoring
            # In production, this would use Twitter API or第三方服务

            tweets = [
                "Just bought more #Bitcoin! 🚀 #crypto",
                "Ethereum gas prices are insane right now 😤",
                "Major exchange listing announcement coming soon!",
                "Technical analysis shows strong bearish divergence",
                "Whales accumulating at these levels 👀",
            ]

            for tweet in tweets:
                sentiment = (
                    "positive"
                    if "🚀" in tweet
                    or "bullish" in tweet.lower()
                    or "accumulating" in tweet.lower()
                    else "negative"
                    if "😤" in tweet or "bearish" in tweet.lower()
                    else "neutral"
                )

                sentiment_score = {
                    "positive": np.random.uniform(0.7, 0.95),
                    "negative": np.random.uniform(0.05, 0.3),
                    "neutral": np.random.uniform(0.4, 0.6),
                }.get(sentiment, 0.5)

                await self.ingest_market_event(
                    {
                        "type": "social_media_trend",
                        "severity": "low",
                        "assets": ["BTC/USDT", "ETH/USDT"],
                        "sentiment": sentiment_score,
                        "confidence": np.random.uniform(
                            0.5, 0.7
                        ),  # Lower confidence for social media
                        "source": "twitter",
                        "headline": tweet[:50] + "...",
                        "description": tweet,
                        "metadata": {
                            "platform": "twitter",
                            "engagement": np.random.randint(100, 10000),
                        },
                    }
                )

        except Exception as e:
            logger.error(f"Twitter news fetch failed: {e}")

    async def _fetch_reddit_news(self) -> None:
        """Fetch crypto posts from Reddit"""
        try:
            # Simulate Reddit monitoring
            # In production, this would use Reddit API or第三方服务

            posts = [
                "Why I'm still bullish on BTC despite the dip",
                "Bearish divergence forming on BTC 4H chart",
                "New Ethereum scaling solution explained",
                "Crypto regulations: What you need to know",
                "Major partnership announcement for DeFi protocol",
            ]

            for post in posts:
                sentiment = (
                    "positive"
                    if "bullish" in post.lower()
                    or "explained" in post.lower()
                    or "partnership" in post.lower()
                    else "negative"
                    if "bearish" in post.lower() or "dip" in post.lower()
                    else "neutral"
                )

                await self.ingest_market_event(
                    {
                        "type": "social_media_trend",
                        "severity": "moderate",
                        "assets": ["BTC/USDT", "ETH/USDT", "USDT"],
                        "sentiment": 0.7
                        if sentiment == "positive"
                        else 0.3
                        if sentiment == "negative"
                        else 0.5,
                        "confidence": np.random.uniform(0.6, 0.8),
                        "source": "reddit",
                        "headline": post[:50] + "...",
                        "description": post,
                        "metadata": {
                            "platform": "reddit",
                            "upvotes": np.random.randint(100, 5000),
                        },
                    }
                )

        except Exception as e:
            logger.error(f"Reddit news fetch failed: {e}")

    async def _fetch_discord_news(self) -> None:
        """Fetch crypto discussions from Discord"""
        try:
            # Simulate Discord monitoring
            # In production, this would use Discord API or第三方服务

            messages = [
                "Whales are buying at current levels",
                "Looking for long entry on ETH",
                "Market sentiment is extremely bullish",
                "Be careful of upcoming event",
                "Technical levels holding strong",
            ]

            for message in messages:
                sentiment = (
                    "positive"
                    if "buying" in message.lower()
                    or "bullish" in message.lower()
                    or "holding" in message.lower()
                    else "negative"
                    if "careful" in message.lower()
                    else "neutral"
                )

                await self.ingest_market_event(
                    {
                        "type": "social_media_trend",
                        "severity": "low",
                        "assets": ["BTC/USDT", "ETH/USDT"],
                        "sentiment": 0.75
                        if sentiment == "positive"
                        else 0.25
                        if sentiment == "negative"
                        else 0.5,
                        "confidence": np.random.uniform(0.5, 0.7),
                        "source": "discord",
                        "headline": message[:40] + "...",
                        "description": message,
                        "metadata": {
                            "platform": "discord",
                            "community": "crypto_traders",
                        },
                    }
                )

        except Exception as e:
            logger.error(f"Discord news fetch failed: {e}")

    async def _event_processing_loop(self) -> None:
        """Background loop for processing events"""
        while True:
            try:
                # Process active events
                for event_id, event in list(self.active_events.items()):
                    await self._process_event(event)

                # Clean up old events
                await self._cleanup_old_events()

                await asyncio.sleep(10)  # Process every 10 seconds

            except Exception as e:
                logger.error(f"Event processing loop error: {e}")
                await asyncio.sleep(5)

    async def _process_event(self, event: MarketEvent) -> None:
        """Process a single market event"""
        try:
            # Check if event has been confirmed
            if await self._is_event_confirmed(event):
                # Generate trading signals
                trading_signals = await self._generate_trading_signals(event)
                event.trading_signals = trading_signals

                # Evaluate trading rules
                triggered_rules = await self._evaluate_trading_rules(event)

                if triggered_rules:
                    # Execute trading actions
                    for rule in triggered_rules:
                        await self._execute_rule_action(rule, event)

        except Exception as e:
            logger.error(f"Event processing failed for {event.event_id}: {e}")

    async def _is_event_confirmed(self, event: MarketEvent) -> bool:
        """Check if event is confirmed (has persisted)"""
        try:
            # High severity events are immediately confirmed
            if event.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]:
                return True

            # Check if event has been present for confirmation time
            event_age = datetime.now() - event.timestamp
            if event_age.total_seconds() >= self.event_confirmation_time:
                return True

            # Check for corroborating evidence
            corroboration_count = sum(
                1
                for other_event in self.active_events.values()
                if (
                    other_event.event_type == event.event_type
                    and set(event.assets_affected) & set(other_event.assets_affected)
                    and other_event.sentiment_score * event.sentiment_score > 0
                )
            )

            if corroboration_count >= 2:
                return True

            return False

        except Exception as e:
            logger.error(f"Event confirmation check failed: {e}")
            return False

    async def _generate_trading_signals(
        self, event: MarketEvent
    ) -> List[Dict[str, Any]]:
        """Generate trading signals from event"""
        signals = []

        try:
            # Generate signals based on event type and sentiment
            if event.sentiment_score > self.bullish_threshold:
                # Bullish signals
                for asset in event.assets_affected:
                    if "/" in asset:  # Trading pair
                        signals.append(
                            {
                                "asset": asset,
                                "action": "buy",
                                "confidence": event.confidence,
                                "sentiment": event.sentiment_score,
                                "strength": (event.sentiment_score - 0.5)
                                * 2,  # Normalize to 0-1
                            }
                        )

            elif event.sentiment_score < self.bearish_threshold:
                # Bearish signals
                for asset in event.assets_affected:
                    if "/" in asset:
                        signals.append(
                            {
                                "asset": asset,
                                "action": "sell",
                                "confidence": event.confidence,
                                "sentiment": event.sentiment_score,
                                "strength": (0.5 - event.sentiment_score) * 2,
                            }
                        )

            return signals

        except Exception as e:
            logger.error(f"Trading signal generation failed: {e}")
            return []

    async def _evaluate_trading_rules(self, event: MarketEvent) -> List[EventRule]:
        """Evaluate trading rules against event"""
        triggered_rules = []

        try:
            for rule_id, rule in self.trading_rules.items():
                if not rule.enabled:
                    continue

                # Check cooldown period
                if rule.last_triggered:
                    time_since_trigger = datetime.now() - rule.last_triggered
                    if time_since_trigger < rule.cooldown_period:
                        continue

                # Check event type matching
                if event.event_type not in rule.event_types:
                    continue

                # Check severity matching
                if event.severity not in rule.severity_filter:
                    continue

                # Check sentiment filter
                min_sentiment, max_sentiment = rule.sentiment_filter
                if not (min_sentiment <= event.sentiment_score <= max_sentiment):
                    continue

                # Check confidence threshold
                if event.confidence < rule.confidence_threshold:
                    continue

                # Check asset matching
                if not (set(rule.assets) & set(event.assets_affected)):
                    continue

                # Check daily trade limit
                today = datetime.now().date()
                rule_trades_today = self.performance_metrics.get(
                    f"{rule_id}_trades_today", 0
                )
                if rule_trades_today >= rule.max_daily_trades:
                    continue

                # Rule triggered
                triggered_rules.append(rule)

                # Update rule stats
                rule.last_triggered = datetime.now()
                rule.trigger_count += 1

                # Update daily trade counter
                self.performance_metrics[f"{rule_id}_trades_today"] = (
                    rule_trades_today + 1
                )

            return triggered_rules

        except Exception as e:
            logger.error(f"Trading rule evaluation failed: {e}")
            return []

    async def _execute_rule_action(self, rule: EventRule, event: MarketEvent) -> None:
        """Execute trading action based on rule"""
        try:
            action = rule.action
            assets = rule.assets

            logger.info(f"Executing rule {rule.rule_id}: {action.value} on {assets}")

            # Calculate position size
            position_size = self.position_size_base * rule.position_size_multiplier

            # Execute action
            if action == TradingAction.BUY:
                await self._execute_buy(assets[0], position_size, event)

            elif action == TradingAction.SELL:
                await self._execute_sell(assets[0], position_size, event)

            elif action == TradingAction.CLOSE_ALL:
                await self._execute_close_all(assets, event)

            elif action == TradingAction.REDUCE_POSITION:
                await self._execute_reduce_position(
                    assets[0], 0.5, event
                )  # Reduce by 50%

            elif action == TradingAction.INCREASE_POSITION:
                await self._execute_increase_position(
                    assets[0], 0.5, event
                )  # Increase by 50%

            elif action == TradingAction.ADJUST_STOPS:
                await self._execute_adjust_stops(assets, event)

            # Update performance metrics
            self.performance_metrics["total_events_processed"] += 1
            self.performance_metrics[f"actions_{action.value}"] += 1

        except Exception as e:
            logger.error(f"Rule action execution failed for {rule.rule_id}: {e}")

    async def _execute_buy(self, asset: str, size: float, event: MarketEvent) -> None:
        """Execute buy order"""
        try:
            # In production, this would execute actual trades
            logger.info(
                f"Event-driven BUY: {asset} size={size} from event {event.event_id}"
            )

            # Store execution record
            execution = {
                "event_id": event.event_id,
                "asset": asset,
                "action": "buy",
                "size": size,
                "timestamp": datetime.now().isoformat(),
                "event_type": event.event_type.value,
                "sentiment": event.sentiment_score,
            }

            await self.cache.set(
                f"event_execution:{datetime.now().timestamp()}", execution, ttl=86400
            )

        except Exception as e:
            logger.error(f"Buy execution failed for {asset}: {e}")

    async def _execute_sell(self, asset: str, size: float, event: MarketEvent) -> None:
        """Execute sell order"""
        try:
            logger.info(
                f"Event-driven SELL: {asset} size={size} from event {event.event_id}"
            )

            execution = {
                "event_id": event.event_id,
                "asset": asset,
                "action": "sell",
                "size": size,
                "timestamp": datetime.now().isoformat(),
                "event_type": event.event_type.value,
                "sentiment": event.sentiment_score,
            }

            await self.cache.set(
                f"event_execution:{datetime.now().timestamp()}", execution, ttl=86400
            )

        except Exception as e:
            logger.error(f"Sell execution failed for {asset}: {e}")

    async def _execute_close_all(self, assets: List[str], event: MarketEvent) -> None:
        """Execute close all positions"""
        try:
            for asset in assets:
                logger.info(
                    f"Event-driven CLOSE ALL: {asset} from event {event.event_id}"
                )

        except Exception as e:
            logger.error(f"Close all execution failed: {e}")

    async def _execute_reduce_position(
        self, asset: str, reduction_pct: float, event: MarketEvent
    ) -> None:
        """Execute position reduction"""
        try:
            logger.info(
                f"Event-driven REDUCE POSITION: {asset} by {reduction_pct:.0%} from event {event.event_id}"
            )

        except Exception as e:
            logger.error(f"Position reduction failed for {asset}: {e}")

    async def _execute_increase_position(
        self, asset: str, increase_pct: float, event: MarketEvent
    ) -> None:
        """Execute position increase"""
        try:
            logger.info(
                f"Event-driven INCREASE POSITION: {asset} by {increase_pct:.0%} from event {event.event_id}"
            )

        except Exception as e:
            logger.error(f"Position increase failed for {asset}: {e}")

    async def _execute_adjust_stops(
        self, assets: List[str], event: MarketEvent
    ) -> None:
        """Execute stop loss adjustments"""
        try:
            for asset in assets:
                logger.info(
                    f"Event-driven ADJUST STOPS: {asset} from event {event.event_id}"
                )

        except Exception as e:
            logger.error(f"Stop adjustment failed: {e}")

    async def _rule_evaluation_loop(self) -> None:
        """Background loop for evaluating trading rules"""
        while True:
            try:
                # Evaluate rules for all active events
                for event in self.active_events.values():
                    await self._evaluate_trading_rules(event)

                await asyncio.sleep(30)  # Evaluate every 30 seconds

            except Exception as e:
                logger.error(f"Rule evaluation loop error: {e}")
                await asyncio.sleep(15)

    async def _load_trading_rules(self) -> None:
        """Load trading rules from configuration"""
        try:
            # Default trading rules
            default_rules = [
                EventRule(
                    rule_id="bullish_news_buy",
                    event_types=[EventType.NEWS_SENTIMENT],
                    severity_filter=[EventSeverity.HIGH, EventSeverity.CRITICAL],
                    sentiment_filter=(0.7, 1.0),
                    confidence_threshold=0.7,
                    assets=["BTC/USDT", "ETH/USDT"],
                    action=TradingAction.BUY,
                    position_size_multiplier=1.2,
                    riskMultiplier=1.0,
                    cooldown_period=timedelta(minutes=30),
                    max_daily_trades=5,
                ),
                EventRule(
                    rule_id="bearish_news_sell",
                    event_types=[EventType.NEWS_SENTIMENT],
                    severity_filter=[EventSeverity.HIGH, EventSeverity.CRITICAL],
                    sentiment_filter=(0.0, 0.3),
                    confidence_threshold=0.7,
                    assets=["BTC/USDT", "ETH/USDT"],
                    action=TradingAction.SELL,
                    position_size_multiplier=0.8,
                    riskMultiplier=0.5,
                    cooldown_period=timedelta(minutes=30),
                    max_daily_trades=5,
                ),
                EventRule(
                    rule_id="sentiment_trend_follow",
                    event_types=[EventType.SOCIAL_MEDIA_TREND],
                    severity_filter=[EventSeverity.MODERATE, EventSeverity.HIGH],
                    sentiment_filter=(0.8, 1.0),
                    confidence_threshold=0.6,
                    assets=["BTC/USDT", "ETH/USDT"],
                    action=TradingAction.INCREASE_POSITION,
                    position_size_multiplier=1.3,
                    riskMultiplier=1.1,
                    cooldown_period=timedelta(minutes=15),
                    max_daily_trades=10,
                ),
                EventRule(
                    rule_id="negative_news_reduce",
                    event_types=[
                        EventType.NEWS_SENTIMENT,
                        EventType.REGULATORY_ANNOUNCEMENT,
                    ],
                    severity_filter=[EventSeverity.CRITICAL],
                    sentiment_filter=(0.0, 0.2),
                    confidence_threshold=0.8,
                    assets=["BTC/USDT", "ETH/USDT"],
                    action=TradingAction.REDUCE_POSITION,
                    position_size_multiplier=0.5,
                    riskMultiplier=0.3,
                    cooldown_period=timedelta(minutes=60),
                    max_daily_trades=3,
                ),
            ]

            for rule in default_rules:
                self.trading_rules[rule.rule_id] = rule

            logger.info(f"Loaded {len(self.trading_rules)} trading rules")

        except Exception as e:
            logger.error(f"Trading rules loading failed: {e}")

    async def _initialize_event_handlers(self) -> None:
        """Initialize event handlers for different event types"""
        try:
            # Register handlers for each event type
            for event_type in self.enabled_event_types:
                if event_type == EventType.NEWS_SENTIMENT:
                    self.event_handlers[event_type] = self._handle_news_sentiment_event
                elif event_type == EventType.PRICE_ANOMALY:
                    self.event_handlers[event_type] = self._handle_price_anomaly_event
                elif event_type == EventType.VOLUME_SPIKE:
                    self.event_handlers[event_type] = self._handle_volume_spike_event
                elif event_type == EventType.SOCIAL_MEDIA_TREND:
                    self.event_handlers[event_type] = self._handle_social_media_event

            logger.info(f"Initialized {len(self.event_handlers)} event handlers")

        except Exception as e:
            logger.error(f"Event handlers initialization failed: {e}")

    async def _handle_news_sentiment_event(self, event: MarketEvent) -> None:
        """Handle news sentiment event"""
        # This could trigger specific trading strategies
        pass

    async def _handle_price_anomaly_event(self, event: MarketEvent) -> None:
        """Handle price anomaly event"""
        # Could trigger volatility-based trading
        pass

    async def _handle_volume_spike_event(self, event: MarketEvent) -> None:
        """Handle volume spike event"""
        # Could trigger breakout trading
        pass

    async def _handle_social_media_event(self, event: MarketEvent) -> None:
        """Handle social media trend event"""
        # Could trigger momentum trading
        pass

    async def _check_event_rate_limits(self) -> None:
        """Check if event rate limits have been exceeded"""
        try:
            now = datetime.now()
            one_hour_ago = now - timedelta(hours=1)

            # Count events in last hour
            recent_events = [
                event for event in self.event_history if event.timestamp >= one_hour_ago
            ]

            if len(recent_events) >= self.max_events_per_hour:
                logger.warning(
                    f"Event rate limit exceeded: {len(recent_events)} events in last hour"
                )
                # Could implement throttling or alerting here

        except Exception as e:
            logger.error(f"Event rate limit check failed: {e}")

    async def _cleanup_old_events(self) -> None:
        """Clean up old events"""
        try:
            # Remove events older than 1 hour
            cutoff_time = datetime.now() - timedelta(hours=1)

            events_to_remove = [
                event_id
                for event_id, event in self.active_events.items()
                if event.timestamp < cutoff_time
            ]

            for event_id in events_to_remove:
                del self.active_events[event_id]

            if events_to_remove:
                logger.debug(f"Cleaned up {len(events_to_remove)} old events")

        except Exception as e:
            logger.error(f"Event cleanup failed: {e}")

    def _log_event(self, event: MarketEvent) -> None:
        """Log market event"""
        try:
            log_level = {
                EventSeverity.LOW: logging.INFO,
                EventSeverity.MODERATE: logging.WARNING,
                EventSeverity.HIGH: logging.ERROR,
                EventSeverity.CRITICAL: logging.CRITICAL,
            }.get(event.severity, logging.INFO)

            logger.log(
                log_level, f"EVENT [{event.event_type.value.upper()}] {event.headline}"
            )
            logger.log(
                log_level,
                f"  Sentiment: {event.sentiment_score:.2f}, Confidence: {event.confidence:.2f}",
            )
            logger.log(log_level, f"  Assets: {', '.join(event.assets_affected)}")

        except Exception as e:
            logger.error(f"Event logging failed: {e}")

    async def get_event_report(self) -> Dict[str, Any]:
        """Get comprehensive event-driven trading report"""
        try:
            # Calculate event statistics
            event_stats = defaultdict(
                lambda: {"count": 0, "avg_sentiment": 0, "avg_confidence": 0}
            )

            for event in self.event_history:
                event_type = event.event_type.value
                event_stats[event_type]["count"] += 1
                event_stats[event_type]["avg_sentiment"] += event.sentiment_score
                event_stats[event_type]["avg_confidence"] += event.confidence

            # Calculate averages
            for event_type in event_stats:
                if event_stats[event_type]["count"] > 0:
                    event_stats[event_type]["avg_sentiment"] /= event_stats[event_type][
                        "count"
                    ]
                    event_stats[event_type]["avg_confidence"] /= event_stats[
                        event_type
                    ]["count"]

            return {
                "active_events": len(self.active_events),
                "total_events_processed": len(self.event_history),
                "recent_events": [
                    {
                        "event_type": e.event_type.value,
                        "severity": e.severity.value,
                        "headline": e.headline,
                        "sentiment": e.sentiment_score,
                        "timestamp": e.timestamp.isoformat(),
                    }
                    for e in list(self.active_events.values())[
                        -10:
                    ]  # Last 10 active events
                ],
                "event_statistics": dict(event_stats),
                "trading_rules_status": {
                    rule_id: {
                        "enabled": rule.enabled,
                        "trigger_count": rule.trigger_count,
                        "last_triggered": rule.last_triggered.isoformat()
                        if rule.last_triggered
                        else None,
                    }
                    for rule_id, rule in self.trading_rules.items()
                },
                "performance_metrics": dict(self.performance_metrics),
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Event report generation failed: {e}")
            return {}

    async def cleanup(self) -> None:
        """Cleanup event-driven trading system"""
        try:
            # Close HTTP session
            if self.http_session:
                await self.http_session.close()

            # Close cache
            await self.cache.close()

            logger.info("Event-Driven Trading System cleaned up")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Utility functions
async def create_event_driven_system(
    config: Dict[str, Any],
) -> EventDrivenTradingSystem:
    """Create and initialize event-driven trading system"""
    system = EventDrivenTradingSystem(config)
    await system.initialize()
    return system


def weighted_average(values: List[float], weights: List[float]) -> float:
    """Calculate weighted average"""
    return (
        sum(v * w for v, w in zip(values, weights)) / sum(weights)
        if weights and values
        else 0.5
    )


def calculate_event_impact_score(event: MarketEvent) -> float:
    """Calculate overall event impact score"""
    severity_weights = {
        EventSeverity.LOW: 0.25,
        EventSeverity.MODERATE: 0.5,
        EventSeverity.HIGH: 0.75,
        EventSeverity.CRITICAL: 1.0,
    }

    severity_multiplier = severity_weights.get(event.severity, 0.5)
    sentiment_multiplier = abs(event.sentiment_score - 0.5) * 2  # Normalize to 0-1

    return (severity_multiplier + sentiment_multiplier) / 2
