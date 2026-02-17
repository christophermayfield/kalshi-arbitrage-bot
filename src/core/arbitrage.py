from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime
import asyncio

from src.core.orderbook import OrderBook, OrderSide
from src.core.predictive_models import EnsembleForecaster, get_arbitrage_timing_signal
from src.core.sentiment_analyzer import SentimentAggregator, get_market_sentiment_signal
from src.core.statistical_arbitrage import (
    StatisticalArbitrageDetector,
    StatisticalArbitrageOpportunity,
)
from src.utils.logging_utils import get_logger

logger = get_logger("arbitrage")


class ArbitrageType(Enum):
    CROSS_MARKET = "cross_market"
    TEMPORAL = "temporal"
    TRIANGULAR = "triangular"
    INTERNAL = "internal"


@dataclass
class ArbitrageOpportunity:
    id: str
    type: ArbitrageType
    market_id_1: str
    market_id_2: Optional[str] = None
    buy_market_id: str = ""
    sell_market_id: str = ""
    buy_price: int = 0
    sell_price: int = 0
    quantity: int = 0
    profit_cents: int = 0
    profit_percent: float = 0.0
    confidence: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    fees: int = 0
    net_profit_cents: int = 0
    execution_window_seconds: int = 30
    risk_level: str = "medium"

    @property
    def gross_margin(self) -> float:
        if self.buy_price > 0:
            return ((self.sell_price - self.buy_price) / self.buy_price) * 100
        return 0.0

    @property
    def is_profitable(self) -> bool:
        return self.net_profit_cents > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "market_id_1": self.market_id_1,
            "market_id_2": self.market_id_2,
            "buy_market_id": self.buy_market_id,
            "sell_market_id": self.sell_market_id,
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
            "quantity": self.quantity,
            "profit_cents": self.profit_cents,
            "profit_percent": self.profit_percent,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "fees": self.fees,
            "net_profit_cents": self.net_profit_cents,
            "risk_level": self.risk_level,
        }


class ArbitrageDetector:
    def __init__(
        self,
        min_profit_cents: int = 10,
        min_profit_percent: float = 0.5,
        fee_rate: float = 0.01,
        min_confidence: float = 0.8,
        enable_predictive_models: bool = True,
        enable_sentiment_analysis: bool = True,
        predictive_weight: float = 0.3,
        sentiment_weight: float = 0.2,
        enable_statistical_arbitrage: bool = False,
        statistical_config: Optional[Dict[str, Any]] = None,
    ):
        self.min_profit_cents = min_profit_cents
        self.min_profit_percent = min_profit_percent
        self.fee_rate = fee_rate
        self.min_confidence = min_confidence
        self.enable_predictive_models = enable_predictive_models
        self.enable_sentiment_analysis = enable_sentiment_analysis
        self.predictive_weight = predictive_weight
        self.sentiment_weight = sentiment_weight
        self.enable_statistical_arbitrage = enable_statistical_arbitrage

        # Initialize ML components
        self.forecaster = EnsembleForecaster() if enable_predictive_models else None
        self.sentiment_aggregator = (
            SentimentAggregator() if enable_sentiment_analysis else None
        )

        # Initialize statistical arbitrage detector
        self.statistical_detector = None
        if enable_statistical_arbitrage:
            from src.core.statistical_arbitrage import StatisticalArbitrageDetector

            self.statistical_detector = StatisticalArbitrageDetector(
                strategies=["mean_reversion", "pairs_trading"],
                config=statistical_config or {},
            )

        # Cache for recent prices and sentiment
        self._price_cache: Dict[str, List[float]] = {}
        self._sentiment_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes

    async def detect_cross_market_arbitrage(
        self, market_1: OrderBook, market_2: OrderBook, event_id: Optional[str] = None
    ) -> List[ArbitrageOpportunity]:
        opportunities = []

        best_bid_1 = market_1.get_best_bid()
        best_ask_1 = market_1.get_best_ask()
        best_bid_2 = market_2.get_best_bid()
        best_ask_2 = market_2.get_best_ask()

        if not all([best_bid_1, best_ask_1, best_bid_2, best_ask_2]):
            return opportunities

        m1_buy_price = best_ask_1.price if best_ask_1 else 0
        m1_sell_price = best_bid_1.price if best_bid_1 else 0
        m2_buy_price = best_ask_2.price if best_ask_2 else 0
        m2_sell_price = best_bid_2.price if best_bid_2 else 0

        if m1_sell_price and m2_buy_price and m1_sell_price > m2_buy_price:
            quantity = min(
                best_bid_1.count if best_bid_1 else 0,
                best_ask_2.count if best_ask_2 else 0,
                100,
            )
            gross_profit = (m1_sell_price - m2_buy_price) * quantity
            fees = int(gross_profit * self.fee_rate * 2)
            net_profit = gross_profit - fees

            if net_profit >= self.min_profit_cents:
                opp_id = f"arb_{market_1.market_id}_{market_2.market_id}_{int(datetime.utcnow().timestamp())}"
                # Create opportunity object first for enhanced confidence calculation
                temp_opp = ArbitrageOpportunity(
                    id=opp_id,
                    type=ArbitrageType.CROSS_MARKET,
                    market_id_1=market_1.market_id,
                    market_id_2=market_2.market_id,
                    buy_market_id=market_2.market_id,
                    sell_market_id=market_1.market_id,
                    buy_price=m2_buy_price,
                    sell_price=m1_sell_price,
                    quantity=quantity,
                    profit_cents=gross_profit,
                    profit_percent=((m1_sell_price - m2_buy_price) / m2_buy_price)
                    * 100,
                    fees=fees,
                    net_profit_cents=net_profit,
                    confidence=0.0,  # Will be updated
                )
                # Calculate enhanced confidence
                temp_opp.confidence = await self._get_enhanced_confidence(
                    temp_opp, market_1, market_2, quantity
                )
                opportunities.append(temp_opp)

        if m2_sell_price and m1_buy_price and m2_sell_price > m1_buy_price:
            quantity = min(
                best_bid_2.count if best_bid_2 else 0,
                best_ask_1.count if best_ask_1 else 0,
                100,
            )
            gross_profit = (m2_sell_price - m1_buy_price) * quantity
            fees = int(gross_profit * self.fee_rate * 2)
            net_profit = gross_profit - fees

            if net_profit >= self.min_profit_cents:
                opp_id = f"arb_{market_2.market_id}_{market_1.market_id}_{int(datetime.utcnow().timestamp())}"
                # Create opportunity object first for enhanced confidence calculation
                temp_opp = ArbitrageOpportunity(
                    id=opp_id,
                    type=ArbitrageType.CROSS_MARKET,
                    market_id_1=market_2.market_id,
                    market_id_2=market_1.market_id,
                    buy_market_id=market_1.market_id,
                    sell_market_id=market_2.market_id,
                    buy_price=m1_buy_price,
                    sell_price=m2_sell_price,
                    quantity=quantity,
                    profit_cents=gross_profit,
                    profit_percent=((m2_sell_price - m1_buy_price) / m1_buy_price)
                    * 100,
                    fees=fees,
                    net_profit_cents=net_profit,
                    confidence=0.0,  # Will be updated
                )
                # Calculate enhanced confidence
                temp_opp.confidence = await self._get_enhanced_confidence(
                    temp_opp, market_2, market_1, quantity
                )
                opportunities.append(temp_opp)

        return opportunities

    def detect_internal_arbitrage(
        self, orderbook: OrderBook, market_id: str
    ) -> List[ArbitrageOpportunity]:
        opportunities = []

        best_bid = orderbook.get_best_bid()
        best_ask = orderbook.get_best_ask()

        if not best_bid or not best_ask:
            return opportunities

        if best_bid and best_ask and best_bid.price > best_ask.price:
            quantity = min(best_bid.count, best_ask.count)
            gross_profit = (best_bid.price - best_ask.price) * quantity
            fees = int(gross_profit * self.fee_rate * 2)
            net_profit = gross_profit - fees

            if net_profit >= self.min_profit_cents:
                opp = ArbitrageOpportunity(
                    id=f"int_{market_id}_{int(datetime.utcnow().timestamp())}",
                    type=ArbitrageType.INTERNAL,
                    market_id_1=market_id,
                    buy_market_id=market_id,
                    sell_market_id=market_id,
                    buy_price=best_ask.price,
                    sell_price=best_bid.price,
                    quantity=quantity,
                    profit_cents=gross_profit,
                    profit_percent=((best_bid.price - best_ask.price) / best_ask.price)
                    * 100,
                    fees=fees,
                    net_profit_cents=net_profit,
                    confidence=1.0,
                    risk_level="low",
                )
                opportunities.append(opp)

        return opportunities

    async def _get_enhanced_confidence(
        self,
        opportunity: "ArbitrageOpportunity",
        market_1: OrderBook,
        market_2: OrderBook,
        quantity: int,
    ) -> float:
        """Calculate enhanced confidence with predictive models and sentiment analysis"""

        # Base confidence calculation
        bid_1 = market_1.get_best_bid()
        ask_1 = market_1.get_best_ask()
        bid_2 = market_2.get_best_bid()
        ask_2 = market_2.get_best_ask()

        conf_1 = (
            market_1.get_fill_probability(
                OrderSide.SELL, quantity, bid_1.price if bid_1 else 0
            )
            if bid_1
            else 0
        )
        conf_2 = (
            market_2.get_fill_probability(
                OrderSide.BUY, quantity, ask_2.price if ask_2 else 0
            )
            if ask_2
            else 0
        )

        _, slippage_1 = (
            market_1.estimate_slippage(OrderSide.SELL, quantity) if market_1 else (0, 1)
        )
        _, slippage_2 = (
            market_2.estimate_slippage(OrderSide.BUY, quantity) if market_2 else (0, 1)
        )

        liquidity_score = (
            (market_1.get_liquidity_score() + market_2.get_liquidity_score()) / 2
            if market_1 and market_2
            else 0
        )

        base_confidence = (
            conf_1 * conf_2 * 0.4
            + (1 - slippage_1) * (1 - slippage_2) * 0.3
            + (
                (market_1.get_liquidity_score() if market_1 else 0)
                + (market_2.get_liquidity_score() if market_2 else 0)
            )
            / 2
            * 0.3
        )

        enhanced_confidence = base_confidence

        # Add predictive model confidence
        if self.enable_predictive_models and self.forecaster:
            predictive_boost = await self._get_predictive_boost(
                opportunity, market_1, market_2
            )
            enhanced_confidence += predictive_boost * self.predictive_weight

        # Add sentiment analysis confidence
        if self.enable_sentiment_analysis and self.sentiment_aggregator:
            sentiment_boost = await self._get_sentiment_boost(opportunity)
            enhanced_confidence += sentiment_boost * self.sentiment_weight

        return min(1.0, enhanced_confidence)

    async def _get_predictive_boost(
        self,
        opportunity: "ArbitrageOpportunity",
        market_1: OrderBook,
        market_2: OrderBook,
    ) -> float:
        """Get confidence boost from predictive models"""
        try:
            # Get recent prices for both markets
            recent_prices_1 = self._get_recent_prices(opportunity.market_id_1)
            recent_prices_2 = (
                self._get_recent_prices(opportunity.market_id_2)
                if opportunity.market_id_2
                else recent_prices_1
            )

            if not recent_prices_1:
                return 0

            # Calculate current spread
            if market_1 and market_2:
                current_spread = (
                    abs(market_1.get_mid_price() - market_2.get_mid_price()) / 2
                )
            else:
                current_spread = market_1.get_spread_percent() if market_1 else 0

            # Get timing signal
            timing_signal = await get_arbitrage_timing_signal(
                opportunity.market_id_1,
                current_spread,
                recent_prices_1,
                prediction_horizon_minutes=5,
            )

            # Boost confidence if timing signal aligns with opportunity
            if timing_signal["signal"] in ["enter_buy", "enter_sell"]:
                return timing_signal["confidence"] * 0.5  # Max 50% boost
            else:
                return 0

        except Exception as e:
            logger.error(f"Error getting predictive boost: {e}")
            return 0

    async def _get_sentiment_boost(self, opportunity: "ArbitrageOpportunity") -> float:
        """Get confidence boost from sentiment analysis"""
        try:
            market_id = opportunity.market_id_1

            # Check cache first
            cache_key = f"sentiment_{market_id}"
            if cache_key in self._sentiment_cache:
                cached_data = self._sentiment_cache[cache_key]
                if (
                    datetime.utcnow() - datetime.fromisoformat(cached_data["timestamp"])
                ).total_seconds() < self.cache_ttl:
                    sentiment_score = cached_data.get("sentiment_score", 0)
                    return (
                        abs(sentiment_score) * 0.3
                    )  # Max 30% boost based on sentiment strength

            # Get fresh sentiment data
            event_keywords = self._extract_event_keywords(market_id)
            sentiment_signal = await get_market_sentiment_signal(
                market_id, event_keywords
            )

            # Cache the result
            self._sentiment_cache[cache_key] = sentiment_signal

            # Boost confidence based on sentiment strength and direction
            sentiment_score = sentiment_signal.get("sentiment_score", 0)
            confidence = sentiment_signal.get("confidence", 0)

            # Strong sentiment increases confidence
            return abs(sentiment_score) * confidence * 0.3

        except Exception as e:
            logger.error(f"Error getting sentiment boost: {e}")
            return 0

    def _get_recent_prices(self, market_id: str, lookback: int = 20) -> List[float]:
        """Get recent prices for predictive modeling"""
        if market_id not in self._price_cache:
            self._price_cache[market_id] = []

        prices = self._price_cache[market_id]
        return prices[-lookback:] if prices else []

    def _extract_event_keywords(self, market_id: str) -> List[str]:
        """Extract event keywords from market ID for sentiment analysis"""
        # This is a simple implementation - customize based on your market ID format
        keywords = []

        # Common prediction market events
        if "election" in market_id.lower():
            keywords.extend(["election", "vote", "campaign", "democrat", "republican"])
        elif "stock" in market_id.lower():
            keywords.extend(["stock", "market", "trading", "finance"])
        elif "crypto" in market_id.lower():
            keywords.extend(["crypto", "bitcoin", "cryptocurrency", "blockchain"])
        elif "weather" in market_id.lower():
            keywords.extend(["weather", "climate", "temperature", "storm"])
        elif "sports" in market_id.lower():
            keywords.extend(["sports", "game", "match", "team"])

        return keywords

    def _calculate_confidence(
        self, market_1: OrderBook, market_2: OrderBook, quantity: int
    ) -> float:
        """Legacy confidence calculation - kept for backward compatibility"""
        conf_1 = (
            market_1.get_fill_probability(
                OrderSide.SELL, quantity, market_1.get_best_bid().price
            )
            if market_1.get_best_bid()
            else 0
        )
        conf_2 = (
            market_2.get_fill_probability(
                OrderSide.BUY, quantity, market_2.get_best_ask().price
            )
            if market_2.get_best_ask()
            else 0
        )

        _, slippage_1 = (
            market_1.estimate_slippage(OrderSide.SELL, quantity) if market_1 else (0, 1)
        )
        _, slippage_2 = (
            market_2.estimate_slippage(OrderSide.BUY, quantity) if market_2 else (0, 1)
        )

        liquidity_score = (
            (market_1.get_liquidity_score() + market_2.get_liquidity_score()) / 2
            if market_1 and market_2
            else 0
        )

        confidence = (
            conf_1 * conf_2 * 0.4
            + (1 - slippage_1) * (1 - slippage_2) * 0.3
            + liquidity_score / 100 * 0.3
        )

        return min(1.0, confidence)

    async def scan_for_opportunities(
        self, orderbooks: Dict[str, OrderBook]
    ) -> List[Union[ArbitrageOpportunity, StatisticalArbitrageOpportunity]]:
        opportunities = []

        # Process internal arbitrage
        internal_results = []
        for market_id, orderbook in orderbooks.items():
            internal_opps = self.detect_internal_arbitrage(orderbook, market_id)
            internal_results.extend(internal_opps)

        # Process traditional arbitrage opportunities
        cross_tasks = []
        market_ids = list(orderbooks.keys())
        for i in range(len(market_ids)):
            for j in range(i + 1, len(market_ids)):
                m1 = orderbooks[market_ids[i]]
                m2 = orderbooks[market_ids[j]]
                cross_tasks.append(self.detect_cross_market_arbitrage(m1, m2))

        # Execute async cross-market tasks
        cross_results = await asyncio.gather(*cross_tasks, return_exceptions=True)

        # Combine traditional arbitrage results
        all_results = [internal_results] + cross_results

        # Collect results and filter exceptions
        for result in all_results:
            if isinstance(result, Exception):
                logger.error(f"Error in opportunity detection: {result}")
            elif isinstance(result, list):
                opportunities.extend(result)

        # Process statistical arbitrage if enabled
        if self.enable_statistical_arbitrage and self.statistical_detector:
            statistical_opps = self.statistical_detector.find_opportunities(orderbooks)
            # Convert statistical opportunities to match expected return type
            for stat_opp in statistical_opps:
                # Map to existing ArbitrageOpportunity structure for compatibility
                trad_opp = ArbitrageOpportunity(
                    id=stat_opp.id,
                    type=ArbitrageType.CROSS_MARKET,  # Map statistical to existing type
                    market_id_1=stat_opp.market_id_1,
                    market_id_2=stat_opp.market_id_2,
                    buy_market_id=stat_opp.market_id_2
                    if stat_opp.quantity_2 > 0
                    else stat_opp.market_id_1,
                    sell_market_id=stat_opp.market_id_1
                    if stat_opp.quantity_1 > 0
                    else stat_opp.market_id_2,
                    buy_price=stat_opp.entry_price_2
                    if stat_opp.quantity_2 > 0
                    else stat_opp.entry_price_1,
                    sell_price=stat_opp.entry_price_1
                    if stat_opp.quantity_1 > 0
                    else stat_opp.entry_price_2,
                    quantity=stat_opp.quantity_1 + stat_opp.quantity_2,
                    profit_cents=stat_opp.expected_profit_cents,
                    profit_percent=stat_opp.profit_margin_percent,
                    fees=stat_opp.expected_profit_cents // 10,  # Approximate
                    net_profit_cents=stat_opp.expected_profit_cents,
                    confidence=min(1.0, stat_opp.confidence * 0.8),  # Adjust confidence
                    risk_level=stat_opp.risk_score,
                    execution_window_seconds=stat_opp.holding_period_hours * 3600,
                )
                opportunities.append(trad_opp)

        # Sort by profit and confidence
        opportunities.sort(
            key=lambda x: x.net_profit_cents * x.confidence, reverse=True
        )

        return opportunities[:20]

    def filter_by_threshold(
        self,
        opportunities: List[ArbitrageOpportunity],
        min_profit_cents: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> List[ArbitrageOpportunity]:
        if min_profit_cents is None:
            min_profit_cents = self.min_profit_cents
        if min_confidence is None:
            min_confidence = self.min_confidence

        return [
            opp
            for opp in opportunities
            if opp.net_profit_cents >= min_profit_cents
            and opp.confidence >= min_confidence
        ]

    def update_price_cache(self, market_id: str, price: float) -> None:
        """Update price cache for predictive modeling"""
        if market_id not in self._price_cache:
            self._price_cache[market_id] = []

        self._price_cache[market_id].append(price)

        # Keep only recent prices (last 100)
        if len(self._price_cache[market_id]) > 100:
            self._price_cache[market_id] = self._price_cache[market_id][-100:]

    def cleanup_caches(self) -> None:
        """Clean up expired cache entries"""
        now = datetime.utcnow()

        # Clean sentiment cache
        expired_keys = [
            key
            for key, data in self._sentiment_cache.items()
            if (now - datetime.fromisoformat(data["timestamp"])).total_seconds()
            > self.cache_ttl
        ]
        for key in expired_keys:
            del self._sentiment_cache[key]
