"""
Kalshi Multi-Market Correlation Strategy
Advanced correlation analysis for Kalshi prediction markets
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import json

from src.core.orderbook import OrderBook
from src.core.arbitrage import ArbitrageOpportunity, ArbitrageType
from src.utils.logging_utils import get_logger
from src.utils.config import Config

logger = get_logger("kalshi_correlation")


class MarketRelationType(Enum):
    """Types of relationships between Kalshi markets"""

    DIRECT_OPPOSITE = "direct_opposite"  # Yes/No on same event
    RELATED_EVENT = "related_event"  # Related but different events
    TEMPORAL = "temporal"  # Same event at different timeframes
    CONDITIONAL = "conditional"  # Conditional relationships
    BASKET_COMPONENT = "basket_component"  # Markets that form a basket
    ARBITRAGE_TRIANGLE = "arbitrage_triangle"  # Three-way arbitrage


@dataclass
class MarketCorrelation:
    """Correlation data between two Kalshi markets"""

    market_id_1: str
    market_id_2: str
    relationship_type: MarketRelationType

    # Correlation metrics
    pearson_correlation: float = 0.0
    spearman_correlation: float = 0.0
    price_correlation: float = 0.0

    # Spread analysis
    price_spread: float = 0.0
    spread_mean: float = 0.0
    spread_std: float = 0.0
    z_score: float = 0.0

    # Liquidity metrics
    combined_liquidity: float = 0.0
    liquidity_ratio: float = 0.0

    # Historical data
    price_history_1: List[float] = field(default_factory=list)
    price_history_2: List[float] = field(default_factory=list)
    correlation_history: List[Tuple[datetime, float]] = field(default_factory=list)

    # Trading metrics
    last_updated: datetime = field(default_factory=datetime.now)
    trade_count: int = 0
    success_rate: float = 0.0

    # Confidence metrics
    correlation_strength: float = 0.0
    prediction_confidence: float = 0.0


@dataclass
class CorrelationOpportunity:
    """Arbitrage opportunity based on market correlations"""

    opportunity_id: str
    correlation: MarketCorrelation

    # Opportunity type and direction
    opportunity_type: str  # "convergence", "divergence", "triangular"
    direction: str  # "long_short", "short_long", "triangular"

    # Financial metrics
    expected_profit_cents: int = 0
    risk_score: float = 0.0
    confidence: float = 0.0

    # Position sizing
    market_1_position: int = 0
    market_2_position: int = 0
    market_3_position: int = 0  # For triangular

    # Price levels
    entry_price_1: int = 0
    entry_price_2: int = 0
    entry_price_3: int = 0
    target_spread: float = 0.0
    stop_loss_spread: float = 0.0

    # Timing
    execution_window_seconds: int = 30
    timestamp: datetime = field(default_factory=datetime.now)

    def to_arbitrage_opportunity(self) -> ArbitrageOpportunity:
        """Convert to standard arbitrage opportunity"""
        arbitrage_type = (
            ArbitrageType.TRIANGULAR
            if self.opportunity_type == "triangular"
            else ArbitrageType.CROSS_MARKET
        )

        return ArbitrageOpportunity(
            id=self.opportunity_id,
            type=arbitrage_type,
            market_id_1=self.correlation.market_id_1,
            market_id_2=self.correlation.market_id_2
            if self.market_3_position == 0
            else None,
            buy_market_id=self.correlation.market_id_1
            if self.market_1_position > 0
            else self.correlation.market_id_2,
            sell_market_id=self.correlation.market_id_2
            if self.market_1_position > 0
            else self.correlation.market_id_1,
            buy_price=min(self.entry_price_1, self.entry_price_2),
            sell_price=max(self.entry_price_1, self.entry_price_2),
            quantity=abs(self.market_1_position),
            profit_cents=self.expected_profit_cents,
            net_profit_cents=self.expected_profit_cents,
            confidence=self.confidence,
            risk_level="high"
            if self.risk_score > 0.7
            else "medium"
            if self.risk_score > 0.4
            else "low",
            execution_window_seconds=self.execution_window_seconds,
        )


class KalshiCorrelationStrategy:
    """
    Advanced correlation strategy for Kalshi prediction markets
    """

    def __init__(self, config: Config):
        self.config = config
        self.correlation_config = config.get("correlation_strategy", {})

        # Strategy parameters
        self.min_correlation_threshold = self.correlation_config.get(
            "min_correlation", 0.6
        )
        self.max_spread_deviation = self.correlation_config.get(
            "max_spread_deviation", 2.0
        )
        self.min_liquidity_threshold = self.correlation_config.get(
            "min_liquidity_threshold", 100
        )
        self.max_positions_per_pair = self.correlation_config.get(
            "max_positions_per_pair", 5
        )

        # Analysis parameters
        self.correlation_lookback_periods = self.correlation_config.get(
            "lookback_periods", 30
        )
        self.spread_analysis_window = self.correlation_config.get("spread_window", 20)
        self.update_interval_seconds = self.correlation_config.get(
            "update_interval", 60
        )

        # Market correlation tracking
        self.market_correlations: Dict[str, MarketCorrelation] = {}
        self.market_metadata: Dict[str, Dict[str, Any]] = {}
        self.market_clusters: Dict[int, List[str]] = {}

        # Price history for analysis
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}

        # Performance tracking
        self.opportunities_found = 0
        self.opportunities_executed = 0
        self.success_rate = 0.0

        logger.info("Kalshi Correlation Strategy initialized")

    async def analyze_markets(
        self, orderbooks: Dict[str, OrderBook]
    ) -> List[CorrelationOpportunity]:
        """Analyze markets for correlation-based arbitrage opportunities"""
        try:
            # Update price history
            await self._update_price_history(orderbooks)

            # Update correlations
            await self._update_correlations(orderbooks)

            # Find opportunities
            opportunities = await self._find_correlation_opportunities()

            self.opportunities_found = len(opportunities)

            logger.info(f"Found {len(opportunities)} correlation opportunities")
            return opportunities

        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            return []

    async def _update_price_history(self, orderbooks: Dict[str, OrderBook]) -> None:
        """Update price history for all markets"""
        current_time = datetime.now()

        for market_id, orderbook in orderbooks.items():
            try:
                # Get mid price
                mid_price = orderbook.get_mid_price()
                if mid_price is None:
                    continue

                # Convert to cents (Kalshi uses integer prices)
                mid_price_cents = int(mid_price)

                # Add to history
                if market_id not in self.price_history:
                    self.price_history[market_id] = []

                self.price_history[market_id].append((current_time, mid_price_cents))

                # Keep only recent history
                max_history = self.correlation_lookback_periods * 2
                if len(self.price_history[market_id]) > max_history:
                    self.price_history[market_id] = self.price_history[market_id][
                        -max_history:
                    ]

            except Exception as e:
                logger.debug(f"Failed to update price history for {market_id}: {e}")

    async def _update_correlations(self, orderbooks: Dict[str, OrderBook]) -> None:
        """Update correlation data for all market pairs"""
        market_ids = list(orderbooks.keys())

        # Generate correlations for market pairs
        for i, market_id_1 in enumerate(market_ids):
            for market_id_2 in market_ids[i + 1 :]:
                try:
                    correlation_key = f"{market_id_1}_{market_id_2}"

                    # Determine relationship type
                    relationship_type = await self._classify_market_relationship(
                        market_id_1, market_id_2
                    )

                    # Calculate correlation
                    correlation = await self._calculate_market_correlation(
                        market_id_1, market_id_2, relationship_type, orderbooks
                    )

                    if correlation:
                        self.market_correlations[correlation_key] = correlation

                except Exception as e:
                    logger.debug(
                        f"Failed to update correlation for {market_id_1}-{market_id_2}: {e}"
                    )

    async def _classify_market_relationship(
        self, market_id_1: str, market_id_2: str
    ) -> MarketRelationType:
        """Classify the relationship type between two markets"""
        # Extract market IDs and titles
        market_1_base = market_id_1.split("-")[0] if "-" in market_id_1 else market_id_1
        market_2_base = market_id_2.split("-")[0] if "-" in market_id_2 else market_id_2

        # Check if they're opposites (Yes/No on same event)
        if market_1_base == market_2_base:
            return MarketRelationType.DIRECT_OPPOSITE

        # Check if they're related by topic/keywords
        keywords_1 = set(market_id_1.lower().split("_"))
        keywords_2 = set(market_id_2.lower().split("_"))

        overlap = keywords_1.intersection(keywords_2)
        if len(overlap) >= 2:  # Significant keyword overlap
            return MarketRelationType.RELATED_EVENT

        # Check temporal relationships
        temporal_keywords = {"today", "tomorrow", "week", "month", "quarter", "year"}
        if any(kw in keywords_1 for kw in temporal_keywords) and any(
            kw in keywords_2 for kw in temporal_keywords
        ):
            return MarketRelationType.TEMPORAL

        # Default to related event
        return MarketRelationType.RELATED_EVENT

    async def _calculate_market_correlation(
        self,
        market_id_1: str,
        market_id_2: str,
        relationship_type: MarketRelationType,
        orderbooks: Dict[str, OrderBook],
    ) -> Optional[MarketCorrelation]:
        """Calculate correlation between two markets"""
        try:
            # Get price histories
            history_1 = self.price_history.get(market_id_1, [])
            history_2 = self.price_history.get(market_id_2, [])

            if len(history_1) < 10 or len(history_2) < 10:
                return None

            # Align time periods
            timestamps_1 = [h[0] for h in history_1]
            timestamps_2 = [h[0] for h in history_2]
            prices_1 = [h[1] for h in history_1]
            prices_2 = [h[1] for h in history_2]

            # Find overlapping time periods
            min_timestamp = max(min(timestamps_1), min(timestamps_2))
            max_timestamp = min(max(timestamps_1), max(timestamps_2))

            # Filter to overlapping period
            filtered_1 = [
                p for t, p in history_1 if min_timestamp <= t <= max_timestamp
            ]
            filtered_2 = [
                p for t, p in history_2 if min_timestamp <= t <= max_timestamp
            ]

            if len(filtered_1) < 10 or len(filtered_2) < 10:
                return None

            # Calculate correlations
            pearson_corr, pearson_p = pearsonr(filtered_1, filtered_2)
            spearman_corr, spearman_p = spearmanr(filtered_1, filtered_2)

            # Handle NaN values
            if np.isnan(pearson_corr) or np.isnan(spearman_corr):
                pearson_corr = 0.0
                spearman_corr = 0.0

            # Calculate price spread
            current_prices = [filtered_1[-1], filtered_2[-1]]
            spread = abs(current_prices[0] - current_prices[1])

            # Calculate spread statistics
            historical_spreads = [
                abs(p1 - p2) for p1, p2 in zip(filtered_1, filtered_2)
            ]
            spread_mean = np.mean(historical_spreads)
            spread_std = np.std(historical_spreads)
            z_score = (spread - spread_mean) / spread_std if spread_std > 0 else 0

            # Get liquidity metrics
            liquidity_1 = self._get_liquidity_score(orderbooks.get(market_id_1))
            liquidity_2 = self._get_liquidity_score(orderbooks.get(market_id_2))
            combined_liquidity = liquidity_1 + liquidity_2
            liquidity_ratio = (
                min(liquidity_1, liquidity_2) / max(liquidity_1, liquidity_2)
                if max(liquidity_1, liquidity_2) > 0
                else 0
            )

            # Calculate correlation strength
            correlation_strength = max(abs(pearson_corr), abs(spearman_corr))

            # Determine prediction confidence
            prediction_confidence = min(
                1.0, correlation_strength * min(1.0, abs(z_score) / 2.0)
            )

            # Create correlation object
            correlation = MarketCorrelation(
                market_id_1=market_id_1,
                market_id_2=market_id_2,
                relationship_type=relationship_type,
                pearson_correlation=pearson_corr,
                spearman_correlation=spearman_corr,
                price_correlation=correlation_strength,
                price_spread=spread,
                spread_mean=spread_mean,
                spread_std=spread_std,
                z_score=abs(z_score),
                combined_liquidity=combined_liquidity,
                liquidity_ratio=liquidity_ratio,
                price_history_1=filtered_1[-30:],  # Keep last 30 points
                price_history_2=filtered_2[-30:],
                correlation_strength=correlation_strength,
                prediction_confidence=prediction_confidence,
            )

            return correlation

        except Exception as e:
            logger.error(
                f"Correlation calculation failed for {market_id_1}-{market_id_2}: {e}"
            )
            return None

    def _get_liquidity_score(self, orderbook: Optional[OrderBook]) -> float:
        """Calculate liquidity score for an orderbook"""
        if not orderbook:
            return 0.0

        try:
            # Get bid/ask depth
            bid_depth = orderbook.get_bid_depth(3)
            ask_depth = orderbook.get_ask_depth(3)

            # Calculate spread quality
            spread_pct = orderbook.get_spread_percent() or 100.0

            # Combined liquidity score
            liquidity = (bid_depth + ask_depth) / max(1.0, spread_pct)

            return min(100.0, liquidity)

        except Exception:
            return 0.0

    async def _find_correlation_opportunities(self) -> List[CorrelationOpportunity]:
        """Find arbitrage opportunities based on correlations"""
        opportunities = []

        for correlation_key, correlation in self.market_correlations.items():
            try:
                # Check for convergence opportunities
                if correlation.relationship_type == MarketRelationType.DIRECT_OPPOSITE:
                    opp = await self._find_direct_opposite_opportunity(correlation)
                    if opp:
                        opportunities.append(opp)

                # Check for related event opportunities
                elif correlation.relationship_type == MarketRelationType.RELATED_EVENT:
                    opp = await self._find_related_event_opportunity(correlation)
                    if opp:
                        opportunities.append(opp)

                # Check for temporal opportunities
                elif correlation.relationship_type == MarketRelationType.TEMPORAL:
                    opp = await self._find_temporal_opportunity(correlation)
                    if opp:
                        opportunities.append(opp)

            except Exception as e:
                logger.debug(f"Failed to find opportunity for {correlation_key}: {e}")

        # Rank opportunities by quality
        opportunities.sort(
            key=lambda x: x.confidence * x.expected_profit_cents, reverse=True
        )

        return opportunities[:10]  # Return top 10 opportunities

    async def _find_direct_opposite_opportunity(
        self, correlation: MarketCorrelation
    ) -> Optional[CorrelationOpportunity]:
        """Find arbitrage opportunity in direct opposite markets (Yes/No pairs)"""
        try:
            # Direct opposite markets should have negative correlation near -1
            if correlation.price_correlation > -0.8:
                return None

            # Check if spread is outside normal range
            if abs(correlation.z_score) < self.max_spread_deviation:
                return None

            # Expected profit from convergence to -1 correlation
            ideal_spread = 0.0  # Yes + No should equal 100 cents
            current_spread = correlation.price_spread
            expected_profit_cents = int(
                abs(current_spread - ideal_spread) * 0.8
            )  # Conservative estimate

            if expected_profit_cents < 10:  # Minimum profit threshold
                return None

            # Calculate position sizes
            position_size = min(
                self.max_positions_per_pair, correlation.combined_liquidity / 10
            )

            # Determine direction
            # If Yes price > No price significantly, short Yes, long No
            latest_price_1 = correlation.price_history_1[-1]
            latest_price_2 = correlation.price_history_2[-1]

            if latest_price_1 > 60:  # Yes is overpriced
                market_1_position = -position_size  # Short Yes
                market_2_position = position_size  # Long No
            else:  # No is overpriced
                market_1_position = position_size  # Long Yes
                market_2_position = -position_size  # Short No

            # Calculate confidence
            confidence = min(
                1.0, abs(correlation.z_score) / 3.0 * correlation.prediction_confidence
            )

            # Risk score
            risk_score = max(0.0, 1.0 - confidence)

            opportunity = CorrelationOpportunity(
                opportunity_id=f"opp_{correlation.market_id_1[:8]}_{correlation.market_id_2[:8]}",
                correlation=correlation,
                opportunity_type="convergence",
                direction="long_short" if market_1_position > 0 else "short_long",
                expected_profit_cents=expected_profit_cents,
                risk_score=risk_score,
                confidence=confidence,
                market_1_position=market_1_position,
                market_2_position=market_2_position,
                entry_price_1=latest_price_1,
                entry_price_2=latest_price_2,
                target_spread=ideal_spread,
                stop_loss_spread=correlation.spread_mean + 2 * correlation.spread_std,
            )

            return opportunity if confidence >= 0.6 else None

        except Exception as e:
            logger.error(f"Direct opposite opportunity detection failed: {e}")
            return None

    async def _find_related_event_opportunity(
        self, correlation: MarketCorrelation
    ) -> Optional[CorrelationOpportunity]:
        """Find arbitrage opportunity in related events"""
        try:
            # Check correlation strength
            if correlation.correlation_strength < self.min_correlation_threshold:
                return None

            # Check for spread deviation
            if abs(correlation.z_score) < 1.5:
                return None

            # Expected profit from correlation normalization
            expected_profit_cents = int(
                abs(correlation.z_score) * correlation.spread_std * 0.6
            )

            if expected_profit_cents < 15:
                return None

            # Position sizing based on correlation strength
            position_multiplier = min(1.0, correlation.correlation_strength)
            position_size = min(
                self.max_positions_per_pair * position_multiplier,
                correlation.combined_liquidity / 15,
            )

            # Get latest prices
            latest_price_1 = correlation.price_history_1[-1]
            latest_price_2 = correlation.price_history_2[-1]

            # Determine direction based on correlation type
            if correlation.price_correlation > 0:  # Positive correlation
                # If prices diverged, bet on convergence
                price_ratio = latest_price_1 / latest_price_2
                if price_ratio > 1.2:  # Market 1 is overpriced relative to 2
                    market_1_position = -position_size
                    market_2_position = position_size
                elif price_ratio < 0.8:  # Market 2 is overpriced relative to 1
                    market_1_position = position_size
                    market_2_position = -position_size
                else:
                    return None
            else:  # Negative correlation
                # If prices moved together, bet on divergence
                avg_price = (latest_price_1 + latest_price_2) / 2
                if latest_price_1 > avg_price and latest_price_2 > avg_price:
                    market_1_position = -position_size
                    market_2_position = position_size
                elif latest_price_1 < avg_price and latest_price_2 < avg_price:
                    market_1_position = position_size
                    market_2_position = -position_size
                else:
                    return None

            # Confidence based on correlation and Z-score
            confidence = min(
                1.0,
                correlation.prediction_confidence
                * min(1.0, abs(correlation.z_score) / 2.5),
            )

            opportunity = CorrelationOpportunity(
                opportunity_id=f"rel_{correlation.market_id_1[:8]}_{correlation.market_id_2[:8]}",
                correlation=correlation,
                opportunity_type="convergence",
                direction="long_short" if market_1_position > 0 else "short_long",
                expected_profit_cents=expected_profit_cents,
                risk_score=1.0 - confidence,
                confidence=confidence,
                market_1_position=market_1_position,
                market_2_position=market_2_position,
                entry_price_1=latest_price_1,
                entry_price_2=latest_price_2,
                target_spread=correlation.spread_mean,
                stop_loss_spread=correlation.spread_mean + 2.5 * correlation.spread_std,
            )

            return opportunity if confidence >= 0.5 else None

        except Exception as e:
            logger.error(f"Related event opportunity detection failed: {e}")
            return None

    async def _find_temporal_opportunity(
        self, correlation: MarketCorrelation
    ) -> Optional[CorrelationOpportunity]:
        """Find arbitrage opportunity in temporal markets"""
        try:
            # Temporal opportunities require stronger correlations
            if correlation.correlation_strength < 0.7:
                return None

            # Check for significant spread deviation
            if abs(correlation.z_score) < 2.0:
                return None

            # Calculate expected profit
            expected_profit_cents = int(
                abs(correlation.z_score) * correlation.spread_std * 0.7
            )

            if expected_profit_cents < 20:
                return None

            # Smaller position sizes for temporal markets (higher uncertainty)
            position_size = min(
                self.max_positions_per_pair * 0.6, correlation.combined_liquidity / 20
            )

            # Get latest prices
            latest_price_1 = correlation.price_history_1[-1]
            latest_price_2 = correlation.price_history_2[-1]

            # Direction based on expected temporal relationship
            # Generally, longer-term markets should be more stable
            price_diff = latest_price_1 - latest_price_2
            if abs(price_diff) < correlation.spread_mean + correlation.spread_std:
                return None

            if price_diff > 0:
                market_1_position = -position_size
                market_2_position = position_size
            else:
                market_1_position = position_size
                market_2_position = -position_size

            # Higher confidence for strong correlations
            confidence = min(1.0, correlation.prediction_confidence * 0.8)

            opportunity = CorrelationOpportunity(
                opportunity_id=f"temp_{correlation.market_id_1[:8]}_{correlation.market_id_2[:8]}",
                correlation=correlation,
                opportunity_type="temporal",
                direction="long_short" if market_1_position > 0 else "short_long",
                expected_profit_cents=expected_profit_cents,
                risk_score=1.0 - confidence,
                confidence=confidence,
                market_1_position=market_1_position,
                market_2_position=market_2_position,
                entry_price_1=latest_price_1,
                entry_price_2=latest_price_2,
                target_spread=correlation.spread_mean,
                stop_loss_spread=correlation.spread_mean + 3 * correlation.spread_std,
                execution_window_seconds=60,  # Longer window for temporal
            )

            return opportunity if confidence >= 0.55 else None

        except Exception as e:
            logger.error(f"Temporal opportunity detection failed: {e}")
            return None

    async def get_correlation_report(self) -> Dict[str, Any]:
        """Get comprehensive correlation strategy report"""
        try:
            correlations = list(self.market_correlations.values())

            # Correlation statistics
            if correlations:
                correlation_values = [c.correlation_strength for c in correlations]
                spread_values = [c.price_spread for c in correlations]
                liquidity_values = [c.combined_liquidity for c in correlations]

                correlation_stats = {
                    "total_pairs": len(correlations),
                    "avg_correlation": float(np.mean(correlation_values)),
                    "max_correlation": float(np.max(correlation_values)),
                    "min_correlation": float(np.min(correlation_values)),
                    "avg_spread": float(np.mean(spread_values)),
                    "avg_liquidity": float(np.mean(liquidity_values)),
                    "strong_correlations": len(
                        [c for c in correlations if c.correlation_strength > 0.8]
                    ),
                }
            else:
                correlation_stats = {"total_pairs": 0}

            # Relationship type distribution
            relationship_counts = {}
            for correlation in correlations:
                rel_type = correlation.relationship_type.value
                relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1

            return {
                "strategy_performance": {
                    "opportunities_found": self.opportunities_found,
                    "opportunities_executed": self.opportunities_executed,
                    "success_rate": self.success_rate,
                },
                "correlation_statistics": correlation_stats,
                "relationship_distribution": relationship_counts,
                "active_correlations": len(
                    [c for c in correlations if abs(c.z_score) > 1.0]
                ),
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Correlation report generation failed: {e}")
            return {}


# Utility function to create strategy instance
def create_kalshi_correlation_strategy(config: Config) -> KalshiCorrelationStrategy:
    """Create and return Kalshi correlation strategy instance"""
    return KalshiCorrelationStrategy(config)
