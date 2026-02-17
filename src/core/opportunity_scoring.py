"""Real-time opportunity scoring with ML-enhanced ranking."""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from src.core.arbitrage import ArbitrageOpportunity, ArbitrageType
from src.core.statistical_arbitrage import StatisticalArbitrageOpportunity
from src.utils.logging_utils import get_logger
from src.utils.performance_cache import get_cache_manager

logger = get_logger("opportunity_scoring")


class OpportunityScore(Enum):
    """Opportunity score levels."""

    EXCELLENT = 9.0
    VERY_GOOD = 7.5
    GOOD = 6.0
    FAIR = 4.0
    POOR = 2.0
    UNSUITABLE = 1.0


@dataclass
class ScoringWeights:
    """Weights for different scoring factors."""

    profit_weight: float = 0.3
    confidence_weight: float = 0.25
    liquidity_weight: float = 0.2
    volatility_weight: float = 0.1
    speed_weight: float = 0.1
    risk_weight: float = 0.05


@dataclass
class RealTimeScore:
    """Real-time opportunity scoring metrics."""

    opportunity_id: str
    total_score: float
    sub_scores: Dict[str, float]
    profit_score: float
    confidence_score: float
    liquidity_score: float
    volatility_score: float
    speed_score: float
    risk_score: float
    timestamp: datetime
    market_conditions: Dict[str, Any]
    execution_priority: int


class OpportunityScorer:
    """Real-time opportunity scorer with ML enhancement."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize opportunity scorer."""
        self.config = config or {}
        self.cache_manager = get_cache_manager()

        # Scoring weights
        self.weights = ScoringWeights(
            profit_weight=self.config.get("scoring.profit_weight", 0.3),
            confidence_weight=self.config.get("scoring.confidence_weight", 0.25),
            liquidity_weight=self.config.get("scoring.liquidity_weight", 0.2),
            volatility_weight=self.config.get("scoring.volatility_weight", 0.1),
            speed_weight=self.config.get("scoring.speed_weight", 0.1),
            risk_weight=self.config.get("scoring.risk_weight", 0.05),
        )

        # Market condition thresholds
        self.thresholds = {
            "min_profit_cents": self.config.get("scoring.min_profit_cents", 10),
            "min_confidence": self.config.get("scoring.min_confidence", 0.6),
            "max_volatility": self.config.get("scoring.max_volatility", 0.5),
            "min_liquidity_score": self.config.get("scoring.min_liquidity_score", 50),
            "max_risk_score": self.config.get("scoring.max_risk_score", 0.8),
        }

        # Market condition history
        self.market_history: Dict[str, List[Dict[str, Any]]] = {}
        self.volatility_window = self.config.get("scoring.volatility_window", 50)

        # Performance tracking
        self.scored_opportunities: Dict[str, RealTimeScore] = {}
        self.performance_history: List[Dict[str, Any]] = []

        logger.info("Opportunity scorer initialized")

    async def score_opportunity(
        self,
        opportunity: Union[ArbitrageOpportunity, StatisticalArbitrageOpportunity],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> RealTimeScore:
        """Score a single opportunity in real-time."""
        opportunity_id = self._get_opportunity_id(opportunity)

        # Get current market conditions
        market_conditions = await self._get_market_conditions(opportunity, market_data)

        # Calculate individual scores
        profit_score = self._calculate_profit_score(opportunity)
        confidence_score = self._calculate_confidence_score(opportunity)
        liquidity_score = self._calculate_liquidity_score(
            opportunity, market_conditions
        )
        volatility_score = self._calculate_volatility_score(
            opportunity, market_conditions
        )
        speed_score = self._calculate_speed_score(opportunity, market_conditions)
        risk_score = self._calculate_risk_score(opportunity, market_conditions)

        # Calculate total weighted score
        sub_scores = {
            "profit": profit_score,
            "confidence": confidence_score,
            "liquidity": liquidity_score,
            "volatility": volatility_score,
            "speed": speed_score,
            "risk": max(0, 10 - risk_score),  # Lower risk = higher score
        }

        total_score = (
            profit_score * self.weights.profit_weight
            + confidence_score * self.weights.confidence_weight
            + liquidity_score * self.weights.liquidity_weight
            + volatility_score * self.weights.volatility_weight
            + speed_score * self.weights.speed_weight
            + sub_scores["risk"] * self.weights.risk_weight
        )

        # Determine execution priority
        execution_priority = self._determine_execution_priority(total_score)

        real_time_score = RealTimeScore(
            opportunity_id=opportunity_id,
            total_score=total_score,
            sub_scores=sub_scores,
            profit_score=profit_score,
            confidence_score=confidence_score,
            liquidity_score=liquidity_score,
            volatility_score=volatility_score,
            speed_score=speed_score,
            risk_score=risk_score,
            timestamp=datetime.utcnow(),
            market_conditions=market_conditions,
            execution_priority=execution_priority,
        )

        # Store score
        self.scored_opportunities[opportunity_id] = real_time_score

        # Cache score
        await self.cache_manager.cache.set_orderbook(
            f"opportunity_score:{opportunity_id}",
            {
                "score": real_time_score.total_score,
                "priority": real_time_score.execution_priority,
                "timestamp": real_time_score.timestamp.isoformat(),
            },
        )

        return real_time_score

    async def score_batch(
        self,
        opportunities: List[
            Union[ArbitrageOpportunity, StatisticalArbitrageOpportunity]
        ],
    ) -> List[RealTimeScore]:
        """Score multiple opportunities concurrently."""
        logger.info(f"Scoring batch of {len(opportunities)} opportunities")

        start_time = time.time()

        # Score all opportunities concurrently
        tasks = [self.score_opportunity(opp) for opp in opportunities]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        scored_opportunities = []
        for i, result in enumerate(results):
            if isinstance(result, RealTimeScore):
                scored_opportunities.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Scoring error for opportunity {i}: {result}")

        # Sort by total score (descending)
        scored_opportunities.sort(key=lambda x: x.total_score, reverse=True)

        # Assign execution priorities
        for i, score in enumerate(scored_opportunities):
            score.execution_priority = i + 1

        execution_time = time.time() - start_time
        logger.info(f"Batch scoring completed in {execution_time:.3f}s")

        # Cache batch result
        await self.cache_manager.cache.set_opportunities(
            [score.total_score for score in scored_opportunities], "latest_scores"
        )

        return scored_opportunities

    def _get_opportunity_id(
        self, opportunity: Union[ArbitrageOpportunity, StatisticalArbitrageOpportunity]
    ) -> str:
        """Extract opportunity ID."""
        return getattr(opportunity, "id", "unknown")

    async def _get_market_conditions(
        self,
        opportunity: Union[ArbitrageOpportunity, StatisticalArbitrageOpportunity],
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get current market conditions for scoring."""
        conditions = additional_data or {}

        # Get market IDs
        market_id_1 = getattr(opportunity, "market_id_1", None)
        market_id_2 = getattr(opportunity, "market_id_2", None)

        # Get cached market data
        for market_id in [market_id_1, market_id_2]:
            if market_id:
                market_data = await self.cache_manager.cache.get_orderbook(market_id)
                if market_data:
                    conditions[f"{market_id}_volume"] = market_data.get("volume", 0)
                    conditions[f"{market_id}_spread"] = market_data.get(
                        "spread_percent", 0
                    )
                    conditions[f"{market_id}_liquidity"] = market_data.get(
                        "liquidity_score", 0
                    )

        # Get market status
        market_status = await self.cache_manager.cache.get_market_status()
        if market_status:
            conditions.update(market_status)

        # Get time-based conditions
        now = datetime.utcnow()
        conditions.update(
            {
                "hour_of_day": now.hour,
                "day_of_week": now.weekday(),
                "is_market_hours": 6 <= now.hour <= 22,  # Trading hours
                "is_weekend": now.weekday() >= 5,
                "market_urgency": self._calculate_market_urgency(conditions),
            }
        )

        return conditions

    def _calculate_profit_score(
        self, opportunity: Union[ArbitrageOpportunity, StatisticalArbitrageOpportunity]
    ) -> float:
        """Calculate profit score (0-10)."""
        profit_cents = getattr(
            opportunity,
            "expected_profit_cents",
            getattr(opportunity, "net_profit_cents", 0),
        )

        if profit_cents <= 0:
            return 0.0

        # Normalize profit score based on threshold
        min_profit = self.thresholds["min_profit_cents"]

        if profit_cents <= min_profit:
            return 1.0  # Minimum score for profitable opportunity

        # Scale score logarithmically for higher profits
        score = 1.0 + 9.0 * (1 - np.exp(-profit_cents / min_profit))
        return min(10.0, max(1.0, score))

    def _calculate_confidence_score(
        self, opportunity: Union[ArbitrageOpportunity, StatisticalArbitrageOpportunity]
    ) -> float:
        """Calculate confidence score (0-10)."""
        confidence = getattr(opportunity, "confidence", 0.5)

        # Scale confidence to 0-10
        score = confidence * 10.0
        return min(10.0, max(1.0, score))

    def _calculate_liquidity_score(
        self,
        opportunity: Union[ArbitrageOpportunity, StatisticalArbitrageOpportunity],
        market_conditions: Dict[str, Any],
    ) -> float:
        """Calculate liquidity score (0-10)."""
        liquidity = 0.0
        total_volume = 0

        # Get liquidity from market conditions
        for attr in ["volume", "liquidity_score"]:
            for key, value in market_conditions.items():
                if attr in key and value:
                    liquidity += value
                    if "volume" in key:
                        total_volume += value

        # Default liquidity from opportunity
        if liquidity == 0:
            liquidity = getattr(opportunity, "quantity", 100)

        # Normalize to 0-10 scale
        min_liquidity = self.thresholds["min_liquidity_score"]

        if liquidity <= min_liquidity:
            return 1.0
        elif liquidity >= 1000:
            return 10.0
        else:
            return 1.0 + 9.0 * ((liquidity - min_liquidity) / (1000 - min_liquidity))

    def _calculate_volatility_score(
        self,
        opportunity: Union[ArbitrageOpportunity, StatisticalArbitrageOpportunity],
        market_conditions: Dict[str, Any],
    ) -> float:
        """Calculate volatility score (0-10)."""
        # Get volatility from opportunity
        risk_score = getattr(opportunity, "risk_score", 0.5)
        volatility = min(1.0, max(0.0, risk_score))

        # Lower volatility = higher score
        max_volatility = self.thresholds["max_volatility"]

        if volatility <= max_volatility / 3:
            return 10.0
        elif volatility <= max_volatility * 2 / 3:
            return 7.0
        elif volatility <= max_volatility:
            return 4.0
        else:
            return 1.0

    def _calculate_speed_score(
        self,
        opportunity: Union[ArbitrageOpportunity, StatisticalArbitrageOpportunity],
        market_conditions: Dict[str, Any],
    ) -> float:
        """Calculate speed score based on market urgency (0-10)."""
        # Higher score during urgent market conditions
        market_urgency = market_conditions.get("market_urgency", 0.5)
        is_market_hours = market_conditions.get("is_market_hours", True)
        hour_of_day = market_conditions.get("hour_of_day", 12)

        base_score = 6.0

        # Market hours bonus
        if is_market_hours:
            base_score += 2.0

        # Time-based adjustments
        if 9 <= hour_of_day <= 16:  # Business hours
            base_score += 1.0

        # Market urgency adjustment
        urgency_bonus = market_urgency * 2.0
        score = min(10.0, base_score + urgency_bonus)

        return max(1.0, score)

    def _calculate_risk_score(
        self,
        opportunity: Union[ArbitrageOpportunity, StatisticalArbitrageOpportunity],
        market_conditions: Dict[str, Any],
    ) -> float:
        """Calculate risk score (0-10, lower is better)."""
        # Extract risk factors
        risk_score = getattr(opportunity, "risk_score", 0.5)

        # Time-based risk (higher risk outside market hours)
        is_market_hours = market_conditions.get("is_market_hours", True)
        time_risk_penalty = 0.0 if is_market_hours else 2.0

        # Weekend risk penalty
        is_weekend = market_conditions.get("is_weekend", False)
        weekend_risk_penalty = 0.0 if not is_weekend else 1.0

        # Combine risk scores
        total_risk = risk_score * 10.0 + time_risk_penalty + weekend_risk_penalty

        # Normalize to 0-10 (lower is better)
        return min(10.0, total_risk)

    def _calculate_market_urgency(self, conditions: Dict[str, Any]) -> float:
        """Calculate market urgency based on conditions."""
        urgency = 0.5  # Default moderate urgency

        # Check for high volatility indicators
        spreads = [value for key, value in conditions.items() if "spread" in key]
        if spreads:
            avg_spread = np.mean([s for s in spreads if s > 0])
            if avg_spread > 5.0:  # High spreads = urgent
                urgency += 0.3

        # Check for unusual volume
        volumes = [value for key, value in conditions.items() if "volume" in key]
        if volumes:
            volume_ratio = min(volumes) / max(volumes) if max(volumes) > 0 else 1
            if volume_ratio < 0.3:  # Low volume relative = urgent
                urgency += 0.2

        return min(1.0, max(0.0, urgency))

    def _determine_execution_priority(self, total_score: float) -> int:
        """Determine execution priority based on total score."""
        if total_score >= OpportunityScore.EXCELLENT.value:
            return 1  # Highest priority
        elif total_score >= OpportunityScore.VERY_GOOD.value:
            return 2
        elif total_score >= OpportunityScore.GOOD.value:
            return 3
        elif total_score >= OpportunityScore.FAIR.value:
            return 4
        else:
            return 5  # Lowest priority

    async def get_top_opportunities(
        self, max_count: int = 10, min_score: float = 5.0
    ) -> List[RealTimeScore]:
        """Get top scored opportunities."""
        # Get all current scores
        current_scores = list(self.scored_opportunities.values())

        # Filter by minimum score and timestamp (last 5 minutes)
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)

        filtered_scores = [
            score
            for score in current_scores
            if (score.total_score >= min_score and score.timestamp >= cutoff_time)
        ]

        # Sort by total score and execution priority
        filtered_scores.sort(
            key=lambda x: (x.total_score, -x.execution_priority), reverse=True
        )

        return filtered_scores[:max_count]

    def get_scoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive scoring statistics."""
        current_scores = list(self.scored_opportunities.values())

        if not current_scores:
            return {
                "total_scored": 0,
                "avg_score": 0,
                "score_distribution": {},
                "sub_score_averages": {},
            }

        # Calculate statistics
        total_scored = len(current_scores)
        avg_score = np.mean([s.total_score for s in current_scores])

        # Score distribution
        score_distribution = {
            "excellent": len(
                [
                    s
                    for s in current_scores
                    if s.total_score >= OpportunityScore.EXCELLENT.value
                ]
            ),
            "very_good": len(
                [
                    s
                    for s in current_scores
                    if OpportunityScore.VERY_GOOD.value
                    <= s.total_score
                    < OpportunityScore.EXCELLENT.value
                ]
            ),
            "good": len(
                [
                    s
                    for s in current_scores
                    if OpportunityScore.GOOD.value
                    <= s.total_score
                    < OpportunityScore.VERY_GOOD.value
                ]
            ),
            "fair": len(
                [
                    s
                    for s in current_scores
                    if OpportunityScore.FAIR.value
                    <= s.total_score
                    < OpportunityScore.GOOD.value
                ]
            ),
            "poor": len(
                [
                    s
                    for s in current_scores
                    if s.total_score < OpportunityScore.FAIR.value
                ]
            ),
        }

        # Sub-score averages
        sub_score_averages = {
            "profit": np.mean([s.profit_score for s in current_scores]),
            "confidence": np.mean([s.confidence_score for s in current_scores]),
            "liquidity": np.mean([s.liquidity_score for s in current_scores]),
            "volatility": np.mean([s.volatility_score for s in current_scores]),
            "speed": np.mean([s.speed_score for s in current_scores]),
            "risk": np.mean([s.risk_score for s in current_scores]),
        }

        return {
            "total_scored": total_scored,
            "avg_score": avg_score,
            "score_distribution": score_distribution,
            "sub_score_averages": sub_score_averages,
            "weights": {
                "profit": self.weights.profit_weight,
                "confidence": self.weights.confidence_weight,
                "liquidity": self.weights.liquidity_weight,
                "volatility": self.weights.volatility_weight,
                "speed": self.weights.speed_weight,
                "risk": self.weights.risk_weight,
            },
        }

    async def cleanup_old_scores(self):
        """Clean up old opportunity scores."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=10)

        old_scores = [
            opp_id
            for opp_id, score in self.scored_opportunities.items()
            if score.timestamp < cutoff_time
        ]

        for opp_id in old_scores:
            del self.scored_opportunities[opp_id]

        logger.info(f"Cleaned up {len(old_scores)} old opportunity scores")

    async def update_weights(self, new_weights: Dict[str, float]):
        """Update scoring weights dynamically."""
        # Validate weights sum to 1.0
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight:.3f}, normalizing to 1.0")
            new_weights = {k: v / total_weight for k, v in new_weights.items()}

        # Update weights
        if "profit" in new_weights:
            self.weights.profit_weight = new_weights["profit"]
        if "confidence" in new_weights:
            self.weights.confidence_weight = new_weights["confidence"]
        if "liquidity" in new_weights:
            self.weights.liquidity_weight = new_weights["liquidity"]
        if "volatility" in new_weights:
            self.weights.volatility_weight = new_weights["volatility"]
        if "speed" in new_weights:
            self.weights.speed_weight = new_weights["speed"]
        if "risk" in new_weights:
            self.weights.risk_weight = new_weights["risk"]

        logger.info(f"Updated scoring weights: {self.weights}")

        # Cache updated weights
        await self.cache_manager.cache.set_orderbook(
            "scoring_weights",
            {
                "profit_weight": self.weights.profit_weight,
                "confidence_weight": self.weights.confidence_weight,
                "liquidity_weight": self.weights.liquidity_weight,
                "volatility_weight": self.weights.volatility_weight,
                "speed_weight": self.weights.speed_weight,
                "risk_weight": self.weights.risk_weight,
            },
        )


class ScoringService:
    """Service for managing opportunity scoring across the system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize scoring service."""
        self.config = config or {}
        self.scorer = OpportunityScorer(config)
        self.background_scoring = True
        self.scoring_task = None

    async def initialize(self):
        """Initialize scoring service."""
        logger.info("Initializing scoring service...")

        # Start background scoring loop
        if self.background_scoring:
            self.scoring_task = asyncio.create_task(self._scoring_loop())

        logger.info("Scoring service initialized")

    async def shutdown(self):
        """Shutdown scoring service."""
        logger.info("Shutting down scoring service...")

        # Cancel background task
        if self.scoring_task:
            self.scoring_task.cancel()
            try:
                await self.scoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Scoring service shutdown complete")

    async def _scoring_loop(self):
        """Background loop for continuous scoring updates."""
        while True:
            try:
                await asyncio.sleep(30)  # 30 seconds

                # Clean up old scores
                await self.scorer.cleanup_old_scores()

                # Update scoring weights based on performance
                await self._update_weights_based_on_performance()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scoring loop error: {e}")

    async def _update_weights_based_on_performance(self):
        """Update scoring weights based on historical performance."""
        # This would analyze past performance and adjust weights
        # For now, keep current weights
        pass

    async def score_and_rank_opportunities(
        self,
        opportunities: List[
            Union[ArbitrageOpportunity, StatisticalArbitrageOpportunity]
        ],
    ) -> List[RealTimeScore]:
        """Score and rank opportunities."""
        scored = await self.scorer.score_batch(opportunities)

        # Log top opportunities
        top_5 = scored[:5]
        logger.info(f"Top 5 opportunities:")
        for i, score in enumerate(top_5, 1):
            logger.info(
                f"  {i}. {score.opportunity_id}: {score.total_score:.2f} (priority: {score.execution_priority})"
            )

        return scored

    def get_scorer(self) -> OpportunityScorer:
        """Get the underlying scorer instance."""
        return self.scorer


# Global scoring service instance
_scoring_service: Optional[ScoringService] = None


def get_scoring_service(config: Optional[Dict[str, Any]] = None) -> ScoringService:
    """Get global scoring service instance."""
    global _scoring_service

    if _scoring_service is None:
        _scoring_service = ScoringService(config)

    return _scoring_service


async def initialize_scoring(config: Optional[Dict[str, Any]] = None) -> ScoringService:
    """Initialize and return scoring service."""
    service = get_scoring_service(config)
    await service.initialize()
    return service


async def shutdown_scoring():
    """Shutdown global scoring service."""
    global _scoring_service

    if _scoring_service is not None:
        await _scoring_service.shutdown()
        _scoring_service = None
