"""
Dynamic Position Sizing with Risk Management Integration
Optimized for high-frequency trading with real-time adjustments
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import deque
import redis.asyncio as redis

from .real_time_risk import RealTimeRiskManager, RiskLevel, RiskType
from ..utils.performance_cache import PerformanceCache

logger = logging.getLogger(__name__)


class PositionSizingStrategy(Enum):
    """Position sizing strategies"""

    FIXED = "fixed"
    VOLATILITY_BASED = "volatility_based"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"
    ADAPTIVE = "adaptive"
    PERCENTAGE_RISK = "percentage_risk"


class SizingFrequency(Enum):
    """Position sizing adjustment frequency"""

    PER_TRADE = "per_trade"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"


@dataclass
class PositionSizeConfig:
    """Configuration for position sizing"""

    strategy: PositionSizingStrategy = PositionSizingStrategy.PERCENTAGE_RISK
    frequency: SizingFrequency = SizingFrequency.PER_TRADE
    base_size: float = 1000.0
    max_size: float = 10000.0
    min_size: float = 100.0

    # Risk parameters
    max_risk_per_trade: float = 0.02  # 2% of portfolio
    max_portfolio_risk: float = 0.1  # 10% of portfolio
    volatility_window: int = 20  # Periods for volatility calc
    correlation_threshold: float = 0.7

    # Kelly criterion parameters
    kelly_fraction: float = 0.25  # Conservative Kelly
    min_win_rate: float = 0.55  # Minimum win rate for Kelly
    confidence_threshold: float = 0.6

    # Adaptive parameters
    adjustment_factor: float = 0.1  # How fast to adjust sizes
    performance_window: int = 50  # Trades to consider for performance

    # Risk parity parameters
    risk_contribution_target: float = 1.0 / 10  # Equal risk across 10 positions


@dataclass
class SizingDecision:
    """Position sizing decision with metadata"""

    symbol: str
    recommended_size: float
    adjusted_size: float
    strategy_used: PositionSizingStrategy
    risk_score: float
    confidence: float
    factors: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Performance metrics for position sizing"""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    avg_holding_time: float = 0.0


class DynamicPositionSizer:
    """
    Advanced dynamic position sizing with real-time risk integration
    and multiple sizing strategies
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RealTimeRiskManager):
        self.config = config
        self.risk_manager = risk_manager
        self.sizing_config = PositionSizingConfig(**config.get("position_sizing", {}))

        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.current_metrics = PerformanceMetrics()

        # Position sizing cache
        self.cache = PerformanceCache(
            redis_url=config.get("redis_url", "redis://localhost:6379"),
            default_ttl=60,  # 1 minute TTL for sizing data
        )

        # Symbol-specific data
        self.symbol_performance: Dict[str, PerformanceMetrics] = {}
        self.volatility_cache: Dict[str, float] = {}
        self.correlation_cache: Dict[str, Dict[str, float]] = {}

        # Sizing history
        self.sizing_history: deque = deque(maxlen=5000)
        self.last_adjustment = {}

        # Portfolio tracking
        self.portfolio_value: float = 0.0
        self.available_capital: float = 0.0

        logger.info(
            f"Dynamic Position Sizer initialized with {self.sizing_config.strategy.value} strategy"
        )

    async def initialize(self) -> None:
        """Initialize position sizer with existing data"""
        try:
            # Load performance history from cache
            await self._load_performance_data()

            # Initialize portfolio value
            await self._update_portfolio_value()

            logger.info("Dynamic Position Sizer initialized successfully")

        except Exception as e:
            logger.error(f"Position Sizer initialization failed: {e}")
            raise

    async def calculate_position_size(
        self,
        symbol: str,
        opportunity_confidence: float,
        expected_return: float = 0.0,
        volatility: Optional[float] = None,
        portfolio_value_override: Optional[float] = None,
    ) -> SizingDecision:
        """
        Calculate optimal position size based on strategy and risk parameters

        Args:
            symbol: Trading symbol
            opportunity_confidence: Confidence level (0-1)
            expected_return: Expected return for the opportunity
            volatility: Current volatility (optional)
            portfolio_value_override: Override portfolio value (optional)

        Returns:
            SizingDecision with recommended size and metadata
        """
        try:
            # Update portfolio value if override provided
            if portfolio_value_override:
                self.portfolio_value = portfolio_value_override
            else:
                await self._update_portfolio_value()

            # Get symbol performance metrics
            symbol_metrics = await self._get_symbol_metrics(symbol)

            # Calculate position size based on strategy
            if self.sizing_config.strategy == PositionSizingStrategy.FIXED:
                size = await self._calculate_fixed_size(symbol)
            elif self.sizing_config.strategy == PositionSizingStrategy.VOLATILITY_BASED:
                size = await self._calculate_volatility_based_size(symbol, volatility)
            elif self.sizing_config.strategy == PositionSizingStrategy.KELLY_CRITERION:
                size = await self._calculate_kelly_size(
                    symbol, opportunity_confidence, expected_return
                )
            elif self.sizing_config.strategy == PositionSizingStrategy.RISK_PARITY:
                size = await self._calculate_risk_parity_size(symbol)
            elif self.sizing_config.strategy == PositionSizingStrategy.ADAPTIVE:
                size = await self._calculate_adaptive_size(
                    symbol, opportunity_confidence
                )
            else:  # PERCENTAGE_RISK
                size = await self._calculate_percentage_risk_size(symbol)

            # Apply risk manager adjustments
            risk_adjusted_size = await self._apply_risk_adjustments(symbol, size)

            # Apply constraints
            final_size = self._apply_constraints(risk_adjusted_size)

            # Calculate risk score and factors
            risk_score, factors = await self._calculate_size_risk_score(
                symbol, final_size
            )

            # Generate warnings
            warnings = await self._generate_warnings(symbol, final_size, risk_score)

            # Create sizing decision
            decision = SizingDecision(
                symbol=symbol,
                recommended_size=size,
                adjusted_size=risk_adjusted_size,
                strategy_used=self.sizing_config.strategy,
                risk_score=risk_score,
                confidence=opportunity_confidence,
                factors=factors,
                warnings=warnings,
            )

            # Store decision
            await self._store_sizing_decision(decision)

            return decision

        except Exception as e:
            logger.error(f"Position size calculation failed for {symbol}: {e}")
            # Return conservative default
            return SizingDecision(
                symbol=symbol,
                recommended_size=self.sizing_config.min_size,
                adjusted_size=self.sizing_config.min_size,
                strategy_used=self.sizing_config.strategy,
                risk_score=1.0,
                confidence=0.0,
                warnings=[f"Calculation error: {str(e)}"],
            )

    async def _calculate_fixed_size(self, symbol: str) -> float:
        """Calculate fixed position size"""
        # Check symbol-specific adjustments
        symbol_adjustment = 1.0
        symbol_metrics = self.symbol_performance.get(symbol)

        if symbol_metrics and symbol_metrics.total_trades > 10:
            # Adjust based on symbol performance
            if symbol_metrics.hit_rate > 0.6:
                symbol_adjustment = 1.2  # Increase size for good performers
            elif symbol_metrics.hit_rate < 0.4:
                symbol_adjustment = 0.8  # Decrease size for poor performers

        size = self.sizing_config.base_size * symbol_adjustment
        return min(size, self.sizing_config.max_size)

    async def _calculate_volatility_based_size(
        self, symbol: str, volatility: Optional[float] = None
    ) -> float:
        """Calculate volatility-adjusted position size"""
        if volatility is None:
            volatility = await self._get_symbol_volatility(symbol)

        # Base size adjusted by inverse volatility
        if volatility > 0:
            volatility_factor = min(2.0, 0.15 / volatility)  # Cap at 2x
        else:
            volatility_factor = 1.0

        size = self.sizing_config.base_size * volatility_factor

        # Additional adjustment for confidence
        confidence_adjustment = 1.0
        symbol_metrics = self.symbol_performance.get(symbol)
        if symbol_metrics:
            if symbol_metrics.hit_rate > 0.6:
                confidence_adjustment = 1.1
            elif symbol_metrics.hit_rate < 0.4:
                confidence_adjustment = 0.9

        return size * confidence_adjustment

    async def _calculate_kelly_size(
        self, symbol: str, confidence: float, expected_return: float
    ) -> float:
        """Calculate Kelly criterion position size"""
        symbol_metrics = self.symbol_performance.get(symbol)

        if not symbol_metrics or symbol_metrics.total_trades < 20:
            # Not enough data for Kelly, use percentage risk
            return await self._calculate_percentage_risk_size(symbol)

        # Calculate win rate and average win/loss
        win_rate = symbol_metrics.hit_rate

        # Skip Kelly if win rate is too low
        if win_rate < self.sizing_config.min_win_rate:
            return await self._calculate_percentage_risk_size(symbol)

        # Calculate average win and loss
        if symbol_metrics.avg_loss > 0:
            win_loss_ratio = symbol_metrics.avg_win / symbol_metrics.avg_loss
        else:
            win_loss_ratio = 1.0

        # Calculate Kelly fraction
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        # Apply conservative Kelly fraction
        kelly_size = kelly_fraction * self.sizing_config.kelly_fraction

        # Convert to dollar amount
        base_kelly = self.portfolio_value * kelly_size

        # Apply confidence adjustment
        confidence_factor = max(0.5, confidence)
        final_size = base_kelly * confidence_factor

        # Cap at maximum size
        return min(final_size, self.sizing_config.max_size)

    async def _calculate_risk_parity_size(self, symbol: str) -> float:
        """Calculate risk parity position size"""
        # Get current positions and their volatilities
        active_symbols = list(self.risk_manager.active_positions.keys())

        if not active_symbols or len(active_symbols) < 2:
            # Fall back to percentage risk if not enough positions
            return await self._calculate_percentage_risk_size(symbol)

        # Calculate volatility for all active symbols
        volatilities = {}
        total_inverse_vol = 0.0

        for sym in active_symbols + [symbol]:
            vol = await self._get_symbol_volatility(sym)
            if vol > 0:
                volatilities[sym] = vol
                total_inverse_vol += 1.0 / vol

        if symbol not in volatilities or total_inverse_vol == 0:
            return await self._calculate_percentage_risk_size(symbol)

        # Risk parity target weight
        target_weight = (1.0 / volatilities[symbol]) / total_inverse_vol

        # Convert to dollar amount
        risk_parity_size = self.portfolio_value * target_weight

        # Apply safety factor
        safety_factor = 0.5  # Conservative risk parity
        final_size = risk_parity_size * safety_factor

        return min(final_size, self.sizing_config.max_size)

    async def _calculate_adaptive_size(self, symbol: str, confidence: float) -> float:
        """Calculate adaptive position size based on recent performance"""
        symbol_metrics = self.symbol_performance.get(symbol)

        if not symbol_metrics or symbol_metrics.total_trades < 10:
            return await self._calculate_percentage_risk_size(symbol)

        # Calculate performance score
        performance_score = 0.0

        # Hit rate component (40% weight)
        performance_score += (
            symbol_metrics.hit_rate - 0.5
        ) * 0.8  # Scale to [-0.4, 0.4]

        # Sharpe ratio component (30% weight)
        if symbol_metrics.sharpe_ratio > 0:
            performance_score += min(0.3, symbol_metrics.sharpe_ratio * 0.1)

        # Recent performance component (30% weight)
        recent_trades = list(self.performance_history)[-20:]  # Last 20 trades
        symbol_recent_trades = [t for t in recent_trades if t["symbol"] == symbol]

        if symbol_recent_trades:
            recent_hit_rate = sum(
                1 for t in symbol_recent_trades if t["pnl"] > 0
            ) / len(symbol_recent_trades)
            performance_score += (recent_hit_rate - 0.5) * 0.6  # Scale to [-0.3, 0.3]

        # Convert performance score to size multiplier
        base_multiplier = 1.0 + performance_score
        multiplier = max(0.5, min(2.0, base_multiplier))  # Cap between 0.5x and 2.0x

        # Calculate base size
        base_size = await self._calculate_percentage_risk_size(symbol)

        # Apply adaptive multiplier
        adaptive_size = base_size * multiplier

        # Apply confidence adjustment
        confidence_factor = max(0.7, confidence)
        final_size = adaptive_size * confidence_factor

        return min(final_size, self.sizing_config.max_size)

    async def _calculate_percentage_risk_size(self, symbol: str) -> float:
        """Calculate position size based on percentage risk"""
        # Get symbol volatility for stop-loss calculation
        volatility = await self._get_symbol_volatility(symbol)

        # Estimate stop-loss distance (2 standard deviations)
        stop_loss_distance = volatility * 2

        if stop_loss_distance == 0:
            stop_loss_distance = 0.1  # Default 10% stop loss

        # Calculate position size based on max risk per trade
        risk_amount = self.portfolio_value * self.sizing_config.max_risk_per_trade
        position_size = risk_amount / stop_loss_distance

        return min(position_size, self.sizing_config.max_size)

    async def _apply_risk_adjustments(self, symbol: str, size: float) -> float:
        """Apply risk manager adjustments to position size"""
        try:
            # Check with risk manager
            is_allowed, alerts = await self.risk_manager.check_position_risk(
                symbol,
                size,
                1.0,
                self.portfolio_value,  # Assuming $1 price for calculation
            )

            if not is_allowed:
                # Find the most restrictive alert
                restrictive_alert = None
                for alert in alerts:
                    if alert.risk_type in [
                        RiskType.POSITION_SIZE,
                        RiskType.PORTFOLIO_EXPOSURE,
                    ]:
                        if (
                            restrictive_alert is None
                            or alert.current_value > restrictive_alert.current_value
                        ):
                            restrictive_alert = alert

                if restrictive_alert:
                    # Adjust size to meet the limit
                    adjusted_size = restrictive_alert.threshold * 0.9  # 90% of limit
                    return adjusted_size

            return size

        except Exception as e:
            logger.error(f"Risk adjustments failed for {symbol}: {e}")
            return size * 0.5  # Conservative fallback

    def _apply_constraints(self, size: float) -> float:
        """Apply min/max constraints"""
        return max(self.sizing_config.min_size, min(size, self.sizing_config.max_size))

    async def _calculate_size_risk_score(
        self, symbol: str, size: float
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate risk score for position size"""
        factors = {}

        # Size risk
        size_factor = min(1.0, size / self.sizing_config.max_size)
        factors["size_risk"] = size_factor

        # Portfolio concentration risk
        portfolio_exposure = await self.risk_manager._calculate_portfolio_exposure()
        new_exposure = portfolio_exposure + size
        concentration_factor = min(1.0, new_exposure / (self.portfolio_value * 0.3))
        factors["concentration_risk"] = concentration_factor

        # Volatility risk
        volatility = await self._get_symbol_volatility(symbol)
        volatility_factor = min(1.0, volatility / 0.2)  # 20% volatility = max risk
        factors["volatility_risk"] = volatility_factor

        # Performance risk
        symbol_metrics = self.symbol_performance.get(symbol)
        if symbol_metrics and symbol_metrics.total_trades > 10:
            performance_factor = max(0, 1.0 - symbol_metrics.hit_rate)
        else:
            performance_factor = 0.5  # Unknown performance
        factors["performance_risk"] = performance_factor

        # Calculate overall risk score (weighted average)
        weights = {
            "size_risk": 0.3,
            "concentration_risk": 0.25,
            "volatility_risk": 0.25,
            "performance_risk": 0.2,
        }

        overall_score = sum(factors[k] * weights[k] for k in factors)

        return overall_score, factors

    async def _generate_warnings(
        self, symbol: str, size: float, risk_score: float
    ) -> List[str]:
        """Generate warnings for position sizing"""
        warnings = []

        # Size warnings
        if size >= self.sizing_config.max_size * 0.9:
            warnings.append("Position size near maximum limit")

        # Risk warnings
        if risk_score > 0.7:
            warnings.append("High risk score detected")
        elif risk_score > 0.5:
            warnings.append("Moderate risk score")

        # Volatility warnings
        volatility = await self._get_symbol_volatility(symbol)
        if volatility > 0.15:
            warnings.append("High volatility detected")

        # Performance warnings
        symbol_metrics = self.symbol_performance.get(symbol)
        if (
            symbol_metrics
            and symbol_metrics.hit_rate < 0.4
            and symbol_metrics.total_trades > 10
        ):
            warnings.append("Poor historical performance for symbol")

        # Portfolio warnings
        portfolio_exposure = await self.risk_manager._calculate_portfolio_exposure()
        if portfolio_exposure > self.portfolio_value * 0.8:
            warnings.append("High portfolio exposure")

        return warnings

    async def _get_symbol_volatility(self, symbol: str) -> float:
        """Get symbol volatility from cache or calculate"""
        try:
            cache_key = f"volatility:{symbol}"
            cached_vol = await self.cache.get(cache_key)

            if cached_vol:
                return float(cached_vol)

            # Try to get from position history
            if symbol in self.risk_manager.position_history:
                history = list(self.risk_manager.position_history[symbol])
                if len(history) >= 20:
                    prices = [p["price"] for p in history]
                    returns = np.diff(np.log(prices))
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized

                    # Cache result
                    await self.cache.set(cache_key, str(volatility), ttl=300)
                    return volatility

            # Default volatility
            return 0.1  # 10% default volatility

        except Exception as e:
            logger.error(f"Volatility calculation failed for {symbol}: {e}")
            return 0.1

    async def _get_symbol_metrics(self, symbol: str) -> PerformanceMetrics:
        """Get or create symbol performance metrics"""
        if symbol not in self.symbol_performance:
            self.symbol_performance[symbol] = PerformanceMetrics()

        return self.symbol_performance[symbol]

    async def _update_portfolio_value(self) -> None:
        """Update portfolio value from risk manager"""
        try:
            metrics = await self.risk_manager.get_risk_metrics()
            self.portfolio_value = metrics.get("portfolio_risk", {}).get(
                "total_value", 10000.0
            )
            self.available_capital = self.portfolio_value * 0.8  # Keep 20% cash
        except Exception as e:
            logger.error(f"Portfolio value update failed: {e}")

    async def _store_sizing_decision(self, decision: SizingDecision) -> None:
        """Store sizing decision in history"""
        try:
            self.sizing_history.append(
                {
                    "timestamp": decision.timestamp,
                    "symbol": decision.symbol,
                    "recommended_size": decision.recommended_size,
                    "adjusted_size": decision.adjusted_size,
                    "strategy": decision.strategy_used.value,
                    "risk_score": decision.risk_score,
                    "confidence": decision.confidence,
                }
            )

            # Cache recent decisions
            await self.cache.set(
                f"sizing_decision:{decision.symbol}:{decision.timestamp.isoformat()}",
                {
                    "symbol": decision.symbol,
                    "size": decision.adjusted_size,
                    "strategy": decision.strategy_used.value,
                    "risk_score": decision.risk_score,
                },
                ttl=3600,
            )

        except Exception as e:
            logger.error(f"Sizing decision storage failed: {e}")

    async def _load_performance_data(self) -> None:
        """Load performance data from cache"""
        try:
            # Load overall performance
            perf_data = await self.cache.get("position_sizing_performance")
            if perf_data:
                data = perf_data
                self.current_metrics = PerformanceMetrics(**data)

            # Load symbol-specific performance
            for symbol in self.symbol_performance.keys():
                symbol_key = f"symbol_performance:{symbol}"
                symbol_data = await self.cache.get(symbol_key)
                if symbol_data:
                    self.symbol_performance[symbol] = PerformanceMetrics(**symbol_data)

            logger.info("Performance data loaded from cache")

        except Exception as e:
            logger.error(f"Performance data loading failed: {e}")

    async def update_trade_result(
        self, symbol: str, pnl: float, holding_time: timedelta, confidence: float
    ) -> None:
        """Update performance metrics after trade completion"""
        try:
            # Update overall metrics
            self.current_metrics.total_trades += 1
            self.current_metrics.total_pnl += pnl
            self.current_metrics.avg_holding_time = (
                self.current_metrics.avg_holding_time
                * (self.current_metrics.total_trades - 1)
                + holding_time.total_seconds()
            ) / self.current_metrics.total_trades

            if pnl > 0:
                self.current_metrics.winning_trades += 1
                self.current_metrics.avg_win = (
                    self.current_metrics.avg_win
                    * (self.current_metrics.winning_trades - 1)
                    + pnl
                ) / self.current_metrics.winning_trades
            else:
                self.current_metrics.losing_trades += 1
                self.current_metrics.avg_loss = (
                    self.current_metrics.avg_loss
                    * (self.current_metrics.losing_trades - 1)
                    + abs(pnl)
                ) / self.current_metrics.losing_trades

            self.current_metrics.hit_rate = (
                self.current_metrics.winning_trades / self.current_metrics.total_trades
            )

            # Update symbol-specific metrics
            symbol_metrics = await self._get_symbol_metrics(symbol)
            symbol_metrics.total_trades += 1
            symbol_metrics.total_pnl += pnl

            if pnl > 0:
                symbol_metrics.winning_trades += 1
                symbol_metrics.avg_win = (
                    symbol_metrics.avg_win * (symbol_metrics.winning_trades - 1) + pnl
                ) / symbol_metrics.winning_trades
            else:
                symbol_metrics.losing_trades += 1
                symbol_metrics.avg_loss = (
                    symbol_metrics.avg_loss * (symbol_metrics.losing_trades - 1)
                    + abs(pnl)
                ) / symbol_metrics.losing_trades

            symbol_metrics.hit_rate = (
                symbol_metrics.winning_trades / symbol_metrics.total_trades
            )

            # Add to performance history
            self.performance_history.append(
                {
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "pnl": pnl,
                    "holding_time": holding_time.total_seconds(),
                    "confidence": confidence,
                }
            )

            # Calculate Sharpe ratio
            if len(self.performance_history) >= 20:
                pnl_series = [p["pnl"] for p in list(self.performance_history)]
                if np.std(pnl_series) > 0:
                    self.current_metrics.sharpe_ratio = np.mean(pnl_series) / np.std(
                        pnl_series
                    )

            # Save updated metrics
            await self._save_performance_data()

        except Exception as e:
            logger.error(f"Trade result update failed: {e}")

    async def _save_performance_data(self) -> None:
        """Save performance metrics to cache"""
        try:
            # Save overall metrics
            await self.cache.set(
                "position_sizing_performance",
                {
                    "total_trades": self.current_metrics.total_trades,
                    "winning_trades": self.current_metrics.winning_trades,
                    "losing_trades": self.current_metrics.losing_trades,
                    "total_pnl": self.current_metrics.total_pnl,
                    "avg_win": self.current_metrics.avg_win,
                    "avg_loss": self.current_metrics.avg_loss,
                    "sharpe_ratio": self.current_metrics.sharpe_ratio,
                    "max_drawdown": self.current_metrics.max_drawdown,
                    "hit_rate": self.current_metrics.hit_rate,
                    "avg_holding_time": self.current_metrics.avg_holding_time,
                },
                ttl=86400,  # 24 hours
            )

            # Save symbol-specific metrics
            for symbol, metrics in self.symbol_performance.items():
                await self.cache.set(
                    f"symbol_performance:{symbol}",
                    {
                        "total_trades": metrics.total_trades,
                        "winning_trades": metrics.winning_trades,
                        "losing_trades": metrics.losing_trades,
                        "total_pnl": metrics.total_pnl,
                        "avg_win": metrics.avg_win,
                        "avg_loss": metrics.avg_loss,
                        "sharpe_ratio": metrics.sharpe_ratio,
                        "max_drawdown": metrics.max_drawdown,
                        "hit_rate": metrics.hit_rate,
                        "avg_holding_time": metrics.avg_holding_time,
                    },
                    ttl=86400,
                )

        except Exception as e:
            logger.error(f"Performance data saving failed: {e}")

    async def get_sizing_report(self) -> Dict[str, Any]:
        """Get comprehensive position sizing report"""
        try:
            recent_decisions = list(self.sizing_history)[-50:]  # Last 50 decisions

            return {
                "current_config": {
                    "strategy": self.sizing_config.strategy.value,
                    "frequency": self.sizing_config.frequency.value,
                    "base_size": self.sizing_config.base_size,
                    "max_size": self.sizing_config.max_size,
                    "min_size": self.sizing_config.min_size,
                },
                "performance_metrics": {
                    "total_trades": self.current_metrics.total_trades,
                    "hit_rate": self.current_metrics.hit_rate,
                    "avg_win": self.current_metrics.avg_win,
                    "avg_loss": self.current_metrics.avg_loss,
                    "sharpe_ratio": self.current_metrics.sharpe_ratio,
                    "total_pnl": self.current_metrics.total_pnl,
                },
                "portfolio_info": {
                    "total_value": self.portfolio_value,
                    "available_capital": self.available_capital,
                },
                "symbol_performance": {
                    symbol: {
                        "trades": metrics.total_trades,
                        "hit_rate": metrics.hit_rate,
                        "avg_win": metrics.avg_win,
                        "avg_loss": metrics.avg_loss,
                    }
                    for symbol, metrics in self.symbol_performance.items()
                    if metrics.total_trades > 0
                },
                "recent_decisions": recent_decisions[-10:],  # Last 10 decisions
                "risk_adjustments": {
                    "total_adjustments": len(
                        [
                            d
                            for d in recent_decisions
                            if d["adjusted_size"] < d["recommended_size"]
                        ]
                    ),
                    "avg_risk_score": np.mean(
                        [d["risk_score"] for d in recent_decisions]
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Sizing report generation failed: {e}")
            return {}

    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            await self._save_performance_data()
            await self.cache.close()
            logger.info("Dynamic Position Sizer cleaned up")
        except Exception as e:
            logger.error(f"Position Sizer cleanup failed: {e}")


# Utility functions
async def create_position_sizer(
    config: Dict[str, Any], risk_manager: RealTimeRiskManager
) -> DynamicPositionSizer:
    """Create and initialize position sizer"""
    sizer = DynamicPositionSizer(config, risk_manager)
    await sizer.initialize()
    return sizer


def calculate_optimal_size(
    portfolio_value: float, risk_per_trade: float, stop_loss_pct: float
) -> float:
    """Calculate optimal position size based on risk parameters"""
    risk_amount = portfolio_value * risk_per_trade
    return risk_amount / stop_loss_pct


def calculate_position_heatmap(sizing_decisions: List[SizingDecision]) -> str:
    """Generate position sizing heatmap"""
    if not sizing_decisions:
        return "No position sizing data available"

    avg_risk_score = np.mean([d.risk_score for d in sizing_decisions])

    if avg_risk_score < 0.3:
        return "🟢 Low Risk Sizing"
    elif avg_risk_score < 0.5:
        return "🟡 Moderate Risk Sizing"
    elif avg_risk_score < 0.7:
        return "🟠 High Risk Sizing"
    else:
        return "🔴 Critical Risk Sizing"
