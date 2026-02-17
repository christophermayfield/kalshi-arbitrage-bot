"""
Auto-Tuning Engine for Dynamic Profit Thresholds
Adaptive parameter optimization based on market conditions and performance feedback
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json
import asyncio

from src.utils.logging_utils import get_logger
from src.utils.config import Config
from src.utils.performance_cache import PerformanceCache

logger = get_logger("auto_tuning")


class MarketRegime(Enum):
    """Market condition regimes"""

    QUIET = "quiet"  # Low volatility, tight spreads
    NORMAL = "normal"  # Normal market conditions
    VOLATILE = "volatile"  # High volatility, wide spreads
    TRENDING = "trending"  # Strong directional movement
    MEAN_REVERTING = "mean_reverting"  # Oscillating around mean


class OptimizationObjective(Enum):
    """Optimization objectives"""

    MAXIMIZE_PROFIT = "maximize_profit"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MAXIMIZE_WIN_RATE = "maximize_win_rate"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    BALANCED = "balanced"


@dataclass
class TuningParameters:
    """Configurable parameters for the arbitrage bot"""

    # Profit thresholds
    min_profit_cents: int = 10
    min_profit_percent: float = 0.5
    confidence_threshold: float = 0.7

    # Risk parameters
    max_position_size: int = 1000
    max_positions_per_hour: int = 10
    risk_tolerance: float = 0.5

    # Execution parameters
    execution_timeout_seconds: int = 30
    max_slippage_percent: float = 2.0
    liquidity_requirement: float = 50.0

    # Market-specific parameters
    spread_multiplier: float = 1.0
    volatility_adjustment: bool = True
    time_decay_factor: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "min_profit_cents": self.min_profit_cents,
            "min_profit_percent": self.min_profit_percent,
            "confidence_threshold": self.confidence_threshold,
            "max_position_size": self.max_position_size,
            "max_positions_per_hour": self.max_positions_per_hour,
            "risk_tolerance": self.risk_tolerance,
            "execution_timeout_seconds": self.execution_timeout_seconds,
            "max_slippage_percent": self.max_slippage_percent,
            "liquidity_requirement": self.liquidity_requirement,
            "spread_multiplier": self.spread_multiplier,
            "volatility_adjustment": self.volatility_adjustment,
            "time_decay_factor": self.time_decay_factor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TuningParameters":
        """Create from dictionary"""
        return cls(**data)


@dataclass
class PerformanceMetrics:
    """Performance metrics for evaluating parameter effectiveness"""

    # Profit metrics
    total_profit_cents: float = 0.0
    avg_profit_per_trade: float = 0.0
    profit_volatility: float = 0.0

    # Risk metrics
    max_drawdown_cents: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Execution metrics
    win_rate: float = 0.0
    avg_execution_time_ms: float = 0.0
    slippage_avg_percent: float = 0.0

    # Opportunity metrics
    opportunities_per_hour: float = 0.0
    success_rate: float = 0.0
    false_positive_rate: float = 0.0

    # Market metrics
    avg_spread_cents: float = 0.0
    avg_liquidity_score: float = 0.0
    market_volatility: float = 0.0

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


class AutoTuningEngine:
    """
    Advanced auto-tuning engine that dynamically adjusts trading parameters
    based on market conditions and performance feedback
    """

    def __init__(self, config: Config):
        self.config = config
        self.tuning_config = config.get("auto_tuning", {})

        # Engine configuration
        self.enabled = self.tuning_config.get("enabled", False)
        self.optimization_objective = OptimizationObjective(
            self.tuning_config.get("objective", "balanced")
        )
        self.retrain_interval_hours = self.tuning_config.get(
            "retrain_interval_hours", 4
        )
        self.min_data_points = self.tuning_config.get("min_data_points", 100)
        self.ensemble_models = self.tuning_config.get("ensemble_models", True)

        # Current parameters and performance tracking
        self.current_parameters = TuningParameters(
            min_profit_cents=config.get("trading.min_profit_cents", 10),
            min_profit_percent=config.get("trading.min_profit_percent", 0.5),
            confidence_threshold=config.get("trading.confidence_threshold", 0.7),
        )

        self.performance_history: List[PerformanceMetrics] = []
        self.parameter_history: List[Tuple[datetime, TuningParameters]] = []

        # Market regime detection
        self.current_regime = MarketRegime.NORMAL
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []

        # ML models for parameter optimization
        self.performance_predictor: Optional[RandomForestRegressor] = None
        self.regime_classifier: Optional[RandomForestRegressor] = None
        self.feature_scaler = StandardScaler()

        # Optimization state
        self.last_optimization_time = datetime.now()
        self.optimization_count = 0
        self.best_parameters = self.current_parameters
        self.best_performance = PerformanceMetrics()

        # Performance cache
        self.cache = PerformanceCache(
            redis_url=config.get("redis.url", "redis://localhost:6379"),
            default_ttl=3600,  # 1 hour
        )

        # Market data for regime detection
        self.market_features_history: List[Dict[str, float]] = []

        logger.info("Auto-Tuning Engine initialized")

    async def update_performance(self, metrics: PerformanceMetrics) -> None:
        """Update performance metrics and trigger optimization if needed"""
        try:
            self.performance_history.append(metrics)

            # Keep only recent history (last 7 days)
            cutoff_time = datetime.now() - timedelta(days=7)
            self.performance_history = [
                m for m in self.performance_history if m.timestamp > cutoff_time
            ]

            # Detect market regime
            await self._detect_market_regime()

            # Check if optimization is needed
            if await self._should_optimize():
                await self.optimize_parameters()

        except Exception as e:
            logger.error(f"Failed to update performance: {e}")

    async def _detect_market_regime(self) -> None:
        """Detect current market regime using market features"""
        try:
            if not self.performance_history:
                self.current_regime = MarketRegime.NORMAL
                return

            recent_metrics = self.performance_history[-20:]  # Last 20 periods

            # Calculate regime indicators
            avg_volatility = np.mean([m.market_volatility for m in recent_metrics])
            avg_spread = np.mean([m.avg_spread_cents for m in recent_metrics])
            profit_volatility = np.std([m.avg_profit_per_trade for m in recent_metrics])
            trend_strength = np.mean([m.profit_volatility for m in recent_metrics])

            # Determine regime
            if avg_volatility < 0.1 and avg_spread < 5:
                regime = MarketRegime.QUIET
            elif avg_volatility > 0.3 or avg_spread > 20:
                regime = MarketRegime.VOLATILE
            elif profit_volatility > 50:
                regime = MarketRegime.TRENDING
            elif trend_strength < 10:
                regime = MarketRegime.MEAN_REVERTING
            else:
                regime = MarketRegime.NORMAL

            # Update regime if changed
            if regime != self.current_regime:
                logger.info(
                    f"Market regime changed: {self.current_regime.value} -> {regime.value}"
                )
                self.current_regime = regime
                self.regime_history.append((datetime.now(), regime))

                # Trigger immediate optimization for regime change
                if self.enabled:
                    await self.optimize_parameters()

        except Exception as e:
            logger.error(f"Failed to detect market regime: {e}")

    async def _should_optimize(self) -> bool:
        """Determine if optimization should be triggered"""
        if not self.enabled:
            return False

        # Check if we have enough data
        if len(self.performance_history) < self.min_data_points:
            return False

        # Check time since last optimization
        time_since_last = datetime.now() - self.last_optimization_time
        if time_since_last < timedelta(hours=self.retrain_interval_hours):
            return False

        # Check performance degradation
        if len(self.performance_history) >= 50:
            recent_performance = np.mean(
                [m.total_profit_cents for m in self.performance_history[-20:]]
            )
            historical_performance = np.mean(
                [m.total_profit_cents for m in self.performance_history[:-20]]
            )

            if recent_performance < historical_performance * 0.8:  # 20% degradation
                logger.info("Performance degradation detected, triggering optimization")
                return True

        # Regular optimization schedule
        return time_since_last >= timedelta(hours=self.retrain_interval_hours)

    async def optimize_parameters(self) -> None:
        """Optimize trading parameters using machine learning"""
        try:
            logger.info("Starting parameter optimization...")

            # Prepare training data
            X, y = await self._prepare_training_data()
            if len(X) < 10:
                logger.warning("Insufficient data for optimization")
                return

            # Train performance predictor
            await self._train_performance_predictor(X, y)

            # Optimize for current regime
            optimized_params = await self._optimize_for_regime()

            if optimized_params:
                # Validate and apply new parameters
                if await self._validate_parameters(optimized_params):
                    self.parameter_history.append(
                        (datetime.now(), self.current_parameters)
                    )
                    self.current_parameters = optimized_params
                    self.optimization_count += 1
                    self.last_optimization_time = datetime.now()

                    logger.info(f"Optimization #{self.optimization_count} completed")
                    logger.info(f"New parameters: {optimized_params.to_dict()}")

                    # Cache optimized parameters
                    await self.cache.set(
                        f"optimized_params_{self.current_regime.value}",
                        optimized_params.to_dict(),
                        ttl=3600,
                    )
                else:
                    logger.warning("Optimized parameters failed validation")
            else:
                logger.warning("Parameter optimization failed")

        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")

    async def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models"""
        try:
            # Create features from historical performance
            features = []
            targets = []

            for i, metrics in enumerate(self.performance_history):
                if i == 0:
                    continue

                # Get previous parameters
                prev_params_idx = i - 1
                if prev_params_idx < len(self.parameter_history):
                    prev_params = self.parameter_history[prev_params_idx][1]

                    # Feature vector
                    feature = [
                        # Market conditions
                        metrics.market_volatility,
                        metrics.avg_spread_cents,
                        metrics.avg_liquidity_score,
                        metrics.opportunities_per_hour,
                        # Previous parameters
                        prev_params.min_profit_cents,
                        prev_params.min_profit_percent,
                        prev_params.confidence_threshold,
                        prev_params.max_position_size,
                        prev_params.risk_tolerance,
                        # Time features
                        metrics.timestamp.hour / 24.0,
                        metrics.timestamp.weekday() / 7.0,
                    ]

                    # Target based on optimization objective
                    if (
                        self.optimization_objective
                        == OptimizationObjective.MAXIMIZE_PROFIT
                    ):
                        target = metrics.total_profit_cents
                    elif (
                        self.optimization_objective
                        == OptimizationObjective.MAXIMIZE_SHARPE
                    ):
                        target = metrics.sharpe_ratio
                    elif (
                        self.optimization_objective
                        == OptimizationObjective.MAXIMIZE_WIN_RATE
                    ):
                        target = metrics.win_rate
                    elif (
                        self.optimization_objective
                        == OptimizationObjective.MINIMIZE_DRAWDOWN
                    ):
                        target = -metrics.max_drawdown_cents
                    else:  # BALANCED
                        # Composite score
                        target = (
                            metrics.sharpe_ratio * 0.4
                            + metrics.win_rate * 0.3
                            + metrics.total_profit_cents / 1000 * 0.2
                            - metrics.max_drawdown_cents / 1000 * 0.1
                        )

                    features.append(feature)
                    targets.append(target)

            return np.array(features), np.array(targets)

        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return np.array([]), np.array([])

    async def _train_performance_predictor(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train ML model to predict performance from parameters"""
        try:
            if len(X) < 10:
                return

            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)

            # Train random forest
            self.performance_predictor = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            self.performance_predictor.fit(X_scaled, y)

            logger.info("Performance predictor trained successfully")

        except Exception as e:
            logger.error(f"Failed to train performance predictor: {e}")

    async def _optimize_for_regime(self) -> Optional[TuningParameters]:
        """Optimize parameters for current market regime"""
        try:
            if not self.performance_predictor:
                return None

            # Get current market conditions
            if not self.performance_history:
                return None

            recent_metrics = self.performance_history[-1]

            def objective_function(params_array: np.ndarray) -> float:
                """Objective function for optimization"""
                try:
                    # Create parameters from array
                    params = TuningParameters(
                        min_profit_cents=int(params_array[0]),
                        min_profit_percent=params_array[1],
                        confidence_threshold=params_array[2],
                        max_position_size=int(params_array[3]),
                        max_positions_per_hour=int(params_array[4]),
                        risk_tolerance=params_array[5],
                        execution_timeout_seconds=int(params_array[6]),
                        max_slippage_percent=params_array[7],
                        liquidity_requirement=params_array[8],
                        spread_multiplier=params_array[9],
                        volatility_adjustment=bool(params_array[10] > 0.5),
                        time_decay_factor=params_array[11],
                    )

                    # Create feature vector
                    features = np.array(
                        [
                            recent_metrics.market_volatility,
                            recent_metrics.avg_spread_cents,
                            recent_metrics.avg_liquidity_score,
                            recent_metrics.opportunities_per_hour,
                            params.min_profit_cents,
                            params.min_profit_percent,
                            params.confidence_threshold,
                            params.max_position_size,
                            params.risk_tolerance,
                            datetime.now().hour / 24.0,
                            datetime.now().weekday() / 7.0,
                        ]
                    ).reshape(1, -1)

                    # Predict performance
                    features_scaled = self.feature_scaler.transform(features)
                    predicted_performance = self.performance_predictor.predict(
                        features_scaled
                    )[0]

                    # Return negative for minimization
                    return -predicted_performance

                except Exception:
                    return 1e6  # Large penalty for invalid parameters

            # Initial guess (current parameters)
            x0 = np.array(
                [
                    self.current_parameters.min_profit_cents,
                    self.current_parameters.min_profit_percent,
                    self.current_parameters.confidence_threshold,
                    self.current_parameters.max_position_size,
                    self.current_parameters.max_positions_per_hour,
                    self.current_parameters.risk_tolerance,
                    self.current_parameters.execution_timeout_seconds,
                    self.current_parameters.max_slippage_percent,
                    self.current_parameters.liquidity_requirement,
                    self.current_parameters.spread_multiplier,
                    1.0 if self.current_parameters.volatility_adjustment else 0.0,
                    self.current_parameters.time_decay_factor,
                ]
            )

            # Bounds for parameters
            bounds = [
                (5, 50),  # min_profit_cents
                (0.1, 2.0),  # min_profit_percent
                (0.5, 0.95),  # confidence_threshold
                (100, 5000),  # max_position_size
                (5, 50),  # max_positions_per_hour
                (0.1, 1.0),  # risk_tolerance
                (10, 120),  # execution_timeout_seconds
                (0.5, 5.0),  # max_slippage_percent
                (10, 200),  # liquidity_requirement
                (0.5, 2.0),  # spread_multiplier
                (0, 1),  # volatility_adjustment
                (0.8, 0.99),  # time_decay_factor
            ]

            # Optimize
            result = minimize(
                objective_function,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 100},
            )

            if result.success:
                optimized_params = TuningParameters(
                    min_profit_cents=int(result.x[0]),
                    min_profit_percent=result.x[1],
                    confidence_threshold=result.x[2],
                    max_position_size=int(result.x[3]),
                    max_positions_per_hour=int(result.x[4]),
                    risk_tolerance=result.x[5],
                    execution_timeout_seconds=int(result.x[6]),
                    max_slippage_percent=result.x[7],
                    liquidity_requirement=result.x[8],
                    spread_multiplier=result.x[9],
                    volatility_adjustment=bool(result.x[10] > 0.5),
                    time_decay_factor=result.x[11],
                )

                return optimized_params

            return None

        except Exception as e:
            logger.error(f"Failed to optimize for regime: {e}")
            return None

    async def _validate_parameters(self, params: TuningParameters) -> bool:
        """Validate optimized parameters"""
        try:
            # Check parameter ranges
            if params.min_profit_cents < 1 or params.min_profit_cents > 100:
                return False

            if params.confidence_threshold < 0.1 or params.confidence_threshold > 1.0:
                return False

            if params.max_position_size < 10 or params.max_position_size > 10000:
                return False

            # Check parameter consistency
            if params.min_profit_percent > params.max_slippage_percent * 2:
                return False

            # Simulate performance (simplified)
            # In production, you'd run backtesting here
            simulated_profit = await self._simulate_performance(params)

            # Check if improvement over current
            current_profit = await self._simulate_performance(self.current_parameters)

            improvement = (simulated_profit - current_profit) / max(
                1, abs(current_profit)
            )

            return improvement > -0.05  # Allow up to 5% degradation

        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            return False

    async def _simulate_performance(self, params: TuningParameters) -> float:
        """Simulate performance for given parameters"""
        try:
            if not self.performance_history:
                return 0.0

            # Use recent performance as baseline
            recent_metrics = self.performance_history[-10:]
            if not recent_metrics:
                return 0.0

            # Simple simulation based on parameter adjustments
            baseline_profit = np.mean([m.total_profit_cents for m in recent_metrics])

            # Apply parameter effects
            profit_factor = 1.0

            # Profit threshold effect
            profit_ratio = (
                params.min_profit_cents / self.current_parameters.min_profit_cents
            )
            profit_factor *= np.sqrt(profit_ratio)  # Diminishing returns

            # Confidence threshold effect
            conf_ratio = (
                params.confidence_threshold
                / self.current_parameters.confidence_threshold
            )
            profit_factor *= (
                1.0 + conf_ratio * 0.2
            )  # Higher confidence improves success

            # Risk tolerance effect
            risk_ratio = params.risk_tolerance / self.current_parameters.risk_tolerance
            profit_factor *= 0.8 + risk_ratio * 0.4  # Risk vs reward balance

            return baseline_profit * profit_factor

        except Exception:
            return 0.0

    def get_current_parameters(self) -> TuningParameters:
        """Get current optimized parameters"""
        return self.current_parameters

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        try:
            return {
                "optimization_enabled": self.enabled,
                "current_regime": self.current_regime.value,
                "optimization_count": self.optimization_count,
                "last_optimization": self.last_optimization_time.isoformat(),
                "objective": self.optimization_objective.value,
                "current_parameters": self.current_parameters.to_dict(),
                "performance_metrics": {
                    "total_profit_cents": self.current_parameters.min_profit_cents,
                    "win_rate": 0.0,  # Would be calculated from recent trades
                    "sharpe_ratio": 0.0,
                },
                "regime_history": [
                    {"timestamp": ts.isoformat(), "regime": regime.value}
                    for ts, regime in self.regime_history[-10:]
                ],
                "parameter_changes": len(self.parameter_history),
                "optimimization_efficiency": self._calculate_optimization_efficiency(),
            }

        except Exception as e:
            logger.error(f"Failed to generate optimization report: {e}")
            return {}

    def _calculate_optimization_efficiency(self) -> float:
        """Calculate optimization efficiency metric"""
        try:
            if len(self.parameter_history) < 2:
                return 0.0

            # Compare performance before and after optimizations
            recent_params = self.parameter_history[-1][0]
            performance_before = [
                m.total_profit_cents
                for m in self.performance_history
                if m.timestamp < recent_params
            ]

            performance_after = [
                m.total_profit_cents
                for m in self.performance_history
                if m.timestamp >= recent_params
            ]

            if not performance_before or not performance_after:
                return 0.0

            avg_before = np.mean(performance_before)
            avg_after = np.mean(performance_after)

            if avg_before == 0:
                return 1.0 if avg_after > 0 else 0.0

            improvement = (avg_after - avg_before) / abs(avg_before)
            return max(0.0, min(1.0, improvement))

        except Exception:
            return 0.0

    async def load_cached_parameters(self, regime: MarketRegime) -> bool:
        """Load cached parameters for a specific regime"""
        try:
            cached_params = await self.cache.get(f"optimized_params_{regime.value}")
            if cached_params:
                self.current_parameters = TuningParameters.from_dict(cached_params)
                logger.info(f"Loaded cached parameters for {regime.value} regime")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to load cached parameters: {e}")
            return False

    async def save_parameters(self) -> None:
        """Save current parameters to persistent storage"""
        try:
            await self.cache.set(
                "current_parameters",
                self.current_parameters.to_dict(),
                ttl=86400,  # 24 hours
            )

        except Exception as e:
            logger.error(f"Failed to save parameters: {e}")


# Utility function to create auto-tuning engine
def create_auto_tuning_engine(config: Config) -> AutoTuningEngine:
    """Create and return auto-tuning engine instance"""
    return AutoTuningEngine(config)
