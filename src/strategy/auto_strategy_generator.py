"""Automated strategy generation and optimization system."""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import asyncio
import logging
from pathlib import Path
import uuid
import itertools
from abc import ABC, abstractmethod

from src.utils.config import Config
from src.utils.logging_utils import get_logger
from src.backtesting.advanced_backtesting import AdvancedBacktester
from src.ml.pipeline import MLPipeline

logger = get_logger(__name__)


class StrategyType(Enum):
    """Strategy types for generation."""

    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    PAIRS_TRADING = "pairs_trading"
    MARKET_NEUTRAL = "market_neutral"
    VOLATILITY_TRADING = "volatility_trading"
    LIQUIDITY_ARBITRAGE = "liquidity_arbitrage"
    PREDICTION_MARKET = "prediction_market"


class OptimizationMethod(Enum):
    """Optimization methods."""

    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    PARTICLE_SWARM = "particle_swarm"


@dataclass
class StrategyTemplate:
    """Strategy generation template."""

    template_id: str
    name: str
    description: str
    strategy_type: StrategyType
    parameter_space: Dict[str, Dict[str, Any]]
    feature_requirements: List[str]
    data_requirements: Dict[str, Any]
    performance_targets: Dict[str, float]
    risk_constraints: Dict[str, float]
    generation_rules: List[Dict[str, Any]]


@dataclass
class GeneratedStrategy:
    """Generated strategy specification."""

    strategy_id: str
    name: str
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    features: List[str]
    entry_conditions: List[Dict[str, Any]]
    exit_conditions: List[Dict[str, Any]]
    risk_management: Dict[str, Any]
    performance_metrics: Dict[str, float]
    backtest_results: Dict[str, Any]
    generation_score: float
    created_at: datetime
    template_id: str
    optimization_iterations: int = 0


class StrategyGenerator(ABC):
    """Abstract base class for strategy generators."""

    @abstractmethod
    def generate_strategy(
        self,
        template: StrategyTemplate,
        market_data: pd.DataFrame,
        optimization_method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
    ) -> GeneratedStrategy:
        """Generate a new strategy based on template."""
        pass

    @abstractmethod
    def optimize_strategy(
        self,
        strategy: GeneratedStrategy,
        market_data: pd.DataFrame,
        optimization_method: OptimizationMethod,
    ) -> GeneratedStrategy:
        """Optimize an existing strategy."""
        pass


class StatisticalArbitrageGenerator(StrategyGenerator):
    """Generator for statistical arbitrage strategies."""

    def __init__(self, config: Config):
        self.config = config
        self.backtester = AdvancedBacktester()

    def generate_strategy(
        self,
        template: StrategyTemplate,
        market_data: pd.DataFrame,
        optimization_method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
    ) -> GeneratedStrategy:
        """Generate statistical arbitrage strategy."""
        try:
            logger.info(f"Generating {template.strategy_type.value} strategy")

            # Generate candidate assets for pairs
            asset_pairs = self._find_candidate_pairs(market_data)

            best_strategy = None
            best_score = float("-inf")

            for pair in asset_pairs[:10]:  # Limit to top 10 pairs
                # Generate strategy parameters
                strategy = self._create_pair_strategy(pair, template, market_data)

                # Optimize strategy
                optimized_strategy = self.optimize_strategy(
                    strategy, market_data, optimization_method
                )

                # Score strategy
                score = self._score_strategy(optimized_strategy, template)

                if score > best_score:
                    best_score = score
                    best_strategy = optimized_strategy

            if best_strategy:
                best_strategy.generation_score = best_score
                logger.info(f"Generated strategy with score: {best_score}")
                return best_strategy
            else:
                raise ValueError("No viable strategy generated")

        except Exception as e:
            logger.error(f"Failed to generate strategy: {e}")
            raise

    def _find_candidate_pairs(self, market_data: pd.DataFrame) -> List[Tuple[str, str]]:
        """Find candidate asset pairs for arbitrage."""
        try:
            assets = market_data.columns.tolist()
            candidates = []

            for asset1, asset2 in itertools.combinations(assets, 2):
                # Calculate cointegration and correlation
                if self._check_cointegration(market_data[asset1], market_data[asset2]):
                    correlation = market_data[asset1].corr(market_data[asset2])
                    if abs(correlation) > 0.7:  # Strong correlation threshold
                        candidates.append((asset1, asset2))

            # Sort by correlation strength
            candidates.sort(
                key=lambda pair: abs(market_data[pair[0]].corr(market_data[pair[1]])),
                reverse=True,
            )

            return candidates

        except Exception as e:
            logger.error(f"Failed to find candidate pairs: {e}")
            return []

    def _check_cointegration(self, series1: pd.Series, series2: pd.Series) -> bool:
        """Check if two series are cointegrated."""
        try:
            from statsmodels.tsa.stattools import coint

            # Remove NaN values
            df = pd.DataFrame({"s1": series1, "s2": series2}).dropna()
            if len(df) < 50:  # Minimum data points
                return False

            # Perform cointegration test
            coint_stat, p_value, _ = coint(df["s1"], df["s2"])

            # Check if p-value is significant (typically < 0.05)
            return p_value < 0.05

        except ImportError:
            logger.warning("statsmodels not available for cointegration test")
            return False
        except Exception as e:
            logger.error(f"Cointegration test failed: {e}")
            return False

    def _create_pair_strategy(
        self,
        pair: Tuple[str, str],
        template: StrategyTemplate,
        market_data: pd.DataFrame,
    ) -> GeneratedStrategy:
        """Create strategy for specific asset pair."""
        strategy_id = str(uuid.uuid4())

        # Generate random parameters within template constraints
        parameters = {}
        for param_name, param_config in template.parameter_space.items():
            param_type = param_config.get("type", "float")

            if param_type == "int":
                min_val = param_config.get("min", 10)
                max_val = param_config.get("max", 50)
                parameters[param_name] = np.random.randint(min_val, max_val + 1)
            elif param_type == "float":
                min_val = param_config.get("min", 0.1)
                max_val = param_config.get("max", 3.0)
                parameters[param_name] = np.random.uniform(min_val, max_val)
            elif param_type == "choice":
                choices = param_config.get("choices", [])
                parameters[param_name] = np.random.choice(choices)

        # Create entry and exit conditions
        entry_conditions = [
            {
                "type": "z_score_threshold",
                "condition": "abs(z_score) > entry_threshold",
                "threshold": parameters.get("entry_threshold", 2.0),
            }
        ]

        exit_conditions = [
            {
                "type": "z_score_revert",
                "condition": "abs(z_score) < exit_threshold",
                "threshold": parameters.get("exit_threshold", 0.5),
            }
        ]

        # Risk management
        risk_management = {
            "max_position_size": parameters.get("max_position_size", 0.1),
            "stop_loss": parameters.get("stop_loss", 0.05),
            "position_timeout": parameters.get(
                "position_timeout", 24 * 60 * 60
            ),  # 24 hours in seconds
        }

        return GeneratedStrategy(
            strategy_id=strategy_id,
            name=f"StatArb_{pair[0]}_{pair[1]}",
            strategy_type=StrategyType.STATISTICAL_ARBITRAGE,
            parameters=parameters,
            features=[
                f"ratio_{pair[0]}_{pair[1]}",
                f"spread_{pair[0]}_{pair[1]}",
                "z_score",
            ],
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_management=risk_management,
            performance_metrics={},
            backtest_results={},
            generation_score=0.0,
            created_at=datetime.utcnow(),
            template_id=template.template_id,
        )

    def optimize_strategy(
        self,
        strategy: GeneratedStrategy,
        market_data: pd.DataFrame,
        optimization_method: OptimizationMethod,
    ) -> GeneratedStrategy:
        """Optimize strategy parameters."""
        try:
            logger.info(f"Optimizing strategy {strategy.strategy_id}")

            if optimization_method == OptimizationMethod.GRID_SEARCH:
                best_params = self._grid_search_optimization(strategy, market_data)
            elif optimization_method == OptimizationMethod.RANDOM_SEARCH:
                best_params = self._random_search_optimization(strategy, market_data)
            elif optimization_method == OptimizationMethod.BAYESIAN:
                best_params = self._bayesian_optimization(strategy, market_data)
            else:
                logger.warning(
                    f"Optimization method {optimization_method} not implemented"
                )
                return strategy

            # Update strategy with optimized parameters
            strategy.parameters.update(best_params)
            strategy.optimization_iterations += 1

            # Re-evaluate performance
            strategy = self._evaluate_strategy(strategy, market_data)

            return strategy

        except Exception as e:
            logger.error(f"Failed to optimize strategy: {e}")
            return strategy

    def _grid_search_optimization(
        self, strategy: GeneratedStrategy, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform grid search optimization."""
        try:
            best_params = {}
            best_score = float("-inf")

            # Define grid for key parameters
            param_grid = {
                "lookback_period": [10, 20, 30, 40, 50],
                "entry_threshold": [1.5, 2.0, 2.5, 3.0],
                "exit_threshold": [0.2, 0.5, 0.8, 1.0],
                "position_size": [0.05, 0.1, 0.15, 0.2],
            }

            # Generate all combinations
            keys = param_grid.keys()
            values = param_grid.values()

            for combination in itertools.product(*values):
                params = dict(zip(keys, combination))

                # Create temporary strategy with these parameters
                temp_strategy = GeneratedStrategy(**asdict(strategy))
                temp_strategy.parameters.update(params)

                # Evaluate strategy
                score = self._evaluate_strategy_score(temp_strategy, market_data)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            return best_params

        except Exception as e:
            logger.error(f"Grid search optimization failed: {e}")
            return strategy.parameters

    def _random_search_optimization(
        self,
        strategy: GeneratedStrategy,
        market_data: pd.DataFrame,
        n_iterations: int = 50,
    ) -> Dict[str, Any]:
        """Perform random search optimization."""
        try:
            best_params = strategy.parameters.copy()
            best_score = self._evaluate_strategy_score(strategy, market_data)

            for _ in range(n_iterations):
                # Generate random parameters
                random_params = self._generate_random_parameters(strategy)

                # Create temporary strategy
                temp_strategy = GeneratedStrategy(**asdict(strategy))
                temp_strategy.parameters.update(random_params)

                # Evaluate strategy
                score = self._evaluate_strategy_score(temp_strategy, market_data)

                if score > best_score:
                    best_score = score
                    best_params = random_params.copy()

            return best_params

        except Exception as e:
            logger.error(f"Random search optimization failed: {e}")
            return strategy.parameters

    def _bayesian_optimization(
        self, strategy: GeneratedStrategy, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform Bayesian optimization."""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer

            # Define parameter space
            param_space = [
                Integer(10, 50, name="lookback_period"),
                Real(1.5, 3.0, name="entry_threshold"),
                Real(0.2, 1.0, name="exit_threshold"),
                Real(0.05, 0.2, name="position_size"),
            ]

            def objective(params):
                """Objective function to minimize (negative score)."""
                param_dict = {
                    "lookback_period": params[0],
                    "entry_threshold": params[1],
                    "exit_threshold": params[2],
                    "position_size": params[3],
                }

                temp_strategy = GeneratedStrategy(**asdict(strategy))
                temp_strategy.parameters.update(param_dict)

                score = self._evaluate_strategy_score(temp_strategy, market_data)
                return -score  # Minimize negative score

            # Perform optimization
            result = gp_minimize(objective, param_space, n_calls=30, random_state=42)

            # Convert best parameters back to dictionary
            best_params = {
                "lookback_period": result.x[0],
                "entry_threshold": result.x[1],
                "exit_threshold": result.x[2],
                "position_size": result.x[3],
            }

            return best_params

        except ImportError:
            logger.warning("scikit-optimize not available for Bayesian optimization")
            return self._random_search_optimization(strategy, market_data, 20)
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            return strategy.parameters

    def _generate_random_parameters(
        self, strategy: GeneratedStrategy
    ) -> Dict[str, Any]:
        """Generate random parameter values."""
        params = {}

        # Generate random values for key parameters
        params["lookback_period"] = np.random.randint(10, 51)
        params["entry_threshold"] = np.random.uniform(1.5, 3.0)
        params["exit_threshold"] = np.random.uniform(0.2, 1.0)
        params["position_size"] = np.random.uniform(0.05, 0.2)

        return params

    def _evaluate_strategy_score(
        self, strategy: GeneratedStrategy, market_data: pd.DataFrame
    ) -> float:
        """Evaluate strategy and return composite score."""
        try:
            # Run backtest
            backtest_results = self._run_backtest(strategy, market_data)

            # Calculate composite score based on multiple metrics
            metrics = backtest_results.get("performance_metrics", {})

            # Weighted score calculation
            score = 0.0

            # Sharpe ratio (40% weight)
            sharpe_ratio = metrics.get("sharpe_ratio", 0)
            score += 0.4 * max(0, sharpe_ratio)

            # Win rate (20% weight)
            win_rate = metrics.get("win_rate", 0)
            score += 0.2 * win_rate

            # Sortino ratio (20% weight)
            sortino_ratio = metrics.get("sortino_ratio", 0)
            score += 0.2 * max(0, sortino_ratio)

            # Calmar ratio (10% weight)
            calmar_ratio = metrics.get("calmar_ratio", 0)
            score += 0.1 * max(0, calmar_ratio)

            # Profit factor (10% weight)
            profit_factor = metrics.get("profit_factor", 1)
            score += 0.1 * min(
                2.0, max(0, (profit_factor - 1) / 2)
            )  # Normalize to 0-2, then 0-1

            return score

        except Exception as e:
            logger.error(f"Failed to evaluate strategy score: {e}")
            return 0.0

    def _run_backtest(
        self, strategy: GeneratedStrategy, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run backtest for strategy."""
        try:
            # Create strategy configuration for backtester
            strategy_config = {
                "name": strategy.name,
                "type": strategy.strategy_type.value,
                "parameters": strategy.parameters,
                "features": strategy.features,
                "entry_conditions": strategy.entry_conditions,
                "exit_conditions": strategy.exit_conditions,
                "risk_management": strategy.risk_management,
            }

            # Run backtest
            results = self.backtester.run_backtest(
                strategy_config=strategy_config,
                market_data=market_data,
                start_date=market_data.index.min(),
                end_date=market_data.index.max(),
                initial_capital=10000.0,
            )

            return results

        except Exception as e:
            logger.error(f"Failed to run backtest: {e}")
            return {"performance_metrics": {}, "trade_results": []}

    def _evaluate_strategy(
        self, strategy: GeneratedStrategy, market_data: pd.DataFrame
    ) -> GeneratedStrategy:
        """Evaluate strategy and update performance metrics."""
        try:
            # Run backtest
            backtest_results = self._run_backtest(strategy, market_data)

            # Update strategy with results
            strategy.backtest_results = backtest_results
            strategy.performance_metrics = backtest_results.get(
                "performance_metrics", {}
            )

            return strategy

        except Exception as e:
            logger.error(f"Failed to evaluate strategy: {e}")
            return strategy

    def _score_strategy(
        self, strategy: GeneratedStrategy, template: StrategyTemplate
    ) -> float:
        """Score strategy against template criteria."""
        try:
            score = 0.0
            metrics = strategy.performance_metrics

            # Check performance targets
            targets = template.performance_targets
            constraints = template.risk_constraints

            # Sharpe ratio
            target_sharpe = targets.get("sharpe_ratio", 1.0)
            actual_sharpe = metrics.get("sharpe_ratio", 0)
            sharpe_score = min(1.0, actual_sharpe / target_sharpe)
            score += 0.3 * sharpe_score

            # Win rate
            target_win_rate = targets.get("win_rate", 0.6)
            actual_win_rate = metrics.get("win_rate", 0)
            win_rate_score = min(1.0, actual_win_rate / target_win_rate)
            score += 0.2 * win_rate_score

            # Max drawdown (risk constraint)
            max_drawdown_limit = constraints.get("max_drawdown", 0.1)
            actual_drawdown = metrics.get("max_drawdown", 1.0)
            drawdown_score = max(0, 1.0 - (actual_drawdown / max_drawdown_limit))
            score += 0.25 * drawdown_score

            # Total return
            target_return = targets.get("total_return", 0.2)
            actual_return = metrics.get("total_return", 0)
            return_score = min(1.0, actual_return / target_return)
            score += 0.25 * return_score

            return score

        except Exception as e:
            logger.error(f"Failed to score strategy: {e}")
            return 0.0


class AutoStrategyGenerator:
    """Main automated strategy generation system."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.templates = self._load_templates()
        self.generators = {
            StrategyType.STATISTICAL_ARBITRAGE: StatisticalArbitrageGenerator(
                self.config
            ),
            # Add other generators as needed
        }

        # ML pipeline for strategy selection
        self.ml_pipeline = MLPipeline()
        self.strategy_performance_history = []

        logger.info("Auto strategy generator initialized")

    def _load_templates(self) -> Dict[str, StrategyTemplate]:
        """Load strategy templates."""
        templates_config = self.config.get("strategy_generation.templates", {})
        templates = {}

        for template_id, template_data in templates_config.items():
            templates[template_id] = StrategyTemplate(
                template_id=template_id,
                name=template_data.get("name", template_id),
                description=template_data.get("description", ""),
                strategy_type=StrategyType(
                    template_data.get("strategy_type", "statistical_arbitrage")
                ),
                parameter_space=template_data.get("parameter_space", {}),
                feature_requirements=template_data.get("feature_requirements", []),
                data_requirements=template_data.get("data_requirements", {}),
                performance_targets=template_data.get("performance_targets", {}),
                risk_constraints=template_data.get("risk_constraints", {}),
                generation_rules=template_data.get("generation_rules", []),
            )

        # Add default templates if none configured
        if not templates:
            templates = self._create_default_templates()

        return templates

    def _create_default_templates(self) -> Dict[str, StrategyTemplate]:
        """Create default strategy templates."""
        return {
            "stat_arb_default": StrategyTemplate(
                template_id="stat_arb_default",
                name="Default Statistical Arbitrage",
                description="Standard statistical arbitrage strategy",
                strategy_type=StrategyType.STATISTICAL_ARBITRAGE,
                parameter_space={
                    "lookback_period": {"type": "int", "min": 10, "max": 50},
                    "entry_threshold": {"type": "float", "min": 1.5, "max": 3.0},
                    "exit_threshold": {"type": "float", "min": 0.2, "max": 1.0},
                    "position_size": {"type": "float", "min": 0.05, "max": 0.2},
                },
                feature_requirements=["price_ratio", "spread", "z_score"],
                data_requirements={"min_data_points": 1000, "frequency": "1h"},
                performance_targets={
                    "sharpe_ratio": 1.5,
                    "win_rate": 0.6,
                    "total_return": 0.2,
                },
                risk_constraints={"max_drawdown": 0.1, "max_position_size": 0.3},
                generation_rules=[
                    {"type": "min_correlation", "value": 0.7},
                    {"type": "cointegration_required", "value": True},
                ],
            )
        }

    async def generate_strategies(
        self,
        market_data: pd.DataFrame,
        num_strategies: int = 5,
        strategy_types: Optional[List[StrategyType]] = None,
        optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN,
    ) -> List[GeneratedStrategy]:
        """Generate multiple strategies."""
        try:
            logger.info(f"Generating {num_strategies} strategies")

            if strategy_types is None:
                strategy_types = [StrategyType.STATISTICAL_ARBITRAGE]

            generated_strategies = []

            for strategy_type in strategy_types:
                # Get relevant templates
                relevant_templates = [
                    template
                    for template in self.templates.values()
                    if template.strategy_type == strategy_type
                ]

                if not relevant_templates:
                    logger.warning(f"No templates found for {strategy_type}")
                    continue

                # Get generator
                generator = self.generators.get(strategy_type)
                if not generator:
                    logger.warning(f"No generator found for {strategy_type}")
                    continue

                # Generate strategies for each template
                for template in relevant_templates:
                    for _ in range(num_strategies // len(relevant_templates)):
                        try:
                            strategy = generator.generate_strategy(
                                template, market_data, optimization_method
                            )
                            generated_strategies.append(strategy)

                        except Exception as e:
                            logger.error(f"Failed to generate strategy: {e}")
                            continue

            # Rank strategies by performance
            generated_strategies.sort(key=lambda s: s.generation_score, reverse=True)

            logger.info(f"Generated {len(generated_strategies)} strategies")
            return generated_strategies[:num_strategies]

        except Exception as e:
            logger.error(f"Failed to generate strategies: {e}")
            return []

    async def optimize_existing_strategies(
        self,
        strategies: List[GeneratedStrategy],
        market_data: pd.DataFrame,
        optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN,
    ) -> List[GeneratedStrategy]:
        """Optimize existing strategies."""
        try:
            logger.info(f"Optimizing {len(strategies)} strategies")

            optimized_strategies = []

            for strategy in strategies:
                generator = self.generators.get(strategy.strategy_type)
                if generator:
                    optimized_strategy = generator.optimize_strategy(
                        strategy, market_data, optimization_method
                    )
                    optimized_strategies.append(optimized_strategy)
                else:
                    optimized_strategies.append(strategy)

            # Re-rank strategies
            optimized_strategies.sort(key=lambda s: s.generation_score, reverse=True)

            logger.info(f"Optimized {len(optimized_strategies)} strategies")
            return optimized_strategies

        except Exception as e:
            logger.error(f"Failed to optimize strategies: {e}")
            return strategies

    async def select_best_strategies(
        self,
        strategies: List[GeneratedStrategy],
        max_strategies: int = 3,
        diversity_factor: float = 0.3,
    ) -> List[GeneratedStrategy]:
        """Select best strategies with diversity consideration."""
        try:
            if len(strategies) <= max_strategies:
                return strategies

            # Rank by performance score
            strategies.sort(key=lambda s: s.generation_score, reverse=True)

            # Select strategies with diversity
            selected = [strategies[0]]  # Always include the best one

            for candidate in strategies[1:]:
                if len(selected) >= max_strategies:
                    break

                # Check diversity against selected strategies
                is_diverse = True

                for selected_strategy in selected:
                    similarity = self._calculate_strategy_similarity(
                        candidate, selected_strategy
                    )

                    if similarity > (1 - diversity_factor):
                        is_diverse = False
                        break

                if is_diverse:
                    selected.append(candidate)

            logger.info(f"Selected {len(selected)} diverse strategies")
            return selected

        except Exception as e:
            logger.error(f"Failed to select best strategies: {e}")
            return strategies[:max_strategies]

    def _calculate_strategy_similarity(
        self, strategy1: GeneratedStrategy, strategy2: GeneratedStrategy
    ) -> float:
        """Calculate similarity between two strategies."""
        try:
            # Different strategy types are completely different
            if strategy1.strategy_type != strategy2.strategy_type:
                return 0.0

            similarity = 0.0

            # Parameter similarity (40% weight)
            param_similarity = self._calculate_parameter_similarity(
                strategy1.parameters, strategy2.parameters
            )
            similarity += 0.4 * param_similarity

            # Feature similarity (30% weight)
            feature_similarity = len(
                set(strategy1.features) & set(strategy2.features)
            ) / max(len(set(strategy1.features) | set(strategy2.features)), 1)
            similarity += 0.3 * feature_similarity

            # Entry/exit condition similarity (30% weight)
            entry_similarity = len(strategy1.entry_conditions) + len(
                strategy2.entry_conditions
            )
            if entry_similarity > 0:
                common_entry = len(
                    set([c.get("type") for c in strategy1.entry_conditions])
                    & set([c.get("type") for c in strategy2.entry_conditions])
                )
                entry_similarity = common_entry / entry_similarity
            else:
                entry_similarity = 1.0

            similarity += 0.3 * entry_similarity

            return similarity

        except Exception as e:
            logger.error(f"Failed to calculate strategy similarity: {e}")
            return 0.0

    def _calculate_parameter_similarity(
        self, params1: Dict[str, Any], params2: Dict[str, Any]
    ) -> float:
        """Calculate parameter similarity."""
        try:
            common_keys = set(params1.keys()) & set(params2.keys())

            if not common_keys:
                return 0.0

            similarities = []

            for key in common_keys:
                val1 = params1[key]
                val2 = params2[key]

                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Numerical parameters - normalized difference
                    max_val = max(abs(val1), abs(val2), 1.0)
                    diff = abs(val1 - val2) / max_val
                    similarity = 1.0 - diff
                elif val1 == val2:
                    # Categorical parameters - exact match
                    similarity = 1.0
                else:
                    similarity = 0.0

                similarities.append(similarity)

            return sum(similarities) / len(similarities)

        except Exception as e:
            logger.error(f"Failed to calculate parameter similarity: {e}")
            return 0.0

    def save_strategy(self, strategy: GeneratedStrategy, file_path: str):
        """Save strategy to file."""
        try:
            strategy_dict = asdict(strategy)
            strategy_dict["created_at"] = strategy.created_at.isoformat()
            strategy_dict["strategy_type"] = strategy.strategy_type.value

            with open(file_path, "w") as f:
                json.dump(strategy_dict, f, indent=2)

            logger.info(f"Strategy saved to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save strategy: {e}")
            raise

    def load_strategy(self, file_path: str) -> GeneratedStrategy:
        """Load strategy from file."""
        try:
            with open(file_path, "r") as f:
                strategy_dict = json.load(f)

            strategy_dict["created_at"] = datetime.fromisoformat(
                strategy_dict["created_at"]
            )
            strategy_dict["strategy_type"] = StrategyType(
                strategy_dict["strategy_type"]
            )

            return GeneratedStrategy(**strategy_dict)

        except Exception as e:
            logger.error(f"Failed to load strategy: {e}")
            raise
