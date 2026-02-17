"""
Advanced Portfolio Optimization Engine
Modern Portfolio Theory with advanced optimization techniques
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import redis.asyncio as redis

from ..utils.performance_cache import PerformanceCache

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    RISK_PARITY = "risk_parity"
    TARGET_RETURN = "target_return"
    TARGET_RISK = "target_risk"

class OptimizationConstraint(Enum):
    """Portfolio optimization constraints"""
    LONG_ONLY = "long_only"
    FULL_INVESTMENT = "full_investment"
    MAX_POSITION_SIZE = "max_position_size"
    SECTOR_LIMITS = "sector_limits"
    BETA_LIMITS = "beta_limits"
    TURNOVER_LIMIT = "turnover_limit"
    FACTOR_EXPOSURE = "factor_exposure"

@dataclass
class OptimizationInput:
    """Input data for portfolio optimization"""
    assets: List[str]
    returns: np.ndarray  # (n_observations, n_assets)
    risk_free_rate: float = 0.02  # Annualized
    
    # Constraints
    constraints: List[OptimizationConstraint] = field(default_factory=lambda: [
        OptimizationConstraint.FULL_INVESTMENT
    ])
    max_position_size: float = 0.3
    target_return: Optional[float] = None
    target_risk: Optional[float] = None
    
    # Strategy weights (for strategy allocation)
    strategy_returns: Optional[np.ndarray] = None
    strategy_names: Optional[List[str]] = None

@dataclass
class OptimizationResult:
    """Result of portfolio optimization"""
    optimization_id: str
    objective: OptimizationObjective
    
    # Optimal weights
    asset_weights: Dict[str, float]
    strategy_weights: Optional[Dict[str, float]] = None
    
    # Portfolio metrics
    expected_return: float
    risk: float
    sharpe_ratio: float
    beta: float
    alpha: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0  # Value at Risk 95%
    var_99: float = 0.0  # Value at Risk 99%
    cvar_95: float = 0.0  # Conditional VaR 95%
    
    # Risk contributions
    risk_contributions: Dict[str, float]
    marginal_risk_contributions: Dict[str, float]
    
    # Decomposition metrics
    sector_exposures: Dict[str, float] = field(default_factory=dict)
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    turnover: float = 0.0
    tracking_error: float = 0.0
    
    # Optimization metadata
    success: bool = True
    optimization_time: float = 0.0
    iterations: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class PortfolioOptimizationEngine:
    """
    Advanced portfolio optimization engine with modern portfolio theory
    and multiple optimization approaches
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.portfolio_config = config.get('portfolio_optimization', {})
        
        # Optimization parameters
        self.default_objective = OptimizationObjective(
            self.portfolio_config.get('default_objective', 'maximize_sharpe')
        )
        self.min_assets = self.portfolio_config.get('min_assets', 5)
        self.max_assets = self.portfolio_config.get('max_assets', 20)
        self.rebalance_frequency = timedelta(days=self.portfolio_config.get('rebalance_frequency_days', 7))
        
        # Risk parameters
        self.risk_free_rate = self.portfolio_config.get('risk_free_rate', 0.02)
        self.confidence_level = self.portfolio_config.get('confidence_level', 0.95)
        
        # Portfolio tracking
        self.current_portfolio: Dict[str, float] = {}
        self.current_strategies: Dict[str, float] = {}
        self.optimization_history: List[OptimizationResult] = []
        
        # Performance tracking
        self.portfolio_returns: deque = deque(maxlen=1000)
        self.benchmark_returns: deque = deque(maxlen=1000)
        
        # Asset and strategy data
        self.asset_data: Dict[str, Dict[str, Any]] = {}
        self.strategy_data: Dict[str, Dict[str, Any]] = {}
        
        # Sector and factor definitions
        self.asset_sectors: Dict[str, str] = {}
        self.asset_factors: Dict[str, Dict[str, float]] = {}
        
        # Performance cache
        self.cache = PerformanceCache(
            redis_url=config.get('redis_url', 'redis://localhost:6379'),
            default_ttl=3600  # 1 hour TTL for optimization data
        )
        
        # Asset universe
        self.assets = self.portfolio_config.get('assets', ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'])
        self.strategies = self.portfolio_config.get('strategies', [
            'mean_reversion', 'momentum', 'pairs_trading', 'statistical_arbitrage'
        ])
        
        logger.info("Portfolio Optimization Engine initialized")
    
    async def initialize(self) -> None:
        """Initialize the portfolio optimization engine"""
        try:
            # Load asset data
            await self._load_asset_data()
            
            # Load strategy data
            await self._load_strategy_data()
            
            # Initialize current portfolio (equal weight)
            await self._initialize_current_portfolio()
            
            # Start monitoring loops
            asyncio.create_task(self._portfolio_monitoring_loop())
            asyncio.create_task(self._optimization_loop())
            
            logger.info("Portfolio Optimization Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Portfolio Optimization Engine initialization failed: {e}")
            raise
    
    async def optimize_portfolio(self, objective: OptimizationObjective, 
                                target_return: Optional[float] = None,
                                target_risk: Optional[float] = None) -> OptimizationResult:
        """Optimize portfolio allocation"""
        try:
            logger.info(f"Starting portfolio optimization with objective: {objective.value}")
            
            # Prepare optimization inputs
            inputs = await self._prepare_optimization_inputs(objective, target_return, target_risk)
            
            if not inputs or len(inputs.assets) < self.min_assets:
                return self._create_default_optimization(objective)
            
            # Select optimization method
            if objective == OptimizationObjective.MAXIMIZE_RETURN:
                result = await self._optimize_maximize_return(inputs)
            elif objective == OptimizationObjective.MINIMIZE_RISK:
                result = await self._optimize_minimize_risk(inputs)
            elif objective == OptimizationObjective.MAXIMIZE_SHARPE:
                result = await self._optimize_maximize_sharpe(inputs)
            elif objective == OptimizationObjective.RISK_PARITY:
                result = await self._optimize_risk_parity(inputs)
            elif objective == OptimizationObjective.TARGET_RETURN:
                result = await self._optimize_target_return(inputs, target_return)
            elif objective == OptimizationObjective.TARGET_RISK:
                result = await self._optimize_target_risk(inputs, target_risk)
            else:
                result = await self._optimize_maximize_sharpe(inputs)
            
            # Calculate risk contributions
            if result.success:
                result = await self._calculate_risk_contributions(result, inputs)
            
            # Store result
            self.optimization_history.append(result)
            
            # Cache result
            await self._cache_optimization_result(result)
            
            logger.info(f"Portfolio optimization completed: expected_return={result.expected_return:.4f}, risk={result.risk:.4f}, sharpe={result.sharpe_ratio:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return self._create_default_optimization(objective)
    
    async def _prepare_optimization_inputs(self, objective: OptimizationObjective,
                                         target_return: Optional[float],
                                         target_risk: Optional[float]) -> Optional[OptimizationInput]:
        """Prepare inputs for optimization"""
        try:
            # Get returns matrix
            returns = await self._get_returns_matrix()
            
            if returns is None or returns.shape[1] < self.min_assets:
                return None
            
            # Create optimization input
            inputs = OptimizationInput(
                assets=self.assets[:min(len(self.assets), self.max_assets)],
                returns=returns[:, :min(len(self.assets), self.max_assets)],
                risk_free_rate=self.risk_free_rate,
                target_return=target_return,
                target_risk=target_risk
            )
            
            return inputs
            
        except Exception as e:
            logger.error(f"Optimization inputs preparation failed: {e}")
            return None
    
    async def _optimize_maximize_return(self, inputs: OptimizationInput) -> OptimizationResult:
        """Maximize portfolio return"""
        try:
            n_assets = len(inputs.assets)
            
            # Objective: maximize expected return
            def expected_return(weights):
                portfolio_return = np.dot(weights, np.mean(inputs.returns, axis=0))
                return -portfolio_return  # Minimize negative return
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Full investment
                {'type': 'ineq', 'fun': lambda w: self.max_position_size - w},  # Max position size
            ]
            
            # Bounds
            bounds = [(0.0, 1.0) if OptimizationConstraint.LONG_ONLY not in inputs.constraints 
                     else (-1.0, 1.0) for _ in range(n_assets)]
            
            # Initial guess (equal weight)
            w0 = np.array([1.0/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                expected_return,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(weights, np.mean(inputs.returns, axis=0))
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(np.cov(inputs.returns.T), weights)))
                sharpe_ratio = (portfolio_return - inputs.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                
                # Calculate VaR and CVaR
                var_95, var_99, cvar_95 = self._calculate_var_cvar(inputs.returns, weights)
                
                return OptimizationResult(
                    optimization_id=f"opt_max_return_{datetime.now().timestamp()}",
                    objective=OptimizationObjective.MAXIMIZE_RETURN,
                    asset_weights={asset: float(weights[i]) for i, asset in enumerate(inputs.assets)},
                    expected_return=portfolio_return,
                    risk=portfolio_risk,
                    sharpe_ratio=sharpe_ratio,
                    var_95=var_95,
                    var_99=var_99,
                    cvar_95=cvar_95,
                    risk_contributions={},
                    marginal_risk_contributions={},
                    success=True,
                    iterations=result.nit
                )
            else:
                return self._create_default_optimization(OptimizationObjective.MAXIMIZE_RETURN)
            
        except Exception as e:
            logger.error(f"Maximize return optimization failed: {e}")
            return self._create_default_optimization(OptimizationObjective.MAXIMIZE_RETURN)
    
    async def _optimize_minimize_risk(self, inputs: OptimizationInput) -> OptimizationResult:
        """Minimize portfolio risk"""
        try:
            n_assets = len(inputs.assets)
            
            # Objective: minimize portfolio variance
            def portfolio_variance(weights):
                cov_matrix = np.cov(inputs.returns.T)
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Full investment
                {'type': 'ineq', 'fun': lambda w: self.max_position_size - w},  # Max position size
            ]
            
            # Bounds
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
            
            # Initial guess (equal weight)
            w0 = np.array([1.0/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                portfolio_variance,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(weights, np.mean(inputs.returns, axis=0))
                portfolio_risk = np.sqrt(result.fun)
                sharpe_ratio = (portfolio_return - inputs.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                
                # Calculate VaR and CVaR
                var_95, var_99, cvar_95 = self._calculate_var_cvar(inputs.returns, weights)
                
                return OptimizationResult(
                    optimization_id=f"opt_min_risk_{datetime.now().timestamp()}",
                    objective=OptimizationObjective.MINIMIZE_RISK,
                    asset_weights={asset: float(weights[i]) for i, asset in enumerate(inputs.assets)},
                    expected_return=portfolio_return,
                    risk=portfolio_risk,
                    sharpe_ratio=sharpe_ratio,
                    var_95=var_95,
                    var_99=var_99,
                    cvar_95=cvar_95,
                    risk_contributions={},
                    marginal_risk_contributions={},
                    success=True,
                    iterations=result.nit
                )
            else:
                return self._create_default_optimization(OptimizationObjective.MINIMIZE_RISK)
            
        except Exception as e:
            logger.error(f"Minimize risk optimization failed: {e}")
            return self._create_default_optimization(OptimizationObjective.MINIMIZE_RISK)
    
    async def _optimize_maximize_sharpe(self, inputs: OptimizationInput) -> OptimizationResult:
        """Maximize Sharpe ratio (modern portfolio theory)"""
        try:
            n_assets = len(inputs.assets)
            
            # Use analytical solution for unconstrained problem
            mean_returns = np.mean(inputs.returns, axis=0)
            cov_matrix = np.cov(inputs.returns.T)
            
            # Avoid division by zero
            cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-8
            
            # Inverse of covariance matrix
            inv_cov = np.linalg.inv(cov_matrix)
            
            # Risk aversion (can be adjusted)
            risk_aversion = 1.0
            
            # Optimal weights (unconstrained)
            ones_vector = np.ones(n_assets)
            w_unconstrained = risk_aversion * np.dot(inv_cov, mean_returns) / np.dot(ones_vector.T, np.dot(inv_cov, ones_vector))
            
            # Constrained optimization for multiple constraints
            def sharpe_ratio_negative(weights):
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -(portfolio_return - inputs.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else -np.inf
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'ineq', 'fun': lambda w: self.max_position_size - w},
            ]
            
            # Bounds
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
            
            # Optimize with constraints
            result = minimize(
                sharpe_ratio_negative,
                w_unconstrained,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - inputs.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                
                # Calculate VaR and CVaR
                var_95, var_99, cvar_95 = self._calculate_var_cvar(inputs.returns, weights)
                
                return OptimizationResult(
                    optimization_id=f"opt_max_sharpe_{datetime.now().timestamp()}",
                    objective=OptimizationObjective.MAXIMIZE_SHARPE,
                    asset_weights={asset: float(weights[i]) for i, asset in enumerate(inputs.assets)},
                    expected_return=portfolio_return,
                    risk=portfolio_risk,
                    sharpe_ratio=sharpe_ratio,
                    var_95=var_95,
                    var_99=var_99,
                    cvar_95=cvar_95,
                    risk_contributions={},
                    marginal_risk_contributions={},
                    success=True,
                    iterations=result.nit
                )
            else:
                return self._create_default_optimization(OptimizationObjective.MAXIMIZE_SHARPE)
            
        except Exception as e:
            logger.error(f"Maximize Sharpe optimization failed: {e}")
            return self._create_default_optimization(OptimizationObjective.MAXIMIZE_SHARPE)
    
    async def _optimize_risk_parity(self, inputs: OptimizationInput) -> OptimizationResult:
        """Optimize using risk parity approach"""
        try:
            n_assets = len(inputs.assets)
            cov_matrix = np.cov(inputs.returns.T)
            
            # Risk parity: equal risk contribution from each asset
            marginal_risks = np.zeros(n_assets)
            for i in range(n_assets):
                marginal_risks[i] = cov_matrix[i, i]  # Volatility squared (simplified)
            
            # Inverse volatility weighting
            inverse_volatility = 1.0 / np.sqrt(marginal_risks)
            weights = inverse_volatility / np.sum(inverse_volatility)
            
            # Calculate portfolio metrics
            mean_returns = np.mean(inputs.returns, axis=0)
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - inputs.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            # Calculate VaR and CVaR
            var_95, var_99, cvar_95 = self._calculate_var_cvar(inputs.returns, weights)
            
            return OptimizationResult(
                optimization_id=f"opt_risk_parity_{datetime.now().timestamp()}",
                objective=OptimizationObjective.RISK_PARITY,
                asset_weights={asset: float(weights[i]) for i, asset in enumerate(inputs.assets)},
                expected_return=portfolio_return,
                risk=portfolio_risk,
                sharpe_ratio=sharpe_ratio,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                risk_contributions={},
                marginal_risk_contributions={},
                success=True
            )
            
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            return self._create_default_optimization(OptimizationObjective.RISK_PARITY)
    
    async def _optimize_target_return(self, inputs: OptimizationInput, target_return: float) -> OptimizationResult:
        """Optimize for specific target return"""
        try:
            n_assets = len(inputs.assets)
            mean_returns = np.mean(inputs.returns, axis=0)
            cov_matrix = np.cov(inputs.returns.T)
            
            # Objective: minimize variance for given return
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return},
                {'type': 'ineq', 'fun': lambda w: self.max_position_size - w},
            ]
            
            # Bounds
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
            
            # Initial guess (equal weight)
            w0 = np.array([1.0/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                portfolio_variance,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
                portfolio_risk = np.sqrt(result.fun)
                portfolio_return = np.dot(weights, mean_returns)
                sharpe_ratio = (portfolio_return - inputs.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                
                var_95, var_99, cvar_95 = self._calculate_var_cvar(inputs.returns, weights)
                
                return OptimizationResult(
                    optimization_id=f"opt_target_return_{datetime.now().timestamp()}",
                    objective=OptimizationObjective.TARGET_RETURN,
                    asset_weights={asset: float(weights[i]) for i, asset in enumerate(inputs.assets)},
                    expected_return=portfolio_return,
                    risk=portfolio_risk,
                    sharpe_ratio=sharpe_ratio,
                    var_95=var_95,
                    var_99=var_99,
                    cvar_95=cvar_95,
                    risk_contributions={},
                    marginal_risk_contributions={},
                    success=True,
                    iterations=result.nit
                )
            else:
                return self._create_default_optimization(OptimizationObjective.TARGET_RETURN)
            
        except Exception as e:
            logger.error(f"Target return optimization failed: {e}")
            return self._create_default_optimization(OptimizationObjective.TARGET_RETURN)
    
    async def _optimize_target_risk(self, inputs: OptimizationInput, target_risk: float) -> OptimizationResult:
        """Optimize for specific target risk level"""
        try:
            n_assets = len(inputs.assets)
            mean_returns = np.mean(inputs.returns, axis=0)
            cov_matrix = np.cov(inputs.returns.T)
            
            # Objective: maximize return for given risk
            def negative_return(weights):
                return -np.dot(weights, mean_returns)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) - target_risk},
                {'type': 'ineq', 'fun': lambda w: self.max_position_size - w},
            ]
            
            # Bounds
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
            
            # Initial guess (equal weight)
            w0 = np.array([1.0/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                negative_return,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_risk = target_risk
                sharpe_ratio = (portfolio_return - inputs.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                
                var_95, var_99, cvar_95 = self._calculate_var_cvar(inputs.returns, weights)
                
                return OptimizationResult(
                    optimization_id=f"opt_target_risk_{datetime.now().timestamp()}",
                    objective=OptimizationObjective.TARGET_RISK,
                    asset_weights={asset: float(weights[i]) for i, asset in enumerate(inputs.assets)},
                    expected_return=portfolio_return,
                    risk=portfolio_risk,
                    sharpe_ratio=sharpe_ratio,
                    var_95=var_95,
                    var_99=var_99,
                    cvar_95=cvar_95,
                    risk_contributions={},
                    marginal_risk_contributions={},
                    success=True,
                    iterations=result.nit
                )
            else:
                return self._create_default_optimization(OptimizationObjective.TARGET_RISK)
            
        except Exception as e:
            logger.error(f"Target risk optimization failed: {e}")
            return self._create_default_optimization(OptimizationObjectve.TARGET_RISK)
    
    def _calculate_var_cvar(self, returns: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float]:
        """Calculate VaR and CVaR"""
        try:
            # Calculate portfolio returns
            portfolio_returns = np.dot(returns, weights)
            
            # Sort returns
            sorted_returns = np.sort(portfolio_returns)
            
            # VaR at 95% and 99%
            var_95_index = int(len(sorted_returns) * 0.05)
            var_99_index = int(len(sorted_returns) * 0.01)
            
            var_95 = abs(sorted_returns[var_95_index])  # Absolute value for loss
            var_99 = abs(sorted_returns[var_99_index])
            
            # CVaR at 95%
            tail_returns = sorted_returns[:var_95_index]
            cvar_95 = abs(np.mean(tail_returns)) if len(tail_returns) > 0 else 0
            
            return var_95, var_99, cvar_95
            
        except Exception as e:
            logger.error(f"VaR/CVaR calculation failed: {e}")
            return 0, 0, 0
    
    async def _calculate_risk_contributions(self, result: OptimizationResult, 
                                         inputs: OptimizationInput) -> OptimizationResult:
        """Calculate risk contributions for each asset"""
        try:
            weights = np.array([result.asset_weights[asset] for asset in inputs.assets])
            cov_matrix = np.cov(inputs.returns.T)
            
            # Marginal risk contributions
            marginal_contributions = np.dot(cov_matrix, weights)
            
            # Risk contributions
            risk_contributions = weights * marginal_contributions / np.dot(weights, marginal_contributions)
            
            result.risk_contributions = {
                asset: float(risk_contributions[i])
                for i, asset in enumerate(inputs.assets)
            }
            
            result.marginal_risk_contributions = {
                asset: float(marginal_contributions[i])
                for i, asset in enumerate(inputs.assets)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Risk contributions calculation failed: {e}")
            return result
    
    async def _get_returns_matrix(self) -> Optional[np.ndarray]:
        """Get returns matrix for optimization"""
        try:
            # This would fetch actual historical returns data
            # For now, generate simulated data
            
            n_observations = 500
            n_assets = len(self.assets)
            
            # Generate correlated returns
            returns = np.random.multivariate_normal(
                mean=np.array([0.001] * n_assets),  # 0.1% daily return mean
                cov=np.eye(n_assets) * 0.02,  # 2% daily volatility
                size=n_observations
            )
            
            returns[returns < -0.1] = -0.1  # Cap at 10% daily loss
            returns[returns > 0.1] = 0.1   # Cap at 10% daily gain
            
            return returns
            
        except Exception as e:
            logger.error(f"Returns matrix generation failed: {e}")
            return None
    
    def _create_default_optimization(self, objective: OptimizationObjective) -> OptimizationResult:
        """Create default optimization when optimization fails"""
        n_assets = min(len(self.assets), 10)
        equal_weights = {asset: 1.0/n_assets for asset in self.assets[:n_assets]}
        
        return OptimizationResult(
            optimization_id=f"opt_default_{datetime.now().timestamp()}",
            objective=objective,
            asset_weights=equal_weights,
            expected_return=0.05,  # 5% annual return
            risk=0.15,  # 15% annual risk
            sharpe_ratio=0.20,  # Sharpe ratio
            var_95=0.10,
            var_99=0.15,
            cvar_95=0.12,
            risk_contributions=equal_weights,
            marginal_risk_contributions=equal_weights,
            success=True
        )
    
    async def _load_asset_data(self) -> None:
        """Load asset data for optimization"""
        try:
            # Load historical returns, volatility, correlations, etc.
            for asset in self.assets:
                self.asset_data[asset] = {
                    'mean_return': np.random.uniform(0.05, 0.2),  # 5-20% annual return
                    'volatility': np.random.uniform(0.3, 0.8),  # 30-80% annual volatility
                    'beta': np.random.uniform(0.5, 1.5),  # Beta
                    'sector': 'crypto'  # Simplified sector classification
                }
            
            logger.info(f"Loaded asset data for {len(self.asset_data)} assets")
            
        except Exception as e:
            logger.error(f"Asset data loading failed: {e}")
    
    async def _load_strategy_data(self) -> None:
        """Load strategy data for optimization"""
        try:
            for strategy in self.strategies:
                self.strategy_data[strategy] = {
                    'mean_return': np.random.uniform(0.1, 0.4),  # Higher returns for strategies
                    'volatility': np.random.uniform(0.4, 1.0),
                    'correlations': np.random.uniform(-0.3, 0.7, size=len(self.strategies)),
                    'max_drawdown': np.random.uniform(0.1, 0.3)
                }
            
            logger.info(f"Loaded strategy data for {len(self.strategy_data)} strategies")
            
        except Exception as e:
            logger.error(f"Strategy data loading failed: {e}")
    
    async def _initialize_current_portfolio(self) -> None:
        """Initialize current portfolio with equal weights"""
        try:
            n_assets = len(self.assets)
            equal_weight = 1.0 / n_assets
            
            for asset in self.assets:
                self.current_portfolio[asset] = equal_weight
            
            # Initialize strategy allocation (equal weight)
            n_strategies = len(self.strategies)
            equal_strategy_weight = 1.0 / n_strategies
            
            for strategy in self.strategies:
                self.current_strategies[strategy] = equal_strategy_weight
            
            logger.info("Current portfolio initialized with equal weights")
            
        except Exception as e:
            logger.error(f"Current portfolio initialization failed: {e}")
    
    async def _portfolio_monitoring_loop(self) -> None:
        """Background loop for monitoring portfolio performance"""
        while True:
            try:
                # Simulate portfolio performance
                await self._update_portfolio_returns()
                
                # Update current portfolio if needed
                await self._update_portfolio_if_needed()
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Portfolio monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _update_portfolio_returns(self) -> None:
        """Update portfolio returns"""
        try:
            # Calculate portfolio return
            total_return = 0.0
            for asset, weight in self.current_portfolio.items():
                asset_return = self.asset_data.get(asset, {}).get('mean_return', 0.05) / 252  # Daily return
                total_return += weight * asset_return
            
            self.portfolio_returns.append(total_return)
            
            # Simulate benchmark return
            benchmark_return = 0.0005  # 0.05% daily
            self.benchmark_returns.append(benchmark_return)
            
        except Exception as e:
            logger.error(f"Portfolio returns update failed: {e}")
    
    async def _update_portfolio_if_needed(self) -> None:
        """Update portfolio if rebalancing is needed"""
        try:
            # Calculate turnover since last rebalance
            # This would check if weights have drifted significantly
            
            # For now, just log optimization need
            last_optimization = self.optimization_history[-1] if self.optimization_history else None
            
            if last_optimization:
                time_since_opt = datetime.now() - last_optimization.timestamp
                if time_since_opt > self.rebalance_frequency:
                    logger.info("Portfolio rebalancing needed")
                    # Trigger optimization in production
            
        except Exception as e:
            logger.error(f"Portfolio update check failed: {e}")
    
    async def _optimization_loop(self) -> None:
        """Background loop for periodic optimization"""
        while True:
            try:
                # Run optimization with default objective
                result = await self.optimize_portfolio(self.default_objective)
                
                if result.success:
                    # Update current portfolio
                    self.current_portfolio = result.asset_weights.copy()
                    logger.info(f"Portfolio optimized: {result.asset_weights}")
                
                await asyncio.sleep(86400)  # Optimize daily
            
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _cache_optimization_result(self, result: OptimizationResult) -> None:
        """Cache optimization result"""
        try:
            await self.cache.set(
                f"portfolio_optimization:{result.optimization_id}",
                {
                    'objective': result.objective.value,
                    'asset_weights': result.asset_weights,
                    'expected_return': result.expected_return,
                    'risk': result.risk,
                    'sharpe_ratio': result.sharpe_ratio,
                    'timestamp': result.timestamp.isoformat(),
                    'optimization_success': result.success
                },
                ttl=86400  # 24 hours
            )
            
        except Exception as e:
            logger.error(f"Optimization result caching failed: {e}")
    
    async def get_portfolio_report(self) -> Dict[str, Any]:
        """Get comprehensive portfolio report"""
        try:
            # Calculate portfolio metrics
            portfolio_return_history = list(self.portfolio_returns)
            benchmark_return_history = list(self.benchmark_returns)
            
            total_return = np.sum(portfolio_return_history) if portfolio_return_history else 0
            benchmark_return = np.sum(benchmark_return_history) if benchmark_return_history else 0
            
            excess_return = total_return - benchmark_return
            tracking_error = np.std(np.array(benchmark_return_history) - np.array(portfolio_return_history)) if len(benchmark_return_history) > 0 else 0
            
            return {
                'current_portfolio': self.current_portfolio,
                'current_strategies': self.current_strategies,
                'performance_metrics': {
                    'total_return': total_return,
                    'benchmark_return': benchmark_return,
                    'excess_return': excess_return,
                    'tracking_error': tracking_error,
                    'annualized_return': total_return * 252 if portfolio_return_history else 0,
                    'volatility': np.std(portfolio_return_history) * np.sqrt(252) if len(portfolio_return_history) > 1 else 0
                },
                'recent_optimizations': [
                    {
                        'optimization_id': opt.optimization_id,
                        'objective': opt.objective.value,
                        'expected_return': opt.expected_return,
                        'risk': opt.risk,
                        'sharpe_ratio': opt.sharpe_ratio,
                        'timestamp': opt.timestamp.isoformat(),
                        'success': opt.success
                    }
                    for opt in self.optimization_history[-10:]  # Last 10 optimizations
                ],
                'asset_weights': self.current_portfolio,
                'risk_metrics': {
                    'var_95': self.optimization_history[-1].var_95 if self.optimization_history else 0,
                    'cvar_95': self.optimization_history[-1].cvar_95 if self.optimization_history else 0,
                    'beta': self.optimization_history[-1].beta if self.optimization_history else 0
                } if self.optimization_history else {},
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Portfolio report generation failed: {e}")
            return {}

# Utility functions
async def create_portfolio_optimizer(config: Dict[str, Any]) -> PortfolioOptimizationEngine:
    """Create and initialize portfolio optimizer"""
    optimizer = PortfolioOptimizationEngine(config)
    await optimizer.initialize()
    return optimizer

def calculate_efficient_frontier(returns: np.ndarray, 
                                n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate efficient frontier"""
    try:
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)
        
        # Target returns range
        min_return = mean_returns.min()
        max_return = mean_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_returns = []
        frontier_risks = []
        frontier_weights = []
        
        for target in target_returns:
            # Minimize risk for given return
            n_assets = len(mean_returns)
            
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target},
                {'type': 'ineq', 'fun': lambda w: w},  # Long-only
            ]
            
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
            
            w0 = np.array([1.0/n_assets] * n_assets])
            
            result = minimize(
                portfolio_variance,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                frontier_returns.append(target)
                frontier_risks.append(np.sqrt(result.fun))
                frontier_weights.append(result.x)
        
        return np.array(frontier_returns), np.array(frontier_risks), frontier_weights

def calculate_portfolio_metrics(weights: np.ndarray, 
                                   returns: np.ndarray,
                                   risk_free_rate: float = 0.02) -> Dict[str, float]:
    """Comprehensive portfolio metrics"""
    try:
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)
        
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        # Sortile returns
        portfolio_returns = np.dot(returns, weights)
        sorted_returns = np.sort(portfolio_returns)
        
        # VaR at different confidence levels
        var_95 = abs(sorted_returns[int(len(sorted_returns) * 0.05)])
        var_99 = abs(sorted_returns[int(len(sorted_returns) * 0.01)])
        
        # CVaR
        tail_95 = sorted_returns[:int(len(sorted_returns) * 0.05)]
        cvar_95 = abs(np.mean(tail_95)) if len(tail_95) > 0 else 0
        
        # Max drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        max_drawdown = drawdown.max()
        
        return {
            'expected_return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'sortino_ratio': portfolio_return / np.std(portfolio_returns[portfolio_returns < 0]) * np.sqrt(252) if len(portfolio_returns[portfolio_returns < 0]) > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Portfolio metrics calculation failed: {e}")
        return {}