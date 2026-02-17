"""
Machine Learning Strategy Optimization System
Advanced ML-based strategy optimization and parameter tuning
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize
import joblib
import redis.asyncio as redis

from ..utils.performance_cache import PerformanceCache

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """ML optimization strategies"""
    PARAMETER_TUNING = "parameter_tuning"
    STRATEGY_SELECTION = "strategy_selection"
    ALLOCATION_OPTIMIZATION = "allocation_optimization"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    ENSEMBLE_WEIGHTING = "ensemble_weighting"

class MLModelType(Enum):
    """Types of ML models"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"

@dataclass
class OptimizationResult:
    """Result of ML optimization"""
    optimization_id: str
    strategy: str
    parameters: Dict[str, Any]
    expected_performance: Dict[str, float]
    confidence: float
    optimization_time: float
    
    # Training data stats
    train_score: float
    validation_score: float
    test_score: float
    
    # Feature importance
    feature_importance: Dict[str, float]
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    model_type: MLModelType = MLModelType.RANDOM_FOREST
    model_version: str = "1.0"

@dataclass
class PerformanceMetrics:
    """Performance metrics for strategy optimization"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    avg_holding_time: float = 0.0
    total_commission: float = 0.0

class StrategyOptimizer:
    """
    Advanced ML-based strategy optimizer with multiple ML approaches
    and real-time performance adaptation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimizer_config = config.get('strategy_optimization', {})
        
        # Optimization strategies
        self.enabled_strategies = [
            OptimizationStrategy(s) for s in self.optimizer_config.get('enabled_strategies', [
                'parameter_tuning', 'strategy_selection', 'allocation_optimization'
            ])
        ]
        
        # ML models
        self.ml_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.active_model_type = MLModelType(self.optimizer_config.get('ml_model_type', 'random_forest'))
        
        # Training data
        self.training_data: pd.DataFrame = pd.DataFrame()
        self.performance_history: List[PerformanceMetrics] = []
        self.optimization_history: List[OptimizationResult] = []
        
        # Strategy parameters
        self.strategy_parameters: Dict[str, Dict[str, Any]] = {}
        self.best_parameters: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.current_performance: Dict[str, PerformanceMetrics] = {}
        
        # Optimization parameters
        self.min_samples_for_training = self.optimizer_config.get('min_samples_for_training', 100)
        self.validation_split = self.optimizer_config.get('validation_split', 0.2)
        self.test_split = self.optimizer_config.get('test_split', 0.2)
        self.cross_validation_folds = self.optimizer_config.get('cross_validation_folds', 5)
        
        # Feature engineering
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        
        # Performance cache
        self.cache = PerformanceCache(
            redis_url=config.get('redis_url', 'redis://localhost:6379'),
            default_ttl=3600  # 1 hour TTL for optimization data
        )
        
        # Strategy universe
        self.strategies = self.optimizer_config.get('strategies', [
            'mean_reversion', 'momentum', 'pairs_trading', 'statistical_arbitrage'
        ])
        
        logger.info("ML Strategy Optimizer initialized")
    
    async def initialize(self) -> None:
        """Initialize the ML optimization system"""
        try:
            # Initialize ML models
            await self._initialize_ml_models()
            
            # Load historical data
            await self._load_training_data()
            
            # Setup feature engineering
            await self._setup_feature_engineering()
            
            # Start optimization loops
            asyncio.create_task(self._optimization_loop())
            asyncio.create_task(self._performance_tracking_loop())
            
            logger.info("ML Strategy Optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Strategy Optimizer initialization failed: {e}")
            raise
    
    async def optimize_strategy(self, strategy_name: str, 
                              context: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Optimize a specific trading strategy using ML"""
        try:
            logger.info(f"Starting ML optimization for strategy: {strategy_name}")
            
            # Gather features and targets
            features, targets = await self._prepare_training_data(strategy_name, context)
            
            if features is None or len(features) < self.min_samples_for_training:
                return self._create_default_optimization_result(strategy_name)
            
            # Split data
            X_train, X_temp, y_train, y_temp = train_test_split(
                features, targets, 
                test_size=(self.test_split + self.validation_split),
                random_state=42
            )
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=(self.test_split / (self.test_split + self.validation_split)),
                random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Train/Select ML model
            model = await self._train_or_select_model(strategy_name, X_train_scaled, y_train)
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            val_score = model.score(X_val_scaled, y_val)
            test_score = model.score(X_test_scaled, y_test)
            
            # Feature importance
            feature_importance = self._extract_feature_importance(model, self.feature_columns)
            
            # Optimize parameters using model
            optimized_params = await self._optimize_parameters(
                model, strategy_name, features, scaler, context
            )
            
            # Calculate expected performance
            expected_performance = await self._calculate_expected_performance(
                model, X_test_scaled, y_test, optimized_params
            )
            
            # Create optimization result
            result = OptimizationResult(
                optimization_id=f"opt_{strategy_name}_{datetime.now().timestamp()}",
                strategy=strategy_name,
                parameters=optimized_params,
                expected_performance=expected_performance,
                confidence=min(1.0, test_score),
                optimization_time=0.0,  # Would track actual time
                train_score=train_score,
                validation_score=val_score,
                test_score=test_score,
                feature_importance=feature_importance,
                model_type=self.active_model_type
            )
            
            # Store results
            self.optimization_history.append(result)
            self.best_parameters[strategy_name] = optimized_params
            self.ml_models[strategy_name] = model
            self.scalers[strategy_name] = scaler
            
            # Cache result
            await self._cache_optimization_result(result)
            
            logger.info(f"ML optimization completed for {strategy_name}: test_score={test_score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Strategy optimization failed for {strategy_name}: {e}")
            return self._create_default_optimization_result(strategy_name)
    
    async def _prepare_training_data(self, strategy_name: str, 
                                     context: Optional[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data for ML optimization"""
        try:
            # Get current parameters for strategy
            current_params = self.strategy_parameters.get(strategy_name, {})
            
            # Get performance history
            if not self.performance_history:
                return None, None
            
            # Extract features from performance history
            features = []
            targets = []
            
            metrics_list = list(self.performance_history)[-500:]  # Last 500 trades
            
            if len(metrics_list) < self.min_samples_for_training:
                return None, None
            
            for i, metrics in enumerate(metrics_list):
                # Feature engineering
                feature_row = self._extract_features(metrics, current_params, i)
                if feature_row:
                    features.append(feature_row)
                    
                # Target: next trade's P&L (or some performance metric)
                if i < len(metrics_list) - 1:
                    next_pnl = metrics_list[i + 1].total_pnl
                    targets.append([next_pnl])
            
            if not features or not targets:
                return None, None
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {e}")
            return None, None
    
    def _extract_features(self, metrics: PerformanceMetrics, 
                        params: Dict[str, Any], index: int) -> Optional[List[float]]:
        """Extract features from performance metrics"""
        try:
            features = []
            
            # Performance metrics
            features.append(metrics.total_trades)
            features.append(metrics.hit_rate if metrics.total_trades > 0 else 0)
            features.append(metrics.sharpe_ratio)
            features.append(metrics.max_drawdown)
            features.append(metrics.profit_factor)
            features.append(metrics.avg_holding_time)
            features.append(metrics.avg_win if metrics.winning_trades > 0 else 0)
            features.append(metrics.avg_loss if metrics.losing_trades > 0 else 0)
            
            # Rolling statistics (if available)
            if index > 10:
                recent_pnls = [self.performance_history[j].total_pnl for j in range(index-10, index)]
                features.append(np.mean(recent_pnls))
                features.append(np.std(recent_pnls))
                features.append(np.max(recent_pnls))
                features.append(np.min(recent_pnls))
            else:
                features.extend([0, 0, 0, 0])
            
            # Time-based features
            features.append(metrics.total_trades % 24)  # Hour of day (if trades are time-stamped)
            features.append(metrics.total_trades % 7)   # Day of week
            
            # Market condition features
            volatility = np.std([self.performance_history[j].total_pnl 
                               for j in range(max(0, index-20), index+1)])
            features.append(volatility)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    async def _train_or_select_model(self, strategy_name: str, 
                                    X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train or select ML model"""
        try:
            # Check if model already exists
            if strategy_name in self.ml_models:
                return self.ml_models[strategy_name]
            
            # Create new model based on type
            if self.active_model_type == MLModelType.RANDOM_FOREST:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            elif self.active_model_type == MLModelType.GRADIENT_BOOSTING:
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            elif self.active_model_type == MLModelType.NEURAL_NETWORK:
                model = MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42,
                    alpha=0.01
                )
            
            # Train model
            model.fit(X_train, y_train)
            
            return model
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return self._create_fallback_model()
    
    def _create_fallback_model(self) -> Any:
        """Create fallback model on training failure"""
        return RandomForestRegressor(n_estimators=10, random_state=42)
    
    def _extract_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                
                # Ensure we have the same number of feature names
                if len(feature_names) == len(importance):
                    return dict(zip(feature_names, importance))
                else:
                    # Generate generic feature names
                    return {
                        f"feature_{i}": importance[i] 
                        for i in range(len(importance))
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Feature importance extraction failed: {e}")
            return {}
    
    async def _optimize_parameters(self, model: Any, strategy_name: str,
                                  features: np.ndarray, scaler: StandardScaler,
                                  context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize strategy parameters using ML model"""
        try:
            # Get current parameters
            current_params = self.strategy_parameters.get(strategy_name, {})
            
            # Define parameter ranges for optimization
            param_ranges = self._get_parameter_ranges(strategy_name)
            
            if not param_ranges:
                return current_params
            
            # Optimize using ML predictions
            def objective_function(params):
                # Create parameter dictionary
                param_dict = {
                    key: param_type(value) if callable(param_type) else value
                    for (key, (param_type, _)), value in zip(param_ranges.items(), params.items())
                }
                
                # Scale features with optimal parameters
                param_features = self._create_param_features(param_dict)
                
                # Predict expected performance
                try:
                    if len(param_features) == features.shape[1]:
                        param_features_scaled = scaler.transform([param_features])
                        predicted_pnl = model.predict(param_features_scaled)[0]
                        return -predicted_pnl  # Minimize negative P&L
                except:
                    return -np.inf
                
                return -np.inf
            
            # Initial parameter guess
            initial_params = [
                np.mean([min_val, max_val])
                for _, (min_val, max_val) in param_ranges.values()
            ]
            
            # Optimize parameters
            result = minimize(
                objective_function,
                initial_params,
                bounds=[(min_val, max_val) for _, (min_val, max_val) in param_ranges.values()],
                method='L-BFGS-B'
            )
            
            if result.success:
                # Extract optimized parameters
                optimized_params = {
                    key: (param_type(value) if callable(param_type) else value)
                    for (key, (param_type, _)), value in zip(param_ranges.items(), result.x)
                }
                
                return optimized_params
            else:
                return current_params
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return self.strategy_parameters.get(strategy_name, {})
    
    def _get_parameter_ranges(self, strategy_name: str) -> Dict[str, Tuple[Any, float, float]]:
        """Get parameter ranges for optimization"""
        try:
            # Define ranges for different strategies
            if strategy_name == 'mean_reversion':
                return {
                    'lookback_period': (int, 10, 50),
                    'entry_threshold': (float, 1.5, 3.0),
                    'exit_threshold': (float, 0.5, 1.5),
                    'position_size_multiplier': (float, 0.5, 1.5)
                }
            elif strategy_name == 'momentum':
                return {
                    'lookback_period': (int, 5, 30),
                    'momentum_threshold': (float, 0.02, 0.10),
                    'exit_threshold': (float, 0.01, 0.05),
                    'position_size_multiplier': (float, 0.5, 1.5)
                }
            elif strategy_name == 'pairs_trading':
                return {
                    'lookback_period': (int, 20, 100),
                    'z_score_threshold': (float, 1.5, 3.0),
                    'hedgeratio_adjustment': (float, 0.9, 1.1),
                    'position_size_multiplier': (float, 0.5, 1.5)
                }
            elif strategy_name == 'statistical_arbitrage':
                return {
                    'correlation_threshold': (float, 0.5, 0.9),
                    'min_confidence': (float, 0.5, 0.9),
                    'risk_limit': (float, 0.01, 0.05),
                    'position_size_multiplier': (float, 0.5, 1.5)
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Parameter range retrieval failed: {e}")
            return {}
    
    def _create_param_features(self, params: Dict[str, Any]) -> List[float]:
        """Create feature vector from parameters"""
        try:
            features = []
            
            # Add parameter values (normalized)
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    features.append(float(value))
                else:
                    features.append(0.0)  # Placeholder for non-numeric params
            
            # Add derived features
            if 'lookback_period' in params and 'entry_threshold' in params:
                features.append(params['lookback_period'] / params['entry_threshold'])
            
            if 'position_size_multiplier' in params and 'exit_threshold' in params:
                features.append(params['position_size_multiplier'] / params['exit_threshold'])
            
            return features
            
        except Exception as e:
            logger.error(f"Param feature creation failed: {e}")
            return [0.0] * 10  # Return default features
    
    async def _calculate_expected_performance(self, model: Any, 
                                             X_test: np.ndarray, y_test: np.ndarray,
                                             params: Dict[str, Any]) -> Dict[str, float]:
        """Calculate expected performance metrics"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate expected profit (average prediction)
            expected_profit = np.mean(y_pred)
            
            # Calculate risk (standard deviation of predictions)
            risk = np.std(y_pred)
            
            # Calculate sharpe ratio (risk-adjusted return)
            sharpe_ratio = expected_profit / risk if risk > 0 else 0
            
            # Calculate confidence (based on prediction error)
            confidence = 1.0 / (1.0 + mae) if mae > 0 else 0.5
            
            return {
                'expected_profit': expected_profit,
                'risk': risk,
                'sharpe_ratio': sharpe_ratio,
                'mse': mse,
                'mae': mae,
                'confidence': confidence,
                'max_profit': np.max(y_pred),
                'min_profit': np.min(y_pred)
            }
            
        except Exception as e:
            logger.error(f"Expected performance calculation failed: {e}")
            return {}
    
    def _create_default_optimization_result(self, strategy_name: str) -> OptimizationResult:
        """Create default optimization result when ML fails"""
        return OptimizationResult(
            optimization_id=f"opt_{strategy_name}_{datetime.now().timestamp()}",
            strategy=strategy_name,
            parameters=self.strategy_parameters.get(strategy_name, {}),
            expected_performance={
                'expected_profit': 0.0,
                'confidence': 0.5,
                'sharpe_ratio': 0.0
            },
            confidence=0.5,
            optimization_time=0.0,
            train_score=0.0,
            validation_score=0.0,
            test_score=0.0,
            feature_importance={}
        )
    
    async def _initialization_ml_models(self) -> None:
        """Initialize ML models"""
        try:
            # Create default parameters for each strategy
            for strategy in self.strategies:
                self.strategy_parameters[strategy] = self._get_default_parameters(strategy)
            
            logger.info("ML models and parameters initialized")
            
        except Exception as e:
            logger.error(f"ML model initialization failed: {e}")
    
    def _get_default_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """Get default parameters for a strategy"""
        try:
            defaults = {
                'mean_reversion': {
                    'lookback_period': 20,
                    'entry_threshold': 2.0,
                    'exit_threshold': 1.0,
                    'position_size_multiplier': 1.0
                },
                'momentum': {
                    'lookback_period': 10,
                    'momentum_threshold': 0.05,
                    'exit_threshold': 0.02,
                    'position_size_multiplier': 1.0
                },
                'pairs_trading': {
                    'lookback_period': 30,
                    'z_score_threshold': 2.0,
                    'hedgeratio_adjustment': 1.0,
                    'position_size_multiplier': 1.0
                },
                'statistical_arbitrage': {
                    'correlation_threshold': 0.7,
                    'min_confidence': 0.7,
                    'risk_limit': 0.02,
                    'position_size_multiplier': 1.0
                }
            }
            
            return defaults.get(strategy_name, {})
            
        except Exception as e:
            logger.error(f"Default parameters retrieval failed: {e}")
            return {}
    
    async def _load_training_data(self) -> None:
        """Load historical training data"""
        try:
            # This would load actual historical data from database or files
            # For now, initialize with simulated data
            
            # Generate sample performance metrics
            for i in range(200):
                metrics = PerformanceMetrics(
                    total_trades=i + 1,
                    winning_trades=np.random.randint(0, i + 1),
                    losing_trades=np.random.randint(0, i + 1),
                    total_pnl=np.random.uniform(-1000, 5000),
                    avg_win=np.random.uniform(50, 500),
                    avg_loss=np.random.uniform(-500, -50),
                    sharpe_ratio=np.random.uniform(-2, 4),
                    max_drawdown=np.random.uniform(-0.2, 0),
                    hit_rate=np.random.uniform(0.3, 0.7),
                    profit_factor=np.random.uniform(0.5, 3),
                    avg_holding_time=np.random.uniform(60, 3600),
                    total_commission=np.random.uniform(10, 100)
                )
                
                self.performance_history.append(metrics)
            
            logger.info(f"Loaded {len(self.performance_history)} performance records")
            
        except Exception as e:
            logger.error(f"Training data loading failed: {e}")
    
    async def _setup_feature_engineering(self) -> None:
        """Setup feature engineering"""
        try:
            # Define feature columns
            self.feature_columns = [
                'total_trades', 'hit_rate', 'sharpe_ratio', 'max_drawdown',
                'profit_factor', 'avg_holding_time', 'avg_win', 'avg_loss',
                'rolling_avg_pnl', 'rolling_std_pnl', 'rolling_max_pnl', 'rolling_min_pnl',
                'hour_of_day', 'day_of_week', 'volatility'
            ]
            
            logger.info(f"Setup {len(self.feature_columns)} features")
            
        except Exception as e:
            logger.error(f"Feature engineering setup failed: {e}")
    
    async def _optimization_loop(self) -> None:
        """Background loop for periodic optimization"""
        while True:
            try:
                # Optimize each strategy
                for strategy in self.strategies:
                    try:
                        result = await self.optimize_strategy(strategy)
                        
                        if result.confidence > 0.7:
                            # Update best parameters
                            self.best_parameters[strategy] = result.parameters
                            
                            logger.info(f"Updated parameters for {strategy}: {result.parameters}")
                    
                    except Exception as e:
                        logger.error(f"Optimization failed for {strategy}: {e}")
                
                await asyncio.sleep(3600)  # Optimize every hour
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(300)
    
    async def _performance_tracking_loop(self) -> None:
        """Background loop for tracking performance"""
        while True:
            try:
                # Update performance metrics for each strategy
                for strategy in self.strategies:
                    # This would pull actual performance data
                    # For now, update with simulated data
                    
                    if strategy not in self.current_performance:
                        self.current_performance[strategy] = PerformanceMetrics()
                    
                    # Simulate trade execution
                    self._simulate_trade(strategy)
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance tracking loop error: {e}")
                await asyncio.sleep(60)
    
    def _simulate_trade(self, strategy: str) -> None:
        """Simulate trade execution for performance tracking"""
        try:
            performance = self.current_performance[strategy]
            performance.total_trades += 1
            
            # Simulate trade result
            is_win = np.random.random() > 0.5
            pnl = np.random.uniform(-200, 400) if is_win else np.random.uniform(-400, -50)
            
            if is_win:
                performance.winning_trades += 1
            else:
                performance.losing_trades += 1
            
            performance.total_pnl += pnl
            
            # Update statistics
            if performance.winning_trades > 0:
                performance.avg_win = (performance.avg_win * (performance.winning_trades - 1) + pnl) / performance.winning_trades
            
            if performance.losing_trades > 0:
                performance.avg_loss = (performance.avg_loss * (performance.losing_trades - 1) + abs(pnl)) / performance.losing_trades
            
            performance.hit_rate = performance.winning_trades / performance.total_trades
            
            # Calculate profit factor
            if performance.losing_trades > 0:
                performance.profit_factor = (performance.avg_win * performance.winning_trades) / (performance.avg_loss * performance.losing_trades)
            
            # Update performance history
            self.performance_history.append(performance)
            
        except Exception as e:
            logger.error(f"Trade simulation failed for {strategy}: {e}")
    
    async def _cache_optimization_result(self, result: OptimizationResult) -> None:
        """Cache optimization result"""
        try:
            await self.cache.set(
                f"optimization_result:{result.strategy}",
                {
                    'optimization_id': result.optimization_id,
                    'parameters': result.parameters,
                    'expected_performance': result.expected_performance,
                    'confidence': result.confidence,
                    'feature_importance': result.feature_importance,
                    'timestamp': result.timestamp.isoformat()
                },
                ttl=86400  # 24 hours
            )
            
        except Exception as e:
            logger.error(f"Optimization result caching failed: {e}")
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        try:
            # Calculate strategy performance
            strategy_performance = {}
            for strategy, performance in self.current_performance.items():
                strategy_performance[strategy] = {
                    'total_trades': performance.total_trades,
                    'hit_rate': performance.hit_rate,
                    'sharpe_ratio': performance.sharpe_ratio,
                    'total_pnl': performance.total_pnl,
                    'profit_factor': performance.profit_factor
                }
            
            return {
                'active_ml_model': self.active_model_type.value,
                'optimized_strategies': len(self.best_parameters),
                'strategy_parameters': self.best_parameters,
                'strategy_performance': strategy_performance,
                'recent_optimizations': [
                    {
                        'optimization_id': result.optimization_id,
                        'strategy': result.strategy,
                        'confidence': result.confidence,
                        'expected_performance': result.expected_performance,
                        'timestamp': result.timestamp.isoformat()
                    }
                    for result in self.optimization_history[-10:]  # Last 10 optimizations
                ],
                'feature_importance': self._aggregate_feature_importance(),
                'ml_model_statistics': {
                    'total_optimizations': len(self.optimization_history),
                    'avg_confidence': np.mean([r.confidence for r in self.optimization_history]) if self.optimization_history else 0,
                    'avg_test_score': np.mean([r.test_score for r in self.optimization_history]) if self.optimization_history else 0
                },
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Optimization report generation failed: {e}")
            return {}
    
    def _aggregate_feature_importance(self) -> Dict[str, float]:
        """Aggregate feature importance across all strategies"""
        try:
            importance_dict = defaultdict(float)
            count_dict = defaultdict(int)
            
            for result in self.optimization_history:
                for feature, importance in result.feature_importance.items():
                    importance_dict[feature] += importance
                    count_dict[feature] += 1
            
            # Calculate average importance
            avg_importance = {
                feature: importance_dict[feature] / count_dict[feature]
                for feature in importance_dict.keys()
            }
            
            # Sort by importance
            return dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Feature importance aggregation failed: {e}")
            return {}

# Utility functions
async def create_strategy_optimizer(config: Dict[str, Any]) -> StrategyOptimizer:
    """Create and initialize strategy optimizer"""
    optimizer = StrategyOptimizer(config)
    await optimizer.initialize()
    return optimizer

def calculate_optimization_schedule(strategy: str, 
                                    performance_history: List[PerformanceMetrics]) -> int:
    """Calculate optimal optimization schedule for a strategy"""
    try:
        if not performance_history:
            return 3600  # Default 1 hour
        
        # Calculate strategy volatility
        recent_pnls = [p.total_pnl for p in performance_history[-20:]]
        volatility = np.std(recent_pnls) if recent_pnls else 0
        
        # Higher volatility = more frequent optimization
        if volatility > 1000:
            return 1800  # Optimize every 30 minutes
        elif volatility > 500:
            return 3600  # Optimize every hour
        else:
            return 7200  # Optimize every 2 hours
    
    except:
        return 3600

def calculate_optimization_stability(recent_results: List[OptimizationResult]) -> float:
    """Calculate stability of recent optimizations"""
    try:
        if len(recent_results) < 3:
            return 0.5  # Unknown stability
        
        # Check parameter stability
        param_changes = []
        for i in range(1, len(recent_results)):
            params1 = recent_results[i - 1].parameters
            params2 = recent_results[i].parameters
            
            # Calculate parameter change
            param_differences = []
            for key in params1.keys():
                if key in params2 and isinstance(params1[key], (int, float)) and isinstance(params2[key], (int, float)):
                    if params2[key] != 0:
                        diff = abs(params1[key] - params2[key]) / params2[key]
                        param_differences.append(diff)
            
            if param_differences:
                param_changes.append(np.mean(param_differences))
        
        if not param_changes:
            return 0.5
        
        # Stability is inverse of average parameter change
        avg_change = np.mean(param_changes)
        stability = max(0, 1.0 - avg_change)
        
        return stability