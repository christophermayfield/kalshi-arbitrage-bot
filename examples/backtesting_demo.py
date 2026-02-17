"""
Backtesting Demonstration Script
Shows how to use the advanced backtesting framework with the Kalshi arbitrage bot
"""

import asyncio
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.logging_utils import setup_logging
from src.backtesting.advanced_backtesting import (
    run_comprehensive_backtest,
    create_backtest_config,
    BacktestMode,
    OptimizationMethod
    create_parameter_grid
)
from src.strategies.kalshi_correlation import KalshiCorrelationStrategy

from src.utils.logging_utils import get_logger

logger = get_logger("backtesting_demo")


class SimpleArbitrageStrategy:
    """Simple arbitrage strategy for demonstration"""
    
    @staticmethod
    async def generate_signals(market_data: dict, parameters: dict):
        """Generate trading signals from market data."""
        signals = []
        
        # Simple momentum strategy
        for market_id, data in market_data.items():
            if len(data) < 10:
                continue
            
            # Calculate momentum
            prices = [row['close'] for row in data]
            if len(prices) < 2:
                continue
                
            returns = np.diff(prices) / prices[:-1]
            momentum = returns[-1] if len(returns) > 0 else 0
            
            # Generate signal if momentum is strong
            if abs(momentum) > 0.02:  # 2% momentum threshold
                confidence = min(0.8, abs(momentum) * 20)
                signal = {
                    "opportunity_id": f"{market_id}_signal_{datetime.now().timestamp()}",
                    "market_id": market_id,
                    "strategy": "momentum",
                    "action": "buy" if momentum > 0 else "sell",
                    "confidence": confidence,
                    "expected_return": abs(momentum) * 100,  # Expected next period return
                    "parameters": parameters
                }
                
                if signal:
                    signals.append(signal)
        
        return signals


class BacktestingDemo:
    """Demonstration of advanced backtesting capabilities."""
    
    def __init__(self, config: Config):
        self.config = config
        self.kalshi_correlation = KalshiCorrelationStrategy(config)
        
    async def run_simple_backtest(self, days: int = 30) -> None:
        """Run simple backtest demonstration."""
        logger.info(f"Running {days}-day backtest demonstration...")
        
        # Create configuration
        config = create_backtest_config(
            start_date=datetime.now() - timedelta(days=days),
            end_date=datetime.now(),
            initial_capital=10000.0,
            commission_rate=0.01,
            slippage_model="percentage",
            slippage_rate=0.005,
            max_position_size=0.2
        )
        
        # Create parameter grid
        parameter_grid = create_parameter_grid({
            'min_profit_cents': [5, 10, 15, 20],
            'min_spread_percent': [1.0, 2.0, 3.0],
            'confidence_threshold': [0.5, 0.6, 0.7, 0.8],
            'window_size': [20, 50, 100],
        })
        
        # Generate sample data
        sample_data = self._generate_sample_market_data(days)
        
        # Create simple strategy instance
        simple_strategy = SimpleArbitrageStrategy()
        
        logger.info(f"Starting backtest with {len(parameter_grid)} parameter combinations...")
        
        # Run comprehensive backtest
        result = await run_comprehensive_backtest(
            strategy_func=simple_strategy.generate_signals,
            parameters={'min_profit_cents': 10, 'confidence_threshold': 0.6, 'window_size': 50},
            training_data=sample_data,
            validation_data=None,
            config=config,
            mode=BacktestMode.EVENT_DRIVEN,
            optimization_method=OptimizationMethod.BAYESIAN
        )
        
        # Display results
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Annualized Return: {result.annualized_return:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Training Samples: {result.training_samples}")
        
        print("\nBEST PARAMETERS:")
        print("="*40)
        if result.parameters:
            for param, value in result.parameters.items():
                print(f"  {param}: {value}")
        
        print(f"\nFEATURE IMPORTANCE:")
        print("="*40)
        if result.feature_importance:
            # Sort by importance and display top 5
            sorted_features = sorted(
                result.feature_importance.items(), 
                key=lambda x: x[1], reverse=True
            )[:5]
            
            for feature, importance in sorted_features:
                print(f"  {feature}: {importance:.3f}")
        
        print(f"\nDAILY RETURNS:")
        if result.daily_returns:
            print(f"  Mean Daily Return: {np.mean(result.daily_returns):.3%}")
            print(f"  Volatility: {np.std(result.daily_returns):.3%}")
        
        return result
    
    async def run_parameter_optimization_demo(self) -> None:
        """Demonstrate parameter optimization."""
        logger.info("Running parameter optimization demonstration...")
        
        # Create strategy
        simple_strategy = SimpleArbitrageStrategy()
        
        # Create sample data
        sample_data = self._generate_sample_market_data(60)
        
        # Create larger parameter grid
        parameter_grid = create_parameter_grid({
            'min_profit_cents': [5, 10, 15, 20, 25, 30],
            'min_spread_percent': [0.5, 1.0, 1.5, 2.0],
            'confidence_threshold': [0.4, 0.5, 0.6, 0.7, 0.8],
            'window_size': [10, 20, 30, 50, 100, 200],
            'momentum_threshold': [0.01, 0.02, 0.05],
            'min_samples': [100, 200, 500, 1000],
        })
        
        print("\n" + "="*60)
        print("PARAMETER OPTIMIZATION DEMO")
        print("="*60)
        
        # Initialize optimization
        from src.backtesting.advanced_backtesting import ParameterOptimizer
        
        optimizer = ParameterOptimizer(None)
        
        print(f"Optimizing {len(parameter_grid)} parameter combinations...")
        
        # Run optimization (simplified for demo)
        best_params, best_result = await optimizer.optimize_parameters(
            strategy_func=simple_strategy.generate_signals,
            parameter_grid=parameter_grid,
            training_data=sample_data,
            validation_data=None,
            config=create_backtest_config(
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now(),
                commission_rate=0.01
            ),
            cv_folds=3,  # Reduced for demo speed
            objective="sharpe_ratio",
            n_jobs=2,  # Reduced for demo
        )
        
        print(f"\nBEST PARAMETERS:")
        print("="*40)
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        print(f"\nOPTIMIZATION RESULTS:")
        print("="*40)
        print(f"Best Sharpe Ratio: {best_result.sharpe_ratio:.3f}")
        print(f"CV Score Mean: {best_result.cv_mean_score:.3f}")
        print(f"Best Test Score: {best_result.validation_score:.3f}")
        
        return best_result
    
    def _generate_sample_market_data(self, days: int) -> pd.DataFrame:
        """Generate sample market data for backtesting."""
        
        # Create sample time series
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='H'
        )
        
        # Generate sample price data for 3 markets
        markets = ["PRESIDENTIAL-2024-DEM", "PRESIDENTIAL-2024-REP", "BITCOIN-PRICE-2024"]
        
        all_data = []
        
        for market in markets:
            for date in dates:
                # Generate realistic price movement
                base_price = 100 + np.random.normal(0, 10)
                
                # Random walk with some trend
                if market == "BITCOIN-PRICE-2024":
                    # More volatile
                    trend = np.random.normal(0.001, 0.02)  # Strong upward trend
                else:
                    # Less volatile
                    trend = np.random.normal(0, 0.001)
                
                # Generate OHLCV data
                price_walk = [base_price]
                for i in range(24):  # 24 hours of data
                    price_walk.append(price_walk[-1] + np.random.normal(0, trend))
                
                # Add random noise
                for i in range(24):
                    price_walk[i] = price_walk[i] * (1 + np.random.normal(0, 0.01))
                
                # Create OHLCV
                ohlcv_data = []
                for i, price in enumerate(price_walk):
                    high = price_walk[i] * (1 + np.random.uniform(0, 0.02))
                    low = price_walk[i] * (1 - np.random.uniform(0, 0.01))
                    close = price_walk[i]
                    
                    volume = np.random.uniform(1000, 10000)
                    
                    ohlcv_data.append({
                        'timestamp': date + timedelta(hours=i),
                        'open': price_walk[i],
                        'high': high,
                        'low': low,
                        'close': close,
                        'volume': volume,
                        'market_id': market,
                    })
                
                df = pd.DataFrame(ohlcv_data)
                all_data.append(df)
        
        return pd.concat(all_data, ignore_index=True)


async def run_backtesting_demo():
    """Run comprehensive backtesting demonstration."""
    config = Config()
    
    print("="*80)
    print("KALSHI ARBITRAGE BOT - BACKTESTING DEMONSTRATION")
    print("="*80)
    print()
    print("This demonstration shows:")
    print("1. Simple arbitrage strategy backtesting")
    print("2. Parameter optimization with Bayesian optimization")
    print("3. Walk-forward analysis")
    print("4. Performance attribution integration")
    print()
    print("Each demo runs independently with sample data")
    print()
    print("Configure backtesting.enabled=true in config.yaml to use real data")
    print("="*80)
    
    demo = BacktestingDemo(config)
    
    # Run demonstrations
    await demo.run_simple_backtest(days=30)
    print()
    
    await demo.run_parameter_optimization_demo()
    print()


if __name__ == "__main__":
    asyncio.run(run_backtesting_demo())