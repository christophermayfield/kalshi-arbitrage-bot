"""
Example strategy demonstrating the advanced backtesting framework
Simple moving average crossover strategy with parameter optimization
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime

from src.backtesting.advanced_backtesting import (
    run_comprehensive_backtest,
    create_parameter_grid,
)
from src.utils.logging_utils import get_logger
from src.utils.config import Config

logger = get_logger("example_strategy")


class MovingAverageStrategy:
    """Example moving average crossover strategy"""

    def __init__(self):
        self.position = 0
        self.entry_price = 0

    @staticmethod
    async def generate_signals(
        market_data: Dict[str, pd.DataFrame], parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on moving average crossover"""
        signals = []

        try:
            # Extract parameters
            fast_period = parameters.get("fast_ma", 10)
            slow_period = parameters.get("slow_ma", 20)
            threshold = parameters.get("threshold", 0.01)

            for market_id, data in market_data.items():
                if len(data) < slow_period:
                    continue

                # Calculate moving averages
                data["fast_ma"] = data["close"].rolling(window=fast_period).mean()
                data["slow_ma"] = data["close"].rolling(window=slow_period).mean()

                # Generate signals
                for i in range(slow_period, len(data)):
                    row = data.iloc[i]

                    # Skip if insufficient data
                    if pd.isna(row["fast_ma"]) or pd.isna(row["slow_ma"]):
                        continue

                    # Calculate signal
                    ma_diff = (row["fast_ma"] - row["slow_ma"]) / row["slow_ma"]

                    if ma_diff > threshold:  # Buy signal
                        signals.append(
                            {
                                "market_id": market_id,
                                "action": "buy",
                                "quantity": 100,  # Fixed quantity for example
                                "price": row["close"],
                                "timestamp": row["timestamp"],
                            }
                        )
                    elif ma_diff < -threshold:  # Sell signal
                        signals.append(
                            {
                                "market_id": market_id,
                                "action": "sell",
                                "quantity": 100,
                                "price": row["close"],
                                "timestamp": row["timestamp"],
                            }
                        )

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")

        return signals


class AdvancedArbitrageStrategy:
    """Advanced arbitrage strategy for backtesting"""

    @staticmethod
    async def generate_signals(
        market_data: Dict[str, pd.DataFrame], parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate arbitrage signals"""
        signals = []

        try:
            # Extract parameters
            min_profit = parameters.get("min_profit_cents", 10)
            max_spread = parameters.get("max_spread_percent", 5.0)
            confidence_threshold = parameters.get("confidence_threshold", 0.7)

            market_ids = list(market_data.keys())

            # Simple cross-market arbitrage
            for i, market_id_1 in enumerate(market_ids):
                for market_id_2 in market_ids[i + 1 :]:
                    data_1 = market_data[market_id_1]
                    data_2 = market_data[market_id_2]

                    if len(data_1) < 10 or len(data_2) < 10:
                        continue

                    # Get latest data
                    latest_1 = data_1.iloc[-1]
                    latest_2 = data_2.iloc[-1]

                    # Calculate spread
                    price_1 = latest_1["close"]
                    price_2 = latest_2["close"]
                    spread = abs(price_1 - price_2)

                    # Check if arbitrage opportunity exists
                    if (
                        spread >= min_profit
                        and spread / max(price_1, price_2) <= max_spread / 100
                    ):
                        # Determine direction
                        if price_1 > price_2:
                            # Sell market 1, buy market 2
                            signals.append(
                                {
                                    "market_id": market_id_1,
                                    "action": "sell",
                                    "quantity": 50,
                                    "price": price_1,
                                    "timestamp": latest_1["timestamp"],
                                }
                            )
                            signals.append(
                                {
                                    "market_id": market_id_2,
                                    "action": "buy",
                                    "quantity": 50,
                                    "price": price_2,
                                    "timestamp": latest_2["timestamp"],
                                }
                            )
                        else:
                            # Buy market 1, sell market 2
                            signals.append(
                                {
                                    "market_id": market_id_1,
                                    "action": "buy",
                                    "quantity": 50,
                                    "price": price_1,
                                    "timestamp": latest_1["timestamp"],
                                }
                            )
                            signals.append(
                                {
                                    "market_id": market_id_2,
                                    "action": "sell",
                                    "quantity": 50,
                                    "price": price_2,
                                    "timestamp": latest_2["timestamp"],
                                }
                            )

        except Exception as e:
            logger.error(f"Arbitrage signal generation failed: {e}")

        return signals


async def run_backtest_example():
    """Run example backtest"""
    try:
        logger.info("Starting example backtest")

        # Load sample data (in production, this would come from database)
        sample_data = generate_sample_data()

        # Define parameter grid for optimization
        parameter_grid = create_parameter_grid(
            {
                "min_profit_cents": [5, 10, 15, 20],
                "max_spread_percent": [2.0, 3.0, 5.0, 8.0],
                "confidence_threshold": [0.5, 0.6, 0.7, 0.8],
            }
        )

        # Run comprehensive backtest with parameter optimization
        results = await run_comprehensive_backtest(
            strategy_func=AdvancedArbitrageStrategy.generate_signals,
            parameters={
                "min_profit_cents": 10,
                "max_spread_percent": 5.0,
                "confidence_threshold": 0.7,
            },  # Default params
            market_data=sample_data,
        )

        # Display results
        print("=== BACKTEST RESULTS ===")
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Annualized Return: {results.annualized_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
        print(f"Maximum Drawdown: {results.max_drawdown:.2%}")
        print(f"Win Rate: {results.win_rate:.2%}")
        print(f"Total Trades: {results.total_trades}")

        # Save results
        save_backtest_results(results)

        logger.info("Example backtest completed successfully")

    except Exception as e:
        logger.error(f"Example backtest failed: {e}")


def generate_sample_data() -> Dict[str, pd.DataFrame]:
    """Generate sample market data for testing"""
    np.random.seed(42)

    # Generate synthetic market data
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="H")

    market_data = {}

    # Generate correlated markets
    base_prices = 100 + np.random.normal(0, 10, len(dates)).cumsum()

    for i, market_id in enumerate(["MARKET_A", "MARKET_B", "MARKET_C"]):
        # Add market-specific noise and drift
        noise = np.random.normal(0, 2, len(dates))
        drift = i * 0.1  # Different drift for each market

        prices = base_prices + noise + drift * np.arange(len(dates))

        # Create OHLCV data
        high_low_range = np.random.uniform(0.001, 0.01, len(dates)) * prices
        opens = prices
        highs = prices + high_low_range / 2
        lows = prices - high_low_range / 2
        closes = prices + np.random.normal(0, 0.5, len(dates))
        volumes = np.random.randint(1000, 10000, len(dates))

        market_data[market_id] = pd.DataFrame(
            {
                "timestamp": dates,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        )

    return market_data


def save_backtest_results(results):
    """Save backtest results to file"""
    try:
        import json

        # Prepare results for JSON serialization
        results_dict = {
            "total_return": results.total_return,
            "annualized_return": results.annualized_return,
            "sharpe_ratio": results.sharpe_ratio,
            "max_drawdown": results.max_drawdown,
            "win_rate": results.win_rate,
            "total_trades": results.total_trades,
            "parameters": results.parameters,
        }

        # Save to file
        with open("backtest_results.json", "w") as f:
            json.dump(results_dict, f, indent=2, default=str)

        logger.info("Results saved to backtest_results.json")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_backtest_example())
