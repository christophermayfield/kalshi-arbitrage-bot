"""
Advanced Backtesting & Parameter Optimization Framework
Professional-grade backtesting with walk-forward analysis, Monte Carlo simulation, and robust optimization
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
import asyncio
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore")

from src.utils.logging_utils import get_logger
from src.utils.config import Config
from src.core.arbitrage import ArbitrageOpportunity, ArbitrageType
from src.core.orderbook import OrderBook
from src.core.portfolio import PortfolioManager
from src.execution.trading import TradingExecutor

logger = get_logger("backtesting")


class BacktestMode(Enum):
    """Backtesting execution modes"""

    VECTORIZED = "vectorized"  # Fast, vectorized backtesting
    EVENT_DRIVEN = "event_driven"  # Realistic, event-by-event simulation
    HYBRID = "hybrid"  # Combination of both


class OptimizationMethod(Enum):
    """Parameter optimization methods"""

    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    PARTICLE_SWARM = "particle_swarm"


@dataclass
class BacktestConfig:
    """Configuration for backtesting runs"""

    # Time parameters
    start_date: datetime
    end_date: datetime
    lookback_window_days: int = 30

    # Execution parameters
    initial_capital: float = 10000.0
    commission_rate: float = 0.01
    slippage_model: str = "linear"  # linear, percentage, fixed
    slippage_rate: float = 0.001

    # Risk parameters
    max_position_size: float = 0.2  # 20% of capital
    max_drawdown_limit: float = 0.15  # 15% max drawdown
    stop_loss_enabled: bool = True
    stop_loss_threshold: float = 0.05  # 5% stop loss

    # Performance parameters
    benchmark_return: float = 0.0
    risk_free_rate: float = 0.02
    rebalance_frequency: str = "daily"  # daily, weekly, monthly

    # Advanced parameters
    enable_short_selling: bool = True
    enable_leverage: bool = False
    leverage_ratio: float = 1.0
    margin_requirement: float = 0.5


@dataclass
class Trade:
    """Individual trade record"""

    timestamp: datetime
    market_id: str
    action: str  # buy, sell, short, cover
    quantity: int
    price: float
    commission: float
    slippage: float
    pnl: float = 0.0
    position_before: int = 0
    position_after: int = 0
    cash_before: float = 0.0
    cash_after: float = 0.0


@dataclass
class BacktestResults:
    """Comprehensive backtest results"""

    # Basic metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # Risk metrics
    var_95: float = 0.0  # Value at Risk 95%
    var_99: float = 0.0  # Value at Risk 99%
    cvar_95: float = 0.0  # Conditional VaR 95%
    beta: float = 0.0
    alpha: float = 0.0

    # Timing metrics
    avg_trade_duration: float = 0.0
    avg_time_between_trades: float = 0.0

    # Detailed data
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    daily_returns: List[Tuple[datetime, float]] = field(default_factory=list)

    # Parameters used
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    backtest_start: datetime = field(default_factory=datetime.now)
    backtest_end: datetime = field(default_factory=datetime.now)
    data_points: int = 0


class MarketDataSimulator:
    """Simulates market data for backtesting"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.orderbooks: Dict[str, List[OrderBook]] = {}

    def load_historical_data(self, market_id: str, data: pd.DataFrame) -> None:
        """Load historical market data"""
        # Ensure data has required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Filter by date range
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        mask = (data["timestamp"] >= self.config.start_date) & (
            data["timestamp"] <= self.config.end_date
        )
        filtered_data = data[mask].copy()

        self.market_data[market_id] = filtered_data

        # Generate orderbook snapshots
        self._generate_orderbooks(market_id, filtered_data)

    def _generate_orderbooks(self, market_id: str, data: pd.DataFrame) -> None:
        """Generate orderbook snapshots from OHLCV data"""
        orderbooks = []

        for _, row in data.iterrows():
            # Simulate orderbook based on OHLCV
            mid_price = (row["high"] + row["low"]) / 2
            spread = (row["high"] - row["low"]) * 0.1  # 10% of range as spread

            # Generate bid/ask levels
            bids = []
            asks = []

            for i in range(5):  # 5 levels each side
                bid_price = mid_price - spread * (i + 1) * 0.2
                ask_price = mid_price + spread * (i + 1) * 0.2

                bid_volume = row["volume"] * np.random.uniform(0.1, 0.3) / 5
                ask_volume = row["volume"] * np.random.uniform(0.1, 0.3) / 5

                bids.append((int(bid_price), int(bid_volume), int(bid_volume)))
                asks.append((int(ask_price), int(ask_volume), int(ask_volume)))

            # Create orderbook
            orderbook = OrderBook(
                market_id=market_id,
                bids=[OrderBookLevel(price=b[0], count=b[1], total=b[2]) for b in bids],
                asks=[OrderBookLevel(price=a[0], count=a[1], total=a[2]) for a in asks],
                timestamp=row["timestamp"].isoformat(),
            )

            orderbooks.append(orderbook)

        self.orderbooks[market_id] = orderbooks

    def get_orderbook(self, market_id: str, timestamp: datetime) -> Optional[OrderBook]:
        """Get orderbook for specific timestamp"""
        if market_id not in self.orderbooks:
            return None

        orderbooks = self.orderbooks[market_id]
        if not orderbooks:
            return None

        # Find closest orderbook to timestamp
        closest_idx = None
        min_diff = timedelta(days=1)

        for i, orderbook in enumerate(orderbooks):
            book_time = datetime.fromisoformat(orderbook.timestamp)
            diff = abs(book_time - timestamp)

            if diff < min_diff:
                min_diff = diff
                closest_idx = i

        return orderbooks[closest_idx] if closest_idx is not None else None


class BacktestEngine:
    """Advanced backtesting engine with multiple execution modes"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.simulator = MarketDataSimulator(config)

        # Portfolio and execution
        self.portfolio = PortfolioManager(
            max_daily_loss=int(config.max_drawdown_limit * config.initial_capital),
            max_open_positions=50,
        )
        self.portfolio.set_balance(config.initial_capital)

        # Tracking
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.current_positions: Dict[str, int] = {}
        self.current_cash: float = config.initial_capital

        logger.info("Backtest engine initialized")

    async def run_backtest(
        self,
        strategy_func: Callable,
        parameters: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame],
        mode: BacktestMode = BacktestMode.EVENT_DRIVEN,
    ) -> BacktestResults:
        """Run backtest with specified strategy and parameters"""
        try:
            logger.info(f"Starting backtest with mode: {mode.value}")

            # Load market data
            for market_id, data in market_data.items():
                self.simulator.load_historical_data(market_id, data)

            # Reset state
            self._reset_state()

            # Run backtest based on mode
            if mode == BacktestMode.VECTORIZED:
                results = await self._run_vectorized_backtest(strategy_func, parameters)
            elif mode == BacktestMode.EVENT_DRIVEN:
                results = await self._run_event_driven_backtest(
                    strategy_func, parameters
                )
            else:  # HYBRID
                results = await self._run_hybrid_backtest(strategy_func, parameters)

            # Calculate comprehensive metrics
            results = self._calculate_comprehensive_metrics(results)
            results.parameters = parameters

            logger.info(f"Backtest completed. Total return: {results.total_return:.2%}")
            return results

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise

    def _reset_state(self) -> None:
        """Reset backtest state"""
        self.trades = []
        self.equity_curve = []
        self.current_positions = {}
        self.current_cash = self.config.initial_capital
        self.portfolio.set_balance(self.config.initial_capital)

    async def _run_vectorized_backtest(
        self, strategy_func: Callable, parameters: Dict[str, Any]
    ) -> BacktestResults:
        """Run fast vectorized backtest"""
        results = BacktestResults()

        # Get all timestamps across markets
        all_timestamps = set()
        for market_id, data in self.simulator.market_data.items():
            all_timestamps.update(data["timestamp"].tolist())

        sorted_timestamps = sorted(all_timestamps)

        # Vectorized processing
        for timestamp in sorted_timestamps:
            # Get market data for all markets at this timestamp
            market_data_at_time = {}
            for market_id, data in self.simulator.market_data.items():
                mask = data["timestamp"] == timestamp
                if mask.any():
                    market_data_at_time[market_id] = data[mask].iloc[0]

            # Generate signals
            signals = await strategy_func(market_data_at_time, parameters)

            # Process signals
            for signal in signals:
                await self._process_signal_vectorized(signal, timestamp)

            # Update equity curve
            portfolio_value = self._calculate_portfolio_value(timestamp)
            self.equity_curve.append((timestamp, portfolio_value))

        # Convert to results format
        results.equity_curve = self.equity_curve
        results.trades = self.trades

        return results

    async def _run_event_driven_backtest(
        self, strategy_func: Callable, parameters: Dict[str, Any]
    ) -> BacktestResults:
        """Run realistic event-driven backtest"""
        results = BacktestResults()

        # Create event timeline
        events = self._create_event_timeline()

        # Process events chronologically
        for event in events:
            await self._process_event(event, strategy_func, parameters)

            # Update equity curve
            portfolio_value = self._calculate_portfolio_value(event["timestamp"])
            self.equity_curve.append((event["timestamp"], portfolio_value))

        results.equity_curve = self.equity_curve
        results.trades = self.trades

        return results

    async def _run_hybrid_backtest(
        self, strategy_func: Callable, parameters: Dict[str, Any]
    ) -> BacktestResults:
        """Run hybrid backtest combining vectorized and event-driven"""
        # Use event-driven for accuracy, but optimize with vectorized calculations
        return await self._run_event_driven_backtest(strategy_func, parameters)

    def _create_event_timeline(self) -> List[Dict[str, Any]]:
        """Create chronological event timeline"""
        events = []

        # Market data events
        for market_id, data in self.simulator.market_data.items():
            for _, row in data.iterrows():
                events.append(
                    {
                        "type": "market_data",
                        "timestamp": row["timestamp"],
                        "market_id": market_id,
                        "data": row,
                    }
                )

        # Sort by timestamp
        events.sort(key=lambda x: x["timestamp"])

        return events

    async def _process_event(
        self, event: Dict[str, Any], strategy_func: Callable, parameters: Dict[str, Any]
    ) -> None:
        """Process individual event"""
        if event["type"] == "market_data":
            # Get current market state
            market_data = {event["market_id"]: event["data"]}

            # Generate signals
            signals = await strategy_func(market_data, parameters)

            # Process signals
            for signal in signals:
                await self._process_signal_event_driven(signal, event["timestamp"])

    async def _process_signal_vectorized(
        self, signal: Dict[str, Any], timestamp: datetime
    ) -> None:
        """Process trading signal in vectorized mode"""
        # Simplified processing for vectorized mode
        market_id = signal.get("market_id")
        action = signal.get("action")
        quantity = signal.get("quantity", 0)

        if action and quantity > 0:
            await self._execute_trade(market_id, action, quantity, timestamp)

    async def _process_signal_event_driven(
        self, signal: Dict[str, Any], timestamp: datetime
    ) -> None:
        """Process trading signal in event-driven mode"""
        market_id = signal.get("market_id")
        action = signal.get("action")
        quantity = signal.get("quantity", 0)

        if action and quantity > 0:
            # Get realistic orderbook
            orderbook = self.simulator.get_orderbook(market_id, timestamp)
            if orderbook:
                await self._execute_trade_with_slippage(
                    market_id, action, quantity, timestamp, orderbook
                )

    async def _execute_trade(
        self, market_id: str, action: str, quantity: int, timestamp: datetime
    ) -> None:
        """Execute trade without slippage (vectorized mode)"""
        # Get price from market data
        data = self.simulator.market_data[market_id]
        mask = data["timestamp"] == timestamp
        if not mask.any():
            return

        price = data[mask].iloc[0]["close"]
        commission = price * quantity * self.config.commission_rate

        # Update positions and cash
        position_before = self.current_positions.get(market_id, 0)
        cash_before = self.current_cash

        if action in ["buy", "long"]:
            self.current_positions[market_id] = position_before + quantity
            self.current_cash -= price * quantity + commission
        elif action in ["sell", "short"]:
            self.current_positions[market_id] = position_before - quantity
            self.current_cash += price * quantity - commission

        # Record trade
        trade = Trade(
            timestamp=timestamp,
            market_id=market_id,
            action=action,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=0.0,
            position_before=position_before,
            position_after=self.current_positions[market_id],
            cash_before=cash_before,
            cash_after=self.current_cash,
        )

        self.trades.append(trade)

    async def _execute_trade_with_slippage(
        self,
        market_id: str,
        action: str,
        quantity: int,
        timestamp: datetime,
        orderbook: OrderBook,
    ) -> None:
        """Execute trade with realistic slippage"""
        # Get execution price with slippage
        if action in ["buy", "long"]:
            # Buy at ask price
            if orderbook.asks:
                base_price = orderbook.asks[0].price
                slippage = self._calculate_slippage(base_price, quantity, "buy")
                execution_price = base_price + slippage
            else:
                return
        else:
            # Sell at bid price
            if orderbook.bids:
                base_price = orderbook.bids[0].price
                slippage = self._calculate_slippage(base_price, quantity, "sell")
                execution_price = base_price - slippage
            else:
                return

        commission = execution_price * quantity * self.config.commission_rate

        # Update positions and cash
        position_before = self.current_positions.get(market_id, 0)
        cash_before = self.current_cash

        if action in ["buy", "long"]:
            self.current_positions[market_id] = position_before + quantity
            self.current_cash -= execution_price * quantity + commission
        elif action in ["sell", "short"]:
            self.current_positions[market_id] = position_before - quantity
            self.current_cash += execution_price * quantity - commission

        # Record trade
        trade = Trade(
            timestamp=timestamp,
            market_id=market_id,
            action=action,
            quantity=quantity,
            price=execution_price,
            commission=commission,
            slippage=execution_price - base_price,
            position_before=position_before,
            position_after=self.current_positions[market_id],
            cash_before=cash_before,
            cash_after=self.current_cash,
        )

        self.trades.append(trade)

    def _calculate_slippage(self, base_price: float, quantity: int, side: str) -> float:
        """Calculate realistic slippage"""
        if self.config.slippage_model == "linear":
            # Linear slippage based on quantity
            slippage_rate = self.config.slippage_rate * (quantity / 1000)
        elif self.config.slippage_model == "percentage":
            # Percentage of price
            slippage_rate = self.config.slippage_rate
        else:  # fixed
            slippage_rate = self.config.slippage_rate

        return base_price * slippage_rate

    def _calculate_portfolio_value(self, timestamp: datetime) -> float:
        """Calculate total portfolio value"""
        total_value = self.current_cash

        # Add value of positions
        for market_id, position in self.current_positions.items():
            if position == 0:
                continue

            # Get current price
            data = self.simulator.market_data[market_id]
            mask = data["timestamp"] <= timestamp
            if mask.any():
                current_price = data[mask].iloc[-1]["close"]
                total_value += position * current_price

        return total_value

    def _calculate_comprehensive_metrics(
        self, results: BacktestResults
    ) -> BacktestResults:
        """Calculate comprehensive performance metrics"""
        if not results.equity_curve:
            return results

        # Extract equity curve data
        timestamps, values = zip(*results.equity_curve)
        equity_series = pd.Series(values, index=timestamps)

        # Basic returns
        returns = equity_series.pct_change().dropna()
        results.total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1

        # Annualized metrics
        days = (timestamps[-1] - timestamps[0]).days
        if days > 0:
            results.annualized_return = (1 + results.total_return) ** (365 / days) - 1
            results.volatility = returns.std() * np.sqrt(365)

        # Risk-adjusted metrics
        if results.volatility > 0:
            results.sharpe_ratio = (
                results.annualized_return - self.config.risk_free_rate
            ) / results.volatility

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(365)
            if downside_std > 0:
                results.sortino_ratio = (
                    results.annualized_return - self.config.risk_free_rate
                ) / downside_std

        # Maximum drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        results.max_drawdown = drawdown.min()

        # Calmar ratio
        if results.max_drawdown != 0:
            results.calmar_ratio = results.annualized_return / abs(results.max_drawdown)

        # Trade statistics
        if results.trades:
            results.total_trades = len(results.trades)

            # Calculate P&L for each trade
            trade_pnls = []
            for i, trade in enumerate(results.trades):
                if i > 0:  # Need previous trade to calculate P&L
                    # Simplified P&L calculation
                    if trade.action in ["sell", "short"]:
                        pnl = (
                            trade.price - results.trades[i - 1].price
                        ) * trade.quantity
                    else:
                        pnl = (
                            trade.price - results.trades[i - 1].price
                        ) * trade.quantity
                    trade.pnl = pnl
                    trade_pnls.append(pnl)

            if trade_pnls:
                winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
                losing_trades = [pnl for pnl in trade_pnls if pnl < 0]

                results.winning_trades = len(winning_trades)
                results.losing_trades = len(losing_trades)
                results.win_rate = len(winning_trades) / len(trade_pnls)

                results.avg_win = np.mean(winning_trades) if winning_trades else 0
                results.avg_loss = np.mean(losing_trades) if losing_trades else 0

                total_wins = sum(winning_trades)
                total_losses = abs(sum(losing_trades)) if losing_trades else 1
                results.profit_factor = total_wins / total_losses

        # Risk metrics
        if len(returns) > 0:
            results.var_95 = np.percentile(returns, 5)
            results.var_99 = np.percentile(returns, 1)

            # Conditional VaR
            var_95_threshold = results.var_95
            tail_returns = returns[returns <= var_95_threshold]
            results.cvar_95 = (
                tail_returns.mean() if len(tail_returns) > 0 else var_95_threshold
            )

        # Store daily returns
        results.daily_returns = list(zip(timestamps[1:], returns.tolist()))

        return results


class ParameterOptimizer:
    """Advanced parameter optimization with multiple methods"""

    def __init__(self, backtest_engine: BacktestEngine):
        self.backtest_engine = backtest_engine
        self.optimization_history: List[Dict[str, Any]] = []

    async def optimize_parameters(
        self,
        strategy_func: Callable,
        parameter_grid: Dict[str, List[Any]],
        market_data: Dict[str, pd.DataFrame],
        method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
        objective: str = "sharpe_ratio",
        cv_folds: int = 5,
        n_jobs: int = -1,
    ) -> Tuple[Dict[str, Any], BacktestResults]:
        """Optimize parameters using specified method"""
        try:
            logger.info(f"Starting parameter optimization with method: {method.value}")

            if method == OptimizationMethod.GRID_SEARCH:
                best_params, best_results = await self._grid_search_optimization(
                    strategy_func,
                    parameter_grid,
                    market_data,
                    objective,
                    cv_folds,
                    n_jobs,
                )
            elif method == OptimizationMethod.RANDOM_SEARCH:
                best_params, best_results = await self._random_search_optimization(
                    strategy_func,
                    parameter_grid,
                    market_data,
                    objective,
                    cv_folds,
                    n_jobs,
                )
            elif method == OptimizationMethod.BAYESIAN:
                best_params, best_results = await self._bayesian_optimization(
                    strategy_func, parameter_grid, market_data, objective, cv_folds
                )
            else:
                raise ValueError(f"Optimization method {method} not yet implemented")

            logger.info(
                f"Optimization completed. Best {objective}: {getattr(best_results, objective):.4f}"
            )
            return best_params, best_results

        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            raise

    async def _grid_search_optimization(
        self,
        strategy_func: Callable,
        parameter_grid: Dict[str, List[Any]],
        market_data: Dict[str, pd.DataFrame],
        objective: str,
        cv_folds: int,
        n_jobs: int,
    ) -> Tuple[Dict[str, Any], BacktestResults]:
        """Grid search optimization"""
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(parameter_grid))

        logger.info(f"Testing {len(param_combinations)} parameter combinations")

        best_score = -np.inf
        best_params = None
        best_results = None

        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []

            for params in param_combinations:
                future = executor.submit(
                    self._evaluate_parameters_cv,
                    strategy_func,
                    params,
                    market_data,
                    objective,
                    cv_folds,
                )
                futures.append((params, future))

            # Collect results
            for params, future in futures:
                try:
                    cv_score, cv_results = future.result(
                        timeout=300
                    )  # 5 minute timeout

                    if cv_score > best_score:
                        best_score = cv_score
                        best_params = params
                        best_results = cv_results

                    logger.debug(f"Params: {params}, Score: {cv_score:.4f}")

                except Exception as e:
                    logger.error(f"Parameter evaluation failed: {e}")

        return best_params, best_results

    async def _random_search_optimization(
        self,
        strategy_func: Callable,
        parameter_grid: Dict[str, List[Any]],
        market_data: Dict[str, pd.DataFrame],
        objective: str,
        cv_folds: int,
        n_jobs: int,
        n_iter: int = 100,
    ) -> Tuple[Dict[str, Any], BacktestResults]:
        """Random search optimization"""
        from sklearn.model_selection import ParameterSampler

        # Generate random parameter combinations
        param_combinations = list(
            ParameterSampler(parameter_grid, n_iter=n_iter, random_state=42)
        )

        logger.info(f"Testing {len(param_combinations)} random parameter combinations")

        best_score = -np.inf
        best_params = None
        best_results = None

        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []

            for params in param_combinations:
                future = executor.submit(
                    self._evaluate_parameters_cv,
                    strategy_func,
                    params,
                    market_data,
                    objective,
                    cv_folds,
                )
                futures.append((params, future))

            # Collect results
            for params, future in futures:
                try:
                    cv_score, cv_results = future.result(timeout=300)

                    if cv_score > best_score:
                        best_score = cv_score
                        best_params = params
                        best_results = cv_results

                except Exception as e:
                    logger.error(f"Parameter evaluation failed: {e}")

        return best_params, best_results

    async def _bayesian_optimization(
        self,
        strategy_func: Callable,
        parameter_grid: Dict[str, List[Any]],
        market_data: Dict[str, pd.DataFrame],
        objective: str,
        cv_folds: int,
        n_calls: int = 50,
    ) -> Tuple[Dict[str, Any], BacktestResults]:
        """Bayesian optimization using Gaussian Process"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args

            # Define search space
            dimensions = []
            param_names = []

            for param_name, param_values in parameter_grid.items():
                param_names.append(param_name)

                if all(isinstance(v, (int, float)) for v in param_values):
                    if isinstance(param_values[0], int):
                        dimensions.append(
                            Integer(
                                min(param_values), max(param_values), name=param_name
                            )
                        )
                    else:
                        dimensions.append(
                            Real(min(param_values), max(param_values), name=param_name)
                        )
                else:
                    dimensions.append(Categorical(param_values, name=param_name))

            @use_named_args(dimensions)
            def objective_function(**params):
                """Objective function for Bayesian optimization"""
                try:
                    cv_score, _ = self._evaluate_parameters_cv(
                        strategy_func, params, market_data, objective, cv_folds
                    )
                    return -cv_score  # Minimize negative score
                except Exception:
                    return 1e6  # Large penalty for failed evaluations

            # Run Bayesian optimization
            result = gp_minimize(
                func=objective_function,
                dimensions=dimensions,
                n_calls=n_calls,
                random_state=42,
                verbose=False,
            )

            # Extract best parameters
            best_params = dict(zip(param_names, result.x))

            # Evaluate best parameters
            _, best_results = self._evaluate_parameters_cv(
                strategy_func, best_params, market_data, objective, cv_folds
            )

            return best_params, best_results

        except ImportError:
            logger.warning("scikit-optimize not installed, falling back to grid search")
            return await self._grid_search_optimization(
                strategy_func, parameter_grid, market_data, objective, cv_folds, 1
            )

    async def _evaluate_parameters_cv(
        self,
        strategy_func: Callable,
        parameters: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame],
        objective: str,
        cv_folds: int,
    ) -> Tuple[float, BacktestResults]:
        """Evaluate parameters using cross-validation"""
        try:
            # Create time series splits
            tscv = TimeSeriesSplit(n_splits=cv_folds)

            scores = []
            all_results = []

            # Get date range
            all_dates = set()
            for data in market_data.values():
                all_dates.update(data["timestamp"].dt.date.tolist())

            sorted_dates = sorted(all_dates)

            # Perform cross-validation
            for fold, (train_start, test_start) in enumerate(tscv.split(sorted_dates)):
                # Define train/test periods
                train_dates = sorted_dates[: train_start[-1] + 1]
                test_dates = sorted_dates[train_start[-1] + 1 : test_start[-1] + 1]

                if len(test_dates) < 5:  # Minimum test period
                    continue

                # Create train/test market data
                train_data = {}
                test_data = {}

                for market_id, data in market_data.items():
                    train_mask = data["timestamp"].dt.date.isin(train_dates)
                    test_mask = data["timestamp"].dt.date.isin(test_dates)

                    train_data[market_id] = data[train_mask]
                    test_data[market_id] = data[test_mask]

                # Configure backtest for test period
                backtest_config = BacktestConfig(
                    start_date=min(test_dates),
                    end_date=max(test_dates),
                    initial_capital=self.backtest_engine.config.initial_capital,
                )

                # Create backtest engine
                test_engine = BacktestEngine(backtest_config)

                # Run backtest
                results = await test_engine.run_backtest(
                    strategy_func, parameters, test_data, BacktestMode.EVENT_DRIVEN
                )

                # Extract objective score
                score = getattr(results, objective, 0)
                scores.append(score)
                all_results.append(results)

            # Return average score and best results
            avg_score = np.mean(scores) if scores else 0
            best_results = (
                max(all_results, key=lambda r: getattr(r, objective, 0))
                if all_results
                else BacktestResults()
            )

            return avg_score, best_results

        except Exception as e:
            logger.error(f"Cross-validation evaluation failed: {e}")
            return 0, BacktestResults()


class WalkForwardAnalyzer:
    """Walk-forward analysis for robust strategy validation"""

    def __init__(self, backtest_engine: BacktestEngine):
        self.backtest_engine = backtest_engine

    async def run_walk_forward_analysis(
        self,
        strategy_func: Callable,
        parameter_grid: Dict[str, List[Any]],
        market_data: Dict[str, pd.DataFrame],
        train_period_months: int = 6,
        test_period_months: int = 1,
        optimization_method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
    ) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        try:
            logger.info("Starting walk-forward analysis")

            # Get date range
            all_dates = set()
            for data in market_data.values():
                all_dates.update(data["timestamp"].dt.date.tolist())

            sorted_dates = sorted(all_dates)

            # Create walk-forward windows
            windows = self._create_walk_forward_windows(
                sorted_dates, train_period_months, test_period_months
            )

            results = []

            for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
                logger.info(f"Walk-forward window {i + 1}/{len(windows)}")

                # Split data
                train_data, test_data = self._split_data_for_window(
                    market_data, train_start, train_end, test_start, test_end
                )

                # Optimize parameters on training data
                optimizer = ParameterOptimizer(self.backtest_engine)
                best_params, _ = await optimizer.optimize_parameters(
                    strategy_func, parameter_grid, train_data, optimization_method
                )

                # Test on out-of-sample data
                test_config = BacktestConfig(
                    start_date=test_start,
                    end_date=test_end,
                    initial_capital=self.backtest_engine.config.initial_capital,
                )

                test_engine = BacktestEngine(test_config)
                test_results = await test_engine.run_backtest(
                    strategy_func, best_params, test_data, BacktestMode.EVENT_DRIVEN
                )

                results.append(
                    {
                        "window": i + 1,
                        "train_period": (train_start, train_end),
                        "test_period": (test_start, test_end),
                        "best_parameters": best_params,
                        "test_results": test_results,
                    }
                )

            # Analyze walk-forward results
            analysis = self._analyze_walk_forward_results(results)

            logger.info("Walk-forward analysis completed")
            return analysis

        except Exception as e:
            logger.error(f"Walk-forward analysis failed: {e}")
            raise

    def _create_walk_forward_windows(
        self, dates: List[datetime], train_months: int, test_months: int
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Create walk-forward windows"""
        windows = []

        step_size_months = test_months  # Roll forward by test period

        current_idx = 0
        while current_idx + train_months + test_months <= len(dates):
            train_start = dates[current_idx]
            train_end = dates[current_idx + train_months - 1]

            test_start = dates[current_idx + train_months]
            test_end = dates[current_idx + train_months + test_months - 1]

            windows.append((train_start, train_end, test_start, test_end))

            current_idx += step_size_months

        return windows

    def _split_data_for_window(
        self,
        market_data: Dict[str, pd.DataFrame],
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Split data for walk-forward window"""
        train_data = {}
        test_data = {}

        for market_id, data in market_data.items():
            train_mask = (data["timestamp"] >= train_start) & (
                data["timestamp"] <= train_end
            )
            test_mask = (data["timestamp"] >= test_start) & (
                data["timestamp"] <= test_end
            )

            train_data[market_id] = data[train_mask]
            test_data[market_id] = data[test_mask]

        return train_data, test_data

    def _analyze_walk_forward_results(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze walk-forward results"""
        if not results:
            return {}

        # Extract performance metrics
        returns = [r["test_results"].total_return for r in results]
        sharpes = [r["test_results"].sharpe_ratio for r in results]
        max_drawdowns = [r["test_results"].max_drawdown for r in results]

        # Calculate statistics
        analysis = {
            "total_windows": len(results),
            "performance": {
                "avg_return": np.mean(returns),
                "std_return": np.std(returns),
                "avg_sharpe": np.mean(sharpes),
                "std_sharpe": np.std(sharpes),
                "avg_max_drawdown": np.mean(max_drawdowns),
                "std_max_drawdown": np.std(max_drawdowns),
                "win_rate": len([r for r in returns if r > 0]) / len(returns),
            },
            "stability": {
                "return_stability": 1
                - (np.std(returns) / max(0.001, np.abs(np.mean(returns)))),
                "sharpe_stability": 1
                - (np.std(sharpes) / max(0.001, np.abs(np.mean(sharpes)))),
                "parameter_stability": self._calculate_parameter_stability(results),
            },
            "detailed_results": results,
        }

        return analysis

    def _calculate_parameter_stability(self, results: List[Dict[str, Any]]) -> float:
        """Calculate parameter stability across windows"""
        try:
            # Get all parameter names
            all_params = set()
            for result in results:
                all_params.update(result["best_parameters"].keys())

            # Calculate coefficient of variation for each parameter
            param_stabilities = []

            for param in all_params:
                values = []
                for result in results:
                    if param in result["best_parameters"]:
                        values.append(result["best_parameters"][param])

                if len(values) > 1:
                    cv = np.std(values) / max(0.001, np.abs(np.mean(values)))
                    stability = 1 - min(1.0, cv)  # Convert to stability score
                    param_stabilities.append(stability)

            return np.mean(param_stabilities) if param_stabilities else 0

        except Exception:
            return 0


# Utility functions
def create_backtest_config(
    start_date: datetime, end_date: datetime, initial_capital: float = 10000.0, **kwargs
) -> BacktestConfig:
    """Create backtest configuration"""
    return BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        **kwargs,
    )


def create_parameter_grid(param_dict: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Create parameter grid for optimization"""
    return param_dict


async def run_comprehensive_backtest(
    strategy_func: Callable,
    parameters: Dict[str, Any],
    market_data: Dict[str, pd.DataFrame],
    config: Optional[BacktestConfig] = None,
) -> BacktestResults:
    """Run comprehensive backtest with all analysis types"""
    if config is None:
        # Default configuration
        start_date = min([data["timestamp"].min() for data in market_data.values()])
        end_date = max([data["timestamp"].max() for data in market_data.values()])
        config = BacktestConfig(start_date=start_date, end_date=end_date)

    # Create backtest engine
    engine = BacktestEngine(config)

    # Run backtest
    results = await engine.run_backtest(
        strategy_func, parameters, market_data, BacktestMode.EVENT_DRIVEN
    )

    return results


class AdvancedBacktester:
    """Wrapper class for advanced backtesting functionality."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            initial_capital=10000.0,
        )
        self.backtest_engine = BacktestEngine(self.config)

    async def run_backtest(
        self,
        strategy_func: Callable,
        parameters: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame],
        mode: BacktestMode = BacktestMode.EVENT_DRIVEN,
    ) -> BacktestResults:
        """Run a backtest with the given strategy and parameters."""
        return await self.backtest_engine.run_backtest(
            strategy_func, parameters, market_data, mode
        )

    async def optimize_parameters(
        self,
        strategy_func: Callable,
        parameter_grid: Dict[str, List[Any]],
        market_data: Dict[str, pd.DataFrame],
        objective: str = "sharpe_ratio",
        method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
    ) -> Tuple[Dict[str, Any], BacktestResults]:
        """Optimize strategy parameters."""
        if method == OptimizationMethod.GRID_SEARCH:
            return await self.backtest_engine.optimize_parameters_grid_search(
                strategy_func, parameter_grid, market_data, objective
            )
        return {}, BacktestResults()

    async def run_walk_forward(
        self,
        strategy_func: Callable,
        parameters: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame],
        train_window_days: int = 90,
        test_window_days: int = 30,
    ) -> List[BacktestResults]:
        """Run walk-forward analysis."""
        return await self.backtest_engine.run_walk_forward_analysis(
            strategy_func, parameters, market_data, train_window_days, test_window_days
        )
