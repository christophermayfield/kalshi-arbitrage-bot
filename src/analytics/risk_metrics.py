"""
Advanced Risk Metrics - VaR, CVaR, and stress testing.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

from src.utils.logging_utils import get_logger

logger = get_logger("risk_metrics")


@dataclass
class RiskMetrics:
    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    cvar_95: float  # Conditional VaR at 95%
    cvar_99: float  # Conditional VaR at 99%
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: Optional[float] = None
    alpha: Optional[float] = None


class RiskAnalyzer:
    def __init__(
        self,
        lookback_days: int = 30,
        confidence_levels: List[float] = [0.95, 0.99],
    ):
        self.lookback_days = lookback_days
        self.confidence_levels = confidence_levels
        self._returns_history: deque = deque(maxlen=1000)
        self._portfolio_values: deque = deque(maxlen=1000)
        self._trade_pnls: List[int] = []

    def add_return(self, return_pct: float) -> None:
        """Add a return observation."""
        self._returns_history.append(return_pct)

    def add_portfolio_value(
        self, value: float, timestamp: Optional[datetime] = None
    ) -> None:
        """Add a portfolio value observation."""
        self._portfolio_values.append((timestamp or datetime.utcnow(), value))

    def add_trade_pnl(self, pnl: int) -> None:
        """Add a trade P&L observation (in cents)."""
        self._trade_pnls.append(pnl)

    def calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk using historical method."""
        if len(self._returns_history) < 2:
            return 0.0

        returns = np.array(self._returns_history)
        var = np.percentile(returns, (1 - confidence) * 100)
        return abs(var) * 100  # Convert to percentage points

    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(self._returns_history) < 2:
            return 0.0

        returns = np.array(self._returns_history)
        var = np.percentile(returns, (1 - confidence) * 100)

        # CVaR is the mean of returns below VaR
        cvar = returns[returns <= var].mean()
        return abs(cvar) * 100 if not np.isnan(cvar) else 0.0

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio values."""
        if len(self._portfolio_values) < 2:
            return 0.0

        values = [v[1] for v in self._portfolio_values]
        values = np.array(values)

        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max

        return abs(drawdowns.min()) * 100  # As percentage

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (annualized)."""
        if len(self._returns_history) < 2:
            return 0.0

        returns = np.array(self._returns_history)

        if np.std(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)

        return sharpe

    def calculate_volatility(self) -> float:
        """Calculate annualized volatility."""
        if len(self._returns_history) < 2:
            return 0.0

        returns = np.array(self._returns_history)
        return np.std(returns) * np.sqrt(252) * 100  # As percentage

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        if len(self._returns_history) < 2:
            return 0.0

        returns = np.array(self._returns_history)
        excess_returns = returns - (risk_free_rate / 252)

        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        return sortino

    def calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        if len(self._portfolio_values) < 2:
            return 0.0

        max_dd = self.calculate_max_drawdown()
        if max_dd == 0:
            return 0.0

        # Calculate annualized return
        values = [v[1] for v in self._portfolio_values]
        total_return = (values[-1] - values[0]) / values[0] if values[0] > 0 else 0

        days = (self._portfolio_values[-1][0] - self._portfolio_values[0][0]).days
        annualized_return = ((1 + total_return) ** (365 / days) - 1) if days > 0 else 0

        return annualized_return / max_dd

    def calculate_win_rate(self) -> float:
        """Calculate win rate from trade P&Ls."""
        if not self._trade_pnls:
            return 0.0

        wins = sum(1 for pnl in self._trade_pnls if pnl > 0)
        return (wins / len(self._trade_pnls)) * 100

    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not self._trade_pnls:
            return 0.0

        gross_profit = sum(pnl for pnl in self._trade_pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in self._trade_pnls if pnl < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def calculate_average_win_loss(self) -> float:
        """Calculate average win to average loss ratio."""
        if not self._trade_pnls:
            return 0.0

        wins = [pnl for pnl in self._trade_pnls if pnl > 0]
        losses = [abs(pnl) for pnl in self._trade_pnls if pnl < 0]

        if not wins or not losses:
            return 0.0

        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)

        return avg_win / avg_loss if avg_loss > 0 else 0.0

    def run_stress_test(
        self,
        scenarios: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Run stress tests with custom scenarios."""
        if scenarios is None:
            scenarios = {
                "market_crash_20": -0.20,
                "market_crash_30": -0.30,
                "volatility_spike_2x": -0.15,
                "liquidity_crisis": -0.25,
                "black_swan": -0.50,
            }

        stress_results = {}

        for scenario_name, shock in scenarios.items():
            # Estimate portfolio impact based on positions
            # This is a simplified model
            impact = shock * 100  # Convert to percentage points
            stress_results[scenario_name] = impact

        return stress_results

    def calculate_rolling_sharpe(self, window: int = 20) -> Optional[float]:
        """Calculate rolling Sharpe ratio over a window."""
        if len(self._returns_history) < window:
            return None

        returns = list(self._returns_history)[-window:]
        returns_array = np.array(returns)

        if np.std(returns_array) == 0:
            return 0.0

        return np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)

    def calculate_rolling_volatility(self, window: int = 20) -> Optional[float]:
        """Calculate rolling volatility over a window."""
        if len(self._returns_history) < window:
            return None

        returns = list(self._returns_history)[-window:]
        returns_array = np.array(returns)

        return np.std(returns_array) * np.sqrt(252) * 100

    def calculate_rolling_drawdown(self, window: int = 20) -> Optional[float]:
        """Calculate rolling maximum drawdown over a window."""
        if len(self._portfolio_values) < 2:
            return None

        values = [v[1] for v in list(self._portfolio_values)[-window:]]
        values_array = np.array(values)

        running_max = np.maximum.accumulate(values_array)
        drawdowns = (values_array - running_max) / running_max

        return abs(np.min(drawdowns)) * 100 if len(drawdowns) > 0 else 0.0

    def calculate_rolling_win_rate(self, window: int = 50) -> Optional[float]:
        """Calculate rolling win rate over a window."""
        if len(self._trade_pnls) < window:
            return None

        recent_pnls = list(self._trade_pnls)[-window:]
        wins = sum(1 for pnl in recent_pnls if pnl > 0)

        return (wins / len(recent_pnls)) * 100

    def get_rolling_metrics(self, window: int = 20) -> Dict[str, Any]:
        """Get all rolling metrics for a given window."""
        return {
            "window": window,
            "sharpe_ratio": self.calculate_rolling_sharpe(window),
            "volatility": self.calculate_rolling_volatility(window),
            "max_drawdown": self.calculate_rolling_drawdown(window),
            "win_rate": self.calculate_rolling_win_rate(window * 2),
        }

    def get_all_metrics(
        self,
        benchmark_returns: Optional[List[float]] = None,
    ) -> RiskMetrics:
        """Calculate all risk metrics."""
        var_95 = self.calculate_var(0.95)
        var_99 = self.calculate_var(0.99)
        cvar_95 = self.calculate_cvar(0.95)
        cvar_99 = self.calculate_cvar(0.99)

        beta, alpha = None, None
        if benchmark_returns and len(benchmark_returns) == len(self._returns_history):
            beta, alpha = self._calculate_beta_alpha(benchmark_returns)

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=self.calculate_max_drawdown(),
            sharpe_ratio=self.calculate_sharpe_ratio(),
            volatility=self.calculate_volatility(),
            beta=beta,
            alpha=alpha,
        )

    def _calculate_beta_alpha(
        self,
        benchmark_returns: List[float],
    ) -> tuple[float, float]:
        """Calculate beta and alpha relative to benchmark."""
        if len(self._returns_history) < 2:
            return None, None

        portfolio_returns = np.array(self._returns_history)
        benchmark = np.array(benchmark_returns)

        if np.std(benchmark) == 0 or np.std(portfolio_returns) == 0:
            return None, None

        covariance = np.cov(portfolio_returns, benchmark)[0, 1]
        benchmark_variance = np.var(benchmark)

        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

        portfolio_return = np.mean(portfolio_returns)
        benchmark_return = np.mean(benchmark)
        alpha = portfolio_return - beta * benchmark_return

        return beta, alpha * 100  # Annualized

    def get_risk_report(self) -> Dict[str, Any]:
        """Get comprehensive risk report."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "var_95": self.calculate_var(0.95),
                "var_99": self.calculate_var(0.99),
                "cvar_95": self.calculate_cvar(0.95),
                "cvar_99": self.calculate_cvar(0.99),
                "max_drawdown": self.calculate_max_drawdown(),
                "sharpe_ratio": self.calculate_sharpe_ratio(),
                "sortino_ratio": self.calculate_sortino_ratio(),
                "calmar_ratio": self.calculate_calmar_ratio(),
                "volatility": self.calculate_volatility(),
                "win_rate": self.calculate_win_rate(),
                "profit_factor": self.calculate_profit_factor(),
                "avg_win_loss_ratio": self.calculate_average_win_loss(),
            },
            "stress_tests": self.run_stress_test(),
            "sample_size": len(self._returns_history),
        }
