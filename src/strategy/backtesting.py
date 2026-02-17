"""
Backtesting Framework - Test strategies on historical data.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger("backtesting")


class BacktestMode(Enum):
    """Backtest execution modes."""

    SIMPLE = "simple"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = 10000.0
    commission_rate: float = 0.01
    slippage_rate: float = 0.001
    max_position_size: float = 0.2
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class BacktestResult:
    """Results from a backtest."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    losing_trades: int
    avg_profit: float
    avg_loss: float
    profit_factor: float
    equity_curve: List[float]
    trade_log: List[Dict[str, Any]] = field(default_factory=list)


class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.equity = config.initial_capital
        self.equity_curve = [self.equity]
        self.trade_log = []
        self.position = None

    def reset(self) -> None:
        """Reset the backtest state."""
        self.equity = self.config.initial_capital
        self.equity_curve = [self.equity]
        self.trade_log = []
        self.position = None

    def apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to execution price."""
        slippage = price * self.config.slippage_rate
        if side == "buy":
            return price + slippage
        return price - slippage

    def apply_commission(self, amount: float) -> float:
        """Apply commission to a trade."""
        return amount * self.config.commission_rate

    def execute_trade(
        self,
        entry_price: float,
        exit_price: float,
        quantity: int,
        side: str,
    ) -> float:
        """Execute a trade and return profit/loss."""
        entry_price = self.apply_slippage(entry_price, side)
        exit_price = self.apply_slippage(exit_price, "sell" if side == "buy" else "buy")

        entry_cost = entry_price * quantity
        exit_revenue = exit_price * quantity

        commission = self.apply_commission(entry_cost + exit_revenue)

        if side == "buy":
            profit = exit_revenue - entry_cost - commission
        else:
            profit = entry_cost - exit_revenue - commission

        self.equity += profit
        self.equity_curve.append(self.equity)

        trade = {
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "side": side,
            "profit": profit,
            "equity_after": self.equity,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.trade_log.append(trade)

        return profit

    def calculate_metrics(self) -> BacktestResult:
        """Calculate performance metrics."""
        if not self.trade_log:
            return BacktestResult(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                profitable_trades=0,
                losing_trades=0,
                avg_profit=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                equity_curve=self.equity_curve,
            )

        profits = [t["profit"] for t in self.trade_log]
        profitable = [p for p in profits if p > 0]
        losing = [p for p in profits if p < 0]

        total_return = (
            self.equity - self.config.initial_capital
        ) / self.config.initial_capital

        returns_array = np.array(profits) / self.config.initial_capital
        if len(returns_array) > 1 and np.std(returns_array) > 0:
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        win_rate = len(profitable) / len(profits) if profits else 0.0

        avg_profit = np.mean(profitable) if profitable else 0.0
        avg_loss = abs(np.mean(losing)) if losing else 0.0

        total_profit = sum(profitable) if profitable else 0
        total_loss = abs(sum(losing)) if losing else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(profits),
            profitable_trades=len(profitable),
            losing_trades=len(losing),
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            equity_curve=self.equity_curve,
            trade_log=self.trade_log,
        )

    def run_strategy(
        self,
        signals: List[Dict[str, Any]],
        prices: Dict[str, List[float]],
    ) -> BacktestResult:
        """Run backtest on trading signals.

        signals format: [{"timestamp": ..., "action": "buy"/"sell", "price": ..., "quantity": ...}]
        """
        self.reset()

        for i, signal in enumerate(signals):
            action = signal.get("action")
            price = signal.get("price")
            quantity = signal.get("quantity", 1)

            if action == "buy" and self.position is None:
                self.position = {
                    "entry_price": price,
                    "quantity": quantity,
                    "entry_index": i,
                }

            elif action == "sell" and self.position is not None:
                profit = self.execute_trade(
                    self.position["entry_price"],
                    price,
                    self.position["quantity"],
                    "buy",
                )
                self.position = None

        if self.position is not None:
            last_price = signals[-1].get("price") if signals else 0
            self.execute_trade(
                self.position["entry_price"],
                last_price,
                self.position["quantity"],
                "buy",
            )
            self.position = None

        return self.calculate_metrics()

    def run_monte_carlo(
        self,
        signals: List[Dict[str, Any]],
        prices: Dict[str, List[float]],
        num_simulations: int = 100,
    ) -> List[BacktestResult]:
        """Run Monte Carlo simulation."""
        results = []

        for _ in range(num_simulations):
            shuffled_signals = signals.copy()
            np.random.shuffle(shuffled_signals)

            result = self.run_strategy(shuffled_signals, prices)
            results.append(result)

        return results
