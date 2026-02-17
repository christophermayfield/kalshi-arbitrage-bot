"""
Position Sizing Optimizer - Optimize position sizes based on risk and Kelly Criterion.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.utils.logging_utils import get_logger

logger = get_logger("position_sizing")


@dataclass
class PositionSize:
    position_id: str
    market_id: str
    recommended_size: float
    risk_percent: float
    kelly_fraction: float
    confidence: float


class PositionSizingOptimizer:
    def __init__(
        self,
        max_position_size: float = 0.2,  # Max 20% of portfolio
        max_total_risk: float = 0.1,  # Max 10% total risk
        kelly_multiplier: float = 0.25,  # Use 25% of Kelly (conservative)
    ):
        self.max_position_size = max_position_size
        self.max_total_risk = max_total_risk
        self.kelly_multiplier = kelly_multiplier

        self._win_rate_history: List[float] = []
        self._avg_win_history: List[float] = []
        self._avg_loss_history: List[float] = []

    def calculate_kelly_fraction(
        self, win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        """Calculate Kelly Criterion for position sizing.

        Kelly % = W - (1-W)/R
        Where W = win rate, R = win/loss ratio
        """
        if avg_loss == 0 or win_rate <= 0:
            return 0.0

        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Apply multiplier for conservatism
        kelly = kelly * self.kelly_multiplier

        # Clamp between 0 and max position size
        return max(0, min(kelly, self.max_position_size))

    def calculate_sharpe_sizing(self, sharpe_ratio: float, volatility: float) -> float:
        """Calculate position size based on Sharpe ratio."""
        if volatility <= 0 or sharpe_ratio <= 0:
            return self.max_position_size / 2

        # Risk parity approach: size inversely to volatility
        base_size = min(sharpe_ratio * 0.1, self.max_position_size)

        return base_size

    def calculate_volatility_sizing(
        self, price: float, target_risk: float = 0.02
    ) -> int:
        """Calculate position size based on volatility.

        Uses ATR-style approach: size based on price volatility
        """
        if price <= 0:
            return 1

        # Simple volatility sizing: risk a fixed percent
        risk_per_unit = price * target_risk

        # Convert to contract count (assuming integer contracts)
        return max(1, int(1 / risk_per_unit) if risk_per_unit > 0 else 1)

    def calculate_optimal_size(
        self,
        market_id: str,
        confidence: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        sharpe_ratio: Optional[float] = None,
        volatility: Optional[float] = None,
        portfolio_value: float = 100000,
    ) -> PositionSize:
        """Calculate optimal position size using multiple factors."""

        # Calculate Kelly-based size
        kelly_size = 0.0
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
            kelly_size = kelly_fraction * portfolio_value

            # Update history
            self._win_rate_history.append(win_rate)
            self._avg_win_history.append(avg_win)
            self._avg_loss_history.append(avg_loss)

        # Calculate Sharpe-based size
        sharpe_size = 0.0
        if sharpe_ratio is not None and volatility is not None:
            sharpe_fraction = self.calculate_sharpe_sizing(sharpe_ratio, volatility)
            sharpe_size = sharpe_fraction * portfolio_value

        # Combine sizes (weighted average)
        if kelly_size > 0 and sharpe_size > 0:
            # Weight by confidence
            combined_size = (
                kelly_size * confidence + sharpe_size * (1 - confidence)
            ) / 2
        elif kelly_size > 0:
            combined_size = kelly_size * confidence
        elif sharpe_size > 0:
            combined_size = sharpe_size
        else:
            combined_size = portfolio_value * self.max_position_size * 0.5

        # Apply constraints
        final_size = min(combined_size, portfolio_value * self.max_position_size)

        risk_percent = final_size / portfolio_value if portfolio_value > 0 else 0

        return PositionSize(
            position_id=f"pos_{market_id}_{int(np.random.randint(0, 10000))}",
            market_id=market_id,
            recommended_size=final_size,
            risk_percent=risk_percent,
            kelly_fraction=kelly_size / portfolio_value if portfolio_value > 0 else 0,
            confidence=confidence,
        )

    def calculate_sizing_from_trades(
        self, trades: List[Dict[str, Any]], portfolio_value: float
    ) -> Dict[str, Any]:
        """Calculate optimal sizing based on historical trades."""
        if not trades:
            return {
                "recommended_size": portfolio_value * 0.1,
                "win_rate": 0.5,
                "avg_win": 100,
                "avg_loss": 100,
                "kelly_fraction": 0.1,
            }

        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]

        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = np.mean([t.get("pnl", 0) for t in wins]) if wins else 100
        avg_loss = abs(np.mean([t.get("pnl", 0) for t in losses])) if losses else 100

        kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)

        # Use historical average if available
        if self._win_rate_history:
            avg_win_rate = np.mean(self._win_rate_history[-20:])
            avg_win_amt = (
                np.mean(self._avg_win_history[-20:])
                if self._avg_win_history
                else avg_win
            )
            avg_loss_amt = (
                np.mean(self._avg_loss_history[-20:])
                if self._avg_loss_history
                else avg_loss
            )

            kelly_fraction = self.calculate_kelly_fraction(
                avg_win_rate, avg_win_amt, avg_loss_amt
            )

        recommended_size = kelly_fraction * portfolio_value

        return {
            "recommended_size": recommended_size,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "kelly_fraction": kelly_fraction,
            "num_trades": len(trades),
        }

    def get_historical_stats(self) -> Dict[str, Any]:
        """Get historical sizing statistics."""
        return {
            "avg_win_rate": np.mean(self._win_rate_history)
            if self._win_rate_history
            else None,
            "avg_win_amount": np.mean(self._avg_win_history)
            if self._avg_win_history
            else None,
            "avg_loss_amount": np.mean(self._avg_loss_history)
            if self._avg_loss_history
            else None,
            "total_observations": len(self._win_rate_history),
        }
