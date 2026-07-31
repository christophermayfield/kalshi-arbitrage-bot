"""Advanced risk management with position-level controls and dynamic sizing."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple
from enum import Enum

logger = logging.getLogger("advanced_risk_manager")


class RiskTier(Enum):
    """Risk tiers based on equity curve."""
    CONSERVATIVE = "conservative"  # High recent losses
    NORMAL = "normal"  # Stable equity
    AGGRESSIVE = "aggressive"  # Winning streak


@dataclass
class PositionRisk:
    """Risk parameters for a single position."""
    market_id: str
    entry_price_cents: int
    quantity: int
    stop_loss_cents: int  # Price at which to exit if loss threshold hit
    take_profit_cents: int  # Price at which to exit if profit target hit
    max_loss_cents: int  # Maximum loss allowed
    max_gain_target_cents: int  # Profit target
    correlation_limit: float = 0.8  # Max correlation with other positions
    holding_period_seconds: int = 3600  # Max holding period


@dataclass
class EquityCurve:
    """Track equity curve over time."""
    timestamps: List[datetime] = field(default_factory=list)
    balances: List[int] = field(default_factory=list)
    peak_equity: int = 0
    max_drawdown: float = 0.0  # As percentage

    def update(self, current_balance: int):
        """Update equity curve."""
        now = datetime.now(timezone.utc)
        self.timestamps.append(now)
        self.balances.append(current_balance)

        if current_balance > self.peak_equity:
            self.peak_equity = current_balance

        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_balance) / self.peak_equity
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

    def get_risk_tier(self) -> RiskTier:
        """Determine risk tier based on recent performance."""
        if not self.balances or len(self.balances) < 10:
            return RiskTier.NORMAL

        # Last 10 updates
        recent_changes = [
            self.balances[i] - self.balances[i - 1]
            for i in range(len(self.balances) - 10, len(self.balances))
        ]

        avg_change = sum(recent_changes) / len(recent_changes)
        win_count = sum(1 for change in recent_changes if change > 0)
        win_rate = win_count / len(recent_changes)

        if self.max_drawdown > 0.20:  # More than 20% drawdown
            return RiskTier.CONSERVATIVE
        elif win_rate > 0.7:  # More than 70% win rate
            return RiskTier.AGGRESSIVE
        else:
            return RiskTier.NORMAL


class AdvancedRiskManager:
    """Advanced risk management system."""

    def __init__(
        self,
        initial_balance_cents: int,
        max_daily_loss_cents: int = 10000,
        max_position_value_cents: int = 50000,
        risk_per_trade_percent: float = 0.02,  # 2% per trade
    ):
        """
        Initialize advanced risk manager.

        Args:
            initial_balance_cents: Starting capital
            max_daily_loss_cents: Daily loss limit
            max_position_value_cents: Max value per position
            risk_per_trade_percent: Risk per trade as % of capital
        """
        self.initial_balance_cents = initial_balance_cents
        self.max_daily_loss_cents = max_daily_loss_cents
        self.max_position_value_cents = max_position_value_cents
        self.risk_per_trade_percent = risk_per_trade_percent

        self.positions: Dict[str, PositionRisk] = {}
        self.daily_loss_cents = 0
        self.day_start_time = datetime.now(timezone.utc)
        self.equity_curve = EquityCurve()

        logger.info(
            f"Advanced risk manager initialized with "
            f"${initial_balance_cents / 100:.2f} capital"
        )

    def can_open_position(
        self,
        market_id: str,
        entry_price_cents: int,
        quantity: int,
        current_balance_cents: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if position can be opened.

        Returns:
            (allowed, reason_if_denied)
        """
        position_value = entry_price_cents * quantity

        # Check 1: Position size limit
        if position_value > self.max_position_value_cents:
            return False, f"Position value ${position_value / 100:.2f} exceeds limit"

        # Check 2: Daily loss limit
        if self.daily_loss_cents >= self.max_daily_loss_cents:
            return False, f"Daily loss limit ${self.max_daily_loss_cents / 100:.2f} reached"

        # Check 3: Sufficient balance
        if position_value > current_balance_cents:
            return False, "Insufficient balance"

        # Check 4: Correlation check
        correlation_issue = self._check_correlation(market_id)
        if correlation_issue:
            return False, correlation_issue

        # Check 5: Risk tier limits
        risk_tier = self.equity_curve.get_risk_tier()
        if risk_tier == RiskTier.CONSERVATIVE:
            return False, "In conservative risk tier due to recent losses"

        return True, None

    def open_position(
        self,
        market_id: str,
        entry_price_cents: int,
        quantity: int,
        stop_loss_percent: float = 0.02,  # 2% stop loss
        take_profit_percent: float = 0.05,  # 5% take profit
    ) -> PositionRisk:
        """
        Open a position with risk parameters.

        Args:
            market_id: Market identifier
            entry_price_cents: Entry price in cents
            quantity: Number of contracts
            stop_loss_percent: Stop loss as % below entry
            take_profit_percent: Take profit as % above entry
        """
        # Calculate risk levels
        stop_loss = int(entry_price_cents * (1 - stop_loss_percent))
        take_profit = int(entry_price_cents * (1 + take_profit_percent))

        max_loss = (entry_price_cents - stop_loss) * quantity
        max_gain = (take_profit - entry_price_cents) * quantity

        position = PositionRisk(
            market_id=market_id,
            entry_price_cents=entry_price_cents,
            quantity=quantity,
            stop_loss_cents=stop_loss,
            take_profit_cents=take_profit,
            max_loss_cents=max_loss,
            max_gain_target_cents=max_gain,
        )

        self.positions[market_id] = position
        logger.info(
            f"Opened position in {market_id}: "
            f"Entry=${entry_price_cents / 100:.2f}, "
            f"SL=${stop_loss / 100:.2f}, TP=${take_profit / 100:.2f}"
        )

        return position

    def close_position(
        self,
        market_id: str,
        exit_price_cents: int,
        realized_pnl_cents: int,
    ) -> bool:
        """
        Close a position and update risk tracking.

        Returns:
            True if position was closed
        """
        if market_id not in self.positions:
            return False

        position = self.positions[market_id]

        # Update daily loss
        if realized_pnl_cents < 0:
            self.daily_loss_cents += abs(realized_pnl_cents)

        # Log position metrics
        logger.info(
            f"Closed position in {market_id}: "
            f"Exit=${exit_price_cents / 100:.2f}, "
            f"PnL=${realized_pnl_cents / 100:.2f}"
        )

        del self.positions[market_id]
        return True

    def should_close_position(
        self,
        market_id: str,
        current_price_cents: int,
        current_balance_cents: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if position should be closed based on risk rules.

        Returns:
            (should_close, reason)
        """
        if market_id not in self.positions:
            return False, None

        position = self.positions[market_id]

        # Check 1: Stop loss hit
        if current_price_cents <= position.stop_loss_cents:
            max_loss = (position.entry_price_cents - current_price_cents) * position.quantity
            return True, f"Stop loss triggered, loss=${max_loss / 100:.2f}"

        # Check 2: Take profit hit
        if current_price_cents >= position.take_profit_cents:
            max_gain = (current_price_cents - position.entry_price_cents) * position.quantity
            return True, f"Take profit triggered, gain=${max_gain / 100:.2f}"

        # Check 3: Holding period exceeded
        if hasattr(position, 'open_time'):
            elapsed = (datetime.now(timezone.utc) - position.open_time).total_seconds()
            if elapsed > position.holding_period_seconds:
                return True, f"Holding period ({position.holding_period_seconds}s) exceeded"

        # Check 4: Daily loss limit approaching
        if self.daily_loss_cents > self.max_daily_loss_cents * 0.9:
            return True, "Approaching daily loss limit"

        return False, None

    def calculate_position_size(
        self,
        market_id: str,
        entry_price_cents: int,
        stop_loss_percent: float,
        current_balance_cents: int,
    ) -> int:
        """
        Calculate position size based on risk management rules.

        Uses Kelly Criterion or fixed risk amount.
        """
        # Maximum risk per trade
        max_risk = int(current_balance_cents * self.risk_per_trade_percent)

        # Calculate quantity based on stop loss
        stop_loss_amount = int(entry_price_cents * stop_loss_percent)
        max_quantity = max_risk // stop_loss_amount if stop_loss_amount > 0 else 1

        # Apply position value limit
        max_quantity_by_value = self.max_position_value_cents // entry_price_cents

        return min(max_quantity, max_quantity_by_value)

    def _check_correlation(self, new_market_id: str) -> Optional[str]:
        """Check if new position would exceed correlation limits."""
        # Simplified: prevent having positions in multiple markets
        # In production, calculate actual correlation between markets
        if len(self.positions) > 0:
            existing_markets = list(self.positions.keys())
            return f"Position already open in {existing_markets[0]}, limit one position"

        return None

    def update_equity(self, current_balance_cents: int):
        """Update equity curve tracking."""
        self.equity_curve.update(current_balance_cents)

    def reset_daily_limits(self):
        """Reset daily limits at start of day."""
        self.daily_loss_cents = 0
        self.day_start_time = datetime.now(timezone.utc)
        logger.info("Daily limits reset")

    def get_portfolio_metrics(self, positions_values: Dict[str, int]) -> Dict:
        """Get portfolio-level metrics."""
        total_position_value = sum(positions_values.values())

        return {
            "current_balance": self.equity_curve.balances[-1] if self.equity_curve.balances else 0,
            "total_position_value": total_position_value,
            "daily_loss_cents": self.daily_loss_cents,
            "daily_loss_remaining": self.max_daily_loss_cents - self.daily_loss_cents,
            "max_drawdown": self.equity_curve.max_drawdown,
            "risk_tier": self.equity_curve.get_risk_tier().value,
            "open_positions": len(self.positions),
        }
