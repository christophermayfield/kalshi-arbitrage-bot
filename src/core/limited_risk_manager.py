"""
Limited Risk Trading Mode Manager

Manages $10-$15 trade size limits with comprehensive risk controls
for small account trading with progressive scaling capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskStatus(Enum):
    """Status for risk checks"""

    ALLOWED = "allowed"
    BLOCKED = "blocked"
    COOLDOWN = "cooldown"
    DAILY_LIMIT = "daily_limit"
    LOSS_LIMIT = "loss_limit"
    SIZE_INVALID = "size_invalid"


@dataclass
class LimitedRiskConfig:
    """Configuration for limited risk trading mode"""

    enabled: bool = True
    auto_enable_balance_cents: int = 100000
    min_trade_cents: int = 1000
    max_trade_cents: int = 1500
    max_daily_trades: int = 10
    max_daily_loss_cents: int = 5000
    cooldown_seconds: int = 60
    require_confirmation: bool = False
    min_profit_after_fees_cents: int = 50
    max_fee_percent_of_profit: float = 50.0
    illiquid_volume_threshold_cents: int = 100000

    progressive_scaling_enabled: bool = True
    scaling_threshold_trades: int = 10
    scaling_profitable_streak_required: bool = True
    scaling_increment_cents: int = 500
    max_trade_cents_scaled: int = 3000

    excluded_markets: List[str] = field(default_factory=list)
    min_contracts_per_trade: int = 1


@dataclass
class DailyStats:
    """Daily trading statistics"""

    trades_executed: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    daily_pnl_cents: int = 0
    gross_profit_cents: int = 0
    total_fees_cents: int = 0
    last_trade_time: Optional[datetime] = None
    last_reset_date: datetime = field(default_factory=lambda: datetime.utcnow().date())

    @property
    def win_rate(self) -> float:
        total = self.trades_won + self.trades_lost
        return (self.trades_won / total * 100) if total > 0 else 0.0

    @property
    def avg_profit_per_trade(self) -> float:
        return (
            self.daily_pnl_cents / self.trades_executed
            if self.trades_executed > 0
            else 0.0
        )

    @property
    def fee_impact_percent(self) -> float:
        if self.gross_profit_cents > 0:
            return (self.total_fees_cents / self.gross_profit_cents) * 100
        return 0.0


class LimitedRiskManager:
    """Manages limited risk trading mode with $10-$15 trade constraints."""

    def __init__(self, config: LimitedRiskConfig):
        self.config = config
        self.daily_stats = DailyStats()
        self.enabled = config.enabled
        self.current_max_trade_cents = config.max_trade_cents
        self.consecutive_wins = 0
        self._scaling_level = 0

        logger.info(f"LimitedRiskManager initialized")
        logger.info(
            f"Trade size: ${config.min_trade_cents / 100:.2f}-${config.max_trade_cents / 100:.2f}"
        )
        logger.info(
            f"Daily limits: {config.max_daily_trades} trades, ${config.max_daily_loss_cents / 100:.2f} loss"
        )

    def check_auto_enable(self, account_balance_cents: int) -> bool:
        """Check if limited risk mode should auto-enable based on account balance."""
        should_enable = account_balance_cents < self.config.auto_enable_balance_cents

        if should_enable and not self.enabled:
            self.enabled = True
            logger.info(
                f"AUTO-ENABLED Limited Risk Mode: Balance ${account_balance_cents / 100:.2f} < threshold"
            )

        return self.enabled

    def validate_trade_size(
        self, quantity: int, price_cents: int
    ) -> Tuple[bool, str, int]:
        """Validate that trade size is within $10-$15 range."""
        trade_value_cents = quantity * price_cents

        if trade_value_cents < self.config.min_trade_cents:
            min_contracts = (
                self.config.min_trade_cents + price_cents - 1
            ) // price_cents
            adjusted_value = min_contracts * price_cents

            if adjusted_value <= self.current_max_trade_cents:
                return True, f"Adjusted to {min_contracts} contracts", min_contracts
            else:
                return False, f"Cannot reach minimum $10", 0

        if trade_value_cents > self.current_max_trade_cents:
            max_contracts = self.current_max_trade_cents // price_cents
            adjusted_value = max_contracts * price_cents

            if adjusted_value >= self.config.min_trade_cents:
                return True, f"Adjusted to {max_contracts} contracts", max_contracts
            else:
                return (
                    False,
                    f"Trade exceeds max ${self.current_max_trade_cents / 100:.2f}",
                    0,
                )

        return True, "Valid trade size", quantity

    def can_execute_trade(self) -> Tuple[bool, RiskStatus, str]:
        """Check if a trade can be executed under current limits."""
        if not self.enabled:
            return True, RiskStatus.ALLOWED, "Limited risk mode disabled"

        if self.daily_stats.trades_executed >= self.config.max_daily_trades:
            return (
                False,
                RiskStatus.DAILY_LIMIT,
                f"Daily trade limit: {self.daily_stats.trades_executed}/{self.config.max_daily_trades}",
            )

        if self.daily_stats.daily_pnl_cents <= -self.config.max_daily_loss_cents:
            return False, RiskStatus.LOSS_LIMIT, f"Daily loss limit reached"

        if self.config.cooldown_seconds > 0 and self.daily_stats.last_trade_time:
            elapsed = (
                datetime.utcnow() - self.daily_stats.last_trade_time
            ).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                remaining = int(self.config.cooldown_seconds - elapsed)
                return False, RiskStatus.COOLDOWN, f"Cooldown: {remaining}s"

        return True, RiskStatus.ALLOWED, "Trade allowed"

    def calculate_fee_impact(
        self, contracts: int, price_cents: int, spread_cents: int
    ) -> Dict[str, Any]:
        """Calculate fee impact on a potential trade using Kalshi formula."""
        price_dollars = price_cents / 100.0

        fee_per_side = 0.07 * contracts * price_dollars * (1 - price_dollars)
        total_fee = fee_per_side * 2
        total_fee_cents = int(total_fee * 100) + 1

        gross_profit_cents = contracts * spread_cents
        net_profit_cents = gross_profit_cents - total_fee_cents

        fee_percent = (
            (total_fee_cents / gross_profit_cents * 100)
            if gross_profit_cents > 0
            else 0
        )

        return {
            "contracts": contracts,
            "price_cents": price_cents,
            "spread_cents": spread_cents,
            "gross_profit_cents": gross_profit_cents,
            "total_fee_cents": total_fee_cents,
            "net_profit_cents": net_profit_cents,
            "fee_percent_of_profit": fee_percent,
            "is_viable": net_profit_cents >= self.config.min_profit_after_fees_cents,
            "fee_too_high": fee_percent > self.config.max_fee_percent_of_profit,
        }

    def is_market_eligible(
        self, market_id: str, daily_volume_cents: int = 0
    ) -> Tuple[bool, str]:
        """Check if a market is eligible for limited risk trading."""
        if market_id in self.config.excluded_markets:
            return False, f"Market excluded"

        if daily_volume_cents < self.config.illiquid_volume_threshold_cents:
            return False, f"Insufficient volume"

        return True, "Market eligible"

    def record_trade(
        self, pnl_cents: int, fee_cents: int, gross_profit_cents: int = 0
    ) -> None:
        """Record a completed trade for tracking and progressive scaling."""
        self.daily_stats.trades_executed += 1
        self.daily_stats.daily_pnl_cents += pnl_cents
        self.daily_stats.total_fees_cents += fee_cents
        self.daily_stats.gross_profit_cents += gross_profit_cents
        self.daily_stats.last_trade_time = datetime.utcnow()

        if pnl_cents > 0:
            self.daily_stats.trades_won += 1
            self.consecutive_wins += 1
        else:
            self.daily_stats.trades_lost += 1
            self.consecutive_wins = 0

        if self.config.progressive_scaling_enabled:
            self._check_progressive_scaling()

        logger.info(
            f"Trade: P&L=${pnl_cents / 100:.2f}, Daily={self.daily_stats.trades_executed}/{self.config.max_daily_trades}"
        )

    def _check_progressive_scaling(self) -> None:
        """Check if we should increase trade size based on performance."""
        total_trades = self.daily_stats.trades_won + self.daily_stats.trades_lost

        if total_trades >= self.config.scaling_threshold_trades:
            if (
                self.config.scaling_profitable_streak_required
                and self.daily_stats.win_rate < 50
            ):
                return

            potential_new_max = (
                self.config.max_trade_cents
                + (self._scaling_level + 1) * self.config.scaling_increment_cents
            )
            new_max = min(potential_new_max, self.config.max_trade_cents_scaled)

            if new_max > self.current_max_trade_cents:
                self._scaling_level += 1
                self.current_max_trade_cents = new_max
                logger.info(
                    f"SCALING: Max trade now ${self.current_max_trade_cents / 100:.2f} (level {self._scaling_level})"
                )

    def reset_daily_stats(self, force: bool = False) -> None:
        """Reset daily statistics (call at market open)."""
        today = datetime.utcnow().date()

        if today != self.daily_stats.last_reset_date or force:
            logger.info(
                f"Reset daily stats. Previous: {self.daily_stats.trades_executed} trades"
            )
            self.daily_stats = DailyStats()
            self.daily_stats.last_reset_date = today
            self.consecutive_wins = 0

            if self.config.progressive_scaling_enabled:
                self._scaling_level = 0
                self.current_max_trade_cents = self.config.max_trade_cents

    def get_status(self) -> Dict[str, Any]:
        """Get current status of limited risk mode."""
        can_trade, status, reason = self.can_execute_trade()

        return {
            "enabled": self.enabled,
            "trade_size_range": {
                "min_cents": self.config.min_trade_cents,
                "max_cents": self.current_max_trade_cents,
            },
            "daily_stats": {
                "trades_executed": self.daily_stats.trades_executed,
                "trades_won": self.daily_stats.trades_won,
                "trades_lost": self.daily_stats.trades_lost,
                "win_rate": round(self.daily_stats.win_rate, 1),
                "daily_pnl_cents": self.daily_stats.daily_pnl_cents,
                "fee_impact_percent": round(self.daily_stats.fee_impact_percent, 1),
            },
            "limits": {
                "max_daily_trades": self.config.max_daily_trades,
                "max_daily_loss_cents": self.config.max_daily_loss_cents,
                "cooldown_seconds": self.config.cooldown_seconds,
            },
            "can_trade": can_trade,
            "status": status.value,
            "reason": reason,
            "scaling_level": self._scaling_level,
        }
