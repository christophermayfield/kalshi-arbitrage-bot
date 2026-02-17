"""
Limited Risk Trading Executor

Wraps TradingExecutor with additional $10-$15 trade constraints
and automatic validation for small account trading.
"""

from typing import Tuple, Optional, Dict, Any
import logging

from src.execution.trading import TradingExecutor
from src.core.arbitrage import ArbitrageOpportunity
from src.core.limited_risk_manager import LimitedRiskManager, RiskStatus

logger = logging.getLogger(__name__)


class LimitedRiskExecutor:
    """
    Executor that enforces limited risk constraints on all trades.

    Wraps the base TradingExecutor and adds:
    - Trade size validation ($10-$15)
    - Daily limit checks (10 trades, $50 loss)
    - Cooldown enforcement (60s between trades)
    - Fee impact validation
    - Market liquidity checks
    - Progressive scaling support
    """

    def __init__(
        self, base_executor: TradingExecutor, risk_manager: LimitedRiskManager
    ):
        self.base_executor = base_executor
        self.risk_manager = risk_manager
        self._bypass_limits = False  # For testing/emergencies

    async def execute_arbitrage(
        self, opportunity: ArbitrageOpportunity
    ) -> Tuple[bool, int]:
        """
        Execute arbitrage with limited risk validation.

        Args:
            opportunity: Arbitrage opportunity to execute

        Returns:
            Tuple of (success, profit_in_cents)
        """
        if not self.risk_manager.enabled or self._bypass_limits:
            # Pass through to base executor
            return await self.base_executor.execute_arbitrage(opportunity)

        # Step 1: Check daily limits
        can_trade, status, reason = self.risk_manager.can_execute_trade()
        if not can_trade:
            logger.warning(f"Trade blocked: {status.value} - {reason}")
            return False, 0

        # Step 2: Validate trade size
        is_valid, msg, adjusted_qty = self.risk_manager.validate_trade_size(
            opportunity.quantity, opportunity.buy_price
        )

        if not is_valid:
            logger.warning(f"Trade size invalid: {msg}")
            return False, 0

        # Adjust quantity if needed
        if adjusted_qty != opportunity.quantity:
            opportunity.quantity = adjusted_qty
            logger.info(f"Adjusted quantity to {adjusted_qty} contracts")

        # Step 3: Calculate fee impact
        spread_cents = opportunity.sell_price - opportunity.buy_price
        fee_analysis = self.risk_manager.calculate_fee_impact(
            opportunity.quantity, opportunity.buy_price, spread_cents
        )

        if not fee_analysis["is_viable"]:
            logger.warning(
                f"Trade not viable: Net profit ${fee_analysis['net_profit_cents'] / 100:.2f} "
                f"below minimum ${self.risk_manager.config.min_profit_after_fees_cents / 100:.2f}"
            )
            return False, 0

        if fee_analysis["fee_too_high"]:
            logger.warning(
                f"Trade rejected: Fees {fee_analysis['fee_percent_of_profit']:.1f}% of profit "
                f"exceeds {self.risk_manager.config.max_fee_percent_of_profit:.1f}% limit"
            )
            return False, 0

        # Step 4: Execute trade
        logger.info(
            f"Executing limited risk trade: {opportunity.quantity} contracts, "
            f"expected net profit ${fee_analysis['net_profit_cents'] / 100:.2f}, "
            f"fees {fee_analysis['fee_percent_of_profit']:.1f}%"
        )

        success, actual_profit = await self.base_executor.execute_arbitrage(opportunity)

        # Step 5: Record trade result
        if success:
            self.risk_manager.record_trade(
                pnl_cents=actual_profit,
                fee_cents=fee_analysis["total_fee_cents"],
                gross_profit_cents=fee_analysis["gross_profit_cents"],
            )
            logger.info(f"Trade successful: ${actual_profit / 100:.2f} profit")
        else:
            logger.warning("Trade failed")

        return success, actual_profit

    def get_status(self) -> Dict[str, Any]:
        """Get current status of limited risk executor."""
        return {
            "enabled": self.risk_manager.enabled,
            "risk_manager_status": self.risk_manager.get_status(),
            "bypass_limits": self._bypass_limits,
        }

    def set_bypass_limits(self, bypass: bool) -> None:
        """
        Set whether to bypass limited risk limits.

        WARNING: Only use for testing or emergencies!
        """
        self._bypass_limits = bypass
        if bypass:
            logger.warning(
                "LIMITED RISK LIMITS BYPASSED - Trading without constraints!"
            )
        else:
            logger.info("Limited risk limits re-enabled")

    async def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.risk_manager.reset_daily_stats(force=True)
        logger.info("Daily stats reset")
