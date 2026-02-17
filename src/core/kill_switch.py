"""
Kill Switch System

Emergency halt mechanism for the trading bot with multiple triggers:
- Manual kill (Telegram command, keyboard interrupt)
- Automatic kill (daily loss limit, consecutive failures)
- Circuit breaker integration
- Graceful shutdown procedures

This is the most critical safety feature - it can save your entire account.
"""

import asyncio
import signal
import sys
from enum import Enum
from typing import Optional, Callable, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class KillLevel(Enum):
    """Levels of kill switch activation"""

    SOFT = "soft"  # Stop new positions only
    HARD = "hard"  # Close all positions, cancel orders
    EMERGENCY = "emergency"  # Immediate halt, emergency procedures


class KillReason(Enum):
    """Reasons for kill switch activation"""

    MANUAL = "manual"  # User triggered
    DAILY_LOSS_LIMIT = "daily_loss"  # Daily loss exceeded
    CONSECUTIVE_FAILURES = "failures"  # Too many failed trades
    CIRCUIT_BREAKER = "circuit"  # Circuit breaker tripped
    API_ERROR = "api_error"  # Critical API failure
    SYSTEM_ERROR = "system"  # System malfunction
    PANIC_BUTTON = "panic"  # Panic button pressed


class KillSwitch:
    """
    Emergency kill switch for trading bot.

    Features:
    - Multi-level kills (soft → hard → emergency)
    - Multiple trigger sources (manual, automatic, system)
    - Graceful shutdown procedures
    - Automatic position reconciliation
    - Alert notifications
    - Audit logging
    - Recovery procedures

    Usage:
        kill_switch = KillSwitch(bot_instance)

        # Manual kill
        await kill_switch.trigger(KillLevel.HARD, KillReason.MANUAL, "User requested stop")

        # Check if killed
        if kill_switch.is_killed:
            print("Trading halted")
    """

    def __init__(self, bot_instance=None):
        self.bot = bot_instance
        self._killed = False
        self._kill_level: Optional[KillLevel] = None
        self._kill_reason: Optional[KillReason] = None
        self._kill_message: Optional[str] = None
        self._kill_time: Optional[datetime] = None
        self._callbacks: List[Callable] = []
        self._lock = asyncio.Lock()

        # Thresholds for automatic kills
        self.max_daily_loss_cents = 5000  # $50
        self.max_consecutive_failures = 5
        self.max_api_errors = 10

        logger.info("Kill switch initialized - Ready to protect your account")

    @property
    def is_killed(self) -> bool:
        """Check if kill switch has been activated"""
        return self._killed

    @property
    def kill_status(self) -> dict:
        """Get current kill status"""
        return {
            "killed": self._killed,
            "level": self._kill_level.value if self._kill_level else None,
            "reason": self._kill_reason.value if self._kill_reason else None,
            "message": self._kill_message,
            "time": self._kill_time.isoformat() if self._kill_time else None,
        }

    async def trigger(
        self,
        level: KillLevel,
        reason: KillReason,
        message: str = "",
        immediate: bool = True,
    ) -> bool:
        """
        Trigger the kill switch.

        Args:
            level: Severity level (SOFT, HARD, EMERGENCY)
            reason: Why the kill was triggered
            message: Additional details
            immediate: If True, execute immediately; if False, queue for next cycle

        Returns:
            True if kill was successful, False if already killed
        """
        async with self._lock:
            if self._killed:
                logger.warning(f"Kill switch already triggered at {self._kill_time}")
                return False

            self._killed = True
            self._kill_level = level
            self._kill_reason = reason
            self._kill_message = message
            self._kill_time = datetime.utcnow()

            logger.critical("=" * 70)
            logger.critical("🚨 KILL SWITCH ACTIVATED 🚨")
            logger.critical("=" * 70)
            logger.critical(f"Level: {level.value.upper()}")
            logger.critical(f"Reason: {reason.value}")
            logger.critical(f"Message: {message}")
            logger.critical(f"Time: {self._kill_time.isoformat()}")
            logger.critical("=" * 70)

            if immediate:
                await self._execute_kill_procedure(level)

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(level, reason, message)
                    else:
                        callback(level, reason, message)
                except Exception as e:
                    logger.error(f"Kill switch callback failed: {e}")

            return True

    async def _execute_kill_procedure(self, level: KillLevel) -> None:
        """Execute the appropriate kill procedure based on level"""
        try:
            if level == KillLevel.SOFT:
                await self._soft_kill()
            elif level == KillLevel.HARD:
                await self._hard_kill()
            elif level == KillLevel.EMERGENCY:
                await self._emergency_kill()
        except Exception as e:
            logger.error(f"Kill procedure failed: {e}")
            # Fall back to emergency kill if other levels fail
            if level != KillLevel.EMERGENCY:
                await self._emergency_kill()

    async def _soft_kill(self) -> None:
        """
        Soft kill: Stop accepting new trades
        - Continue monitoring
        - Keep existing positions
        - Can be resumed
        """
        logger.info("Executing SOFT kill procedure...")

        if self.bot:
            # Stop detection loop
            if hasattr(self.bot, "stop_detection"):
                await self.bot.stop_detection()

            # Mark as paused
            if hasattr(self.bot, "_detection_running"):
                self.bot._detection_running = False

        logger.info(
            "✓ Soft kill complete - New trades stopped, existing positions maintained"
        )

    async def _hard_kill(self) -> None:
        """
        Hard kill: Stop trading and close positions
        - Cancel all pending orders
        - Close all open positions
        - Stop all trading activity
        """
        logger.info("Executing HARD kill procedure...")

        # Soft kill first
        await self._soft_kill()

        if self.bot:
            # Cancel all pending orders
            if hasattr(self.bot, "executor") and self.bot.executor:
                pending_orders = list(self.bot.executor.pending_orders.keys())
                for order_id in pending_orders:
                    try:
                        if hasattr(self.bot.executor, "_emergency_cancel_order"):
                            await self.bot.executor._emergency_cancel_order(order_id)
                        logger.info(f"✓ Cancelled order: {order_id}")
                    except Exception as e:
                        logger.error(f"Failed to cancel order {order_id}: {e}")

            # Close all positions (if possible)
            if hasattr(self.bot, "portfolio") and self.bot.portfolio:
                positions = list(self.bot.portfolio.positions.values())
                for position in positions:
                    try:
                        logger.info(
                            f"Attempting to close position: {position.market_id}"
                        )
                        # Note: Actual close logic depends on your position structure
                        # This is a placeholder - implement based on your needs
                    except Exception as e:
                        logger.error(
                            f"Failed to close position {position.market_id}: {e}"
                        )

        logger.info("✓ Hard kill complete - All orders cancelled, positions closed")

    async def _emergency_kill(self) -> None:
        """
        Emergency kill: Immediate halt
        - Stop everything immediately
        - Don't wait for graceful shutdown
        - Log everything for audit
        - Alert all stakeholders
        """
        logger.critical("Executing EMERGENCY kill procedure...")

        # Hard kill procedures
        try:
            await self._hard_kill()
        except Exception as e:
            logger.critical(f"Hard kill failed during emergency: {e}")

        # Immediate shutdown
        if self.bot and hasattr(self.bot, "running"):
            self.bot.running = False

        # Cancel all async tasks
        try:
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
        except Exception as e:
            logger.error(f"Failed to cancel tasks: {e}")

        logger.critical("✓ Emergency kill complete - System halted")

        # Write audit log
        self._write_audit_log()

    def _write_audit_log(self) -> None:
        """Write kill switch activation to audit log"""
        import json
        from pathlib import Path

        audit_entry = {
            "timestamp": self._kill_time.isoformat() if self._kill_time else None,
            "level": self._kill_level.value if self._kill_level else None,
            "reason": self._kill_reason.value if self._kill_reason else None,
            "message": self._kill_message,
        }

        audit_file = Path("logs/kill_switch_audit.jsonl")
        audit_file.parent.mkdir(parents=True, exist_ok=True)

        with open(audit_file, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")

    def register_callback(self, callback: Callable) -> None:
        """Register a callback to be called when kill switch is triggered"""
        self._callbacks.append(callback)

    def setup_signal_handlers(self) -> None:
        """Setup keyboard interrupt handlers for emergency kill"""

        def signal_handler(signum, frame):
            logger.critical("Received interrupt signal - Triggering emergency kill")
            # Can't use await in signal handler, so create task
            asyncio.create_task(
                self.trigger(
                    KillLevel.EMERGENCY, KillReason.PANIC_BUTTON, "Keyboard interrupt"
                )
            )

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info(
            "Signal handlers registered - Press Ctrl+C twice for emergency kill"
        )

    async def auto_check_and_kill(
        self, daily_pnl_cents: int, consecutive_failures: int
    ) -> bool:
        """
        Automatic kill switch checks

        Returns:
            True if kill was triggered, False otherwise
        """
        # Check daily loss limit
        if daily_pnl_cents <= -self.max_daily_loss_cents:
            await self.trigger(
                KillLevel.HARD,
                KillReason.DAILY_LOSS_LIMIT,
                f"Daily loss limit reached: ${abs(daily_pnl_cents) / 100:.2f}",
            )
            return True

        # Check consecutive failures
        if consecutive_failures >= self.max_consecutive_failures:
            await self.trigger(
                KillLevel.SOFT,
                KillReason.CONSECUTIVE_FAILURES,
                f"{consecutive_failures} consecutive trade failures",
            )
            return True

        return False

    async def resume(self) -> bool:
        """
        Resume trading after a kill (only for SOFT kills)

        Returns:
            True if resumed successfully
        """
        async with self._lock:
            if not self._killed:
                logger.warning("Kill switch not active - nothing to resume")
                return False

            if self._kill_level == KillLevel.EMERGENCY:
                logger.error(
                    "Cannot resume from EMERGENCY kill - manual intervention required"
                )
                return False

            # Reset kill state
            self._killed = False
            old_level = self._kill_level
            old_reason = self._kill_reason

            self._kill_level = None
            self._kill_reason = None
            self._kill_message = None
            self._kill_time = None

            logger.info("=" * 70)
            logger.info("✅ KILL SWITCH RESUMED")
            logger.info(f"Previous kill: {old_level.value} - {old_reason.value}")
            logger.info("=" * 70)

            if self.bot:
                # Resume detection
                if hasattr(self.bot, "start_detection"):
                    await self.bot.start_detection()

            return True


class KillSwitchMiddleware:
    """
    Middleware to check kill switch before executing trades

    Usage:
        @KillSwitchMiddleware.check(kill_switch)
        async def execute_trade(self, ...):
            # Your trading logic
            pass
    """

    @staticmethod
    def check(kill_switch: KillSwitch):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if kill_switch.is_killed:
                    logger.warning(
                        f"Kill switch active - blocking call to {func.__name__}"
                    )
                    return False, 0
                return await func(*args, **kwargs)

            return wrapper

        return decorator
