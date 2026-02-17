"""
Telegram Bot Integration

Provides remote monitoring and control via Telegram:
- Real-time alerts for trades, errors, and important events
- Commands to check status, P&L, and positions
- Emergency kill switch via /stop command
- Daily summaries and reports

Setup:
1. Create a bot with @BotFather on Telegram
2. Get your bot token
3. Add your chat ID to config
4. Start the bot
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from telegram import Update, Bot
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.utils.config import Config

logger = logging.getLogger(__name__)


class TelegramBot:
    """
    Telegram bot for monitoring and controlling the trading bot.

    Commands:
    /start - Start the bot and show welcome message
    /status - Check if bot is running and healthy
    /pnl - Show today's profit/loss
    /positions - Show current open positions
    /history - Show recent trades
    /stop - Emergency stop (kill switch)
    /resume - Resume trading after stop
    /balance - Show account balance
    /limits - Show daily limits status
    /alerts - Show recent alerts
    /help - Show all available commands
    """

    def __init__(self, config: Config, bot_instance=None):
        self.config = config
        self.bot_instance = bot_instance
        self.application: Optional[Application] = None
        self._running = False

        # Get config
        self.token = config.get("telegram.token", "")
        self.chat_id = config.get("telegram.chat_id", "")

        if not self.token:
            logger.warning("Telegram token not configured - bot will not start")
        if not self.chat_id:
            logger.warning("Telegram chat ID not configured - bot will not start")

    async def start(self) -> None:
        """Start the Telegram bot"""
        if not self.token or not self.chat_id:
            logger.error("Cannot start Telegram bot: missing token or chat_id")
            return

        try:
            self.application = Application.builder().token(self.token).build()

            # Add command handlers
            self.application.add_handler(CommandHandler("start", self._cmd_start))
            self.application.add_handler(CommandHandler("status", self._cmd_status))
            self.application.add_handler(CommandHandler("pnl", self._cmd_pnl))
            self.application.add_handler(
                CommandHandler("positions", self._cmd_positions)
            )
            self.application.add_handler(CommandHandler("history", self._cmd_history))
            self.application.add_handler(CommandHandler("stop", self._cmd_stop))
            self.application.add_handler(CommandHandler("resume", self._cmd_resume))
            self.application.add_handler(CommandHandler("balance", self._cmd_balance))
            self.application.add_handler(CommandHandler("limits", self._cmd_limits))
            self.application.add_handler(CommandHandler("alerts", self._cmd_alerts))
            self.application.add_handler(CommandHandler("help", self._cmd_help))

            # Start the bot
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()

            self._running = True
            logger.info("✅ Telegram bot started successfully")

            # Send startup message
            await self.send_message("🚀 Trading bot started and ready!")

        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")

    async def stop(self) -> None:
        """Stop the Telegram bot"""
        if self.application:
            await self.application.stop()
            self._running = False
            logger.info("Telegram bot stopped")

    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send a message to the configured chat"""
        if not self.application or not self._running:
            return False

        try:
            await self.application.bot.send_message(
                chat_id=self.chat_id, text=message, parse_mode=parse_mode
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def send_trade_notification(self, trade_data: Dict[str, Any]) -> None:
        """Send a notification about a trade"""
        profit = trade_data.get("net_profit_cents", 0)
        profit_emoji = "🟢" if profit > 0 else "🔴"

        message = f"""
{profit_emoji} <b>Trade Executed</b>

📊 Market: <code>{trade_data.get("market_id", "N/A")}</code>
💰 Profit: ${profit / 100:.2f}
📈 Win Rate: {trade_data.get("win_rate", 0):.1f}%
📊 Daily Trades: {trade_data.get("trades_today", 0)}/10
"""
        await self.send_message(message)

    async def send_alert(self, level: str, message: str) -> None:
        """Send an alert notification"""
        emoji_map = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨", "success": "✅"}
        emoji = emoji_map.get(level, "ℹ️")

        formatted_message = f"""
{emoji} <b>Alert - {level.upper()}</b>

{message}

<i>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</i>
"""
        await self.send_message(formatted_message)

    async def send_daily_summary(self, summary: Dict[str, Any]) -> None:
        """Send daily trading summary"""
        pnl = summary.get("daily_pnl_cents", 0)
        pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"

        message = f"""
📊 <b>Daily Trading Summary</b>

{pnl_emoji} P&L: ${pnl / 100:.2f}
📈 Win Rate: {summary.get("win_rate", 0):.1f}%
✅ Wins: {summary.get("trades_won", 0)}
❌ Losses: {summary.get("trades_lost", 0)}
📊 Total Trades: {summary.get("trades_executed", 0)}
💵 Balance: ${summary.get("balance_cents", 0) / 100:.2f}
🔥 Current Streak: {summary.get("consecutive_wins", 0)} wins

<i>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</i>
"""
        await self.send_message(message)

    # Command Handlers

    async def _cmd_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command"""
        welcome_message = """
🤖 <b>Kalshi Arbitrage Bot</b>

Welcome! I can help you monitor and control your trading bot.

<b>Available Commands:</b>
/status - Check bot health
/pnl - Today's profit/loss
/positions - Current positions
/history - Recent trades
/balance - Account balance
/limits - Daily limits
/alerts - Recent alerts
/stop - Emergency stop
/help - Show all commands

<i>Use these commands anytime to check on your bot!</i>
"""
        await update.message.reply_text(welcome_message, parse_mode="HTML")

    async def _cmd_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /status command"""
        if not self.bot_instance:
            await update.message.reply_text("❌ Bot instance not connected")
            return

        try:
            status = self.bot_instance.get_health_status()

            status_emoji = "🟢" if status.get("status") == "healthy" else "🔴"
            running = "✅ Running" if status.get("running") else "⏹️ Stopped"

            message = f"""
{status_emoji} <b>Bot Status</b>

{running}
📊 Status: {status.get("status", "unknown")}
🕐 Last Update: {status.get("timestamp", "N/A")}
📈 Markets Tracked: {status.get("components", {}).get("websocket", {}).get("subscribed_markets", 0)}
🔒 Circuit Breaker: {status.get("components", {}).get("circuit_breaker", {}).get("state", "unknown")}
"""
            await update.message.reply_text(message, parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"❌ Error getting status: {e}")

    async def _cmd_pnl(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /pnl command"""
        if not self.bot_instance:
            await update.message.reply_text("❌ Bot instance not connected")
            return

        try:
            # Get limited risk status if available
            if hasattr(self.bot_instance, "get_limited_risk_status"):
                lr_status = self.bot_instance.get_limited_risk_status()
                daily_stats = lr_status.get("daily_stats", {})

                pnl = daily_stats.get("daily_pnl_cents", 0)
                pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"

                message = f"""
{pnl_emoji} <b>Today's P&L</b>

💰 Net P&L: ${pnl / 100:.2f}
📈 Win Rate: {daily_stats.get("win_rate", 0):.1f}%
✅ Wins: {daily_stats.get("trades_won", 0)}
❌ Losses: {daily_stats.get("trades_lost", 0)}
📊 Total Trades: {daily_stats.get("trades_executed", 0)}/10

<i>Updated: {datetime.now().strftime("%H:%M:%S")}</i>
"""
            else:
                message = "📊 P&L tracking not available in current mode"

            await update.message.reply_text(message, parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"❌ Error getting P&L: {e}")

    async def _cmd_positions(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /positions command"""
        if not self.bot_instance:
            await update.message.reply_text("❌ Bot instance not connected")
            return

        try:
            # Get positions from portfolio
            if hasattr(self.bot_instance, "portfolio"):
                positions = self.bot_instance.portfolio.positions

                if not positions:
                    await update.message.reply_text("📭 No open positions")
                    return

                message = "📊 <b>Open Positions</b>\n\n"
                for pos in positions.values():
                    message += f"• {pos.market_id}: {pos.quantity} contracts\n"

                await update.message.reply_text(message, parse_mode="HTML")
            else:
                await update.message.reply_text("📭 No positions data available")

        except Exception as e:
            await update.message.reply_text(f"❌ Error getting positions: {e}")

    async def _cmd_history(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /history command"""
        if not self.bot_instance:
            await update.message.reply_text("❌ Bot instance not connected")
            return

        try:
            # Get recent trades from limited risk metrics if available
            if hasattr(self.bot_instance, "limited_risk_metrics"):
                recent = self.bot_instance.limited_risk_metrics.get_recent_trades(5)

                if not recent:
                    await update.message.reply_text("📭 No recent trades")
                    return

                message = "📈 <b>Recent Trades</b>\n\n"
                for trade in recent:
                    profit_emoji = "🟢" if trade["net_profit"] > 0 else "🔴"
                    message += f"{profit_emoji} {trade['market_id']}: ${trade['net_profit']:.2f}\n"

                await update.message.reply_text(message, parse_mode="HTML")
            else:
                await update.message.reply_text("📭 Trade history not available")

        except Exception as e:
            await update.message.reply_text(f"❌ Error getting history: {e}")

    async def _cmd_stop(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /stop command - Emergency kill switch"""
        if not self.bot_instance:
            await update.message.reply_text("❌ Bot instance not connected")
            return

        try:
            # Trigger kill switch
            if hasattr(self.bot_instance, "kill_switch"):
                from src.core.kill_switch import KillLevel, KillReason

                success = await self.bot_instance.kill_switch.trigger(
                    KillLevel.HARD, KillReason.MANUAL, "Triggered via Telegram by user"
                )

                if success:
                    await update.message.reply_text(
                        "🛑 <b>EMERGENCY STOP ACTIVATED</b>\n\n"
                        "All trading has been halted.\n"
                        "Use /resume to restart trading.",
                        parse_mode="HTML",
                    )
                else:
                    await update.message.reply_text("⚠️ Kill switch already active")
            else:
                await update.message.reply_text("❌ Kill switch not available")

        except Exception as e:
            await update.message.reply_text(f"❌ Error stopping bot: {e}")

    async def _cmd_resume(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /resume command"""
        if not self.bot_instance:
            await update.message.reply_text("❌ Bot instance not connected")
            return

        try:
            if hasattr(self.bot_instance, "kill_switch"):
                success = await self.bot_instance.kill_switch.resume()

                if success:
                    await update.message.reply_text(
                        "✅ <b>Trading Resumed</b>\n\nThe bot is now trading again.",
                        parse_mode="HTML",
                    )
                else:
                    await update.message.reply_text(
                        "⚠️ Cannot resume - either not stopped or emergency kill"
                    )
            else:
                await update.message.reply_text("❌ Kill switch not available")

        except Exception as e:
            await update.message.reply_text(f"❌ Error resuming bot: {e}")

    async def _cmd_balance(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /balance command"""
        if not self.bot_instance:
            await update.message.reply_text("❌ Bot instance not connected")
            return

        try:
            if hasattr(self.bot_instance, "portfolio"):
                balance = self.bot_instance.portfolio.cash_balance

                message = f"""
💰 <b>Account Balance</b>

Available: ${balance / 100:.2f}

<i>Use /pnl to see today's profit/loss</i>
"""
                await update.message.reply_text(message, parse_mode="HTML")
            else:
                await update.message.reply_text("❌ Balance data not available")

        except Exception as e:
            await update.message.reply_text(f"❌ Error getting balance: {e}")

    async def _cmd_limits(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /limits command"""
        if not self.bot_instance:
            await update.message.reply_text("❌ Bot instance not connected")
            return

        try:
            if hasattr(self.bot_instance, "limited_risk_manager"):
                lr = self.bot_instance.limited_risk_manager
                stats = lr.daily_stats
                config = lr.config

                trades_remaining = config.max_daily_trades - stats.trades_executed
                loss_remaining = config.max_daily_loss_cents + stats.daily_pnl_cents

                message = f"""
📊 <b>Daily Limits Status</b>

Trades: {stats.trades_executed}/{config.max_daily_trades} ({trades_remaining} remaining)
Loss Limit: ${max(0, loss_remaining) / 100:.2f} remaining
Cooldown: {config.cooldown_seconds}s between trades

<i>Limited Risk Mode: {"Active" if lr.enabled else "Inactive"}</i>
"""
                await update.message.reply_text(message, parse_mode="HTML")
            else:
                await update.message.reply_text("❌ Limits data not available")

        except Exception as e:
            await update.message.reply_text(f"❌ Error getting limits: {e}")

    async def _cmd_alerts(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /alerts command"""
        if not self.bot_instance:
            await update.message.reply_text("❌ Bot instance not connected")
            return

        try:
            if hasattr(self.bot_instance, "limited_risk_metrics"):
                alerts = self.bot_instance.limited_risk_metrics.get_alert_status()

                if not alerts:
                    await update.message.reply_text("✅ No active alerts")
                    return

                message = "🚨 <b>Active Alerts</b>\n\n"
                for alert in alerts:
                    emoji = (
                        "🔴"
                        if alert["level"] == "critical"
                        else "⚠️"
                        if alert["level"] == "warning"
                        else "ℹ️"
                    )
                    message += f"{emoji} {alert['message']}\n"

                await update.message.reply_text(message, parse_mode="HTML")
            else:
                await update.message.reply_text("ℹ️ Alert system not available")

        except Exception as e:
            await update.message.reply_text(f"❌ Error getting alerts: {e}")

    async def _cmd_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /help command"""
        help_message = """
🤖 <b>Kalshi Arbitrage Bot - Help</b>

<b>Monitoring Commands:</b>
/status - Check if bot is running
/pnl - Today's profit/loss
/balance - Account balance
/positions - Current open positions
/history - Recent trades
/limits - Daily limits status
/alerts - Active alerts

<b>Control Commands:</b>
/stop - Emergency stop (kill switch)
/resume - Resume trading after stop

<b>Other:</b>
/help - Show this message
/start - Show welcome message

<b>Tips:</b>
• Use /status frequently to check bot health
• Set up alerts for important events
• /stop immediately if something seems wrong
• Check /limits to see how many trades remaining

<i>Questions? Check the documentation or contact support.</i>
"""
        await update.message.reply_text(help_message, parse_mode="HTML")


# Convenience functions for sending notifications


async def send_trade_alert(bot: TelegramBot, trade_data: Dict[str, Any]) -> None:
    """Send a trade notification"""
    if bot and bot._running:
        await bot.send_trade_notification(trade_data)


async def send_daily_summary(bot: TelegramBot, summary: Dict[str, Any]) -> None:
    """Send daily summary"""
    if bot and bot._running:
        await bot.send_daily_summary(summary)


async def send_alert(bot: TelegramBot, level: str, message: str) -> None:
    """Send an alert"""
    if bot and bot._running:
        await bot.send_alert(level, message)
