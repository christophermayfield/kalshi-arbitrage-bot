"""
Limited Risk Trading Metrics

Analytics and monitoring for limited risk trading mode.
Provides real-time dashboard data and performance tracking.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from src.core.limited_risk_manager import LimitedRiskManager

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single trade"""

    timestamp: datetime
    market_id: str
    contracts: int
    entry_price_cents: int
    exit_price_cents: int
    gross_profit_cents: int
    fees_cents: int
    net_profit_cents: int
    fee_percent: float


class LimitedRiskMetrics:
    """
    Metrics and analytics for limited risk trading mode.

    Tracks:
    - Trade history
    - P&L by time period
    - Fee impact analysis
    - Performance vs limits
    - Progressive scaling progress
    """

    def __init__(self, risk_manager: LimitedRiskManager):
        self.risk_manager = risk_manager
        self.trade_history: List[TradeRecord] = []
        self._max_history_size = 1000

    def record_trade(
        self,
        market_id: str,
        contracts: int,
        entry_price_cents: int,
        exit_price_cents: int,
        gross_profit_cents: int,
        fees_cents: int,
        net_profit_cents: int,
    ) -> None:
        """Record a completed trade."""
        fee_percent = (
            (fees_cents / gross_profit_cents * 100) if gross_profit_cents > 0 else 0
        )

        record = TradeRecord(
            timestamp=datetime.utcnow(),
            market_id=market_id,
            contracts=contracts,
            entry_price_cents=entry_price_cents,
            exit_price_cents=exit_price_cents,
            gross_profit_cents=gross_profit_cents,
            fees_cents=fees_cents,
            net_profit_cents=net_profit_cents,
            fee_percent=fee_percent,
        )

        self.trade_history.append(record)

        # Trim history if too large
        if len(self.trade_history) > self._max_history_size:
            self.trade_history = self.trade_history[-self._max_history_size :]

    def get_today_summary(self) -> Dict[str, Any]:
        """Get summary of today's trading."""
        stats = self.risk_manager.daily_stats

        return {
            "trades": {
                "executed": stats.trades_executed,
                "won": stats.trades_won,
                "lost": stats.trades_lost,
                "remaining": self.risk_manager.config.max_daily_trades
                - stats.trades_executed,
                "win_rate": round(stats.win_rate, 1),
            },
            "profit_loss": {
                "net_cents": stats.daily_pnl_cents,
                "net_dollars": round(stats.daily_pnl_cents / 100, 2),
                "gross_cents": stats.gross_profit_cents,
                "avg_per_trade": round(stats.avg_profit_per_trade / 100, 2),
            },
            "fees": {
                "total_cents": stats.total_fees_cents,
                "total_dollars": round(stats.total_fees_cents / 100, 2),
                "impact_percent": round(stats.fee_impact_percent, 1),
            },
            "limits": {
                "max_trades": self.risk_manager.config.max_daily_trades,
                "max_loss_cents": self.risk_manager.config.max_daily_loss_cents,
                "current_drawdown_cents": max(0, -stats.daily_pnl_cents),
                "remaining_loss_cents": self.risk_manager.config.max_daily_loss_cents
                - max(0, -stats.daily_pnl_cents),
            },
            "scaling": {
                "current_level": self.risk_manager._scaling_level,
                "current_max_trade_cents": self.risk_manager.current_max_trade_cents,
                "base_max_cents": self.risk_manager.config.max_trade_cents,
                "threshold_trades": self.risk_manager.config.scaling_threshold_trades,
                "progress": min(
                    stats.trades_executed
                    / self.risk_manager.config.scaling_threshold_trades
                    * 100,
                    100,
                ),
            },
        }

    def get_recent_trades(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent trades."""
        recent = self.trade_history[-n:] if self.trade_history else []

        return [
            {
                "timestamp": trade.timestamp.isoformat(),
                "market_id": trade.market_id,
                "contracts": trade.contracts,
                "entry_price": trade.entry_price_cents / 100,
                "exit_price": trade.exit_price_cents / 100,
                "gross_profit": trade.gross_profit_cents / 100,
                "fees": trade.fees_cents / 100,
                "net_profit": trade.net_profit_cents / 100,
                "fee_percent": round(trade.fee_percent, 1),
            }
            for trade in reversed(recent)
        ]

    def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Get performance report for the last N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent_trades = [t for t in self.trade_history if t.timestamp >= cutoff]

        if not recent_trades:
            return {"error": "No trades in the specified period"}

        total_trades = len(recent_trades)
        winning_trades = [t for t in recent_trades if t.net_profit_cents > 0]
        losing_trades = [t for t in recent_trades if t.net_profit_cents <= 0]

        total_net_pnl = sum(t.net_profit_cents for t in recent_trades)
        total_fees = sum(t.fees_cents for t in recent_trades)

        avg_win = (
            sum(t.net_profit_cents for t in winning_trades) / len(winning_trades)
            if winning_trades
            else 0
        )
        avg_loss = (
            sum(t.net_profit_cents for t in losing_trades) / len(losing_trades)
            if losing_trades
            else 0
        )

        return {
            "period_days": days,
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(len(winning_trades) / total_trades * 100, 1)
            if total_trades > 0
            else 0,
            "total_net_pnl_dollars": round(total_net_pnl / 100, 2),
            "total_fees_dollars": round(total_fees / 100, 2),
            "avg_win_dollars": round(avg_win / 100, 2),
            "avg_loss_dollars": round(avg_loss / 100, 2),
            "profit_factor": abs(
                sum(t.net_profit_cents for t in winning_trades)
                / sum(t.net_profit_cents for t in losing_trades)
            )
            if losing_trades and sum(t.net_profit_cents for t in losing_trades) != 0
            else float("inf"),
            "avg_fee_percent": round(
                sum(t.fee_percent for t in recent_trades) / len(recent_trades), 1
            )
            if recent_trades
            else 0,
        }

    def get_alert_status(self) -> List[Dict[str, Any]]:
        """Get current alerts/warnings."""
        alerts = []
        stats = self.risk_manager.daily_stats
        config = self.risk_manager.config

        # Trade limit warning
        if stats.trades_executed >= config.max_daily_trades * 0.8:
            alerts.append(
                {
                    "level": "warning"
                    if stats.trades_executed < config.max_daily_trades
                    else "critical",
                    "type": "trade_limit",
                    "message": f"Trade limit: {stats.trades_executed}/{config.max_daily_trades} trades used",
                }
            )

        # Loss limit warning
        loss_pct = abs(stats.daily_pnl_cents) / config.max_daily_loss_cents * 100
        if loss_pct >= 80:
            alerts.append(
                {
                    "level": "warning" if loss_pct < 100 else "critical",
                    "type": "loss_limit",
                    "message": f"Loss limit: ${abs(stats.daily_pnl_cents) / 100:.2f} / ${config.max_daily_loss_cents / 100:.2f}",
                }
            )

        # Fee impact warning
        if stats.fee_impact_percent > 40:
            alerts.append(
                {
                    "level": "warning",
                    "type": "high_fees",
                    "message": f"High fee impact: {stats.fee_impact_percent:.1f}% of profits",
                }
            )

        # Progressive scaling notification
        if config.progressive_scaling_enabled:
            progress = stats.trades_executed / config.scaling_threshold_trades * 100
            if progress >= 80 and progress < 100:
                alerts.append(
                    {
                        "level": "info",
                        "type": "scaling_soon",
                        "message": f"Approaching scaling threshold: {stats.trades_executed}/{config.scaling_threshold_trades} trades",
                    }
                )
            elif self.risk_manager._scaling_level > 0:
                alerts.append(
                    {
                        "level": "success",
                        "type": "scaled",
                        "message": f"Progressive scaling active: Level {self.risk_manager._scaling_level} (${self.risk_manager.current_max_trade_cents / 100:.2f} max)",
                    }
                )

        return alerts

    def export_trade_history(self, filepath: str) -> None:
        """Export trade history to CSV."""
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "market_id",
                    "contracts",
                    "entry_price",
                    "exit_price",
                    "gross_profit",
                    "fees",
                    "net_profit",
                    "fee_percent",
                ]
            )

            for trade in self.trade_history:
                writer.writerow(
                    [
                        trade.timestamp.isoformat(),
                        trade.market_id,
                        trade.contracts,
                        trade.entry_price_cents / 100,
                        trade.exit_price_cents / 100,
                        trade.gross_profit_cents / 100,
                        trade.fees_cents / 100,
                        trade.net_profit_cents / 100,
                        round(trade.fee_percent, 2),
                    ]
                )

        logger.info(f"Exported {len(self.trade_history)} trades to {filepath}")
