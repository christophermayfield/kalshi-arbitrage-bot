"""Advanced monitoring and alerting system."""

import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Callable, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio

logger = logging.getLogger("advanced_monitoring")


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    LOG = "log"
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"


@dataclass
class PerformanceMetric:
    """Performance metric."""
    name: str
    value: float
    unit: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert notification."""
    level: AlertLevel
    title: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: Dict[str, Any] = field(default_factory=dict)
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])


class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: List[PerformanceMetric] = []
        self.max_metrics = 10000  # Keep last N metrics

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        tags: Dict[str, str] = None,
    ):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            tags=tags or {},
        )
        self.metrics.append(metric)

        # Keep only last N metrics
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]

    def get_metrics(self, name: str = None) -> List[PerformanceMetric]:
        """Get metrics by name."""
        if not name:
            return self.metrics
        return [m for m in self.metrics if m.name == name]

    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Calculate statistics for a metric."""
        metrics = self.get_metrics(name)
        if not metrics:
            return {}

        values = [m.value for m in metrics]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
        }


class AlertManager:
    """Manage alert generation and delivery."""

    def __init__(self):
        """Initialize alert manager."""
        self.alerts: List[Alert] = []
        self.alert_handlers: Dict[AlertChannel, Callable] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.max_alerts = 1000

    def register_handler(self, channel: AlertChannel, handler: Callable):
        """Register alert delivery handler."""
        self.alert_handlers[channel] = handler
        logger.info(f"Registered alert handler for {channel.value}")

    async def send_alert(self, alert: Alert):
        """Send alert through configured channels."""
        self.alerts.append(alert)

        # Keep only last N alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]

        logger.log(
            getattr(logging, alert.level.value.upper()),
            f"[{alert.level.value.upper()}] {alert.title}: {alert.message}",
        )

        # Send through registered channels
        for channel in alert.channels:
            if channel in self.alert_handlers:
                try:
                    handler = self.alert_handlers[channel]
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel.value}: {e}")

    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, float]], bool],
        alert_level: AlertLevel = AlertLevel.WARNING,
        title: str = "",
        message_template: str = "",
    ):
        """Add alert rule."""
        rule = {
            "name": name,
            "condition": condition,
            "alert_level": alert_level,
            "title": title or name,
            "message_template": message_template,
        }
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {name}")

    async def check_rules(self, metrics_snapshot: Dict[str, float]):
        """Check all alert rules against current metrics."""
        for rule in self.alert_rules:
            try:
                if rule["condition"](metrics_snapshot):
                    alert = Alert(
                        level=rule["alert_level"],
                        title=rule["title"],
                        message=rule["message_template"].format(**metrics_snapshot),
                    )
                    await self.send_alert(alert)
            except Exception as e:
                logger.error(f"Error checking rule {rule['name']}: {e}")


class HealthChecker:
    """System health checks."""

    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable] = {}
        self.last_check_results: Dict[str, bool] = {}

    def register_check(self, name: str, check_func: Callable):
        """Register a health check."""
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    async def run_all_checks(self) -> Dict[str, bool]:
        """Run all health checks."""
        results = {}

        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                results[name] = result
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = False

        self.last_check_results = results
        return results

    def is_healthy(self) -> bool:
        """Check if system is healthy (all checks pass)."""
        return all(self.last_check_results.values()) if self.last_check_results else False

    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status."""
        return {
            "healthy": self.is_healthy(),
            "checks": self.last_check_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class PerformanceMonitor:
    """Monitor bot performance metrics."""

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance monitor."""
        self.metrics = metrics_collector
        self.start_time = datetime.now(timezone.utc)

    def record_trade_execution(
        self,
        market_id: str,
        execution_time_ms: float,
        profit_cents: int,
    ):
        """Record trade execution metrics."""
        self.metrics.record_metric(
            "execution_time_ms",
            execution_time_ms,
            "ms",
            {"market": market_id},
        )
        self.metrics.record_metric(
            "trade_profit_cents",
            profit_cents,
            "cents",
            {"market": market_id},
        )

    def record_api_call(
        self,
        endpoint: str,
        response_time_ms: float,
        success: bool,
    ):
        """Record API call metrics."""
        self.metrics.record_metric(
            "api_response_time_ms",
            response_time_ms,
            "ms",
            {"endpoint": endpoint},
        )
        self.metrics.record_metric(
            "api_success",
            1.0 if success else 0.0,
            "bool",
            {"endpoint": endpoint},
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        return {
            "uptime_seconds": uptime,
            "execution_time_avg": self.metrics.get_metric_stats("execution_time_ms"),
            "trade_profit_stats": self.metrics.get_metric_stats("trade_profit_cents"),
            "api_response_time_avg": self.metrics.get_metric_stats("api_response_time_ms"),
        }


# Default alert rules configuration
DEFAULT_ALERT_RULES = [
    {
        "name": "high_error_rate",
        "condition": lambda m: m.get("error_rate", 0) > 0.05,  # >5% errors
        "alert_level": AlertLevel.ERROR,
        "title": "High Error Rate",
        "message": "Error rate is {error_rate:.1%}",
    },
    {
        "name": "api_latency_high",
        "condition": lambda m: m.get("api_latency_ms", 0) > 5000,  # >5s
        "alert_level": AlertLevel.WARNING,
        "title": "High API Latency",
        "message": "API latency is {api_latency_ms:.0f}ms",
    },
    {
        "name": "balance_low",
        "condition": lambda m: m.get("balance_cents", float('inf')) < 100000,  # <$1000
        "alert_level": AlertLevel.WARNING,
        "title": "Low Balance",
        "message": "Available balance is ${balance:.2f}",
    },
    {
        "name": "daily_loss_limit",
        "condition": lambda m: m.get("daily_loss_percent", 0) > 0.90,  # >90% of limit
        "alert_level": AlertLevel.CRITICAL,
        "title": "Daily Loss Limit Approaching",
        "message": "Daily loss is {daily_loss_percent:.0%} of limit",
    },
]
