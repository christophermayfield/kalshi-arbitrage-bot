from datetime import datetime
from typing import Any, Dict, List, Optional
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from src.utils.logging_utils import get_logger

logger = get_logger("monitoring")


class MetricsCollector:
    def __init__(self, port: int = 8000):
        self.port = port
        self.opportunities_found = Counter(
            'arbitrage_opportunities_total',
            'Total arbitrage opportunities found'
        )
        self.opportunities_executed = Counter(
            'arbitrage_executed_total',
            'Total arbitrage opportunities executed'
        )
        self.profit_total = Counter(
            'arbitrage_profit_cents_total',
            'Total profit in cents'
        )
        self.orderbook_scans = Counter(
            'orderbook_scans_total',
            'Total orderbook scans'
        )
        self.api_errors = Counter(
            'api_errors_total',
            'Total API errors'
        )
        self.active_positions = Gauge(
            'active_positions',
            'Number of active positions'
        )
        self.cash_balance = Gauge(
            'cash_balance_cents',
            'Cash balance in cents'
        )
        self.scan_duration = Histogram(
            'scan_duration_seconds',
            'Time to scan markets',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )
        self.execution_duration = Histogram(
            'execution_duration_seconds',
            'Time to execute arbitrage',
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0]
        )

    def start(self) -> None:
        start_http_server(self.port)
        logger.info(f"Metrics server started on port {self.port}")

    def record_opportunity(self, profit: int) -> None:
        self.opportunities_found.inc()
        self.profit_total.inc(profit)

    def record_execution(self, success: bool, duration: float) -> None:
        self.opportunities_executed.inc()
        self.execution_duration.observe(duration)

    def record_scan(self, duration: float) -> None:
        self.orderbook_scans.inc()
        self.scan_duration.observe(duration)

    def record_api_error(self) -> None:
        self.api_errors.inc()

    def update_positions(self, count: int) -> None:
        self.active_positions.set(count)

    def update_balance(self, balance: int) -> None:
        self.cash_balance.set(balance)


class HealthChecker:
    def __init__(self):
        self.last_heartbeat = datetime.utcnow()
        self.last_error: Optional[str] = None
        self.error_count = 0
        self.startup_time = datetime.utcnow()

    def is_healthy(self) -> bool:
        return self.error_count < 5

    def record_error(self, error: str) -> None:
        self.last_error = error
        self.error_count += 1

    def record_success(self) -> None:
        self.last_heartbeat = datetime.utcnow()

    def get_status(self) -> Dict[str, Any]:
        return {
            'healthy': self.is_healthy(),
            'uptime_seconds': (datetime.utcnow() - self.startup_time).total_seconds(),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'last_error': self.last_error,
            'error_count': self.error_count
        }


class NotificationManager:
    def __init__(self, enabled: bool = False, webhook_url: str = ""):
        self.enabled = enabled
        self.webhook_url = webhook_url
        self.notification_queue: List[Dict] = []

    async def send_notification(
        self,
        title: str,
        message: str,
        level: str = "info"
    ) -> None:
        if not self.enabled:
            return

        notification = {
            'title': title,
            'message': message,
            'level': level,
            'timestamp': datetime.utcnow().isoformat()
        }

        logger.info(f"[NOTIFICATION] {title}: {message}")

        if self.webhook_url:
            try:
                await self._send_webhook(notification)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")

    async def _send_webhook(self, notification: Dict) -> None:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.webhook_url,
                json=notification,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status != 200:
                    logger.error(f"Webhook failed: {await response.text()}")

    async def send_opportunity_alert(
        self,
        opportunity_id: str,
        profit_cents: int,
        confidence: float
    ) -> None:
        await self.send_notification(
            title="Arbitrage Opportunity",
            message=f"Found opportunity {opportunity_id}: {profit_cents} cents profit, {confidence:.1%} confidence",
            level="info"
        )

    async def send_execution_alert(
        self,
        opportunity_id: str,
        success: bool,
        profit_cents: int
    ) -> None:
        level = "success" if success else "error"
        message = f"Executed {opportunity_id}: {'Success' if success else 'Failed'} - {profit_cents} cents"
        await self.send_notification(
            title="Arbitrage Execution",
            message=message,
            level=level
        )

    async def send_error_alert(self, error: str) -> None:
        await self.send_notification(
            title="Bot Error",
            message=error,
            level="error"
        )


class MonitoringSystem:
    def __init__(self, config: Dict[str, Any]):
        self.metrics = MetricsCollector(
            port=config.get('metrics_port', 8000)
        )
        self.health = HealthChecker()
        self.notifications = NotificationManager(
            enabled=config.get('notification_enabled', False),
            webhook_url=config.get('notification_webhook', '')
        )
        self.config = config

    def start(self) -> None:
        if self.config.get('enabled', True):
            self.metrics.start()
            logger.info("Monitoring system started")

    async def check_health(self) -> Dict[str, Any]:
        return self.health.get_status()

    async def record_scan(self, duration: float) -> None:
        self.metrics.record_scan(duration)
        self.health.record_success()

    async def record_execution(self, success: bool, duration: float) -> None:
        self.metrics.record_execution(success, duration)

    async def record_error(self, error: str) -> None:
        self.health.record_error(error)
        self.metrics.record_api_error()
        await self.notifications.send_error_alert(error)

    async def update_balance(self, balance: int) -> None:
        self.metrics.update_balance(balance)

    async def update_positions(self, count: int) -> None:
        self.metrics.update_positions(count)
