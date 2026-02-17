"""
Real-Time Risk Dashboard and Monitoring System
Professional-grade monitoring with real-time updates and alerts
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import redis.asyncio as redis

from .real_time_risk import RealTimeRiskManager, RiskLevel, RiskType
from .position_sizing_enhanced import DynamicPositionSizer
from .circuit_breaker_enhanced import EnhancedCircuitBreaker
from .stop_loss_manager import AutomatedStopManager
from ..utils.performance_cache import PerformanceCache

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Metric definition"""

    name: str
    metric_type: MetricType
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class AlertDefinition:
    """Alert definition"""

    name: str
    metric_name: str
    condition: str
    threshold: float
    severity: AlertLevel
    duration: int = 300  # 5 minutes default
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class DashboardWidget:
    """Dashboard widget definition"""

    widget_id: str
    widget_type: str  # chart, table, metric, alert
    title: str
    metrics: List[str]
    config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(
        default_factory=lambda: {"x": 0, "y": 0, "w": 6, "h": 4}
    )
    refresh_interval: int = 5  # seconds


@dataclass
class RiskSnapshot:
    """Risk system snapshot at a point in time"""

    timestamp: datetime
    portfolio_value: float
    total_exposure: float
    current_drawdown: float
    risk_scores: Dict[str, float]
    active_positions: int
    circuit_breaker_state: str
    total_alerts: int
    system_health: float


class RealTimeRiskDashboard:
    """
    Real-time risk monitoring dashboard with comprehensive metrics
    and alert management
    """

    def __init__(
        self,
        config: Dict[str, Any],
        risk_manager: RealTimeRiskManager,
        position_sizer: DynamicPositionSizer,
        circuit_breaker: EnhancedCircuitBreaker,
        stop_manager: AutomatedStopManager,
    ):
        self.config = config
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self.circuit_breaker = circuit_breaker
        self.stop_manager = stop_manager

        # Dashboard configuration
        self.dashboard_config = config.get("dashboard", {})
        self.refresh_interval = self.dashboard_config.get("refresh_interval", 5)
        self.history_retention = self.dashboard_config.get(
            "history_retention", 3600
        )  # 1 hour

        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.alert_definitions: Dict[str, AlertDefinition] = {}

        # Dashboard widgets
        self.widgets: Dict[str, DashboardWidget] = {}
        self.dashboard_layout = {}

        # Alert management
        self.active_alerts: Dict[str, Dict] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_escalations: Dict[str, datetime] = {}

        # Risk snapshots
        self.risk_snapshots: deque = deque(maxlen=720)  # 1 hour of 5-second snapshots

        # Performance cache
        self.cache = PerformanceCache(
            redis_url=config.get("redis_url", "redis://localhost:6379"),
            default_ttl=10,  # 10 second TTL for dashboard data
        )

        # WebSocket connections for real-time updates
        self.websocket_connections: Set[Any] = set()

        # Initialize metric definitions
        self._initialize_metric_definitions()
        self._initialize_alert_definitions()
        self._initialize_widgets()

        logger.info("Real-Time Risk Dashboard initialized")

    def _initialize_metric_definitions(self) -> None:
        """Initialize metric definitions"""
        # Portfolio metrics
        self.metric_definitions["portfolio_value"] = MetricDefinition(
            "portfolio_value", MetricType.GAUGE, "Total portfolio value", unit="$"
        )
        self.metric_definitions["total_exposure"] = MetricDefinition(
            "total_exposure", MetricType.GAUGE, "Total portfolio exposure", unit="$"
        )
        self.metric_definitions["current_drawdown"] = MetricDefinition(
            "current_drawdown",
            MetricType.GAUGE,
            "Current drawdown percentage",
            unit="%",
        )
        self.metric_definitions["var_95"] = MetricDefinition(
            "var_95", MetricType.GAUGE, "Value at Risk 95%", unit="$"
        )

        # Position metrics
        self.metric_definitions["active_positions"] = MetricDefinition(
            "active_positions", MetricType.GAUGE, "Number of active positions"
        )
        self.metric_definitions["position_risk_score"] = MetricDefinition(
            "position_risk_score", MetricType.GAUGE, "Average position risk score"
        )
        self.metric_definitions["unrealized_pnl"] = MetricDefinition(
            "unrealized_pnl", MetricType.GAUGE, "Total unrealized P&L", unit="$"
        )
        self.metric_definitions["realized_pnl"] = MetricDefinition(
            "realized_pnl", MetricType.GAUGE, "Total realized P&L", unit="$"
        )

        # Trading metrics
        self.metric_definitions["opportunities_detected"] = MetricDefinition(
            "opportunities_detected", MetricType.COUNTER, "Total opportunities detected"
        )
        self.metric_definitions["trades_executed"] = MetricDefinition(
            "trades_executed", MetricType.COUNTER, "Total trades executed"
        )
        self.metric_definitions["execution_latency"] = MetricDefinition(
            "execution_latency",
            MetricType.HISTOGRAM,
            "Trade execution latency",
            unit="ms",
        )
        self.metric_definitions["hit_rate"] = MetricDefinition(
            "hit_rate", MetricType.GAUGE, "Trade hit rate percentage", unit="%"
        )

        # System metrics
        self.metric_definitions["circuit_breaker_state"] = MetricDefinition(
            "circuit_breaker_state",
            MetricType.GAUGE,
            "Circuit breaker state (0=closed, 1=half-open, 2=open)",
        )
        self.metric_definitions["system_error_rate"] = MetricDefinition(
            "system_error_rate", MetricType.GAUGE, "System error rate", unit="%"
        )
        self.metric_definitions["cache_hit_rate"] = MetricDefinition(
            "cache_hit_rate", MetricType.GAUGE, "Cache hit rate", unit="%"
        )
        self.metric_definitions["api_response_time"] = MetricDefinition(
            "api_response_time", MetricType.HISTOGRAM, "API response time", unit="ms"
        )

        # Risk management metrics
        self.metric_definitions["risk_alerts"] = MetricDefinition(
            "risk_alerts", MetricType.COUNTER, "Total risk alerts generated"
        )
        self.metric_definitions["stop_loss_triggers"] = MetricDefinition(
            "stop_loss_triggers", MetricType.COUNTER, "Stop loss triggers"
        )
        self.metric_definitions["take_profit_triggers"] = MetricDefinition(
            "take_profit_triggers", MetricType.COUNTER, "Take profit triggers"
        )
        self.metric_definitions["position_size_adjustments"] = MetricDefinition(
            "position_size_adjustments",
            MetricType.COUNTER,
            "Automatic position size adjustments",
        )

    def _initialize_alert_definitions(self) -> None:
        """Initialize alert definitions"""
        # Portfolio risk alerts
        self.alert_definitions["high_drawdown"] = AlertDefinition(
            "High Drawdown Alert",
            "current_drawdown",
            "greater_than",
            0.05,  # 5%
            AlertLevel.WARNING,
            description="Portfolio drawdown exceeded 5%",
        )

        self.alert_definitions["critical_drawdown"] = AlertDefinition(
            "Critical Drawdown Alert",
            "current_drawdown",
            "greater_than",
            0.10,  # 10%
            AlertLevel.CRITICAL,
            description="Portfolio drawdown exceeded 10%",
        )

        # Position risk alerts
        self.alert_definitions["high_exposure"] = AlertDefinition(
            "High Exposure Alert",
            "total_exposure",
            "greater_than",
            0.8,
            AlertLevel.WARNING,
            description="Portfolio exposure exceeded 80%",
        )

        self.alert_definitions["circuit_breaker_open"] = AlertDefinition(
            "Circuit Breaker Alert",
            "circuit_breaker_state",
            "equals",
            2,  # Open state
            AlertLevel.CRITICAL,
            description="Circuit breaker is open - trading halted",
        )

        # System alerts
        self.alert_definitions["high_error_rate"] = AlertDefinition(
            "High Error Rate Alert",
            "system_error_rate",
            "greater_than",
            0.05,  # 5%
            AlertLevel.WARNING,
            description="System error rate exceeded 5%",
        )

        self.alert_definitions["low_cache_hit_rate"] = AlertDefinition(
            "Low Cache Hit Rate Alert",
            "cache_hit_rate",
            "less_than",
            0.7,  # 70%
            AlertLevel.WARNING,
            description="Cache hit rate below 70%",
        )

    def _initialize_widgets(self) -> None:
        """Initialize dashboard widgets"""
        # Portfolio overview widget
        self.widgets["portfolio_overview"] = DashboardWidget(
            widget_id="portfolio_overview",
            widget_type="metrics_grid",
            title="Portfolio Overview",
            metrics=[
                "portfolio_value",
                "total_exposure",
                "current_drawdown",
                "unrealized_pnl",
            ],
            config={"grid_columns": 2, "show_change": True},
            position={"x": 0, "y": 0, "w": 12, "h": 4},
        )

        # Risk metrics widget
        self.widgets["risk_metrics"] = DashboardWidget(
            widget_id="risk_metrics",
            widget_type="chart",
            title="Risk Metrics Over Time",
            metrics=["current_drawdown", "position_risk_score"],
            config={"chart_type": "line", "time_range": "1h"},
            position={"x": 0, "y": 4, "w": 8, "h": 6},
        )

        # Active positions widget
        self.widgets["active_positions"] = DashboardWidget(
            widget_id="active_positions",
            widget_type="table",
            title="Active Positions",
            metrics=["active_positions"],
            config={"columns": ["symbol", "size", "pnl", "risk_score"]},
            position={"x": 8, "y": 4, "w": 4, "h": 6},
        )

        # Trading performance widget
        self.widgets["trading_performance"] = DashboardWidget(
            widget_id="trading_performance",
            widget_type="chart",
            title="Trading Performance",
            metrics=["hit_rate", "realized_pnl"],
            config={"chart_type": "area", "time_range": "24h"},
            position={"x": 0, "y": 10, "w": 6, "h": 6},
        )

        # System health widget
        self.widgets["system_health"] = DashboardWidget(
            widget_id="system_health",
            widget_type="status",
            title="System Health",
            metrics=["circuit_breaker_state", "system_error_rate", "cache_hit_rate"],
            config={"show_status_indicators": True},
            position={"x": 6, "y": 10, "w": 6, "h": 6},
        )

        # Alerts widget
        self.widgets["alerts"] = DashboardWidget(
            widget_id="alerts",
            widget_type="alert_list",
            title="Active Alerts",
            metrics=["risk_alerts"],
            config={"max_alerts": 10, "group_by_severity": True},
            position={"x": 0, "y": 16, "w": 12, "h": 4},
        )

    async def start_monitoring(self) -> None:
        """Start real-time monitoring"""
        try:
            logger.info("Starting real-time risk monitoring")

            # Create monitoring task
            asyncio.create_task(self._monitoring_loop())

            # Create alert checking task
            asyncio.create_task(self._alert_checking_loop())

            logger.info("Real-time risk monitoring started")

        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            raise

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                # Collect metrics from all components
                await self._collect_metrics()

                # Create risk snapshot
                await self._create_risk_snapshot()

                # Update WebSocket clients
                await self._broadcast_updates()

                # Cache dashboard data
                await self._cache_dashboard_data()

                await asyncio.sleep(self.refresh_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(1)

    async def _collect_metrics(self) -> None:
        """Collect metrics from all risk components"""
        try:
            timestamp = datetime.now()

            # Collect risk manager metrics
            risk_metrics = await self.risk_manager.get_risk_metrics()

            if risk_metrics:
                portfolio_risk = risk_metrics.get("portfolio_risk", {})
                market_conditions = risk_metrics.get("market_conditions", {})
                risk_scores = risk_metrics.get("risk_scores", {})

                # Store portfolio metrics
                self.metrics["portfolio_value"].append(
                    (timestamp, portfolio_risk.get("total_value", 0))
                )
                self.metrics["total_exposure"].append(
                    (timestamp, portfolio_risk.get("total_exposure", 0))
                )
                self.metrics["current_drawdown"].append(
                    (timestamp, portfolio_risk.get("current_drawdown", 0))
                )
                self.metrics["var_95"].append(
                    (timestamp, portfolio_risk.get("var_95", 0))
                )

                # Store risk scores
                for score_name, score_value in risk_scores.items():
                    metric_name = f"risk_score_{score_name}"
                    self.metrics[metric_name].append((timestamp, score_value))

            # Collect position sizer metrics
            sizing_report = await self.position_sizer.get_sizing_report()

            if sizing_report:
                performance_metrics = sizing_report.get("performance_metrics", {})
                portfolio_info = sizing_report.get("portfolio_info", {})

                self.metrics["active_positions"].append(
                    (
                        timestamp,
                        len(
                            [
                                p
                                for p in self.risk_manager.active_positions.values()
                                if p != 0
                            ]
                        ),
                    )
                )
                self.metrics["position_risk_score"].append(
                    (
                        timestamp,
                        np.mean(
                            [
                                d.risk_score
                                for d in self.position_sizer.sizing_history[-10:]
                            ]
                        ),
                    )
                    if self.position_sizer.sizing_history
                    else 0
                )
                self.metrics["hit_rate"].append(
                    (timestamp, performance_metrics.get("hit_rate", 0))
                )
                self.metrics["realized_pnl"].append(
                    (timestamp, performance_metrics.get("total_pnl", 0))
                )

            # Collect stop manager metrics
            positions_status = await self.stop_manager.get_positions_status()

            if positions_status:
                self.metrics["unrealized_pnl"].append(
                    (timestamp, positions_status.get("total_unrealized_pnl", 0))
                )
                self.metrics["realized_pnl"].append(
                    (timestamp, positions_status.get("total_realized_pnl", 0))
                )
                self.metrics["stop_loss_triggers"].append(
                    (
                        timestamp,
                        len(
                            [
                                e
                                for e in self.stop_manager.execution_history
                                if "stop_loss" in e.get("reason", "")
                            ]
                        ),
                    )
                )
                self.metrics["take_profit_triggers"].append(
                    (
                        timestamp,
                        len(
                            [
                                e
                                for e in self.stop_manager.execution_history
                                if "take_profit" in e.get("reason", "")
                            ]
                        ),
                    )
                )

            # Collect circuit breaker metrics
            breaker_status = await self.circuit_breaker.get_status()

            if breaker_status:
                state_value = {"closed": 0, "half_open": 1, "open": 2}.get(
                    breaker_status.get("state", "closed"), 0
                )
                self.metrics["circuit_breaker_state"].append((timestamp, state_value))

                metrics_data = breaker_status.get("metrics", {})
                self.metrics["risk_alerts"].append(
                    (timestamp, metrics_data.get("total_triggers", 0))
                )

            # Collect system metrics
            await self._collect_system_metrics(timestamp)

        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")

    async def _collect_system_metrics(self, timestamp: datetime) -> None:
        """Collect system-level metrics"""
        try:
            # System error rate (mock - would integrate with actual error tracking)
            error_rate = np.random.uniform(0, 0.02)  # 0-2% error rate
            self.metrics["system_error_rate"].append((timestamp, error_rate))

            # Cache hit rate (from performance cache)
            # This would be available from the actual cache implementation
            cache_hit_rate = 0.85  # Mock 85% hit rate
            self.metrics["cache_hit_rate"].append((timestamp, cache_hit_rate))

            # API response time (mock)
            response_time = np.random.uniform(50, 200)  # 50-200ms
            self.metrics["api_response_time"].append((timestamp, response_time))

            # Execution latency (mock)
            latency = np.random.uniform(10, 100)  # 10-100ms
            self.metrics["execution_latency"].append((timestamp, latency))

            # Trading metrics (mock)
            opportunities = np.random.randint(0, 5)
            trades = np.random.randint(0, opportunities)

            self.metrics["opportunities_detected"].append((timestamp, opportunities))
            self.metrics["trades_executed"].append((timestamp, trades))

        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")

    async def _create_risk_snapshot(self) -> None:
        """Create risk system snapshot"""
        try:
            timestamp = datetime.now()

            # Get current portfolio value
            portfolio_value = self._get_latest_metric("portfolio_value", 0)
            total_exposure = self._get_latest_metric("total_exposure", 0)
            current_drawdown = self._get_latest_metric("current_drawdown", 0)

            # Get risk scores
            risk_scores = {}
            for metric_name in self.metrics:
                if metric_name.startswith("risk_score_"):
                    score_name = metric_name.replace("risk_score_", "")
                    risk_scores[score_name] = self._get_latest_metric(metric_name, 0)

            # Get active positions count
            active_positions = int(self._get_latest_metric("active_positions", 0))

            # Get circuit breaker state
            circuit_state = self._get_latest_metric("circuit_breaker_state", 0)
            circuit_state_name = {0: "closed", 1: "half_open", 2: "open"}.get(
                circuit_state, "unknown"
            )

            # Get total alerts
            total_alerts = len(self.active_alerts)

            # Calculate system health score (0-1)
            error_rate = self._get_latest_metric("system_error_rate", 0)
            cache_hit_rate = self._get_latest_metric("cache_hit_rate", 1)
            system_health = max(0, 1 - (error_rate * 10) - ((1 - cache_hit_rate) * 2))

            # Create snapshot
            snapshot = RiskSnapshot(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                total_exposure=total_exposure,
                current_drawdown=current_drawdown,
                risk_scores=risk_scores,
                active_positions=active_positions,
                circuit_breaker_state=circuit_state_name,
                total_alerts=total_alerts,
                system_health=system_health,
            )

            self.risk_snapshots.append(snapshot)

        except Exception as e:
            logger.error(f"Risk snapshot creation failed: {e}")

    async def _alert_checking_loop(self) -> None:
        """Check for alert conditions"""
        while True:
            try:
                await self._check_alert_conditions()
                await asyncio.sleep(10)  # Check alerts every 10 seconds

            except Exception as e:
                logger.error(f"Alert checking loop error: {e}")
                await asyncio.sleep(5)

    async def _check_alert_conditions(self) -> None:
        """Check all alert conditions"""
        try:
            current_time = datetime.now()

            for alert_id, alert_def in self.alert_definitions.items():
                metric_value = self._get_latest_metric(alert_def.metric_name)

                if metric_value is None:
                    continue

                # Check alert condition
                triggered = self._evaluate_condition(
                    metric_value, alert_def.condition, alert_def.threshold
                )

                if triggered:
                    if alert_id not in self.active_alerts:
                        # New alert
                        await self._trigger_alert(alert_id, alert_def, metric_value)
                    else:
                        # Check for escalation
                        alert_start = self.alert_escalations.get(alert_id, current_time)
                        duration = (current_time - alert_start).total_seconds()

                        if (
                            duration > alert_def.duration * 2
                        ):  # Escalate after 2x duration
                            await self._escalate_alert(
                                alert_id, alert_def, metric_value
                            )
                else:
                    # Clear alert if it exists
                    if alert_id in self.active_alerts:
                        await self._clear_alert(alert_id)

        except Exception as e:
            logger.error(f"Alert condition checking failed: {e}")

    def _evaluate_condition(
        self, value: float, condition: str, threshold: float
    ) -> bool:
        """Evaluate alert condition"""
        try:
            if condition == "greater_than":
                return value > threshold
            elif condition == "less_than":
                return value < threshold
            elif condition == "equals":
                return abs(value - threshold) < 0.001  # Small tolerance
            elif condition == "not_equals":
                return abs(value - threshold) >= 0.001
            else:
                return False

        except Exception:
            return False

    async def _trigger_alert(
        self, alert_id: str, alert_def: AlertDefinition, metric_value: float
    ) -> None:
        """Trigger a new alert"""
        try:
            alert_data = {
                "alert_id": alert_id,
                "name": alert_def.name,
                "severity": alert_def.severity.value,
                "metric": alert_def.metric_name,
                "current_value": metric_value,
                "threshold": alert_def.threshold,
                "condition": alert_def.condition,
                "description": alert_def.description,
                "triggered_at": datetime.now().isoformat(),
                "status": "active",
            }

            self.active_alerts[alert_id] = alert_data
            self.alert_escalations[alert_id] = datetime.now()
            self.alert_history.append(alert_data)

            # Log alert
            logger.warning(
                f"ALERT TRIGGERED: {alert_def.name} - {alert_def.description}"
            )

            # Send to WebSocket clients
            await self._broadcast_alert(alert_data)

            # Cache alert
            await self.cache.set(
                f"alert:{alert_id}",
                alert_data,
                ttl=3600,  # 1 hour
            )

        except Exception as e:
            logger.error(f"Alert triggering failed for {alert_id}: {e}")

    async def _escalate_alert(
        self, alert_id: str, alert_def: AlertDefinition, metric_value: float
    ) -> None:
        """Escalate an existing alert"""
        try:
            if alert_id in self.active_alerts:
                alert_data = self.active_alerts[alert_id]
                alert_data["status"] = "escalated"
                alert_data["escalated_at"] = datetime.now().isoformat()

                # Increase severity if not already emergency
                if alert_def.severity != AlertLevel.EMERGENCY:
                    alert_data["severity"] = AlertLevel.CRITICAL.value

                logger.error(f"ALERT ESCALATED: {alert_def.name}")
                await self._broadcast_alert(alert_data)

        except Exception as e:
            logger.error(f"Alert escalation failed for {alert_id}: {e}")

    async def _clear_alert(self, alert_id: str) -> None:
        """Clear an active alert"""
        try:
            if alert_id in self.active_alerts:
                alert_data = self.active_alerts[alert_id]
                alert_data["status"] = "cleared"
                alert_data["cleared_at"] = datetime.now().isoformat()

                # Remove from active alerts
                del self.active_alerts[alert_id]
                if alert_id in self.alert_escalations:
                    del self.alert_escalations[alert_id]

                logger.info(f"ALERT CLEARED: {alert_data['name']}")
                await self._broadcast_alert(alert_data)

        except Exception as e:
            logger.error(f"Alert clearing failed for {alert_id}: {e}")

    async def _broadcast_updates(self) -> None:
        """Broadcast updates to WebSocket clients"""
        try:
            if not self.websocket_connections:
                return

            # Prepare update data
            update_data = {
                "type": "dashboard_update",
                "timestamp": datetime.now().isoformat(),
                "metrics": self._prepare_metrics_data(),
                "alerts": list(self.active_alerts.values()),
                "snapshot": self._prepare_latest_snapshot(),
            }

            # Send to all connected clients
            message = json.dumps(update_data)

            for websocket in self.websocket_connections.copy():
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.warning(f"Failed to send update to WebSocket client: {e}")
                    self.websocket_connections.discard(websocket)

        except Exception as e:
            logger.error(f"Update broadcasting failed: {e}")

    async def _broadcast_alert(self, alert_data: Dict) -> None:
        """Broadcast alert to WebSocket clients"""
        try:
            if not self.websocket_connections:
                return

            alert_message = {
                "type": "alert",
                "timestamp": datetime.now().isoformat(),
                "alert": alert_data,
            }

            message = json.dumps(alert_message)

            for websocket in self.websocket_connections.copy():
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.warning(f"Failed to send alert to WebSocket client: {e}")
                    self.websocket_connections.discard(websocket)

        except Exception as e:
            logger.error(f"Alert broadcasting failed: {e}")

    def _prepare_metrics_data(self) -> Dict[str, Any]:
        """Prepare metrics data for dashboard"""
        metrics_data = {}

        for metric_name, metric_values in self.metrics.items():
            if metric_values:
                latest_value = metric_values[-1][1]
                definition = self.metric_definitions.get(metric_name)

                metrics_data[metric_name] = {
                    "value": latest_value,
                    "timestamp": metric_values[-1][0].isoformat(),
                    "definition": asdict(definition) if definition else None,
                    "history": [
                        {"timestamp": ts.isoformat(), "value": val}
                        for ts, val in list(metric_values)[-20:]  # Last 20 values
                    ],
                }

        return metrics_data

    def _prepare_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Prepare latest risk snapshot"""
        if not self.risk_snapshots:
            return None

        snapshot = self.risk_snapshots[-1]
        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "portfolio_value": snapshot.portfolio_value,
            "total_exposure": snapshot.total_exposure,
            "current_drawdown": snapshot.current_drawdown,
            "risk_scores": snapshot.risk_scores,
            "active_positions": snapshot.active_positions,
            "circuit_breaker_state": snapshot.circuit_breaker_state,
            "total_alerts": snapshot.total_alerts,
            "system_health": snapshot.system_health,
        }

    def _get_latest_metric(self, metric_name: str, default: Any = None) -> Any:
        """Get latest value of a metric"""
        try:
            if metric_name in self.metrics and self.metrics[metric_name]:
                return self.metrics[metric_name][-1][1]
            return default
        except Exception:
            return default

    async def _cache_dashboard_data(self) -> None:
        """Cache dashboard data for persistence"""
        try:
            # Cache latest metrics
            metrics_data = self._prepare_metrics_data()
            await self.cache.set(
                "dashboard_metrics",
                metrics_data,
                ttl=60,  # 1 minute
            )

            # Cache active alerts
            await self.cache.set(
                "dashboard_alerts", list(self.active_alerts.values()), ttl=60
            )

            # Cache latest snapshot
            latest_snapshot = self._prepare_latest_snapshot()
            if latest_snapshot:
                await self.cache.set("dashboard_snapshot", latest_snapshot, ttl=60)

        except Exception as e:
            logger.error(f"Dashboard data caching failed: {e}")

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        try:
            return {
                "widgets": {
                    widget_id: asdict(widget)
                    for widget_id, widget in self.widgets.items()
                },
                "metrics": self._prepare_metrics_data(),
                "alerts": list(self.active_alerts.values()),
                "snapshot": self._prepare_latest_snapshot(),
                "config": {
                    "refresh_interval": self.refresh_interval,
                    "history_retention": self.history_retention,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Dashboard data retrieval failed: {e}")
            return {}

    async def get_metrics_history(
        self, metric_names: List[str], time_range: str = "1h"
    ) -> Dict[str, List[Dict]]:
        """Get historical metrics data"""
        try:
            history = {}

            for metric_name in metric_names:
                if metric_name in self.metrics:
                    # Filter by time range
                    cutoff_time = datetime.now() - timedelta(hours=1)  # Default 1 hour

                    if time_range == "24h":
                        cutoff_time = datetime.now() - timedelta(hours=24)
                    elif time_range == "7d":
                        cutoff_time = datetime.now() - timedelta(days=7)

                    filtered_data = [
                        {"timestamp": ts.isoformat(), "value": val}
                        for ts, val in self.metrics[metric_name]
                        if ts >= cutoff_time
                    ]

                    history[metric_name] = filtered_data

            return history

        except Exception as e:
            logger.error(f"Metrics history retrieval failed: {e}")
            return {}

    def register_websocket(self, websocket: Any) -> None:
        """Register WebSocket connection for real-time updates"""
        self.websocket_connections.add(websocket)
        logger.info(
            f"WebSocket client registered. Total connections: {len(self.websocket_connections)}"
        )

    def unregister_websocket(self, websocket: Any) -> None:
        """Unregister WebSocket connection"""
        self.websocket_connections.discard(websocket)
        logger.info(
            f"WebSocket client unregistered. Total connections: {len(self.websocket_connections)}"
        )

    async def cleanup(self) -> None:
        """Cleanup dashboard resources"""
        try:
            # Close all WebSocket connections
            for websocket in self.websocket_connections.copy():
                try:
                    await websocket.close()
                except Exception:
                    pass
            self.websocket_connections.clear()

            # Save final data
            await self._cache_dashboard_data()

            # Close cache
            await self.cache.close()

            logger.info("Real-Time Risk Dashboard cleaned up")

        except Exception as e:
            logger.error(f"Dashboard cleanup failed: {e}")


# Utility functions
def calculate_dashboard_health_score(metrics: Dict[str, Any]) -> float:
    """Calculate overall dashboard health score"""
    try:
        health_factors = []

        # Error rate (lower is better)
        error_rate = metrics.get("system_error_rate", {}).get("value", 0)
        health_factors.append(max(0, 1 - error_rate * 10))  # Scale error rate impact

        # Cache hit rate (higher is better)
        cache_hit_rate = metrics.get("cache_hit_rate", {}).get("value", 0.8)
        health_factors.append(cache_hit_rate)

        # Circuit breaker state (closed is best)
        circuit_state = metrics.get("circuit_breaker_state", {}).get("value", 0)
        circuit_health = max(0, 1 - circuit_state * 0.5)  # Penalize open states
        health_factors.append(circuit_health)

        # Alert count (lower is better)
        alert_count = len(metrics.get("alerts", []))
        alert_health = max(0, 1 - alert_count * 0.1)  # Each alert reduces health
        health_factors.append(alert_health)

        return np.mean(health_factors) if health_factors else 0.0

    except Exception:
        return 0.0


def generate_risk_heatmap(risk_scores: Dict[str, float]) -> str:
    """Generate risk score heatmap"""
    if not risk_scores:
        return "🟢 No Risk Data"

    avg_risk = np.mean(list(risk_scores.values()))

    if avg_risk < 0.3:
        return "🟢 Low Risk"
    elif avg_risk < 0.5:
        return "🟡 Moderate Risk"
    elif avg_risk < 0.7:
        return "🟠 High Risk"
    else:
        return "🔴 Critical Risk"
