"""
Audit Log / Event Store - Immutable audit trail for all trading activities.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger("audit_log")


class EventType(Enum):
    # Trading events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"

    # Arbitrage events
    OPPORTUNITY_DETECTED = "opportunity_detected"
    ARBITRAGE_EXECUTED = "arbitrage_executed"
    ARBITRAGE_FAILED = "arbitrage_failed"

    # Risk events
    RISK_LIMIT_HIT = "risk_limit_hit"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    DAILY_LOSS_LIMIT_HIT = "daily_loss_limit_hit"

    # System events
    BOT_STARTED = "bot_started"
    BOT_STOPPED = "bot_stopped"
    CONFIG_CHANGED = "config_changed"

    # Anomaly events
    ANOMALY_DETECTED = "anomaly_detected"


class EventSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    event_id: str
    event_type: str
    timestamp: str
    severity: str
    correlation_id: Optional[str]
    market_id: Optional[str]
    data: Dict[str, Any]
    user: Optional[str] = None


class AuditLog:
    def __init__(self, storage_path: str = "data/audit.log"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._event_counter = 0

    def _generate_event_id(self) -> str:
        self._event_counter += 1
        return f"evt_{self._event_counter}_{int(datetime.utcnow().timestamp())}"

    def log_event(
        self,
        event_type: EventType,
        severity: EventSeverity,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None,
        market_id: Optional[str] = None,
        user: Optional[str] = None,
    ) -> AuditEvent:
        """Log an audit event."""

        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=event_type.value,
            timestamp=datetime.utcnow().isoformat(),
            severity=severity.value,
            correlation_id=correlation_id,
            market_id=market_id,
            data=data,
            user=user,
        )

        # Write to file
        self._write_event(event)

        logger.debug(f"Audit event logged: {event.event_type}")

        return event

    def _write_event(self, event: AuditEvent) -> None:
        """Write event to storage."""
        try:
            with open(self.storage_path, "a") as f:
                f.write(json.dumps(asdict(event)) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit event: {e}")

    def query_events(
        self,
        event_type: Optional[EventType] = None,
        severity: Optional[EventSeverity] = None,
        market_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query audit events with filters."""
        events = []

        try:
            with open(self.storage_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        event_data = json.loads(line)
                        event = AuditEvent(**event_data)

                        # Apply filters
                        if event_type and event.event_type != event_type.value:
                            continue
                        if severity and event.severity != severity.value:
                            continue
                        if market_id and event.market_id != market_id:
                            continue
                        if (
                            start_time
                            and datetime.fromisoformat(event.timestamp) < start_time
                        ):
                            continue
                        if (
                            end_time
                            and datetime.fromisoformat(event.timestamp) > end_time
                        ):
                            continue

                        events.append(event)

                        if len(events) >= limit:
                            break

                    except Exception:
                        continue

        except FileNotFoundError:
            pass

        return events

    def get_recent_events(self, hours: int = 24, limit: int = 100) -> List[AuditEvent]:
        """Get recent events."""
        start_time = datetime.utcnow() - timedelta(hours=hours)
        return self.query_events(start_time=start_time, limit=limit)

    def get_events_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of events in time period."""
        events = self.get_recent_events(hours=hours, limit=10000)

        if not events:
            return {
                "total_events": 0,
                "by_type": {},
                "by_severity": {},
            }

        by_type = {}
        by_severity = {}

        for event in events:
            by_type[event.event_type] = by_type.get(event.event_type, 0) + 1
            by_severity[event.severity] = by_severity.get(event.severity, 0) + 1

        return {
            "total_events": len(events),
            "by_type": by_type,
            "by_severity": by_severity,
        }

    def get_trade_history(self, limit: int = 100) -> List[AuditEvent]:
        """Get trade-related events."""
        trade_types = [
            EventType.ORDER_SUBMITTED.value,
            EventType.ORDER_FILLED.value,
            EventType.ARBITRAGE_EXECUTED.value,
        ]

        events = []

        try:
            with open(self.storage_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        event_data = json.loads(line)
                        if event_data["event_type"] in trade_types:
                            events.append(AuditEvent(**event_data))
                            if len(events) >= limit:
                                break
                    except Exception:
                        continue

        except FileNotFoundError:
            pass

        return events


from datetime import timedelta
