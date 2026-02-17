"""Comprehensive audit logging and compliance system for arbitrage bot."""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
import sqlite3
import threading
import logging
from pathlib import Path
from contextlib import contextmanager
import asyncio
from queue import Queue, Empty
import pandas as pd

from src.utils.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AuditEventType(Enum):
    """Audit event types."""

    TRADE_EXECUTION = "trade_execution"
    ORDER_SUBMISSION = "order_submission"
    ORDER_CANCELLATION = "order_cancellation"
    STRATEGY_CHANGE = "strategy_change"
    CONFIG_CHANGE = "config_change"
    ACCESS_ATTEMPT = "access_attempt"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    ERROR_EVENT = "error_event"
    DATA_ACCESS = "data_access"
    API_CALL = "api_call"
    USER_ACTION = "user_action"
    COMPLIANCE_CHECK = "compliance_check"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    POSITION_CHANGE = "position_change"
    BALANCE_CHANGE = "balance_change"


class ComplianceLevel(Enum):
    """Compliance levels."""

    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    FULL = "full"


@dataclass
class AuditEvent:
    """Audit event data structure."""

    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    user_id: Optional[str]
    session_id: Optional[str]
    source: str
    severity: str  # low, medium, high, critical
    description: str
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    outcome: str  # success, failure, partial
    compliance_tags: List[str]
    risk_score: float = 0.0
    related_events: List[str] = None

    def __post_init__(self):
        if self.related_events is None:
            self.related_events = []


@dataclass
class ComplianceRule:
    """Compliance rule definition."""

    rule_id: str
    name: str
    description: str
    level: ComplianceLevel
    enabled: bool
    conditions: Dict[str, Any]
    actions: List[str]
    alert_threshold: int = 1
    time_window: timedelta = timedelta(hours=24)


class AuditLogger:
    """Comprehensive audit logging system."""

    def __init__(self, db_path: str = "audit_logs.db", config: Optional[Config] = None):
        self.db_path = db_path
        self.config = config or Config()
        self.compliance_level = ComplianceLevel(
            self.config.get("compliance.level", "standard")
        )

        # Event queue for async processing
        self.event_queue = Queue()
        self.batch_size = self.config.get("audit.batch_size", 100)
        self.flush_interval = self.config.get("audit.flush_interval", 60)

        # Compliance rules
        self.compliance_rules = self._load_compliance_rules()

        # Initialize database
        self._init_database()

        # Start background processing
        self._stop_event = threading.Event()
        self._processor_thread = threading.Thread(
            target=self._process_events, daemon=True
        )
        self._processor_thread.start()

        logger.info("Audit logger initialized")

    def _init_database(self):
        """Initialize audit database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Audit events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    source TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    outcome TEXT NOT NULL,
                    compliance_tags TEXT,
                    risk_score REAL DEFAULT 0.0,
                    related_events TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Compliance violations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS compliance_violations (
                    violation_id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    details TEXT,
                    status TEXT DEFAULT 'open',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES audit_events (event_id)
                )
            """)

            # Audit trail table for critical events
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_trail (
                    trail_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT,
                    hash_value TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES audit_events (event_id)
                )
            """)

            # Create indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events(event_type)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_events(user_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_severity ON audit_events(severity)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_compliance_timestamp ON compliance_violations(timestamp)"
            )

            conn.commit()

    def _load_compliance_rules(self) -> Dict[str, ComplianceRule]:
        """Load compliance rules from configuration."""
        rules_config = self.config.get("compliance.rules", {})
        rules = {}

        for rule_id, rule_data in rules_config.items():
            rules[rule_id] = ComplianceRule(
                rule_id=rule_id,
                name=rule_data.get("name", rule_id),
                description=rule_data.get("description", ""),
                level=ComplianceLevel(rule_data.get("level", "standard")),
                enabled=rule_data.get("enabled", True),
                conditions=rule_data.get("conditions", {}),
                actions=rule_data.get("actions", []),
                alert_threshold=rule_data.get("alert_threshold", 1),
                time_window=timedelta(hours=rule_data.get("time_window_hours", 24)),
            )

        return rules

    def log_event(
        self,
        event_type: AuditEventType,
        description: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        source: str = "system",
        severity: str = "medium",
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        outcome: str = "success",
        compliance_tags: Optional[List[str]] = None,
        related_events: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Log an audit event."""
        try:
            # Generate event ID
            event_id = self._generate_event_id()

            # Create audit event
            event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.utcnow(),
                event_type=event_type,
                user_id=user_id,
                session_id=session_id,
                source=source,
                severity=severity,
                description=description,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent,
                outcome=outcome,
                compliance_tags=compliance_tags or [],
                related_events=related_events or [],
                **kwargs,
            )

            # Calculate risk score
            event.risk_score = self._calculate_risk_score(event)

            # Add to queue for processing
            self.event_queue.put(event)

            # Check compliance rules
            if self.compliance_level != ComplianceLevel.NONE:
                asyncio.create_task(self._check_compliance(event))

            return event_id

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            raise

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        timestamp = datetime.utcnow().isoformat()
        random_data = str(id(object())) + str(threading.get_ident())
        return hashlib.sha256(f"{timestamp}{random_data}".encode()).hexdigest()[:16]

    def _calculate_risk_score(self, event: AuditEvent) -> float:
        """Calculate risk score for audit event."""
        base_score = 0.0

        # Event type risk weights
        risk_weights = {
            AuditEventType.TRADE_EXECUTION: 0.3,
            AuditEventType.ORDER_SUBMISSION: 0.2,
            AuditEventType.STRATEGY_CHANGE: 0.4,
            AuditEventType.CONFIG_CHANGE: 0.3,
            AuditEventType.RISK_LIMIT_BREACH: 0.8,
            AuditEventType.ACCESS_ATTEMPT: 0.2,
            AuditEventType.ERROR_EVENT: 0.3,
            AuditEventType.SYSTEM_SHUTDOWN: 0.4,
        }

        base_score += risk_weights.get(event.event_type, 0.1)

        # Severity multiplier
        severity_multipliers = {"low": 0.5, "medium": 1.0, "high": 1.5, "critical": 2.0}

        base_score *= severity_multipliers.get(event.severity, 1.0)

        # Outcome adjustment
        if event.outcome == "failure":
            base_score *= 1.2

        # Time-based risk (after hours trading, etc.)
        current_hour = event.timestamp.hour
        if current_hour < 9 or current_hour > 16:  # Outside market hours
            base_score *= 1.1

        return min(base_score, 1.0)

    async def _check_compliance(self, event: AuditEvent):
        """Check event against compliance rules."""
        try:
            for rule_id, rule in self.compliance_rules.items():
                if not rule.enabled:
                    continue

                if self._evaluate_rule(event, rule):
                    await self._handle_compliance_violation(event, rule)

        except Exception as e:
            logger.error(f"Compliance check failed: {e}")

    def _evaluate_rule(self, event: AuditEvent, rule: ComplianceRule) -> bool:
        """Evaluate if event violates compliance rule."""
        conditions = rule.conditions

        # Event type condition
        if "event_types" in conditions:
            if event.event_type.value not in conditions["event_types"]:
                return False

        # Severity condition
        if "min_severity" in conditions:
            severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            if (
                severity_levels.get(event.severity, 0)
                < severity_levels[conditions["min_severity"]]
            ):
                return False

        # Risk score condition
        if "min_risk_score" in conditions:
            if event.risk_score < conditions["min_risk_score"]:
                return False

        # Custom conditions based on event details
        if "custom_conditions" in conditions:
            for condition in conditions["custom_conditions"]:
                if not self._evaluate_custom_condition(event, condition):
                    return False

        return True

    def _evaluate_custom_condition(
        self, event: AuditEvent, condition: Dict[str, Any]
    ) -> bool:
        """Evaluate custom condition logic."""
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")

        if not all([field, operator, value is not None]):
            return True

        event_value = self._get_nested_value(event, field)

        if event_value is None:
            return True

        if operator == "equals":
            return event_value == value
        elif operator == "not_equals":
            return event_value != value
        elif operator == "greater_than":
            return float(event_value) > float(value)
        elif operator == "less_than":
            return float(event_value) < float(value)
        elif operator == "contains":
            return value in str(event_value)

        return True

    def _get_nested_value(self, obj: Any, path: str) -> Any:
        """Get nested value from object using dot notation."""
        keys = path.split(".")
        current = obj

        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            elif hasattr(current, key):
                current = getattr(current, key)
            else:
                return None

        return current

    async def _handle_compliance_violation(
        self, event: AuditEvent, rule: ComplianceRule
    ):
        """Handle compliance rule violation."""
        try:
            violation_id = self._generate_event_id()

            # Create violation record
            violation = {
                "violation_id": violation_id,
                "rule_id": rule.rule_id,
                "event_id": event.event_id,
                "timestamp": datetime.utcnow().isoformat(),
                "severity": self._map_severity_to_compliance(event.severity),
                "description": f"Compliance rule '{rule.name}' violated: {event.description}",
                "details": {
                    "event": asdict(event),
                    "rule": asdict(rule),
                    "conditions_met": True,
                },
            }

            # Store violation
            self._store_compliance_violation(violation)

            # Execute rule actions
            for action in rule.actions:
                await self._execute_compliance_action(action, violation, event, rule)

            logger.warning(f"Compliance violation detected: {rule.name}")

        except Exception as e:
            logger.error(f"Failed to handle compliance violation: {e}")

    def _map_severity_to_compliance(self, severity: str) -> str:
        """Map event severity to compliance severity."""
        mapping = {
            "low": "informational",
            "medium": "warning",
            "high": "violation",
            "critical": "critical",
        }
        return mapping.get(severity, "warning")

    def _store_compliance_violation(self, violation: Dict[str, Any]):
        """Store compliance violation in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO compliance_violations
                (violation_id, rule_id, event_id, timestamp, severity, description, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    violation["violation_id"],
                    violation["rule_id"],
                    violation["event_id"],
                    violation["timestamp"],
                    violation["severity"],
                    violation["description"],
                    json.dumps(violation["details"]),
                ),
            )
            conn.commit()

    async def _execute_compliance_action(
        self,
        action: str,
        violation: Dict[str, Any],
        event: AuditEvent,
        rule: ComplianceRule,
    ):
        """Execute compliance action."""
        try:
            if action == "alert":
                await self._send_compliance_alert(violation, event, rule)
            elif action == "block":
                await self._block_operation(event)
            elif action == "require_approval":
                await self._require_approval(event, rule)
            elif action == "log_only":
                # Already logged, no additional action needed
                pass
            else:
                logger.warning(f"Unknown compliance action: {action}")

        except Exception as e:
            logger.error(f"Failed to execute compliance action {action}: {e}")

    async def _send_compliance_alert(
        self, violation: Dict[str, Any], event: AuditEvent, rule: ComplianceRule
    ):
        """Send compliance alert."""
        alert_data = {
            "type": "compliance_violation",
            "severity": violation["severity"],
            "rule_name": rule.name,
            "event_description": event.description,
            "timestamp": violation["timestamp"],
            "user_id": event.user_id,
            "risk_score": event.risk_score,
        }

        # Send to monitoring system
        # TODO: Integrate with alerting system
        logger.warning(f"Compliance alert: {alert_data}")

    async def _block_operation(self, event: AuditEvent):
        """Block operation that caused violation."""
        logger.warning(
            f"Blocking operation due to compliance violation: {event.event_id}"
        )
        # TODO: Implement operation blocking logic

    async def _require_approval(self, event: AuditEvent, rule: ComplianceRule):
        """Require approval for operation."""
        logger.info(f"Requiring approval for operation: {event.event_id}")
        # TODO: Implement approval workflow

    def _process_events(self):
        """Background thread to process audit events."""
        batch_events = []
        last_flush = datetime.utcnow()

        while not self._stop_event.is_set():
            try:
                # Try to get event from queue
                try:
                    event = self.event_queue.get(timeout=1.0)
                    batch_events.append(event)
                except Empty:
                    pass

                # Check if we should flush the batch
                now = datetime.utcnow()
                should_flush = len(batch_events) >= self.batch_size or (
                    batch_events and (now - last_flush).seconds >= self.flush_interval
                )

                if should_flush and batch_events:
                    self._flush_events(batch_events)
                    batch_events.clear()
                    last_flush = now

            except Exception as e:
                logger.error(f"Error processing audit events: {e}")

        # Flush remaining events on shutdown
        if batch_events:
            self._flush_events(batch_events)

    def _flush_events(self, events: List[AuditEvent]):
        """Flush batch of events to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for event in events:
                    cursor.execute(
                        """
                        INSERT INTO audit_events
                        (event_id, timestamp, event_type, user_id, session_id, source,
                         severity, description, details, ip_address, user_agent, outcome,
                         compliance_tags, risk_score, related_events)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            event.event_id,
                            event.timestamp.isoformat(),
                            event.event_type.value,
                            event.user_id,
                            event.session_id,
                            event.source,
                            event.severity,
                            event.description,
                            json.dumps(event.details),
                            event.ip_address,
                            event.user_agent,
                            event.outcome,
                            json.dumps(event.compliance_tags),
                            event.risk_score,
                            json.dumps(event.related_events),
                        ),
                    )

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to flush audit events: {e}")

    @contextmanager
    def audit_context(
        self,
        operation: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ):
        """Context manager for audit operations."""
        start_time = datetime.utcnow()
        event_id = None

        try:
            # Log operation start
            event_id = self.log_event(
                event_type=AuditEventType.USER_ACTION,
                description=f"Started operation: {operation}",
                user_id=user_id,
                session_id=session_id,
                **kwargs,
            )

            yield

            # Log operation success
            self.log_event(
                event_type=AuditEventType.USER_ACTION,
                description=f"Completed operation: {operation}",
                user_id=user_id,
                session_id=session_id,
                related_events=[event_id] if event_id else None,
                **kwargs,
            )

        except Exception as e:
            # Log operation failure
            self.log_event(
                event_type=AuditEventType.ERROR_EVENT,
                description=f"Failed operation: {operation} - {str(e)}",
                user_id=user_id,
                session_id=session_id,
                severity="high",
                details={"error": str(e), "traceback": traceback.format_exc()},
                related_events=[event_id] if event_id else None,
                **kwargs,
            )
            raise

    def query_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        severity: Optional[str] = None,
        min_risk_score: Optional[float] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query audit events with filters."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Build query
                query = "SELECT * FROM audit_events WHERE 1=1"
                params = []

                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date.isoformat())

                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date.isoformat())

                if event_types:
                    placeholders = ",".join(["?" for _ in event_types])
                    query += f" AND event_type IN ({placeholders})"
                    params.extend([et.value for et in event_types])

                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)

                if severity:
                    query += " AND severity = ?"
                    params.append(severity)

                if min_risk_score:
                    query += " AND risk_score >= ?"
                    params.append(min_risk_score)

                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                cursor.execute(query, params)
                rows = cursor.fetchall()

                # Convert to list of dictionaries
                events = []
                for row in rows:
                    event = dict(row)
                    event["details"] = json.loads(event["details"] or "{}")
                    event["compliance_tags"] = json.loads(
                        event["compliance_tags"] or "[]"
                    )
                    event["related_events"] = json.loads(
                        event["related_events"] or "[]"
                    )
                    events.append(event)

                return events

        except Exception as e:
            logger.error(f"Failed to query audit events: {e}")
            return []

    def get_compliance_report(
        self, start_date: datetime, end_date: datetime, rule_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate compliance report."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get violations
                query = """
                    SELECT cv.*, ae.event_type, ae.description as event_description
                    FROM compliance_violations cv
                    JOIN audit_events ae ON cv.event_id = ae.event_id
                    WHERE cv.timestamp >= ? AND cv.timestamp <= ?
                """
                params = [start_date.isoformat(), end_date.isoformat()]

                if rule_id:
                    query += " AND cv.rule_id = ?"
                    params.append(rule_id)

                cursor.execute(query, params)
                violations = cursor.fetchall()

                # Calculate statistics
                total_violations = len(violations)
                violations_by_severity = {}
                violations_by_rule = {}

                for violation in violations:
                    severity = violation[3]  # severity column
                    rule = violation[2]  # rule_id column

                    violations_by_severity[severity] = (
                        violations_by_severity.get(severity, 0) + 1
                    )
                    violations_by_rule[rule] = violations_by_rule.get(rule, 0) + 1

                return {
                    "period": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat(),
                    },
                    "summary": {
                        "total_violations": total_violations,
                        "violations_by_severity": violations_by_severity,
                        "violations_by_rule": violations_by_rule,
                    },
                    "violations": [
                        {
                            "violation_id": v[0],
                            "rule_id": v[2],
                            "timestamp": v[3],
                            "severity": v[4],
                            "description": v[5],
                            "event_type": v[8],
                            "event_description": v[9],
                        }
                        for v in violations
                    ],
                }

        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            return {}

    def export_audit_data(
        self,
        format: str = "csv",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Union[str, pd.DataFrame]:
        """Export audit data."""
        try:
            events = self.query_events(
                start_date=start_date, end_date=end_date, limit=10000
            )

            df = pd.DataFrame(events)

            if format.lower() == "csv":
                return df.to_csv(index=False)
            elif format.lower() == "json":
                return df.to_json(orient="records", date_format="iso")
            else:
                return df

        except Exception as e:
            logger.error(f"Failed to export audit data: {e}")
            raise

    def cleanup_old_events(self, retention_days: int = 365) -> int:
        """Clean up old audit events based on retention policy."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Delete old events
                cursor.execute(
                    "DELETE FROM audit_events WHERE timestamp < ?",
                    (cutoff_date.isoformat(),),
                )
                deleted_count = cursor.rowcount

                # Delete old violations
                cursor.execute(
                    "DELETE FROM compliance_violations WHERE timestamp < ?",
                    (cutoff_date.isoformat(),),
                )

                # Delete old trail entries
                cursor.execute(
                    "DELETE FROM audit_trail WHERE timestamp < ?",
                    (cutoff_date.isoformat(),),
                )

                conn.commit()

                logger.info(f"Cleaned up {deleted_count} old audit events")
                return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old events: {e}")
            return 0

    def shutdown(self):
        """Shutdown audit logger gracefully."""
        logger.info("Shutting down audit logger...")

        # Stop background processing
        self._stop_event.set()
        self._processor_thread.join(timeout=30)

        # Flush remaining events
        remaining_events = []
        while not self.event_queue.empty():
            try:
                remaining_events.append(self.event_queue.get_nowait())
            except Empty:
                break

        if remaining_events:
            self._flush_events(remaining_events)

        logger.info("Audit logger shutdown complete")


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# Decorators for easy audit logging
def audit_operation(
    operation: str, event_type: AuditEventType = AuditEventType.USER_ACTION
):
    """Decorator to audit function calls."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()

            with audit_logger.audit_context(
                operation=operation,
                user_id=kwargs.get("user_id"),
                session_id=kwargs.get("session_id"),
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def audit_trade_execution(func):
    """Decorator specifically for trade execution functions."""

    def wrapper(*args, **kwargs):
        audit_logger = get_audit_logger()

        return audit_logger.log_event(
            event_type=AuditEventType.TRADE_EXECUTION,
            description=f"Trade executed via {func.__name__}",
            details={
                "function": func.__name__,
                "args": str(args)[:1000],  # Limit string length
                "kwargs": str(kwargs)[:1000],
            },
            user_id=kwargs.get("user_id"),
            session_id=kwargs.get("session_id"),
        )

        return func(*args, **kwargs)

    return wrapper
