"""
Anomaly Detection - Detect anomalous trading behavior and performance.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

from src.utils.logging_utils import get_logger

logger = get_logger("anomaly_detection")


@dataclass
class Anomaly:
    anomaly_id: str
    anomaly_type: str
    severity: str  # low, medium, high, critical
    description: str
    metric_name: str
    value: float
    expected_range: tuple[float, float]
    timestamp: datetime


class AnomalyDetector:
    def __init__(
        self,
        z_threshold: float = 3.0,
        lookback_period: int = 100,
    ):
        self.z_threshold = z_threshold
        self.lookback_period = lookback_period

        self._metric_history: Dict[str, deque] = {}
        self._anomalies: List[Anomaly] = []
        self._anomaly_counter = 0

    def add_observation(
        self, metric_name: str, value: float, timestamp: Optional[datetime] = None
    ) -> None:
        """Add a metric observation."""
        if metric_name not in self._metric_history:
            self._metric_history[metric_name] = deque(maxlen=self.lookback_period)

        self._metric_history[metric_name].append(
            {"value": value, "timestamp": timestamp or datetime.utcnow()}
        )

    def detect_z_score_anomaly(
        self, metric_name: str, value: float
    ) -> Optional[Anomaly]:
        """Detect anomaly using z-score method."""
        if (
            metric_name not in self._metric_history
            or len(self._metric_history[metric_name]) < 10
        ):
            return None

        values = [obs["value"] for obs in self._metric_history[metric_name]]
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return None

        z_score = (value - mean) / std

        if abs(z_score) > self.z_threshold:
            self._anomaly_counter += 1
            severity = self._get_severity(abs(z_score))

            return Anomaly(
                anomaly_id=f"zscore_{self._anomaly_counter}",
                anomaly_type="z_score",
                severity=severity,
                description=f"Z-score anomaly detected: {z_score:.2f} standard deviations",
                metric_name=metric_name,
                value=value,
                expected_range=(
                    mean - self.z_threshold * std,
                    mean + self.z_threshold * std,
                ),
                timestamp=datetime.utcnow(),
            )

        return None

    def detect_moving_average_anomaly(
        self, metric_name: str, value: float, window: int = 20
    ) -> Optional[Anomaly]:
        """Detect anomaly using moving average deviation."""
        if (
            metric_name not in self._metric_history
            or len(self._metric_history[metric_name]) < window
        ):
            return None

        values = [
            obs["value"] for obs in list(self._metric_history[metric_name])[-window:]
        ]
        ma = np.mean(values)
        std = np.std(values)

        if std == 0:
            return None

        deviation = abs(value - ma) / std

        if deviation > self.z_threshold:
            self._anomaly_counter += 1
            severity = self._get_severity(deviation)

            return Anomaly(
                anomaly_id=f"ma_{self._anomaly_counter}",
                anomaly_type="moving_average",
                severity=severity,
                description=f"Value deviates {deviation:.2f} std from {window}-period MA",
                metric_name=metric_name,
                value=value,
                expected_range=(
                    ma - self.z_threshold * std,
                    ma + self.z_threshold * std,
                ),
                timestamp=datetime.utcnow(),
            )

        return None

    def detect_rate_of_change_anomaly(
        self, metric_name: str, value: float, threshold: float = 2.0
    ) -> Optional[Anomaly]:
        """Detect anomaly based on rate of change."""
        if (
            metric_name not in self._metric_history
            or len(self._metric_history[metric_name]) < 2
        ):
            return None

        history = list(self._metric_history[metric_name])
        prev_value = history[-1]["value"]

        if prev_value == 0:
            return None

        change_pct = abs(value - prev_value) / prev_value

        if change_pct > threshold:
            self._anomaly_counter += 1
            severity = "high" if change_pct > 0.5 else "medium"

            return Anomaly(
                anomaly_id=f"roc_{self._anomaly_counter}",
                anomaly_type="rate_of_change",
                severity=severity,
                description=f"Rapid change: {change_pct * 100:.1f}% from previous value",
                metric_name=metric_name,
                value=value,
                expected_range=(prev_value * 0.5, prev_value * 1.5),
                timestamp=datetime.utcnow(),
            )

        return None

    def detect_threshold_anomaly(
        self, metric_name: str, value: float, min_threshold: float, max_threshold: float
    ) -> Optional[Anomaly]:
        """Detect anomaly when value exceeds threshold."""
        if min_threshold <= value <= max_threshold:
            return None

        self._anomaly_counter += 1
        severity = (
            "critical"
            if value < min_threshold * 0.5 or value > max_threshold * 2
            else "high"
        )

        return Anomaly(
            anomaly_id=f"thresh_{self._anomaly_counter}",
            anomaly_type="threshold",
            severity=severity,
            description=f"Value {value} outside threshold [{min_threshold}, {max_threshold}]",
            metric_name=metric_name,
            value=value,
            expected_range=(min_threshold, max_threshold),
            timestamp=datetime.utcnow(),
        )

    def check_all_anomalies(self, metrics: Dict[str, float]) -> List[Anomaly]:
        """Check all metrics for anomalies."""
        anomalies = []

        for metric_name, value in metrics.items():
            self.add_observation(metric_name, value)

            z_anomaly = self.detect_z_score_anomaly(metric_name, value)
            if z_anomaly:
                anomalies.append(z_anomaly)
                self._anomalies.append(z_anomaly)

            ma_anomaly = self.detect_moving_average_anomaly(metric_name, value)
            if ma_anomaly:
                anomalies.append(ma_anomaly)
                self._anomalies.append(ma_anomaly)

        return anomalies

    def get_recent_anomalies(
        self, hours: int = 24, severity: Optional[str] = None
    ) -> List[Anomaly]:
        """Get recent anomalies."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        anomalies = [a for a in self._anomalies if a.timestamp >= cutoff]

        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]

        return sorted(anomalies, key=lambda x: x.timestamp, reverse=True)

    def get_anomaly_stats(self) -> Dict[str, Any]:
        """Get anomaly detection statistics."""
        if not self._anomalies:
            return {
                "total_anomalies": 0,
                "by_severity": {},
                "by_type": {},
            }

        by_severity = {}
        by_type = {}

        for anomaly in self._anomalies:
            by_severity[anomaly.severity] = by_severity.get(anomaly.severity, 0) + 1
            by_type[anomaly.anomaly_type] = by_type.get(anomaly.anomaly_type, 0) + 1

        return {
            "total_anomalies": len(self._anomalies),
            "by_severity": by_severity,
            "by_type": by_type,
            "metrics_tracked": len(self._metric_history),
        }

    def _get_severity(self, z_score: float) -> str:
        """Determine severity based on z-score."""
        if z_score > 5:
            return "critical"
        elif z_score > 4:
            return "high"
        elif z_score > 3:
            return "medium"
        return "low"

    def clear_anomalies(self, before: Optional[datetime] = None) -> int:
        """Clear old anomalies."""
        if before is None:
            before = datetime.utcnow() - timedelta(days=7)

        initial_count = len(self._anomalies)
        self._anomalies = [a for a in self._anomalies if a.timestamp >= before]
        return initial_count - len(self._anomalies)
