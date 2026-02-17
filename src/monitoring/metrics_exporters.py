import aiohttp
from typing import Any, Dict, List, Optional
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
import json

from src.utils.logging_utils import get_logger

logger = get_logger("metrics_exporters")


class MetricsExporter:
    def __init__(self):
        self._registry = CollectorRegistry()

    def get_metrics(self) -> bytes:
        return generate_latest(self._registry)

    def get_content_type(self) -> str:
        return CONTENT_TYPE_LATEST


class JSONMetricsExporter:
    def __init__(self):
        self._metrics: Dict[str, Any] = {}

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        key = self._make_key(name, labels)
        self._metrics[key] = {
            'type': 'gauge',
            'value': value,
            'labels': labels or {}
        }

    def increment_counter(self, name: str, labels: Dict[str, str] = None, amount: float = 1.0) -> None:
        key = self._make_key(name, labels)
        if key in self._metrics:
            self._metrics[key]['value'] += amount
        else:
            self._metrics[key] = {
                'type': 'counter',
                'value': amount,
                'labels': labels or {}
            }

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
        buckets: Optional[List[float]] = None
    ) -> None:
        key = self._make_key(name, labels)
        if key in self._metrics:
            if 'values' not in self._metrics[key]:
                self._metrics[key]['values'] = []
            self._metrics[key]['values'].append(value)
        else:
            self._metrics[key] = {
                'type': 'histogram',
                'value': value,
                'values': [value],
                'labels': labels or {},
                'buckets': buckets
            }

    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        if labels:
            label_str = '_'.join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}[{label_str}]"
        return name

    def get_metrics(self) -> str:
        output = {}
        for key, data in self._metrics.items():
            output[key] = {
                'value': data['value'],
                'type': data['type'],
                'labels': data['labels']
            }
        return json.dumps(output, indent=2)

    def clear(self) -> None:
        self._metrics.clear()


class StatsDMetricsExporter:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8125,
        prefix: str = "arbitrage"
    ):
        self.host = host
        self.port = port
        self.prefix = prefix
        self._socket = None

    def _get_socket(self):
        if self._socket is None:
            import socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return self._socket

    def _send(self, metric: str) -> None:
        try:
            sock = self._get_socket()
            sock.sendto(metric.encode(), (self.host, self.port))
        except Exception as e:
            logger.debug(f"Failed to send StatsD metric: {e}")

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        metric_name = f"{self.prefix}.{name}"
        if labels:
            metric_name += f".{'_'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
        self._send(f"{metric_name}:{value}|g")

    def increment_counter(self, name: str, labels: Dict[str, str] = None, amount: float = 1.0) -> None:
        metric_name = f"{self.prefix}.{name}"
        if labels:
            metric_name += f".{'_'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
        self._send(f"{metric_name}:{amount}|c")

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
        buckets: Optional[List[float]] = None
    ) -> None:
        metric_name = f"{self.prefix}.{name}"
        if labels:
            metric_name += f".{'_'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
        self._send(f"{metric_name}:{value}|h")

    def timing(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        metric_name = f"{self.prefix}.{name}"
        if labels:
            metric_name += f".{'_'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
        self._send(f"{metric_name}:{value}|ms")


class OpenTelemetryMetricsExporter:
    def __init__(
        self,
        endpoint: str = "http://localhost:4318/v1/metrics",
        headers: Optional[Dict[str, str]] = None,
        enabled: bool = False
    ):
        self.enabled = enabled
        self.endpoint = endpoint
        self.headers = headers or {}
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def export(self, metrics: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False

        payload = {
            'metrics': [
                {
                    'name': k,
                    'value': v['value'],
                    'type': v['type'],
                    'labels': v.get('labels', {})
                }
                for k, v in metrics.items()
            ]
        }

        try:
            session = await self._get_session()
            async with session.post(
                self.endpoint,
                json=payload,
                headers=self.headers
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Failed to export OpenTelemetry metrics: {e}")
            return False

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None


class MultiMetricsExporter:
    def __init__(self):
        self.exporters: List[Any] = []

    def add_exporter(self, exporter: Any) -> None:
        self.exporters.append(exporter)

    async def export_all(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        results = {}
        for exporter in self.exporters:
            try:
                if hasattr(exporter, 'export'):
                    result = await exporter.export(metrics)
                    results[type(exporter).__name__] = result
                elif hasattr(exporter, 'get_metrics'):
                    metrics_data = exporter.get_metrics()
                    results[type(exporter).__name__] = True
            except Exception as e:
                logger.warning(f"Failed to export with {type(exporter).__name__}: {e}")
                results[type(exporter).__name__] = False
        return results


def create_metrics_exporter(config: Dict[str, Any]) -> MultiMetricsExporter:
    exporter = MultiMetricsExporter()

    if config.get('statsd', {}).get('enabled'):
        statsd_config = config['statsd']
        exporter.add_exporter(StatsDMetricsExporter(
            host=statsd_config.get('host', 'localhost'),
            port=statsd_config.get('port', 8125),
            prefix=statsd_config.get('prefix', 'arbitrage')
        ))

    if config.get('opentelemetry', {}).get('enabled'):
        otel_config = config['opentelemetry']
        exporter.add_exporter(OpenTelemetryMetricsExporter(
            endpoint=otel_config.get('endpoint', 'http://localhost:4318/v1/metrics'),
            headers=otel_config.get('headers'),
            enabled=True
        ))

    exporter.add_exporter(JSONMetricsExporter())

    return exporter
