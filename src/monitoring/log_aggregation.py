import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from logging.handlers import SysLogHandler
import threading

from src.utils.logging_utils import get_logger

logger = get_logger("log_aggregation")


class LogAggregator:
    def __init__(
        self,
        enabled: bool = False,
        syslog_host: Optional[str] = None,
        syslog_port: int = 514,
        json_format: bool = True,
        tags: Optional[Dict[str, str]] = None
    ):
        self.enabled = enabled
        self.json_format = json_format
        self.tags = tags or {}
        self._syslog_handler: Optional[SysLogHandler] = None
        self._elastic_client = None

        if syslog_host:
            self._setup_syslog(syslog_host, syslog_port)

    def _setup_syslog(self, host: str, port: int) -> None:
        try:
            self._syslog_handler = SysLogHandler(
                address=(host, port),
                facility=SysLogHandler.LOG_LOCAL0
            )
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            self._syslog_handler.setFormatter(formatter)
        except Exception as e:
            logger.warning(f"Failed to setup syslog: {e}")

    def configure_logger(self, logger: logging.Logger) -> None:
        if not self.enabled:
            return

        if self._syslog_handler:
            logger.addHandler(self._syslog_handler)

        if self.json_format:
            self._add_json_handler(logger)

    def _add_json_handler(self, logger: logging.Logger) -> None:
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }

                if record.exc_info:
                    log_data['exception'] = self.formatException(record.exc_info)

                if hasattr(record, 'extra_data'):
                    log_data.update(record.extra_data)

                log_data.update(self.tags)

                return json.dumps(log_data)

        json_handler = logging.StreamHandler()
        json_handler.setFormatter(JsonFormatter())
        json_handler.setLevel(logging.DEBUG)
        logger.addHandler(json_handler)

    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        logger.info(f"Trade executed: {json.dumps(trade_data)}")

    def log_opportunity(self, opportunity_data: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        logger.info(f"Opportunity detected: {json.dumps(opportunity_data)}")

    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None) -> None:
        if not self.enabled:
            return
        logger.error(f"{error_type}: {error_message} | Context: {json.dumps(context or {})}")


class ElasticsearchLogger:
    def __init__(
        self,
        hosts: Optional[list] = None,
        index: str = "arbitrage-logs",
        username: Optional[str] = None,
        password: Optional[str] = None,
        enabled: bool = False
    ):
        self.enabled = enabled
        self.index = index
        self._client = None

        if enabled and hosts:
            try:
                from elasticsearch import Elasticsearch
                self._client = Elasticsearch(
                    hosts=hosts,
                    basic_auth=(username, password) if username else None
                )
            except ImportError:
                logger.warning("elasticsearch package not installed")

    async def log(self, document: Dict[str, Any]) -> None:
        if not self.enabled or not self._client:
            return

        try:
            document['@timestamp'] = datetime.utcnow().isoformat()
            self._client.index(index=self.index, document=document)
        except Exception as e:
            logger.warning(f"Failed to send log to Elasticsearch: {e}")

    async def log_trade(self, trade_data: Dict[str, Any]) -> None:
        await self.log({
            'type': 'trade',
            'data': trade_data
        })

    async def log_opportunity(self, opp_data: Dict[str, Any]) -> None:
        await self.log({
            'type': 'opportunity',
            'data': opp_data
        })

    async def log_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None) -> None:
        await self.log({
            'type': 'metric',
            'name': metric_name,
            'value': value,
            'tags': tags or {}
        })


class LokiLogger:
    def __init__(
        self,
        url: str = "http://localhost:3100",
        labels: Optional[Dict[str, str]] = None,
        enabled: bool = False
    ):
        self.enabled = enabled
        self.url = url.rstrip('/')
        self.labels = labels or {'job': 'arbitrage-bot'}
        self._batch: list = []
        self._batch_size = 100
        self._flush_interval = 5
        self._last_flush = datetime.utcnow()
        self._lock = threading.Lock()

    async def _send_batch(self) -> None:
        if not self._batch:
            return

        batch = self._batch[:]
        self._batch = []

        payload = {
            'streams': [
                {
                    'stream': self.labels,
                    'values': [
                        [
                            str(int(datetime.utcnow().timestamp() * 1e9)),
                            json.dumps(entry)
                        ]
                        for entry in batch
                    ]
                }
            ]
        }

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.url}/loki/api/v1/push",
                    json=payload,
                    timeout=10.0
                )
        except Exception as e:
            logger.warning(f"Failed to send logs to Loki: {e}")

    async def log(self, message: str, level: str = "info", extra_labels: Dict[str, str] = None) -> None:
        if not self.enabled:
            return

        entry = {
            'level': level,
            'message': message,
            'labels': extra_labels or {}
        }

        with self._lock:
            self._batch.append(entry)

            if len(self._batch) >= self._batch_size:
                await self._send_batch()
            elif (datetime.utcnow() - self._last_flush).total_seconds() > self._flush_interval:
                await self._send_batch()
                self._last_flush = datetime.utcnow()

    async def flush(self) -> None:
        await self._send_batch()


def create_log_aggregator(config: Dict[str, Any]) -> LogAggregator:
    return LogAggregator(
        enabled=config.get('enabled', False),
        syslog_host=config.get('syslog_host'),
        syslog_port=config.get('syslog_port', 514),
        json_format=config.get('json_format', True),
        tags=config.get('tags')
    )
