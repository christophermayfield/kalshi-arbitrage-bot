import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional
from datetime import datetime

from src.utils.logging_utils import get_logger

logger = get_logger("circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3


@dataclass
class CircuitBreakerStats:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    last_failure_time: Optional[str] = None
    last_success_time: Optional[str] = None
    state_transitions: int = 0
    current_state: CircuitState = CircuitState.CLOSED


class CircuitBreaker:
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        self._stats = CircuitBreakerStats()

    @property
    def state(self) -> CircuitState:
        # Check if circuit should transition from OPEN to HALF_OPEN
        if (
            self._state == CircuitState.OPEN
            and self._last_failure_time
            and time.time() - self._last_failure_time >= self.config.timeout_seconds
        ):
            asyncio.create_task(self._transition_to_half_open())
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        return self._stats

    async def _transition_to_half_open(self) -> None:
        async with self._lock:
            if self._state == CircuitState.OPEN:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._stats.state_transitions += 1
                logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN")

    async def _transition_to_closed(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._stats.state_transitions += 1
        logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED")

    async def _transition_to_open(self) -> None:
        self._state = CircuitState.OPEN
        self._last_failure_time = time.time()
        self._stats.last_failure_time = datetime.utcnow().isoformat()
        self._stats.state_transitions += 1
        logger.warning(f"Circuit {self.name}: CLOSED/HALF_OPEN -> OPEN")

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        async with self._lock:
            self._stats.total_calls += 1

            # Check if circuit should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if (
                    self._last_failure_time
                    and time.time() - self._last_failure_time
                    >= self.config.timeout_seconds
                ):
                    await self._transition_to_half_open()
                else:
                    self._stats.failed_calls += 1
                    raise CircuitOpenError(f"Circuit {self.name} is open")

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._stats.failed_calls += 1
                    raise CircuitOpenError(
                        f"Circuit {self.name} half-open call limit reached"
                    )
                self._half_open_calls += 1

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            async with self._lock:
                self._stats.successful_calls += 1
                self._last_success_time = time.time()
                self._stats.last_success_time = datetime.utcnow().isoformat()
                self._success_count += 1

                if self._state == CircuitState.HALF_OPEN:
                    if self._success_count >= self.config.success_threshold:
                        await self._transition_to_closed()

            return result

        except Exception:
            async with self._lock:
                self._stats.failed_calls += 1
                self._last_failure_time = time.time()
                self._stats.last_failure_time = datetime.utcnow().isoformat()
                self._failure_count += 1

                if self._state == CircuitState.CLOSED:
                    if self._failure_count >= self.config.failure_threshold:
                        await self._transition_to_open()

                elif self._state == CircuitState.HALF_OPEN:
                    await self._transition_to_open()

            raise

    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    def reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_success_time = None
        self._half_open_calls = 0
        self._stats = CircuitBreakerStats()
        logger.info(f"Circuit {self.name}: Reset to CLOSED")


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open."""

    pass


class CircuitBreakerManager:
    def __init__(self):
        self._circuits: Dict[str, CircuitBreaker] = {}

    def get_or_create(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        if name not in self._circuits:
            self._circuits[name] = CircuitBreaker(name, config)
        return self._circuits[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        return self._circuits.get(name)

    def remove(self, name: str) -> None:
        if name in self._circuits:
            del self._circuits[name]

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "state": circuit.state.value,
                "stats": {
                    "total_calls": circuit.stats.total_calls,
                    "successful_calls": circuit.stats.successful_calls,
                    "failed_calls": circuit.stats.failed_calls,
                    "state_transitions": circuit.stats.state_transitions,
                },
            }
            for name, circuit in self._circuits.items()
        }

    def reset_all(self) -> None:
        for circuit in self._circuits.values():
            circuit.reset()
