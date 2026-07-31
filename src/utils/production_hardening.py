"""Production hardening utilities for reliability and resilience."""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Callable, Any, Optional, Dict
from functools import wraps
import signal
from enum import Enum

logger = logging.getLogger("production_hardening")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures detected, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Exception = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception types to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 2:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info("Circuit breaker CLOSED - service recovered")

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        logger.warning(f"Circuit breaker failure count: {self.failure_count}")

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error("Circuit breaker OPEN - failing fast")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if not self.last_failure_time:
            return True

        elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_calls: int = 100, time_window: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed per time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: list = []

    def is_allowed(self) -> bool:
        """Check if call is allowed under rate limit."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.time_window)

        # Remove old calls
        self.calls = [call_time for call_time in self.calls if call_time > cutoff]

        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True

        return False

    async def call_with_limit(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with rate limiting."""
        if not self.is_allowed():
            raise Exception("Rate limit exceeded")

        return await func(*args, **kwargs)


class GracefulShutdown:
    """Graceful shutdown handler."""

    def __init__(self):
        """Initialize graceful shutdown handler."""
        self.shutdown_event = asyncio.Event()
        self.pending_tasks: set = set()

    def register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown_event.set()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    async def wait_for_shutdown(self):
        """Wait for shutdown signal."""
        await self.shutdown_event.wait()

    def register_task(self, task: asyncio.Task):
        """Register a task to be tracked."""
        self.pending_tasks.add(task)
        task.add_done_callback(self.pending_tasks.discard)

    async def cancel_all_tasks(self):
        """Cancel all pending tasks."""
        logger.info(f"Cancelling {len(self.pending_tasks)} pending tasks")
        for task in self.pending_tasks:
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete
        if self.pending_tasks:
            await asyncio.gather(*self.pending_tasks, return_exceptions=True)

        logger.info("All pending tasks cancelled")


class ConnectionPool:
    """Simple connection pool for managing resources."""

    def __init__(self, factory: Callable, pool_size: int = 10):
        """
        Initialize connection pool.

        Args:
            factory: Function to create new connections
            pool_size: Size of the pool
        """
        self.factory = factory
        self.pool_size = pool_size
        self.available: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self.all_connections: list = []
        self.initialized = False

    async def initialize(self):
        """Initialize the connection pool."""
        if self.initialized:
            return

        for _ in range(self.pool_size):
            conn = await self.factory()
            self.all_connections.append(conn)
            await self.available.put(conn)

        self.initialized = True
        logger.info(f"Connection pool initialized with {self.pool_size} connections")

    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self.initialized:
            await self.initialize()

        return await self.available.get()

    async def release(self, conn):
        """Release a connection back to the pool."""
        await self.available.put(conn)

    async def close_all(self):
        """Close all connections in the pool."""
        logger.info("Closing all connections in pool")
        for conn in self.all_connections:
            if hasattr(conn, 'close'):
                await conn.close()
            elif hasattr(conn, 'aclose'):
                await conn.aclose()

        self.initialized = False


class RetryPolicy:
    """Retry policy with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ):
        """
        Initialize retry policy.

        Args:
            max_retries: Maximum number of retries
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)

        logger.error(f"All {self.max_retries + 1} attempts failed")
        raise last_exception


class StateRecovery:
    """Recovery mechanism for state consistency."""

    def __init__(self, state_file: str = "bot_state.json"):
        """
        Initialize state recovery.

        Args:
            state_file: File to persist state
        """
        self.state_file = state_file
        self.state: Dict = {}

    async def save_state(self, state: Dict):
        """Save state to file."""
        import json
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
            logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def load_state(self) -> Dict:
        """Load state from file."""
        import json
        try:
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
            logger.info(f"State loaded from {self.state_file}")
            return self.state
        except FileNotFoundError:
            logger.info(f"No previous state file found: {self.state_file}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return {}

    async def reconcile_positions(self, actual_positions: Dict) -> Dict:
        """Reconcile saved positions with actual positions."""
        discrepancies = {}

        for market_id, saved_qty in self.state.get("positions", {}).items():
            actual_qty = actual_positions.get(market_id, 0)
            if saved_qty != actual_qty:
                discrepancies[market_id] = {
                    "saved": saved_qty,
                    "actual": actual_qty,
                    "difference": actual_qty - saved_qty,
                }
                logger.warning(
                    f"Position discrepancy for {market_id}: "
                    f"saved={saved_qty}, actual={actual_qty}"
                )

        return discrepancies


def with_circuit_breaker(circuit_breaker: CircuitBreaker):
    """Decorator for circuit breaker protection."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await circuit_breaker.call_async(func, *args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def with_retry(retry_policy: RetryPolicy):
    """Decorator for retry logic."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_policy.execute(func, *args, **kwargs)

        return wrapper

    return decorator
