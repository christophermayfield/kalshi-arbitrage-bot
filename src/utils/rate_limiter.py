import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
from datetime import datetime, timedelta

from src.utils.logging_utils import get_logger

logger = get_logger("rate_limiter")


@dataclass
class RateLimitConfig:
    requests_per_second: float = 5.0
    requests_per_minute: float = 300.0
    burst_limit: int = 10
    retry_after_base: float = 1.0


class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, 0.0
            else:
                wait_time = (tokens - self.tokens) / self.rate
                return False, wait_time


class RateLimiter:
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._second_bucket = TokenBucket(
            rate=self.config.requests_per_second,
            capacity=int(self.config.burst_limit)
        )
        self._minute_bucket = TokenBucket(
            rate=self.config.requests_per_minute / 60,
            capacity=self.config.burst_limit
        )
        self._request_counts: Dict[str, Dict[str, int]] = {}
        self._window_start: Optional[datetime] = None

    async def acquire(self, key: str = "default") -> Tuple[bool, float]:
        second_allowed, second_wait = await self._second_bucket.consume()
        if not second_allowed:
            return False, second_wait

        minute_allowed, minute_wait = await self._minute_bucket.consume()
        if not minute_allowed:
            await self._second_bucket.consume(-1)
            return False, minute_wait

        self._record_request(key)
        return True, 0.0

    def _record_request(self, key: str) -> None:
        now = datetime.utcnow()
        if self._window_start is None or now - self._window_start > timedelta(minutes=1):
            self._window_start = now
            self._request_counts.clear()

        if key not in self._request_counts:
            self._request_counts[key] = {}
        self._request_counts[key]['count'] = self._request_counts[key].get('count', 0) + 1
        self._request_counts[key]['timestamp'] = now.isoformat()

    def get_remaining(self, key: str = "default") -> Dict[str, int]:
        return {
            'per_second': int(self._second_bucket.tokens),
            'per_minute': int(self._minute_bucket.tokens)
        }

    def get_usage(self, key: str = "default") -> Dict[str, Any]:
        return {
            'request_count': self._request_counts.get(key, {}).get('count', 0),
            'window_start': self._window_start.isoformat() if self._window_start else None
        }


class AdaptiveRateLimiter(RateLimiter):
    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        error_weight: float = 1.5,
        success_weight: float = 0.9
    ):
        super().__init__(config)
        self.error_weight = error_weight
        self.success_weight = success_weight
        self._error_count = 0
        self._success_count = 0
        self._current_multiplier = 1.0

    async def acquire(self, key: str = "default") -> Tuple[bool, float]:
        success, wait = await super().acquire(key)
        if not success:
            self._error_count += 1
            self._update_multiplier()
            return False, wait * self._current_multiplier
        self._success_count += 1
        return True, 0.0

    def _update_multiplier(self) -> None:
        error_rate = self._error_count / max(1, self._error_count + self._success_count)
        if error_rate > 0.1:
            self._current_multiplier = min(3.0, self._current_multiplier * self.error_weight)
        else:
            self._current_multiplier = max(0.5, self._current_multiplier * self.success_weight)


class RateLimitedClient:
    def __init__(
        self,
        rate_limiter: RateLimiter,
        retry_on_rate_limit: bool = True
    ):
        self.rate_limiter = rate_limiter
        self.retry_on_rate_limit = retry_on_rate_limit

    async def request(
        self,
        func: Callable,
        *args,
        key: str = "default",
        max_retries: int = 3,
        **kwargs
    ) -> Any:
        for attempt in range(max_retries):
            allowed, wait_time = await self.rate_limiter.acquire(key)

            if not allowed and self.retry_on_rate_limit:
                logger.warning(f"Rate limited, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                continue

            if not allowed:
                raise RateLimitExceeded(f"Rate limit exceeded, try again in {wait_time:.2f}s")

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
            except Exception:
                self.rate_limiter._error_count += 1
                raise

        raise RateLimitExceeded("Max retries exceeded")


class RateLimitExceeded(Exception):
    pass


class RateLimitMiddleware:
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter

    async def __call__(self, request: Any, call_next: Callable) -> Any:
        key = getattr(request, 'client', {}).get('host', 'unknown')
        allowed, wait = await self.rate_limiter.acquire(key)

        if not allowed:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "retry_after": wait}
            )

        return await call_next(request)


def create_rate_limiter_from_config(config: Dict[str, Any]) -> RateLimiter:
    api_config = config.get('api', {})
    return RateLimiter(
        RateLimitConfig(
            requests_per_second=api_config.get('rate_limit_rps', 5.0),
            requests_per_minute=api_config.get('rate_limit_rpm', 300.0),
            burst_limit=api_config.get('rate_limit_burst', 10)
        )
    )
