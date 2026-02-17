import asyncio
import uuid
from typing import Optional, Dict
from dataclasses import dataclass
from contextlib import asynccontextmanager

from src.utils.redis_client import RedisClient
from src.utils.logging_utils import get_logger

logger = get_logger("distributed_lock")


@dataclass
class LockConfig:
    timeout_seconds: float = 30.0
    retry_delay: float = 0.1
    max_retries: int = 50
    extend_interval: float = 10.0
    auto_extend: bool = True


class DistributedLock:
    def __init__(
        self,
        redis_client: RedisClient,
        lock_name: str,
        lock_id: Optional[str] = None,
        config: Optional[LockConfig] = None
    ):
        self.redis = redis_client
        self.lock_name = lock_name
        self.lock_id = lock_id or str(uuid.uuid4())
        self.config = config or LockConfig()
        self._held = False
        self._extend_task: Optional[asyncio.Task] = None

    async def acquire(self, blocking: bool = True) -> bool:
        lock_key = f"lock:{self.lock_name}"

        for attempt in range(self.config.max_retries):
            acquired = await self.redis._client.set(
                lock_key,
                self.lock_id,
                nx=True,
                ex=int(self.config.timeout_seconds)
            )

            if acquired:
                self._held = True
                logger.debug(f"Acquired lock: {self.lock_name}")

                if self.config.auto_extend:
                    self._start_extend_task()

                return True

            if not blocking:
                return False

            await asyncio.sleep(self.config.retry_delay)

        logger.warning(f"Failed to acquire lock: {self.lock_name}")
        return False

    async def release(self) -> bool:
        if not self._held:
            return False

        self._stop_extend_task()

        lock_key = f"lock:{self.lock_name}"
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        try:
            result = await self.redis._client.eval(
                script,
                keys=[lock_key],
                args=[self.lock_id]
            )
            success = result == 1
            if success:
                self._held = False
                logger.debug(f"Released lock: {self.lock_name}")
            return success
        except Exception as e:
            logger.error(f"Failed to release lock: {e}")
            return False

    async def extend(self, timeout_seconds: Optional[float] = None) -> bool:
        if not self._held:
            return False

        timeout = timeout_seconds or self.config.timeout_seconds
        lock_key = f"lock:{self.lock_name}"

        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("expire", KEYS[1], ARGV[2])
        else
            return 0
        end
        """

        try:
            result = await self.redis._client.eval(
                script,
                keys=[lock_key],
                args=[self.lock_id, int(timeout)]
            )
            return result == 1
        except Exception as e:
            logger.error(f"Failed to extend lock: {e}")
            return False

    def _start_extend_task(self) -> None:
        if self._extend_task is None or self._extend_task.done():
            self._extend_task = asyncio.create_task(self._extend_loop())

    def _stop_extend_task(self) -> None:
        if self._extend_task and not self._extend_task.done():
            self._extend_task.cancel()
            try:
                await self._extend_task
            except asyncio.CancelledError:
                pass

    async def _extend_loop(self) -> None:
        while self._held:
            await asyncio.sleep(self.config.extend_interval)
            if self._held:
                await self.extend()

    @property
    def is_held(self) -> bool:
        return self._held

    @asynccontextmanager
    async def context(self, blocking: bool = True):
        acquired = await self.acquire(blocking)
        if not acquired:
            raise LockAcquisitionError(f"Could not acquire lock: {self.lock_name}")
        try:
            yield self
        finally:
            await self.release()


class LockManager:
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self._active_locks: Dict[str, DistributedLock] = {}

    def acquire_lock(
        self,
        lock_name: str,
        config: Optional[LockConfig] = None
    ) -> DistributedLock:
        lock = DistributedLock(self.redis, lock_name, config=config)
        self._active_locks[lock_name] = lock
        return lock

    def release_lock(self, lock_name: str) -> None:
        if lock_name in self._active_locks:
            lock = self._active_locks[lock_name]
            asyncio.create_task(lock.release())
            del self._active_locks[lock_name]

    async def cleanup(self) -> None:
        for lock in list(self._active_locks.values()):
            await lock.release()
        self._active_locks.clear()


class LockAcquisitionError(Exception):
    pass


async def with_lock(
    redis_client: RedisClient,
    lock_name: str,
    func,
    *args,
    config: Optional[LockConfig] = None,
    **kwargs
):
    lock = DistributedLock(redis_client, lock_name, config=config)
    async with lock.context():
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)
