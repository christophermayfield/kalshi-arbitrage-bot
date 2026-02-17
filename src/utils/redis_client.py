import asyncio
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from contextlib import asynccontextmanager
import redis.asyncio as redis

from src.utils.logging_utils import get_logger

logger = get_logger("redis_client")


class RedisClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50,
        prefix: str = "arbitrage"
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.prefix = prefix
        self._client: Optional[redis.Redis] = None
        self._lock_manager: Optional[redis.Redis] = None

    async def connect(self) -> None:
        self._client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            max_connections=self.max_connections,
            decode_responses=True
        )
        self._lock_manager = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            max_connections=10
        )
        await self._client.ping()
        logger.info(f"Connected to Redis at {self.host}:{self.port}")

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._lock_manager:
            await self._lock_manager.aclose()
            self._lock_manager = None
        logger.info("Disconnected from Redis")

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Optional[str]:
        return await self._client.get(self._key(key))

    async def set(
        self,
        key: str,
        value: str,
        expire_seconds: Optional[int] = None
    ) -> None:
        await self._client.set(self._key(key), value, ex=expire_seconds)

    async def delete(self, key: str) -> None:
        await self._client.delete(self._key(key))

    async def exists(self, key: str) -> bool:
        return await self._client.exists(self._key(key)) > 0

    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        value = await self.get(key)
        if value:
            return json.loads(value)
        return None

    async def set_json(
        self,
        key: str,
        value: Dict[str, Any],
        expire_seconds: Optional[int] = None
    ) -> None:
        await self.set(key, json.dumps(value), expire_seconds)

    async def get_pickle(self, key: str) -> Optional[Any]:
        value = await self._client.get(self._key(key))
        if value:
            return pickle.loads(value)
        return None

    async def set_pickle(
        self,
        key: str,
        value: Any,
        expire_seconds: Optional[int] = None
    ) -> None:
        await self._client.set(self._key(key), pickle.dumps(value), ex=expire_seconds)

    async def hget(self, name: str, key: str) -> Optional[str]:
        return await self._client.hget(self._key(name), key)

    async def hset(self, name: str, key: str, value: str) -> None:
        await self._client.hset(self._key(name), key, value)

    async def hgetall(self, name: str) -> Dict[str, str]:
        return await self._client.hgetall(self._key(name))

    async def incr(self, key: str) -> int:
        return await self._client.incr(self._key(key))

    async def zadd(self, name: str, score: float, member: str) -> None:
        await self._client.zadd(self._key(name), {member: score})

    async def zrangebyscore(
        self,
        name: str,
        min_score: float,
        max_score: float
    ) -> List[str]:
        return await self._client.zrangebyscore(self._key(name), min_score, max_score)

    async def expire(self, key: str, seconds: int) -> None:
        await self._client.expire(self._key(key), seconds)

    @asynccontextmanager
    async def lock(
        self,
        name: str,
        timeout: float = 10.0,
        blocking: bool = True
    ):
        lock = self._lock_manager.lock(
            self._key(name),
            timeout=timeout,
            blocking=blocking
        )
        async with lock:
            yield

    async def acquire_distributed_lock(
        self,
        lock_name: str,
        lock_id: str,
        expire_seconds: int = 30
    ) -> bool:
        return await self._client.set(
            self._key(f"lock:{lock_name}"),
            lock_id,
            nx=True,
            ex=expire_seconds
        )

    async def release_distributed_lock(
        self,
        lock_name: str,
        lock_id: str
    ) -> bool:
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        result = await self._client.eval(
            script,
            keys=[self._key(f"lock:{lock_name}")],
            args=[lock_id]
        )
        return result == 1

    async def get_or_set(
        self,
        key: str,
        factory: callable,
        expire_seconds: Optional[int] = None
    ) -> Any:
        cached = await self.get_pickle(key)
        if cached is not None:
            return cached

        value = await factory()
        await self.set_pickle(key, value, expire_seconds)
        return value

    async def cache_market_data(
        self,
        market_id: str,
        data: Dict[str, Any],
        ttl: int = 60
    ) -> None:
        await self.set_json(f"market:{market_id}", data, ttl)

    async def get_market_data(self, market_id: str) -> Optional[Dict[str, Any]]:
        return await self.get_json(f"market:{market_id}")

    async def cache_orderbook(
        self,
        market_id: str,
        data: Dict[str, Any],
        ttl: int = 5
    ) -> None:
        await self.set_json(f"orderbook:{market_id}", data, ttl)

    async def get_orderbook(self, market_id: str) -> Optional[Dict[str, Any]]:
        return await self.get_json(f"orderbook:{market_id}")

    async def increment_counter(self, name: str, amount: int = 1) -> int:
        return await self._client.incrby(self._key(name), amount)

    async def get_counter(self, name: str) -> int:
        value = await self._client.get(self._key(name))
        return int(value) if value else 0

    async def record_trade(self, trade_data: Dict[str, Any]) -> int:
        trade_id = await self.increment_counter("trade_id")
        trade_data["id"] = trade_id
        trade_data["timestamp"] = datetime.utcnow().isoformat()
        await self.set_json(f"trade:{trade_id}", trade_data, 86400 * 7)
        await self.zadd(
            "trades_by_time",
            datetime.utcnow().timestamp(),
            str(trade_id)
        )
        return trade_id

    async def get_recent_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        trade_ids = await self.zrangebyscore(
            "trades_by_time",
            0,
            datetime.utcnow().timestamp()
        )
        trades = []
        for tid in trade_ids[-limit:]:
            trade = await self.get_json(f"trade:{tid}")
            if trade:
                trades.append(trade)
        return trades

    async def health_check(self) -> Tuple[bool, str]:
        try:
            await self._client.ping()
            info = await self._client.info("memory")
            used_memory = info.get("used_memory_human", "unknown")
            return True, f"Connected, memory: {used_memory}"
        except Exception as e:
            return False, str(e)


class CacheManager:
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self._default_ttl = {
            'market': 60,
            'orderbook': 5,
            'position': 300,
            'config': 3600,
            'stats': 10
        }

    def get_ttl(self, cache_type: str) -> int:
        return self._default_ttl.get(cache_type, 60)

    async def invalidate(self, pattern: str) -> int:
        keys = await self.redis._client.keys(self.redis._key(pattern))
        if keys:
            return await self.redis._client.delete(*keys)
        return 0

    async def warm_cache(self, market_ids: List[str]) -> None:
        from src.clients.kalshi_client import KalshiClient
        client = KalshiClient.__new__(KalshiClient)
        for market_id in market_ids:
            try:
                data = await asyncio.coroutine(lambda: client.get_market_orderbook(market_id))()
                await self.redis.cache_orderbook(market_id, data, self.get_ttl('orderbook'))
                logger.debug(f"Warmed cache for {market_id}")
            except Exception as e:
                logger.warning(f"Failed to warm cache for {market_id}: {e}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        info = await self.redis._client.info("stats")
        clients_info = await self.redis._client.info("clients")
        return {
            'keyspace_hits': info.get('keyspace_hits', 0),
            'keyspace_misses': info.get('keyspace_misses', 0),
            'hit_rate': self._calculate_hit_rate(info),
            'connected_clients': clients_info.get('connected_clients', 0) if clients_info else 0,
            'used_memory': info.get('used_memory_human', 'unknown')
        }

    def _calculate_hit_rate(self, info: Dict) -> float:
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100
