"""High-performance Redis caching layer for arbitrage bot."""

import asyncio
import json
import logging
import pickle
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import redis.asyncio as redis
from dataclasses import asdict

from src.utils.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger("redis_cache")


class PerformanceCache:
    """High-performance cache layer with TTL and data compression."""

    def __init__(self, config: Config):
        """Initialize Redis cache with performance optimizations."""
        self.config = config
        self.redis_client = None
        self.cache_stats = {"hits": 0, "misses": 0, "sets": 0, "dels": 0, "errors": 0}

        # Cache TTL configurations
        self.ttl_configs = {
            "orderbook": 30,  # 30 seconds
            "market_status": 60,  # 1 minute
            "opportunities": 10,  # 10 seconds
            "position": 300,  # 5 minutes
            "portfolio": 60,  # 1 minute
            "metrics": 5,  # 5 seconds
            "rates": 3600,  # 1 hour
            "statistics": 1800,  # 30 minutes
        }

        # Connection pool settings
        self.pool_size = 20
        self.connection_timeout = 5
        self.socket_timeout = 5
        self.max_connections = 100

        self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis connection pool."""
        try:
            redis_host = self.config.get("cache.redis_host", "localhost")
            redis_port = self.config.get("cache.redis_port", 6379)
            redis_db = self.config.get("cache.redis_db", 0)
            redis_password = self.config.get("cache.redis_password")

            # Create connection pool
            self.redis_client = redis.ConnectionPool(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                max_connections=self.max_connections,
                socket_connect_timeout=self.connection_timeout,
                socket_timeout=self.socket_timeout,
                retry_on_timeout=True,
                decode_responses=False,  # Keep binary for performance
                encoding="utf-8",
            )

            # Test connection
            with self.redis_client.get_connection() as conn:
                conn.ping()

            logger.info("Redis cache initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None

    async def get_orderbook(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Get orderbook from cache with ultra-low latency."""
        if not self.redis_client:
            return None

        key = f"orderbook:{market_id}"
        try:
            async with self.redis_client.get_connection() as conn:
                data = await conn.get(key)

                if data:
                    self.cache_stats["hits"] += 1
                    return pickle.loads(data)
                else:
                    self.cache_stats["misses"] += 1
                    return None

        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
            self.cache_stats["errors"] += 1
            return None

    async def set_orderbook(self, market_id: str, orderbook: Dict[str, Any]) -> bool:
        """Set orderbook in cache with compression."""
        if not self.redis_client:
            return False

        key = f"orderbook:{market_id}"
        ttl = self.ttl_configs["orderbook"]

        try:
            # Compress data for Redis
            data = pickle.dumps(orderbook, protocol=pickle.HIGHEST_PROTOCOL)

            async with self.redis_client.get_connection() as conn:
                await conn.setex(key, ttl, data)
                self.cache_stats["sets"] += 1
                return True

        except Exception as e:
            logger.error(f"Cache set error for {key}: {e}")
            self.cache_stats["errors"] += 1
            return False

    async def get_opportunities(
        self, strategy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get cached arbitrage opportunities."""
        if not self.redis_client:
            return []

        key = f"opportunities:{strategy or 'all'}"
        try:
            async with self.redis_client.get_connection() as conn:
                data = await conn.get(key)

                if data:
                    self.cache_stats["hits"] += 1
                    return pickle.loads(data)
                else:
                    self.cache_stats["misses"] += 1
                    return []

        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
            self.cache_stats["errors"] += 1
            return []

    async def set_opportunities(
        self, opportunities: List[Dict[str, Any]], strategy: Optional[str] = None
    ) -> bool:
        """Set arbitrage opportunities in cache."""
        if not self.redis_client:
            return False

        key = f"opportunities:{strategy or 'all'}"
        ttl = self.ttl_configs["opportunities"]

        try:
            data = pickle.dumps(opportunities, protocol=pickle.HIGHEST_PROTOCOL)

            async with self.redis_client.get_connection() as conn:
                await conn.setex(key, ttl, data)
                self.cache_stats["sets"] += 1
                return True

        except Exception as e:
            logger.error(f"Cache set error for {key}: {e}")
            self.cache_stats["errors"] += 1
            return False

    async def get_market_status(self) -> Optional[Dict[str, Any]]:
        """Get cached exchange/market status."""
        if not self.redis_client:
            return None

        key = "market:status"
        try:
            async with self.redis_client.get_connection() as conn:
                data = await conn.get(key)

                if data:
                    self.cache_stats["hits"] += 1
                    return json.loads(data)
                else:
                    self.cache_stats["misses"] += 1
                    return None

        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
            self.cache_stats["errors"] += 1
            return None

    async def set_market_status(self, status: Dict[str, Any]) -> bool:
        """Set exchange/market status in cache."""
        if not self.redis_client:
            return False

        key = "market:status"
        ttl = self.ttl_configs["market_status"]

        try:
            data = json.dumps(status, default=str)

            async with self.redis_client.get_connection() as conn:
                await conn.setex(key, ttl, data)
                self.cache_stats["sets"] += 1
                return True

        except Exception as e:
            logger.error(f"Cache set error for {key}: {e}")
            self.cache_stats["errors"] += 1
            return False

    async def get_portfolio_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get cached portfolio snapshot."""
        if not self.redis_client:
            return None

        key = "portfolio:snapshot"
        try:
            async with self.redis_client.get_connection() as conn:
                data = await conn.get(key)

                if data:
                    self.cache_stats["hits"] += 1
                    return pickle.loads(data)
                else:
                    self.cache_stats["misses"] += 1
                    return None

        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
            self.cache_stats["errors"] += 1
            return None

    async def set_portfolio_snapshot(self, portfolio_data: Dict[str, Any]) -> bool:
        """Set portfolio snapshot in cache."""
        if not self.redis_client:
            return False

        key = "portfolio:snapshot"
        ttl = self.ttl_configs["portfolio"]

        try:
            data = pickle.dumps(portfolio_data, protocol=pickle.HIGHEST_PROTOCOL)

            async with self.redis_client.get_connection() as conn:
                await conn.setex(key, ttl, data)
                self.cache_stats["sets"] += 1
                return True

        except Exception as e:
            logger.error(f"Cache set error for {key}: {e}")
            self.cache_stats["errors"] += 1
            return False

    async def increment_metrics(self, metric_name: str, value: int = 1) -> bool:
        """Increment performance metrics counter."""
        if not self.redis_client:
            return False

        key = f"metrics:{metric_name}"

        try:
            async with self.redis_client.get_connection() as conn:
                await conn.incrby(key, value)
                await conn.expire(key, self.ttl_configs["metrics"])
                return True

        except Exception as e:
            logger.error(f"Cache increment error for {key}: {e}")
            self.cache_stats["errors"] += 1
            return False

    async def get_metrics(self, metric_name: str) -> int:
        """Get performance metric value."""
        if not self.redis_client:
            return 0

        key = f"metrics:{metric_name}"
        try:
            async with self.redis_client.get_connection() as conn:
                value = await conn.get(key)
                return int(value) if value else 0

        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
            self.cache_stats["errors"] += 1
            return 0

    async def delete_keys(self, pattern: str) -> int:
        """Delete multiple keys by pattern."""
        if not self.redis_client:
            return 0

        try:
            async with self.redis_client.get_connection() as conn:
                keys = await conn.keys(pattern)
                if keys:
                    deleted = await conn.delete(*keys)
                    self.cache_stats["dels"] += deleted
                    return deleted
                return 0

        except Exception as e:
            logger.error(f"Cache delete error for pattern {pattern}: {e}")
            self.cache_stats["errors"] += 1
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self.redis_client:
            return self.cache_stats

        try:
            async with self.redis_client.get_connection() as conn:
                info = await conn.info()

                return {
                    **self.cache_stats,
                    "redis_info": {
                        "used_memory": info.get("used_memory", 0),
                        "connected_clients": info.get("connected_clients", 0),
                        "total_commands_processed": info.get(
                            "total_commands_processed", 0
                        ),
                        "keyspace_hits": info.get("keyspace_hits", 0),
                        "keyspace_misses": info.get("keyspace_misses", 0),
                    },
                    "hit_rate": (
                        self.cache_stats["hits"]
                        / max(1, self.cache_stats["hits"] + self.cache_stats["misses"])
                    )
                    * 100,
                }

        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return self.cache_stats

    async def warm_cache(self, market_ids: List[str]) -> bool:
        """Warm up cache with essential data for markets."""
        logger.info(f"Warming cache for {len(market_ids)} markets")

        success_count = 0
        for market_id in market_ids:
            # Pre-populate cache structure
            try:
                await self.set_orderbook(market_id, {})
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to warm cache for {market_id}: {e}")

        logger.info(
            f"Cache warming completed: {success_count}/{len(market_ids)} markets"
        )
        return success_count == len(market_ids)

    async def cleanup_expired_data(self) -> int:
        """Clean up expired cache entries."""
        if not self.redis_client:
            return 0

        try:
            # Clean up old metrics
            await self.delete_keys("metrics:*")

            # Clean up very old orderbooks
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            cutoff_timestamp = int(cutoff_time.timestamp())

            # Note: This would require additional Redis modules for efficient cleanup
            # For now, let TTL handle natural expiration

            logger.info("Cache cleanup completed")
            return 0

        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            return 0

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive cache health check."""
        health_status = {
            "redis_connected": False,
            "cache_performance": "unknown",
            "memory_usage": "unknown",
            "error_rate": 0,
        }

        if not self.redis_client:
            return health_status

        try:
            async with self.redis_client.get_connection() as conn:
                # Test basic connectivity
                await conn.ping()
                health_status["redis_connected"] = True

                # Get memory usage
                info = await conn.info()
                memory_bytes = info.get("used_memory", 0)
                memory_mb = memory_bytes / (1024 * 1024)
                health_status["memory_usage"] = f"{memory_mb:.2f}MB"

                # Calculate error rate
                total_ops = (
                    self.cache_stats["hits"]
                    + self.cache_stats["misses"]
                    + self.cache_stats["sets"]
                    + self.cache_stats["dels"]
                )
                if total_ops > 0:
                    health_status["error_rate"] = (
                        self.cache_stats["errors"] / total_ops * 100
                    )

                # Determine performance
                hit_rate = (
                    self.cache_stats["hits"]
                    / max(1, self.cache_stats["hits"] + self.cache_stats["misses"])
                ) * 100

                if hit_rate > 80:
                    health_status["cache_performance"] = "excellent"
                elif hit_rate > 60:
                    health_status["cache_performance"] = "good"
                elif hit_rate > 40:
                    health_status["cache_performance"] = "fair"
                else:
                    health_status["cache_performance"] = "poor"

        except Exception as e:
            logger.error(f"Cache health check error: {e}")

        return health_status


class CacheManager:
    """High-level cache management with smart preloading and cleanup."""

    def __init__(self, config: Config):
        """Initialize cache manager with performance optimizations."""
        self.config = config
        self.cache = PerformanceCache(config)
        self.preload_tasks = []
        self.cleanup_task = None
        self.performance_monitoring = True

    async def initialize(self):
        """Initialize cache system with warmup."""
        logger.info("Initializing cache manager...")

        # Warm cache with essential data
        essential_markets = self.config.get("cache.essential_markets", [])
        if essential_markets:
            await self.cache.warm_cache(essential_markets)

        # Start background cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Start performance monitoring
        if self.performance_monitoring:
            asyncio.create_task(self._performance_monitoring_loop())

        logger.info("Cache manager initialized")

    async def shutdown(self):
        """Gracefully shutdown cache system."""
        logger.info("Shutting down cache manager...")

        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Final cleanup
        await self.cache.cleanup_expired_data()

        logger.info("Cache manager shutdown complete")

    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                await self.cache.cleanup_expired_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _performance_monitoring_loop(self):
        """Background performance monitoring."""
        while True:
            try:
                await asyncio.sleep(60)  # 1 minute
                stats = await self.cache.get_cache_stats()

                # Log performance metrics
                hit_rate = stats.get("hit_rate", 0)
                error_rate = stats.get("error_rate", 0)

                if hit_rate < 50:
                    logger.warning(f"Low cache hit rate: {hit_rate:.1f}%")

                if error_rate > 5:
                    logger.warning(f"High cache error rate: {error_rate:.1f}%")

                # Update metrics dashboard
                await self.cache.increment_metrics("cache_hit_rate", int(hit_rate))
                await self.cache.increment_metrics("cache_error_rate", int(error_rate))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(config: Optional[Config] = None) -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager

    if _cache_manager is None and config is not None:
        _cache_manager = CacheManager(config)

    return _cache_manager


async def initialize_cache(config: Config) -> CacheManager:
    """Initialize and return cache manager."""
    manager = get_cache_manager(config)
    await manager.initialize()
    return manager


async def shutdown_cache():
    """Shutdown global cache manager."""
    global _cache_manager

    if _cache_manager is not None:
        await _cache_manager.shutdown()
        _cache_manager = None
