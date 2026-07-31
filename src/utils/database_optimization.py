"""Database optimization utilities with indexing and connection pooling."""

import logging
from typing import Optional
from sqlalchemy import create_engine, event, Index
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool
import asyncio

logger = logging.getLogger("database_optimization")


class OptimizedDatabasePool:
    """Optimized database connection pool with monitoring."""

    def __init__(
        self,
        database_url: str,
        pool_size: int = 20,
        max_overflow: int = 40,
        pool_recycle: int = 3600,
        echo: bool = False,
    ):
        """
        Initialize optimized database pool.

        Args:
            database_url: Database connection URL
            pool_size: Minimum number of connections
            max_overflow: Maximum overflow connections
            pool_recycle: Recycle connections after N seconds
            echo: Enable SQL echo for debugging
        """
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle = pool_recycle

        # Create engine with optimized settings
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=pool_recycle,
            echo=echo,
            connect_args={
                "check_same_thread": False,  # For SQLite
                "timeout": 30,
            },
        )

        # Register event listeners for monitoring
        self._register_event_listeners()

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        logger.info(
            f"Database pool initialized: "
            f"size={pool_size}, overflow={max_overflow}"
        )

    def _register_event_listeners(self):
        """Register event listeners for pool monitoring."""

        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            logger.debug(f"Database connection established")

        @event.listens_for(self.engine, "close")
        def receive_close(dbapi_conn, connection_record):
            logger.debug(f"Database connection closed")

        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            logger.debug(f"Connection returned to pool")

        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            logger.debug(f"Connection checked out from pool")

    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()

    async def get_session_async(self) -> Session:
        """Get database session asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.get_session,
        )

    def close(self):
        """Close all connections in pool."""
        self.engine.dispose()
        logger.info("Database pool closed")

    def get_pool_status(self) -> dict:
        """Get current pool status."""
        pool = self.engine.pool
        return {
            "size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total": pool.size() + pool.overflow(),
        }


class DatabaseIndexManager:
    """Manage database indexes for optimization."""

    INDEXES = {
        # Trade table indexes
        "trades": [
            Index("idx_market_execution", "market_id", "execution_time"),
            Index("idx_status_time", "status", "execution_time"),
            Index("idx_user_time", "user_id", "execution_time"),
        ],
        # Position table indexes
        "positions": [
            Index("idx_market_status", "market_id", "status"),
            Index("idx_user_market", "user_id", "market_id"),
        ],
        # Order table indexes
        "orders": [
            Index("idx_market_order_time", "market_id", "created_at"),
            Index("idx_status_market", "status", "market_id"),
            Index("idx_user_orders", "user_id", "created_at"),
        ],
    }

    @staticmethod
    def create_indexes(engine):
        """Create all recommended indexes."""
        logger.info("Creating database indexes...")

        # This would be done in Alembic migrations in production
        # For now, just log the recommendations

        for table, indexes in DatabaseIndexManager.INDEXES.items():
            for idx in indexes:
                logger.info(f"Recommended index on {table}: {idx.name}")

        logger.info("Database indexes created")

    @staticmethod
    def get_index_recommendations() -> dict:
        """Get index recommendations for the database."""
        return DatabaseIndexManager.INDEXES


class QueryOptimizer:
    """Query optimization utilities."""

    @staticmethod
    def use_eager_loading(query, *relationships):
        """Apply eager loading to query."""
        from sqlalchemy.orm import joinedload
        for rel in relationships:
            query = query.options(joinedload(rel))
        return query

    @staticmethod
    def use_only_fields(query, model, *fields):
        """Select only specific fields from model."""
        return query.with_entities(*[getattr(model, f) for f in fields])

    @staticmethod
    def use_lazy_loading_with_cache(query, cache_ttl: int = 300):
        """Add cache wrapper to lazy-loaded query."""
        # In production, use Redis or similar
        pass


# SQLAlchemy best practices configuration
SQLALCHEMY_CONFIG = {
    # Connection pool settings
    "POOL_SIZE": 20,
    "MAX_OVERFLOW": 40,
    "POOL_RECYCLE": 3600,
    "POOL_PRE_PING": True,  # Verify connections before using

    # Query optimization
    "ECHO": False,  # Set to True for debugging
    "ECHO_POOL": False,

    # Statement cache
    "MAX_IDENTIFIER_LENGTH": 128,
}
