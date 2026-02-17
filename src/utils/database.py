from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session, declarative_base
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
import enum
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger("database")

Base = declarative_base()


class OrderStatus(enum.Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(enum.Enum):
    YES = "yes"
    NO = "no"


class TradeType(enum.Enum):
    BUY = "buy"
    SELL = "sell"


class Market(Base):
    __tablename__ = "markets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String(255), unique=True, nullable=False, index=True)
    event_id = Column(String(255), index=True)
    series_id = Column(String(255), index=True)
    title = Column(Text)
    description = Column(Text)
    status = Column(String(50), default="open")
    current_price = Column(Float)
    volume = Column(Integer, default=0)
    liquidity = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    settled_at = Column(DateTime, nullable=True)
    settlement_price = Column(Float, nullable=True)

    orders = relationship("Order", back_populates="market")
    positions = relationship("Position", back_populates="market")


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(255), unique=True, nullable=False, index=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    external_id = Column(String(255), nullable=True)
    side = Column(String(10), nullable=False)
    order_type = Column(String(20), nullable=False)
    price = Column(Integer, nullable=False)
    count = Column(Integer, nullable=False)
    filled_count = Column(Integer, default=0)
    remaining_count = Column(Integer, default=0)
    status = Column(String(20), default=OrderStatus.PENDING.value)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    filled_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    market = relationship("Market", back_populates="orders")
    fills = relationship("OrderFill", back_populates="order")

    __table_args__ = (
        Index("idx_order_market_status", "market_id", "status"),
        Index("idx_order_created", "created_at"),
    )


class OrderFill(Base):
    __tablename__ = "order_fills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    fill_id = Column(String(255), unique=True, nullable=False)
    price = Column(Integer, nullable=False)
    count = Column(Integer, nullable=False)
    fees = Column(Integer, default=0)
    filled_at = Column(DateTime, default=datetime.utcnow)

    order = relationship("Order", back_populates="fills")


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    side = Column(String(10), nullable=False)
    quantity = Column(Integer, default=0)
    avg_cost = Column(Integer, default=0)
    realized_pnl = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    market = relationship("Market", back_populates="positions")

    __table_args__ = (Index("idx_position_market_side", "market_id", "side"),)


class ArbitrageOpportunity(Base):
    __tablename__ = "arbitrage_opportunities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    opportunity_id = Column(String(255), unique=True, nullable=False, index=True)
    type = Column(String(50), nullable=False)
    market_id_1 = Column(String(255), nullable=False)
    market_id_2 = Column(String(255), nullable=True)
    buy_market_id = Column(String(255), nullable=False)
    sell_market_id = Column(String(255), nullable=False)
    buy_price = Column(Integer, nullable=False)
    sell_price = Column(Integer, nullable=False)
    quantity = Column(Integer, nullable=False)
    gross_profit = Column(Integer, nullable=False)
    fees = Column(Integer, default=0)
    net_profit = Column(Integer, nullable=False)
    confidence = Column(Float, default=0.0)
    risk_level = Column(String(20), default="medium")
    status = Column(String(20), default="detected")
    detected_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime, nullable=True)
    execution_result = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_opportunity_status", "status"),
        Index("idx_opportunity_detected", "detected_at"),
    )


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(255), unique=True, nullable=False, index=True)
    opportunity_id = Column(String(255), index=True)
    market_id = Column(String(255), nullable=False)
    side = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Integer, nullable=False)
    exit_price = Column(Integer, nullable=True)
    pnl = Column(Integer, default=0)
    fees = Column(Integer, default=0)
    status = Column(String(20), default="open")
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_trade_status", "status"),
        Index("idx_trade_opened", "opened_at"),
    )


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cash_balance = Column(Integer, default=0)
    positions_value = Column(Integer, default=0)
    total_value = Column(Integer, default=0)
    open_positions = Column(Integer, default=0)
    daily_pnl = Column(Integer, default=0)
    total_pnl = Column(Integer, default=0)
    metrics = Column(SQLiteJSON, nullable=True)


class BotConfig(Base):
    __tablename__ = "bot_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(255), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MarketDataSnapshot(Base):
    __tablename__ = "market_data_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    best_bid = Column(Integer, nullable=True)
    best_ask = Column(Integer, nullable=True)
    mid_price = Column(Float, nullable=True)
    spread = Column(Integer, nullable=True)
    spread_percent = Column(Float, nullable=True)

    bid_depth_5 = Column(Integer, default=0)
    ask_depth_5 = Column(Integer, default=0)
    total_volume = Column(Integer, default=0)
    liquidity_score = Column(Float, nullable=True)

    orderbook_data = Column(SQLiteJSON, nullable=True)

    __table_args__ = (Index("idx_market_data_market_time", "market_id", "timestamp"),)


class PriceHistory(Base):
    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    price = Column(Float, nullable=False)
    volume = Column(Integer, default=0)

    __table_args__ = (Index("idx_price_market_time", "market_id", "timestamp"),)


class Database:
    def __init__(self, database_url: str = "sqlite:///data/arbitrage.db"):
        self.database_url = database_url
        self._engine = None
        self._session_factory = None

    def create_engine(self):
        from sqlalchemy import create_engine

        self._engine = create_engine(
            self.database_url, echo=False, pool_pre_ping=True, pool_recycle=3600
        )
        return self._engine

    def get_session(self) -> Session:
        if not self._session_factory:
            from sqlalchemy.orm import sessionmaker

            self._session_factory = sessionmaker(bind=self.create_engine())
        return self._session_factory()

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        if not self._engine:
            self.create_engine()
        with self._engine.connect() as conn:
            return pd.read_sql(sql, conn, params=params)

    def execute(self, sql: str, params: tuple = ()) -> None:
        if not self._engine:
            self.create_engine()
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(text(sql), params)

    def create_tables(self) -> None:
        if not self._engine:
            self.create_engine()
        Base.metadata.create_all(self._engine)
        logger.info("Database tables created")

    def drop_tables(self) -> None:
        if self._engine:
            Base.metadata.drop_all(self._engine)
            logger.info("Database tables dropped")

    async def save_order(self, order_data: Dict[str, Any]) -> Order:
        session = self.get_session()
        try:
            order = Order(
                order_id=order_data["order_id"],
                market_id=order_data["market_id"],
                side=order_data["side"],
                order_type=order_data["order_type"],
                price=order_data["price"],
                count=order_data["count"],
                status=order_data.get("status", "pending"),
            )
            session.add(order)
            session.commit()
            logger.info(f"Saved order: {order.order_id}")
            return order
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save order: {e}")
            raise
        finally:
            session.close()

    async def save_position(self, position_data: Dict[str, Any]) -> Position:
        session = self.get_session()
        try:
            position = Position(
                market_id=position_data["market_id"],
                side=position_data["side"],
                quantity=position_data["quantity"],
                avg_cost=position_data["avg_cost"],
                realized_pnl=position_data.get("realized_pnl", 0),
            )
            session.add(position)
            session.commit()
            return position
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def save_opportunity(self, opp_data: Dict[str, Any]) -> ArbitrageOpportunity:
        session = self.get_session()
        try:
            opp = ArbitrageOpportunity(
                opportunity_id=opp_data["opportunity_id"],
                type=opp_data["type"],
                market_id_1=opp_data["market_id_1"],
                market_id_2=opp_data.get("market_id_2"),
                buy_market_id=opp_data["buy_market_id"],
                sell_market_id=opp_data["sell_market_id"],
                buy_price=opp_data["buy_price"],
                sell_price=opp_data["sell_price"],
                quantity=opp_data["quantity"],
                gross_profit=opp_data["gross_profit"],
                fees=opp_data.get("fees", 0),
                net_profit=opp_data["net_profit"],
                confidence=opp_data.get("confidence", 0.0),
                risk_level=opp_data.get("risk_level", "medium"),
            )
            session.add(opp)
            session.commit()
            return opp
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def save_trade(self, trade_data: Dict[str, Any]) -> Trade:
        session = self.get_session()
        try:
            trade = Trade(
                trade_id=trade_data["trade_id"],
                opportunity_id=trade_data.get("opportunity_id"),
                market_id=trade_data["market_id"],
                side=trade_data["side"],
                quantity=trade_data["quantity"],
                entry_price=trade_data["entry_price"],
                exit_price=trade_data.get("exit_price"),
                pnl=trade_data.get("pnl", 0),
                fees=trade_data.get("fees", 0),
                status=trade_data.get("status", "open"),
            )
            session.add(trade)
            session.commit()
            return trade
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def record_portfolio_snapshot(
        self, stats: Dict[str, Any]
    ) -> PortfolioSnapshot:
        session = self.get_session()
        try:
            snapshot = PortfolioSnapshot(
                cash_balance=stats.get("cash_balance", 0),
                positions_value=stats.get("positions_value", 0),
                total_value=stats.get("total_value", 0),
                open_positions=stats.get("open_positions", 0),
                daily_pnl=stats.get("daily_pnl", 0),
                total_pnl=stats.get("total_pnl", 0),
                metrics=stats.get("metrics"),
            )
            session.add(snapshot)
            session.commit()
            return snapshot
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def get_open_orders(self) -> List[Order]:
        session = self.get_session()
        try:
            return (
                session.query(Order)
                .filter(
                    Order.status.in_(
                        [
                            OrderStatus.PENDING.value,
                            OrderStatus.SUBMITTED.value,
                            OrderStatus.PARTIAL.value,
                        ]
                    )
                )
                .all()
            )
        finally:
            session.close()

    async def get_open_positions(self) -> List[Position]:
        session = self.get_session()
        try:
            return session.query(Position).filter(Position.quantity > 0).all()
        finally:
            session.close()

    async def get_recent_trades(self, limit: int = 100) -> List[Trade]:
        session = self.get_session()
        try:
            return (
                session.query(Trade).order_by(Trade.opened_at.desc()).limit(limit).all()
            )
        finally:
            session.close()

    async def get_portfolio_history(
        self, start_time: datetime, end_time: Optional[datetime] = None
    ) -> List[PortfolioSnapshot]:
        session = self.get_session()
        try:
            query = session.query(PortfolioSnapshot).filter(
                PortfolioSnapshot.timestamp >= start_time
            )
            if end_time:
                query = query.filter(PortfolioSnapshot.timestamp <= end_time)
            return query.order_by(PortfolioSnapshot.timestamp).all()
        finally:
            session.close()

    async def cleanup_old_data(self, days: int = 30) -> int:
        session = self.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            from sqlalchemy import delete

            result = session.execute(
                delete(PortfolioSnapshot).where(PortfolioSnapshot.timestamp < cutoff)
            )
            session.commit()
            return result.rowcount if hasattr(result, "rowcount") else 0
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def save_market_data_snapshot(
        self, data: Dict[str, Any]
    ) -> MarketDataSnapshot:
        """Save a market data snapshot for historical analysis."""
        session = self.get_session()
        try:
            snapshot = MarketDataSnapshot(
                market_id=data["market_id"],
                best_bid=data.get("best_bid"),
                best_ask=data.get("best_ask"),
                mid_price=data.get("mid_price"),
                spread=data.get("spread"),
                spread_percent=data.get("spread_percent"),
                bid_depth_5=data.get("bid_depth_5", 0),
                ask_depth_5=data.get("ask_depth_5", 0),
                total_volume=data.get("total_volume", 0),
                liquidity_score=data.get("liquidity_score"),
                orderbook_data=data.get("orderbook_data"),
            )
            session.add(snapshot)
            session.commit()
            return snapshot
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save market data snapshot: {e}")
            raise
        finally:
            session.close()

    async def save_price_history(self, data: Dict[str, Any]) -> PriceHistory:
        """Save a price history point."""
        session = self.get_session()
        try:
            price_record = PriceHistory(
                market_id=data["market_id"],
                price=data["price"],
                volume=data.get("volume", 0),
            )
            session.add(price_record)
            session.commit()
            return price_record
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save price history: {e}")
            raise
        finally:
            session.close()

    async def get_market_data_snapshots(
        self,
        market_id: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[MarketDataSnapshot]:
        """Get historical market data snapshots."""
        session = self.get_session()
        try:
            query = session.query(MarketDataSnapshot).filter(
                MarketDataSnapshot.market_id == market_id,
                MarketDataSnapshot.timestamp >= start_time,
            )
            if end_time:
                query = query.filter(MarketDataSnapshot.timestamp <= end_time)
            return (
                query.order_by(MarketDataSnapshot.timestamp.desc()).limit(limit).all()
            )
        finally:
            session.close()

    async def get_price_history(
        self,
        market_id: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> List[PriceHistory]:
        """Get historical price data for a market."""
        session = self.get_session()
        try:
            query = session.query(PriceHistory).filter(
                PriceHistory.market_id == market_id,
                PriceHistory.timestamp >= start_time,
            )
            if end_time:
                query = query.filter(PriceHistory.timestamp <= end_time)
            return query.order_by(PriceHistory.timestamp).all()
        finally:
            session.close()
