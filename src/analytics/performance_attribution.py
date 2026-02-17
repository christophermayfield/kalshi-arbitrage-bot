"""
Real-Time Performance Attribution & Analytics System
Enterprise-grade performance tracking with granular attribution and advanced analytics
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import uuid

from src.utils.logging_utils import get_logger
from src.utils.config import Config
from src.core.arbitrage import ArbitrageOpportunity
from src.core.orderbook import OrderBook

logger = get_logger("performance_attribution")


class AttributionDimension(Enum):
    """Performance attribution dimensions"""

    STRATEGY = "strategy"
    MARKET = "market"
    TIMEFRAME = "timeframe"
    MARKET_CONDITION = "market_condition"
    EXECUTION_ALGORITHM = "execution_algorithm"
    RISK_FACTOR = "risk_factor"
    SECTOR = "sector"
    LIQUIDITY_BUCKET = "liquidity_bucket"


class PerformanceMetric(Enum):
    """Performance metrics types"""

    RETURN = "return"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR = "calmar"
    ALPHA = "alpha"
    BETA = "beta"
    INFORMATION_RATIO = "information_ratio"


@dataclass
class TradeExecution:
    """Detailed trade execution record"""

    execution_id: str
    opportunity_id: str
    strategy: str
    market_id: str

    # Execution details
    action: str  # buy, sell, short, cover
    quantity: int
    intended_price: float
    execution_price: float
    execution_timestamp: datetime

    # Cost analysis
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    total_cost: float = 0.0

    # Timing metrics
    decision_to_execution_ms: float = 0.0
    order_duration_ms: float = 0.0

    # Market conditions
    market_volatility: float = 0.0
    liquidity_score: float = 0.0
    spread_at_execution: float = 0.0

    # Attribution tags
    tags: List[str] = field(default_factory=list)

    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""

    timestamp: datetime
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Risk metrics
    current_exposure: float = 0.0
    var_95: float = 0.0
    beta: float = 0.0

    # Attribution data
    strategy_performance: Dict[str, float] = field(default_factory=dict)
    market_performance: Dict[str, float] = field(default_factory=dict)

    # Market conditions
    market_regime: str = "unknown"
    volatility_regime: str = "unknown"


@dataclass
class AttributionReport:
    """Comprehensive attribution report"""

    report_id: str
    start_time: datetime
    end_time: datetime
    period_days: int

    # Overall performance
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Attribution breakdowns
    strategy_attribution: Dict[str, float] = field(default_factory=dict)
    market_attribution: Dict[str, float] = field(default_factory=dict)
    timeframe_attribution: Dict[str, float] = field(default_factory=dict)
    condition_attribution: Dict[str, float] = field(default_factory=dict)

    # Execution quality
    avg_slippage: float = 0.0
    avg_execution_time_ms: float = 0.0
    execution_quality_score: float = 0.0

    # Risk analysis
    risk_adjusted_performance: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Detailed metrics
    trade_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Recommendations
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PerformanceDatabase:
    """High-performance database for performance data"""

    def __init__(self, db_path: str = "data/performance.db"):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Trade executions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trade_executions (
                        execution_id TEXT PRIMARY KEY,
                        opportunity_id TEXT,
                        strategy TEXT,
                        market_id TEXT,
                        action TEXT,
                        quantity INTEGER,
                        intended_price REAL,
                        execution_price REAL,
                        execution_timestamp TEXT,
                        commission REAL,
                        slippage REAL,
                        market_impact REAL,
                        total_cost REAL,
                        decision_to_execution_ms REAL,
                        order_duration_ms REAL,
                        market_volatility REAL,
                        liquidity_score REAL,
                        spread_at_execution REAL,
                        tags TEXT,
                        realized_pnl REAL,
                        unrealized_pnl REAL
                    )
                """)

                # Performance snapshots table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        total_pnl REAL,
                        realized_pnl REAL,
                        unrealized_pnl REAL,
                        total_trades INTEGER,
                        winning_trades INTEGER,
                        losing_trades INTEGER,
                        current_exposure REAL,
                        var_95 REAL,
                        beta REAL,
                        strategy_performance TEXT,
                        market_performance TEXT,
                        market_regime TEXT,
                        volatility_regime TEXT
                    )
                """)

                # Attribution reports table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS attribution_reports (
                        report_id TEXT PRIMARY KEY,
                        start_time TEXT,
                        end_time TEXT,
                        period_days INTEGER,
                        total_return REAL,
                        annualized_return REAL,
                        sharpe_ratio REAL,
                        max_drawdown REAL,
                        strategy_attribution TEXT,
                        market_attribution TEXT,
                        timeframe_attribution TEXT,
                        condition_attribution TEXT,
                        avg_slippage REAL,
                        avg_execution_time_ms REAL,
                        execution_quality_score REAL,
                        risk_adjusted_performance TEXT,
                        correlation_matrix TEXT,
                        trade_count INTEGER,
                        win_rate REAL,
                        profit_factor REAL,
                        insights TEXT,
                        recommendations TEXT
                    )
                """)

                # Create indexes for performance
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_executions_timestamp ON trade_executions(execution_timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_executions_strategy ON trade_executions(strategy)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_executions_market ON trade_executions(market_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON performance_snapshots(timestamp)"
                )

                conn.commit()
                logger.info("Performance database initialized")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def store_execution(self, execution: TradeExecution) -> None:
        """Store trade execution"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO trade_executions VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """,
                    (
                        execution.execution_id,
                        execution.opportunity_id,
                        execution.strategy,
                        execution.market_id,
                        execution.action,
                        execution.quantity,
                        execution.intended_price,
                        execution.execution_price,
                        execution.execution_timestamp.isoformat(),
                        execution.commission,
                        execution.slippage,
                        execution.market_impact,
                        execution.total_cost,
                        execution.decision_to_execution_ms,
                        execution.order_duration_ms,
                        execution.market_volatility,
                        execution.liquidity_score,
                        execution.spread_at_execution,
                        json.dumps(execution.tags),
                        execution.realized_pnl,
                        execution.unrealized_pnl,
                    ),
                )

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to store execution: {e}")

    def store_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Store performance snapshot"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO performance_snapshots VALUES (
                        NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """,
                    (
                        snapshot.timestamp.isoformat(),
                        snapshot.total_pnl,
                        snapshot.realized_pnl,
                        snapshot.unrealized_pnl,
                        snapshot.total_trades,
                        snapshot.winning_trades,
                        snapshot.losing_trades,
                        snapshot.current_exposure,
                        snapshot.var_95,
                        snapshot.beta,
                        json.dumps(snapshot.strategy_performance),
                        json.dumps(snapshot.market_performance),
                        snapshot.market_regime,
                        snapshot.volatility_regime,
                    ),
                )

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to store snapshot: {e}")

    def get_executions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        strategy: Optional[str] = None,
        market_id: Optional[str] = None,
    ) -> List[TradeExecution]:
        """Retrieve trade executions with filters"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM trade_executions WHERE 1=1"
                params = []

                if start_time:
                    query += " AND execution_timestamp >= ?"
                    params.append(start_time.isoformat())

                if end_time:
                    query += " AND execution_timestamp <= ?"
                    params.append(end_time.isoformat())

                if strategy:
                    query += " AND strategy = ?"
                    params.append(strategy)

                if market_id:
                    query += " AND market_id = ?"
                    params.append(market_id)

                query += " ORDER BY execution_timestamp"

                cursor.execute(query, params)
                rows = cursor.fetchall()

                executions = []
                for row in rows:
                    execution = TradeExecution(
                        execution_id=row[0],
                        opportunity_id=row[1],
                        strategy=row[2],
                        market_id=row[3],
                        action=row[4],
                        quantity=row[5],
                        intended_price=row[6],
                        execution_price=row[7],
                        execution_timestamp=datetime.fromisoformat(row[8]),
                        commission=row[9],
                        slippage=row[10],
                        market_impact=row[11],
                        total_cost=row[12],
                        decision_to_execution_ms=row[13],
                        order_duration_ms=row[14],
                        market_volatility=row[15],
                        liquidity_score=row[16],
                        spread_at_execution=row[17],
                        tags=json.loads(row[18]) if row[18] else [],
                        realized_pnl=row[19],
                        unrealized_pnl=row[20],
                    )
                    executions.append(execution)

                return executions

        except Exception as e:
            logger.error(f"Failed to retrieve executions: {e}")
            return []


class RealTimeAttributionEngine:
    """Real-time performance attribution engine"""

    def __init__(self, config: Config):
        self.config = config
        self.attribution_config = config.get("performance_attribution", {})

        # Database
        self.db = PerformanceDatabase()

        # Real-time tracking
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        self.strategy_performance: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.market_performance: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Performance snapshots
        self.snapshots: deque = deque(maxlen=10000)
        self.last_snapshot_time = datetime.now()

        # Attribution cache
        self.attribution_cache: Dict[str, AttributionReport] = {}
        self.cache_ttl = timedelta(minutes=5)

        # Background processing
        self.running = False
        self.snapshot_task: Optional[asyncio.Task] = None

        # Configuration
        self.snapshot_interval_seconds = self.attribution_config.get(
            "snapshot_interval", 60
        )
        self.attribution_window_days = self.attribution_config.get(
            "attribution_window", 30
        )
        self.enable_real_time_attribution = self.attribution_config.get(
            "enable_real_time", True
        )

        logger.info("Real-time attribution engine initialized")

    async def start(self) -> None:
        """Start the attribution engine"""
        try:
            self.running = True

            # Start background snapshot task
            self.snapshot_task = asyncio.create_task(self._snapshot_loop())

            logger.info("Real-time attribution engine started")

        except Exception as e:
            logger.error(f"Failed to start attribution engine: {e}")
            raise

    async def stop(self) -> None:
        """Stop the attribution engine"""
        try:
            self.running = False

            if self.snapshot_task:
                self.snapshot_task.cancel()
                try:
                    await self.snapshot_task
                except asyncio.CancelledError:
                    pass

            logger.info("Real-time attribution engine stopped")

        except Exception as e:
            logger.error(f"Failed to stop attribution engine: {e}")

    async def record_execution(self, execution: TradeExecution) -> None:
        """Record a trade execution for attribution"""
        try:
            # Store in database
            self.db.store_execution(execution)

            # Update real-time tracking
            await self._update_real_time_tracking(execution)

            # Update positions
            await self._update_positions(execution)

            # Trigger attribution update if enabled
            if self.enable_real_time_attribution:
                await self._update_attribution_cache(execution)

            logger.debug(f"Recorded execution: {execution.execution_id}")

        except Exception as e:
            logger.error(f"Failed to record execution: {e}")

    async def _update_real_time_tracking(self, execution: TradeExecution) -> None:
        """Update real-time performance tracking"""
        try:
            # Update strategy performance
            strategy_key = execution.strategy
            strategy_perf = {
                "timestamp": execution.execution_timestamp,
                "pnl": execution.realized_pnl,
                "slippage": execution.slippage,
                "execution_time_ms": execution.order_duration_ms,
                "quantity": execution.quantity,
            }
            self.strategy_performance[strategy_key].append(strategy_perf)

            # Update market performance
            market_key = execution.market_id
            market_perf = {
                "timestamp": execution.execution_timestamp,
                "pnl": execution.realized_pnl,
                "slippage": execution.slippage,
                "volatility": execution.market_volatility,
                "liquidity": execution.liquidity_score,
            }
            self.market_performance[market_key].append(market_perf)

        except Exception as e:
            logger.error(f"Failed to update real-time tracking: {e}")

    async def _update_positions(self, execution: TradeExecution) -> None:
        """Update current positions"""
        try:
            market_id = execution.market_id

            if market_id not in self.current_positions:
                self.current_positions[market_id] = {
                    "quantity": 0,
                    "avg_price": 0.0,
                    "unrealized_pnl": 0.0,
                    "last_update": datetime.now(),
                }

            position = self.current_positions[market_id]

            # Update position based on execution
            if execution.action in ["buy", "long"]:
                new_quantity = position["quantity"] + execution.quantity
                if new_quantity != 0:
                    position["avg_price"] = (
                        position["avg_price"] * position["quantity"]
                        + execution.execution_price * execution.quantity
                    ) / new_quantity
                position["quantity"] = new_quantity

            elif execution.action in ["sell", "short"]:
                position["quantity"] -= execution.quantity
                # Calculate realized P&L
                if position["quantity"] >= 0:  # Closing long position
                    realized_pnl = (
                        execution.execution_price - position["avg_price"]
                    ) * execution.quantity
                else:  # Closing short position
                    realized_pnl = (
                        position["avg_price"] - execution.execution_price
                    ) * execution.quantity

                # Update execution with realized P&L
                execution.realized_pnl = realized_pnl

            position["last_update"] = execution.execution_timestamp

            # Clean up closed positions
            if position["quantity"] == 0:
                position["avg_price"] = 0.0
                position["unrealized_pnl"] = 0.0

        except Exception as e:
            logger.error(f"Failed to update positions: {e}")

    async def _update_attribution_cache(self, execution: TradeExecution) -> None:
        """Update attribution cache with new execution"""
        try:
            # Invalidate relevant cache entries
            cache_keys_to_remove = []

            for cache_key in self.attribution_cache.keys():
                if (
                    execution.strategy in cache_key
                    or execution.market_id in cache_key
                    or "all" in cache_key
                ):
                    cache_keys_to_remove.append(cache_key)

            for key in cache_keys_to_remove:
                del self.attribution_cache[key]

        except Exception as e:
            logger.error(f"Failed to update attribution cache: {e}")

    async def _snapshot_loop(self) -> None:
        """Background loop for creating performance snapshots"""
        while self.running:
            try:
                await self._create_performance_snapshot()
                await asyncio.sleep(self.snapshot_interval_seconds)

            except Exception as e:
                logger.error(f"Snapshot loop error: {e}")
                await asyncio.sleep(5)

    async def _create_performance_snapshot(self) -> None:
        """Create performance snapshot"""
        try:
            now = datetime.now()

            # Calculate current performance metrics
            total_pnl = sum(exec.realized_pnl for exec in self.db.get_executions())
            unrealized_pnl = sum(
                pos["unrealized_pnl"] for pos in self.current_positions.values()
            )

            # Get recent executions for statistics
            recent_executions = self.db.get_executions(
                start_time=now - timedelta(hours=24)
            )

            total_trades = len(recent_executions)
            winning_trades = len([e for e in recent_executions if e.realized_pnl > 0])
            losing_trades = len([e for e in recent_executions if e.realized_pnl < 0])

            # Calculate strategy performance
            strategy_perf = {}
            for strategy, perf_data in self.strategy_performance.items():
                if perf_data:
                    recent_perf = [
                        p
                        for p in perf_data
                        if p["timestamp"] > now - timedelta(hours=24)
                    ]
                    if recent_perf:
                        strategy_perf[strategy] = sum(p["pnl"] for p in recent_perf)

            # Calculate market performance
            market_perf = {}
            for market, perf_data in self.market_performance.items():
                if perf_data:
                    recent_perf = [
                        p
                        for p in perf_data
                        if p["timestamp"] > now - timedelta(hours=24)
                    ]
                    if recent_perf:
                        market_perf[market] = sum(p["pnl"] for p in recent_perf)

            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=now,
                total_pnl=total_pnl,
                realized_pnl=total_pnl,
                unrealized_pnl=unrealized_pnl,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                strategy_performance=strategy_perf,
                market_performance=market_perf,
                market_regime=self._detect_market_regime(),
                volatility_regime=self._detect_volatility_regime(),
            )

            # Store snapshot
            self.snapshots.append(snapshot)
            self.db.store_snapshot(snapshot)
            self.last_snapshot_time = now

        except Exception as e:
            logger.error(f"Failed to create performance snapshot: {e}")

    def _detect_market_regime(self) -> str:
        """Detect current market regime"""
        try:
            # Simple regime detection based on recent volatility
            recent_executions = self.db.get_executions(
                start_time=datetime.now() - timedelta(hours=6)
            )

            if len(recent_executions) < 10:
                return "insufficient_data"

            volatilities = [
                e.market_volatility
                for e in recent_executions
                if e.market_volatility > 0
            ]

            if not volatilities:
                return "unknown"

            avg_volatility = np.mean(volatilities)

            if avg_volatility < 0.1:
                return "quiet"
            elif avg_volatility < 0.3:
                return "normal"
            elif avg_volatility < 0.5:
                return "volatile"
            else:
                return "extreme"

        except Exception:
            return "unknown"

    def _detect_volatility_regime(self) -> str:
        """Detect current volatility regime"""
        return self._detect_market_regime()  # Same logic for now

    async def get_attribution_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        dimensions: List[AttributionDimension] = None,
    ) -> AttributionReport:
        """Generate comprehensive attribution report"""
        try:
            # Default to last 30 days if no time range specified
            if not end_time:
                end_time = datetime.now()
            if not start_time:
                start_time = end_time - timedelta(days=self.attribution_window_days)

            # Check cache first
            cache_key = f"attribution_{start_time.isoformat()}_{end_time.isoformat()}"
            if cache_key in self.attribution_cache:
                cached_report = self.attribution_cache[cache_key]
                if datetime.now() - cached_report.end_time < self.cache_ttl:
                    return cached_report

            # Generate new report
            report = await self._generate_attribution_report(
                start_time, end_time, dimensions
            )

            # Cache report
            self.attribution_cache[cache_key] = report

            return report

        except Exception as e:
            logger.error(f"Failed to generate attribution report: {e}")
            # Return empty report on error
            return AttributionReport(
                report_id=str(uuid.uuid4()),
                start_time=start_time or datetime.now(),
                end_time=end_time or datetime.now(),
                period_days=0,
            )

    async def _generate_attribution_report(
        self,
        start_time: datetime,
        end_time: datetime,
        dimensions: List[AttributionDimension],
    ) -> AttributionReport:
        """Generate attribution report"""
        try:
            # Get executions for the period
            executions = self.db.get_executions(start_time, end_time)

            if not executions:
                return AttributionReport(
                    report_id=str(uuid.uuid4()),
                    start_time=start_time,
                    end_time=end_time,
                    period_days=(end_time - start_time).days,
                )

            # Calculate basic performance metrics
            total_pnl = sum(e.realized_pnl for e in executions)
            total_trades = len(executions)
            winning_trades = len([e for e in executions if e.realized_pnl > 0])
            losing_trades = len([e for e in executions if e.realized_pnl < 0])

            # Calculate attribution breakdowns
            strategy_attribution = self._calculate_strategy_attribution(executions)
            market_attribution = self._calculate_market_attribution(executions)
            timeframe_attribution = self._calculate_timeframe_attribution(executions)
            condition_attribution = self._calculate_condition_attribution(executions)

            # Calculate execution quality metrics
            avg_slippage = (
                np.mean([e.slippage for e in executions]) if executions else 0
            )
            avg_execution_time = (
                np.mean([e.order_duration_ms for e in executions]) if executions else 0
            )
            execution_quality_score = self._calculate_execution_quality_score(
                executions
            )

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                executions, start_time, end_time
            )

            # Generate insights and recommendations
            insights = self._generate_insights(
                executions, strategy_attribution, market_attribution
            )
            recommendations = self._generate_recommendations(executions, risk_metrics)

            # Create report
            report = AttributionReport(
                report_id=str(uuid.uuid4()),
                start_time=start_time,
                end_time=end_time,
                period_days=(end_time - start_time).days,
                total_return=total_pnl / 10000,  # Assuming $10k starting capital
                strategy_attribution=strategy_attribution,
                market_attribution=market_attribution,
                timeframe_attribution=timeframe_attribution,
                condition_attribution=condition_attribution,
                avg_slippage=avg_slippage,
                avg_execution_time_ms=avg_execution_time,
                execution_quality_score=execution_quality_score,
                risk_adjusted_performance=risk_metrics,
                trade_count=total_trades,
                win_rate=winning_trades / max(1, total_trades),
                profit_factor=self._calculate_profit_factor(executions),
                insights=insights,
                recommendations=recommendations,
            )

            return report

        except Exception as e:
            logger.error(f"Failed to generate attribution report: {e}")
            raise

    def _calculate_strategy_attribution(
        self, executions: List[TradeExecution]
    ) -> Dict[str, float]:
        """Calculate strategy-level attribution"""
        try:
            strategy_pnl = defaultdict(float)

            for execution in executions:
                strategy_pnl[execution.strategy] += execution.realized_pnl

            total_pnl = sum(strategy_pnl.values())

            if total_pnl == 0:
                return {k: 0.0 for k in strategy_pnl.keys()}

            # Return percentage attribution
            return {k: v / total_pnl for k, v in strategy_pnl.items()}

        except Exception:
            return {}

    def _calculate_market_attribution(
        self, executions: List[TradeExecution]
    ) -> Dict[str, float]:
        """Calculate market-level attribution"""
        try:
            market_pnl = defaultdict(float)

            for execution in executions:
                market_pnl[execution.market_id] += execution.realized_pnl

            total_pnl = sum(market_pnl.values())

            if total_pnl == 0:
                return {k: 0.0 for k in market_pnl.keys()}

            return {k: v / total_pnl for k, v in market_pnl.items()}

        except Exception:
            return {}

    def _calculate_timeframe_attribution(
        self, executions: List[TradeExecution]
    ) -> Dict[str, float]:
        """Calculate timeframe-level attribution"""
        try:
            timeframe_pnl = defaultdict(float)

            for execution in executions:
                # Categorize by hour of day
                hour = execution.execution_timestamp.hour
                timeframe_key = f"{hour:02d}:00-{hour + 1:02d}:00"
                timeframe_pnl[timeframe_key] += execution.realized_pnl

            total_pnl = sum(timeframe_pnl.values())

            if total_pnl == 0:
                return {k: 0.0 for k in timeframe_pnl.keys()}

            return {k: v / total_pnl for k, v in timeframe_pnl.items()}

        except Exception:
            return {}

    def _calculate_condition_attribution(
        self, executions: List[TradeExecution]
    ) -> Dict[str, float]:
        """Calculate market condition attribution"""
        try:
            condition_pnl = defaultdict(float)

            for execution in executions:
                # Categorize by volatility regime
                if execution.market_volatility < 0.1:
                    condition = "low_volatility"
                elif execution.market_volatility < 0.3:
                    condition = "normal_volatility"
                else:
                    condition = "high_volatility"

                condition_pnl[condition] += execution.realized_pnl

            total_pnl = sum(condition_pnl.values())

            if total_pnl == 0:
                return {k: 0.0 for k in condition_pnl.keys()}

            return {k: v / total_pnl for k, v in condition_pnl.items()}

        except Exception:
            return {}

    def _calculate_execution_quality_score(
        self, executions: List[TradeExecution]
    ) -> float:
        """Calculate overall execution quality score"""
        try:
            if not executions:
                return 0.0

            # Factors: slippage, execution time, market impact
            avg_slippage = np.mean([abs(e.slippage) for e in executions])
            avg_execution_time = np.mean([e.order_duration_ms for e in executions])
            avg_market_impact = np.mean([abs(e.market_impact) for e in executions])

            # Normalize and combine (lower is better for all metrics)
            slippage_score = max(0, 1 - avg_slippage * 100)  # Assume 1% slippage is bad
            time_score = max(0, 1 - avg_execution_time / 10000)  # 10 seconds is bad
            impact_score = max(0, 1 - avg_market_impact * 100)  # 1% impact is bad

            # Weighted average
            quality_score = slippage_score * 0.4 + time_score * 0.3 + impact_score * 0.3

            return quality_score

        except Exception:
            return 0.0

    def _calculate_risk_metrics(
        self, executions: List[TradeExecution], start_time: datetime, end_time: datetime
    ) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        try:
            if not executions:
                return {}

            # Calculate daily returns
            daily_returns = defaultdict(float)

            for execution in executions:
                date = execution.execution_timestamp.date()
                daily_returns[date] += execution.realized_pnl

            returns_series = list(daily_returns.values())

            if len(returns_series) < 2:
                return {}

            # Calculate metrics
            mean_return = np.mean(returns_series)
            std_return = np.std(returns_series)

            # Sharpe ratio (assuming 0 risk-free rate for simplicity)
            sharpe = mean_return / std_return if std_return > 0 else 0

            # Sortino ratio (downside deviation)
            downside_returns = [r for r in returns_series if r < 0]
            if downside_returns:
                downside_std = np.std(downside_returns)
                sortino = mean_return / downside_std if downside_std > 0 else 0
            else:
                sortino = sharpe

            # Maximum drawdown
            cumulative_returns = np.cumsum(returns_series)
            rolling_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - rolling_max
            max_drawdown = np.min(drawdowns)

            return {
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "max_drawdown": max_drawdown,
                "volatility": std_return,
                "mean_return": mean_return,
            }

        except Exception:
            return {}

    def _calculate_profit_factor(self, executions: List[TradeExecution]) -> float:
        """Calculate profit factor"""
        try:
            winning_trades = [e.realized_pnl for e in executions if e.realized_pnl > 0]
            losing_trades = [
                abs(e.realized_pnl) for e in executions if e.realized_pnl < 0
            ]

            if not losing_trades:
                return float("inf") if winning_trades else 0

            total_wins = sum(winning_trades)
            total_losses = sum(losing_trades)

            return total_wins / total_losses if total_losses > 0 else 0

        except Exception:
            return 0

    def _generate_insights(
        self,
        executions: List[TradeExecution],
        strategy_attribution: Dict[str, float],
        market_attribution: Dict[str, float],
    ) -> List[str]:
        """Generate performance insights"""
        insights = []

        try:
            # Best performing strategy
            if strategy_attribution:
                best_strategy = max(strategy_attribution.items(), key=lambda x: x[1])
                insights.append(
                    f"Best performing strategy: {best_strategy[0]} ({best_strategy[1]:.1%} of P&L)"
                )

            # Best performing market
            if market_attribution:
                best_market = max(market_attribution.items(), key=lambda x: x[1])
                insights.append(
                    f"Best performing market: {best_market[0]} ({best_market[1]:.1%} of P&L)"
                )

            # Execution quality
            avg_slippage = (
                np.mean([abs(e.slippage) for e in executions]) if executions else 0
            )
            if avg_slippage > 0.005:  # 0.5%
                insights.append(
                    "High slippage detected - consider improving execution algorithms"
                )

            # Trade frequency
            trades_per_day = len(executions) / max(
                1,
                (
                    executions[-1].execution_timestamp
                    - executions[0].execution_timestamp
                ).days,
            )
            if trades_per_day > 50:
                insights.append("High trading frequency - monitor for overtrading")
            elif trades_per_day < 5:
                insights.append(
                    "Low trading frequency - consider relaxing entry criteria"
                )

        except Exception:
            pass

        return insights

    def _generate_recommendations(
        self, executions: List[TradeExecution], risk_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        try:
            # Risk-based recommendations
            if risk_metrics.get("max_drawdown", 0) < -0.1:  # 10% drawdown
                recommendations.append(
                    "Consider reducing position sizes to limit drawdown"
                )

            if risk_metrics.get("sharpe_ratio", 0) < 0.5:
                recommendations.append(
                    "Low risk-adjusted returns - review strategy parameters"
                )

            # Execution-based recommendations
            avg_execution_time = (
                np.mean([e.order_duration_ms for e in executions]) if executions else 0
            )
            if avg_execution_time > 5000:  # 5 seconds
                recommendations.append(
                    "Slow execution - optimize order routing and reduce latency"
                )

            # Strategy-based recommendations
            strategy_performance = defaultdict(list)
            for execution in executions:
                strategy_performance[execution.strategy].append(execution.realized_pnl)

            underperforming_strategies = []
            for strategy, pnls in strategy_performance.items():
                if np.mean(pnls) < 0:
                    underperforming_strategies.append(strategy)

            if underperforming_strategies:
                recommendations.append(
                    f"Review underperforming strategies: {', '.join(underperforming_strategies)}"
                )

        except Exception:
            pass

        return recommendations

    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics"""
        try:
            now = datetime.now()

            # Get recent performance
            recent_executions = self.db.get_executions(
                start_time=now - timedelta(hours=1)
            )

            # Calculate metrics
            hourly_pnl = sum(e.realized_pnl for e in recent_executions)
            hourly_trades = len(recent_executions)
            hourly_win_rate = len(
                [e for e in recent_executions if e.realized_pnl > 0]
            ) / max(1, hourly_trades)

            # Get current positions
            total_exposure = sum(
                abs(pos["quantity"] * pos["avg_price"])
                for pos in self.current_positions.values()
            )

            # Get strategy performance
            strategy_metrics = {}
            for strategy, perf_data in self.strategy_performance.items():
                if perf_data:
                    recent_perf = [
                        p
                        for p in perf_data
                        if p["timestamp"] > now - timedelta(hours=1)
                    ]
                    if recent_perf:
                        strategy_metrics[strategy] = {
                            "pnl": sum(p["pnl"] for p in recent_perf),
                            "trades": len(recent_perf),
                            "avg_slippage": np.mean(
                                [p["slippage"] for p in recent_perf]
                            ),
                        }

            return {
                "timestamp": now.isoformat(),
                "hourly_pnl": hourly_pnl,
                "hourly_trades": hourly_trades,
                "hourly_win_rate": hourly_win_rate,
                "total_exposure": total_exposure,
                "open_positions": len(
                    [
                        pos
                        for pos in self.current_positions.values()
                        if pos["quantity"] != 0
                    ]
                ),
                "strategy_metrics": strategy_metrics,
                "market_regime": self._detect_market_regime(),
                "last_snapshot": self.last_snapshot_time.isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {e}")
            return {}


# Utility functions
def create_attribution_engine(config: Config) -> RealTimeAttributionEngine:
    """Create and return attribution engine instance"""
    return RealTimeAttributionEngine(config)


async def record_arbitrage_execution(
    attribution_engine: RealTimeAttributionEngine,
    opportunity: ArbitrageOpportunity,
    execution_details: Dict[str, Any],
) -> None:
    """Record arbitrage execution for attribution"""
    try:
        execution = TradeExecution(
            execution_id=str(uuid.uuid4()),
            opportunity_id=opportunity.id,
            strategy=opportunity.type.value,
            market_id=opportunity.market_id_1,
            action=execution_details.get("action", "unknown"),
            quantity=execution_details.get("quantity", 0),
            intended_price=execution_details.get("intended_price", 0),
            execution_price=execution_details.get("execution_price", 0),
            execution_timestamp=datetime.now(),
            commission=execution_details.get("commission", 0),
            slippage=execution_details.get("slippage", 0),
            market_impact=execution_details.get("market_impact", 0),
            decision_to_execution_ms=execution_details.get(
                "decision_to_execution_ms", 0
            ),
            order_duration_ms=execution_details.get("order_duration_ms", 0),
            market_volatility=execution_details.get("market_volatility", 0),
            liquidity_score=execution_details.get("liquidity_score", 0),
            spread_at_execution=execution_details.get("spread_at_execution", 0),
            tags=execution_details.get("tags", []),
            realized_pnl=execution_details.get("realized_pnl", 0),
        )

        await attribution_engine.record_execution(execution)

    except Exception as e:
        logger.error(f"Failed to record arbitrage execution: {e}")
