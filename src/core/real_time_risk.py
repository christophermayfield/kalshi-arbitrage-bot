"""
Real-Time Risk Management System
Enhanced with dynamic position limits and comprehensive risk controls
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import redis.asyncio as redis

from ..utils.performance_cache import PerformanceCache

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk severity levels"""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RiskType(Enum):
    """Types of risk to monitor"""

    POSITION_SIZE = "position_size"
    PORTFOLIO_EXPOSURE = "portfolio_exposure"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"
    DRAWN_DOWN = "draw_down"
    MARKET_STRESS = "market_stress"


@dataclass
class RiskAlert:
    """Risk alert data structure"""

    risk_type: RiskType
    level: RiskLevel
    message: str
    current_value: float
    threshold: float
    timestamp: datetime
    symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionLimit:
    """Dynamic position limit configuration"""

    symbol: str
    base_limit: float  # Base position size
    volatility_adjustment: float = 1.0
    correlation_adjustment: float = 1.0
    portfolio_adjustment: float = 1.0
    effective_limit: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def calculate_effective_limit(
        self,
        volatility_factor: float,
        correlation_factor: float,
        portfolio_factor: float,
    ) -> float:
        """Calculate adjusted position limit"""
        self.volatility_adjustment = volatility_factor
        self.correlation_adjustment = correlation_factor
        self.portfolio_adjustment = portfolio_factor
        self.effective_limit = (
            self.base_limit * volatility_factor * correlation_factor * portfolio_factor
        )
        self.last_updated = datetime.now()
        return self.effective_limit


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""

    total_value: float = 0.0
    total_exposure: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    sharpe_ratio: float = 0.0
    beta: float = 0.0
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    concentration: Dict[str, float] = field(default_factory=dict)


@dataclass
class MarketCondition:
    """Current market condition metrics"""

    volatility_index: float = 0.0
    market_stress_score: float = 0.0
    liquidity_score: float = 0.0
    correlation_trend: float = 0.0
    volume_anomaly: bool = False
    price_anomaly: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class RealTimeRiskManager:
    """
    Real-time risk management system with dynamic position limits
    and comprehensive monitoring capabilities
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_limits = config.get("risk_limits", {})
        self.position_limits: Dict[str, PositionLimit] = {}
        self.active_alerts: List[RiskAlert] = []
        self.risk_history: deque = deque(maxlen=10000)
        self.market_conditions: MarketCondition = MarketCondition()
        self.portfolio_risk = PortfolioRisk()

        # Performance cache for risk metrics
        self.cache = PerformanceCache(
            redis_url=config.get("redis_url", "redis://localhost:6379"),
            default_ttl=30,  # 30 second TTL for risk data
        )

        # Risk thresholds
        self.thresholds = {
            "max_position_size": 0.05,  # 5% of portfolio
            "max_portfolio_exposure": 1.0,  # 100% of portfolio
            "max_correlation": 0.7,
            "max_volatility": 0.15,
            "min_liquidity": 0.3,
            "max_concentration": 0.3,
            "max_drawdown": 0.1,
            "stress_threshold": 0.7,
        }

        # Update thresholds from config
        self.thresholds.update(self.risk_limits.get("thresholds", {}))

        # Position tracking
        self.active_positions: Dict[str, float] = defaultdict(float)
        self.position_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.pnl_history: deque = deque(maxlen=1000)

        # Risk calculation intervals
        self.last_risk_update = datetime.now()
        self.market_update_interval = timedelta(seconds=30)
        self.position_update_interval = timedelta(seconds=5)

        logger.info("Real-Time Risk Manager initialized with dynamic limits")

    async def initialize(self) -> None:
        """Initialize risk management system"""
        try:
            # Initialize position limits from config
            base_limits = self.risk_limits.get("base_position_limits", {})
            for symbol, limit in base_limits.items():
                self.position_limits[symbol] = PositionLimit(
                    symbol=symbol, base_limit=limit
                )

            # Load existing risk data from cache
            await self._load_risk_state()

            logger.info(
                f"Risk Manager initialized with {len(self.position_limits)} position limits"
            )

        except Exception as e:
            logger.error(f"Risk Manager initialization failed: {e}")
            raise

    async def check_position_risk(
        self,
        symbol: str,
        new_position_size: float,
        price: float,
        portfolio_value: float,
    ) -> Tuple[bool, List[RiskAlert]]:
        """
        Check if a new position complies with risk limits

        Returns:
            Tuple of (is_allowed, risk_alerts)
        """
        alerts = []

        try:
            # Get effective position limit
            position_limit = await self._get_position_limit(symbol)

            # Check position size limit
            current_position = self.active_positions[symbol]
            total_position = abs(current_position + new_position_size)
            position_value = total_position * price

            if position_value > position_limit.effective_limit:
                alerts.append(
                    RiskAlert(
                        risk_type=RiskType.POSITION_SIZE,
                        level=RiskLevel.HIGH,
                        message=f"Position size {position_value:.2f} exceeds limit {position_limit.effective_limit:.2f}",
                        current_value=position_value,
                        threshold=position_limit.effective_limit,
                        timestamp=datetime.now(),
                        symbol=symbol,
                    )
                )

            # Check portfolio exposure
            portfolio_exposure = await self._calculate_portfolio_exposure()
            new_exposure = portfolio_exposure + (new_position_size * price)

            if (
                new_exposure
                > self.thresholds["max_portfolio_exposure"] * portfolio_value
            ):
                alerts.append(
                    RiskAlert(
                        risk_type=RiskType.PORTFOLIO_EXPOSURE,
                        level=RiskLevel.HIGH,
                        message=f"Portfolio exposure {new_exposure:.2f} exceeds limit {self.thresholds['max_portfolio_exposure'] * portfolio_value:.2f}",
                        current_value=new_exposure,
                        threshold=self.thresholds["max_portfolio_exposure"]
                        * portfolio_value,
                        timestamp=datetime.now(),
                    )
                )

            # Check concentration risk
            concentration = await self._calculate_concentration()
            if symbol in concentration:
                symbol_concentration = concentration[symbol]
                if symbol_concentration > self.thresholds["max_concentration"]:
                    alerts.append(
                        RiskAlert(
                            risk_type=RiskType.CONCENTRATION,
                            level=RiskLevel.MODERATE,
                            message=f"Concentration {symbol_concentration:.2%} exceeds limit {self.thresholds['max_concentration']:.2%}",
                            current_value=symbol_concentration,
                            threshold=self.thresholds["max_concentration"],
                            timestamp=datetime.now(),
                            symbol=symbol,
                        )
                    )

            # Check correlation risk
            if new_position_size != 0:
                correlation_alerts = await self._check_correlation_risk(
                    symbol, new_position_size
                )
                alerts.extend(correlation_alerts)

            # Store alerts
            for alert in alerts:
                await self._store_alert(alert)

            return len(alerts) == 0, alerts

        except Exception as e:
            logger.error(f"Risk check failed for {symbol}: {e}")
            return False, [
                RiskAlert(
                    risk_type=RiskType.POSITION_SIZE,
                    level=RiskLevel.CRITICAL,
                    message=f"Risk check error: {str(e)}",
                    current_value=0,
                    threshold=0,
                    timestamp=datetime.now(),
                    symbol=symbol,
                )
            ]

    async def update_position(
        self, symbol: str, size_change: float, price: float
    ) -> None:
        """Update position tracking"""
        try:
            old_size = self.active_positions[symbol]
            self.active_positions[symbol] += size_change

            # Track position history
            timestamp = datetime.now()
            self.position_history[symbol].append(
                {
                    "timestamp": timestamp,
                    "size": self.active_positions[symbol],
                    "price": price,
                }
            )

            # Update portfolio value and P&L
            pnl_change = size_change * price
            self.pnl_history.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "pnl_change": pnl_change,
                    "cumulative_pnl": sum(p["pnl_change"] for p in self.pnl_history),
                }
            )

            # Update portfolio risk metrics
            await self._update_portfolio_risk()

            # Cache updated position
            await self.cache.set(
                f"position:{symbol}",
                {
                    "size": self.active_positions[symbol],
                    "price": price,
                    "timestamp": timestamp.isoformat(),
                },
                ttl=60,
            )

            logger.debug(
                f"Updated position {symbol}: {old_size} -> {self.active_positions[symbol]}"
            )

        except Exception as e:
            logger.error(f"Position update failed for {symbol}: {e}")

    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics"""
        try:
            await self._update_portfolio_risk()

            return {
                "portfolio_risk": {
                    "total_value": self.portfolio_risk.total_value,
                    "total_exposure": self.portfolio_risk.total_exposure,
                    "current_drawdown": self.portfolio_risk.current_drawdown,
                    "max_drawdown": self.portfolio_risk.max_drawdown,
                    "var_95": self.portfolio_risk.var_95,
                    "sharpe_ratio": self.portfolio_risk.sharpe_ratio,
                    "beta": self.portfolio_risk.beta,
                },
                "market_conditions": {
                    "volatility_index": self.market_conditions.volatility_index,
                    "market_stress_score": self.market_conditions.market_stress_score,
                    "liquidity_score": self.market_conditions.liquidity_score,
                    "correlation_trend": self.market_conditions.correlation_trend,
                    "volume_anomaly": self.market_conditions.volume_anomaly,
                    "price_anomaly": self.market_conditions.price_anomaly,
                },
                "position_limits": {
                    symbol: {
                        "base_limit": pos.base_limit,
                        "effective_limit": pos.effective_limit,
                        "adjustments": {
                            "volatility": pos.volatility_adjustment,
                            "correlation": pos.correlation_adjustment,
                            "portfolio": pos.portfolio_adjustment,
                        },
                    }
                    for symbol, pos in self.position_limits.items()
                },
                "active_alerts": [
                    {
                        "type": alert.risk_type.value,
                        "level": alert.level.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "symbol": alert.symbol,
                    }
                    for alert in self.active_alerts[-10:]  # Last 10 alerts
                ],
                "risk_scores": await self._calculate_risk_scores(),
            }

        except Exception as e:
            logger.error(f"Risk metrics retrieval failed: {e}")
            return {}

    async def _get_position_limit(self, symbol: str) -> PositionLimit:
        """Get or create position limit for symbol"""
        if symbol not in self.position_limits:
            # Create default limit based on portfolio value and risk parameters
            base_limit = self.risk_limits.get("default_position_limit", 1000.0)
            self.position_limits[symbol] = PositionLimit(
                symbol=symbol, base_limit=base_limit
            )

        position_limit = self.position_limits[symbol]

        # Update effective limit with current market conditions
        volatility_factor = await self._calculate_volatility_adjustment(symbol)
        correlation_factor = await self._calculate_correlation_adjustment(symbol)
        portfolio_factor = await self._calculate_portfolio_adjustment()

        position_limit.calculate_effective_limit(
            volatility_factor, correlation_factor, portfolio_factor
        )

        return position_limit

    async def _calculate_volatility_adjustment(self, symbol: str) -> float:
        """Calculate volatility-based adjustment factor"""
        try:
            # Get recent price data for volatility calculation
            cache_key = f"volatility:{symbol}"
            cached_vol = await self.cache.get(cache_key)

            if cached_vol:
                volatility = float(cached_vol)
            else:
                # Calculate from position history if available
                history = list(self.position_history[symbol])
                if len(history) >= 20:
                    prices = [p["price"] for p in history]
                    returns = np.diff(np.log(prices))
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized

                    await self.cache.set(cache_key, str(volatility), ttl=300)
                else:
                    volatility = self.thresholds[
                        "max_volatility"
                    ]  # Default high volatility

            # Convert to adjustment factor (lower volatility = higher limit)
            if volatility <= self.thresholds["max_volatility"]:
                return 1.0  # Normal limit
            elif volatility <= self.thresholds["max_volatility"] * 2:
                return 0.7  # 30% reduction
            else:
                return 0.5  # 50% reduction

        except Exception as e:
            logger.error(f"Volatility adjustment calculation failed for {symbol}: {e}")
            return 0.5  # Conservative default

    async def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Calculate correlation-based adjustment factor"""
        try:
            # Calculate average correlation with existing positions
            total_correlation = 0.0
            count = 0

            for other_symbol in self.active_positions:
                if other_symbol != symbol and self.active_positions[other_symbol] != 0:
                    correlation = await self._get_correlation(symbol, other_symbol)
                    total_correlation += abs(correlation)
                    count += 1

            if count == 0:
                return 1.0  # No correlation risk

            avg_correlation = total_correlation / count

            # Convert to adjustment factor
            if avg_correlation <= self.thresholds["max_correlation"]:
                return 1.0  # Normal limit
            elif avg_correlation <= 0.85:
                return 0.8  # 20% reduction
            else:
                return 0.6  # 40% reduction

        except Exception as e:
            logger.error(f"Correlation adjustment calculation failed for {symbol}: {e}")
            return 0.7  # Conservative default

    async def _calculate_portfolio_adjustment(self) -> float:
        """Calculate portfolio-level adjustment factor"""
        try:
            # Base on current drawdown and market stress
            drawdown_factor = 1.0
            if (
                self.portfolio_risk.current_drawdown
                > self.thresholds["max_drawdown"] * 0.5
            ):
                drawdown_factor = 0.8

            stress_factor = 1.0
            if (
                self.market_conditions.market_stress_score
                > self.thresholds["stress_threshold"] * 0.7
            ):
                stress_factor = 0.7

            return min(drawdown_factor, stress_factor)

        except Exception as e:
            logger.error(f"Portfolio adjustment calculation failed: {e}")
            return 0.8  # Conservative default

    async def _calculate_portfolio_exposure(self) -> float:
        """Calculate total portfolio exposure"""
        try:
            total_exposure = 0.0
            for symbol, size in self.active_positions.items():
                if size != 0:
                    # Get current price from cache
                    cache_key = f"price:{symbol}"
                    price_data = await self.cache.get(cache_key)
                    if price_data:
                        price = float(price_data)
                        total_exposure += abs(size * price)

            return total_exposure

        except Exception as e:
            logger.error(f"Portfolio exposure calculation failed: {e}")
            return 0.0

    async def _calculate_concentration(self) -> Dict[str, float]:
        """Calculate concentration by symbol"""
        try:
            total_value = await self._calculate_portfolio_exposure()
            if total_value == 0:
                return {}

            concentration = {}
            for symbol, size in self.active_positions.items():
                if size != 0:
                    cache_key = f"price:{symbol}"
                    price_data = await self.cache.get(cache_key)
                    if price_data:
                        price = float(price_data)
                        value = abs(size * price)
                        concentration[symbol] = value / total_value

            return concentration

        except Exception as e:
            logger.error(f"Concentration calculation failed: {e}")
            return {}

    async def _check_correlation_risk(
        self, symbol: str, new_position: float
    ) -> List[RiskAlert]:
        """Check correlation risk with existing positions"""
        alerts = []

        try:
            for other_symbol, existing_position in self.active_positions.items():
                if other_symbol != symbol and existing_position != 0:
                    correlation = await self._get_correlation(symbol, other_symbol)

                    if abs(correlation) > self.thresholds["max_correlation"]:
                        # Check if new position would increase exposure
                        if (
                            np.sign(new_position) == np.sign(existing_position)
                            and correlation > 0
                        ):
                            alerts.append(
                                RiskAlert(
                                    risk_type=RiskType.CORRELATION,
                                    level=RiskLevel.MODERATE,
                                    message=f"High correlation {correlation:.3f} with {other_symbol}",
                                    current_value=correlation,
                                    threshold=self.thresholds["max_correlation"],
                                    timestamp=datetime.now(),
                                    symbol=symbol,
                                    metadata={"correlated_symbol": other_symbol},
                                )
                            )

            return alerts

        except Exception as e:
            logger.error(f"Correlation risk check failed for {symbol}: {e}")
            return []

    async def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        try:
            cache_key = f"correlation:{symbol1}:{symbol2}"
            cached_corr = await self.cache.get(cache_key)

            if cached_corr:
                return float(cached_corr)

            # Calculate from price histories
            history1 = list(self.position_history[symbol1])
            history2 = list(self.position_history[symbol2])

            if len(history1) >= 20 and len(history2) >= 20:
                prices1 = [p["price"] for p in history1[-20:]]
                prices2 = [p["price"] for p in history2[-20:]]

                # Calculate correlation
                correlation = np.corrcoef(prices1, prices2)[0, 1]

                # Cache result
                await self.cache.set(cache_key, str(correlation), ttl=600)

                return correlation if not np.isnan(correlation) else 0.0

            return 0.0  # Default

        except Exception as e:
            logger.error(f"Correlation calculation failed for {symbol1}-{symbol2}: {e}")
            return 0.0

    async def _update_portfolio_risk(self) -> None:
        """Update portfolio risk metrics"""
        try:
            # Update portfolio value
            self.portfolio_risk.total_value = await self._calculate_portfolio_exposure()

            # Update current drawdown
            if len(self.pnl_history) >= 2:
                pnl_values = [p["cumulative_pnl"] for p in self.pnl_history]
                peak = max(pnl_values)
                current = pnl_values[-1]

                if peak > 0:
                    self.portfolio_risk.current_drawdown = (peak - current) / peak
                    self.portfolio_risk.max_drawdown = max(
                        self.portfolio_risk.max_drawdown,
                        self.portfolio_risk.current_drawdown,
                    )

            # Calculate VaR (simplified)
            if len(self.pnl_history) >= 50:
                pnl_changes = [p["pnl_change"] for p in list(self.pnl_history)[-50:]]
                self.portfolio_risk.var_95 = np.percentile(pnl_changes, 5)

            # Update market conditions
            await self._update_market_conditions()

        except Exception as e:
            logger.error(f"Portfolio risk update failed: {e}")

    async def _update_market_conditions(self) -> None:
        """Update market condition metrics"""
        try:
            # Simplified market stress calculation
            # In production, this would use external market data

            # Calculate portfolio volatility as stress indicator
            if len(self.pnl_history) >= 20:
                pnl_changes = [p["pnl_change"] for p in list(self.pnl_history)[-20:]]
                volatility = np.std(pnl_changes)
                self.market_conditions.volatility_index = volatility

                # Market stress score (0-1)
                if volatility > 0:
                    self.market_conditions.market_stress_score = min(
                        1.0, volatility / 0.1
                    )

            self.market_conditions.timestamp = datetime.now()

        except Exception as e:
            logger.error(f"Market conditions update failed: {e}")

    async def _calculate_risk_scores(self) -> Dict[str, float]:
        """Calculate overall risk scores"""
        try:
            scores = {}

            # Position risk score
            position_violations = len(
                [a for a in self.active_alerts if a.risk_type == RiskType.POSITION_SIZE]
            )
            scores["position_risk"] = min(1.0, position_violations / 3.0)

            # Portfolio risk score
            if self.portfolio_risk.max_drawdown > 0:
                scores["portfolio_risk"] = min(
                    1.0,
                    self.portfolio_risk.current_drawdown
                    / self.thresholds["max_drawdown"],
                )
            else:
                scores["portfolio_risk"] = 0.0

            # Market risk score
            scores["market_risk"] = self.market_conditions.market_stress_score

            # Overall risk score
            scores["overall_risk"] = np.mean(list(scores.values()))

            return scores

        except Exception as e:
            logger.error(f"Risk score calculation failed: {e}")
            return {"overall_risk": 0.5}

    async def _store_alert(self, alert: RiskAlert) -> None:
        """Store risk alert"""
        try:
            self.active_alerts.append(alert)
            self.risk_history.append(alert)

            # Store in cache for monitoring
            await self.cache.set(
                f"alert:{alert.timestamp.isoformat()}",
                {
                    "type": alert.risk_type.value,
                    "level": alert.level.value,
                    "message": alert.message,
                    "symbol": alert.symbol,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                },
                ttl=3600,  # 1 hour
            )

            # Log alert
            log_level = {
                RiskLevel.LOW: logging.INFO,
                RiskLevel.MODERATE: logging.WARNING,
                RiskLevel.HIGH: logging.ERROR,
                RiskLevel.CRITICAL: logging.CRITICAL,
                RiskLevel.EMERGENCY: logging.CRITICAL,
            }.get(alert.level, logging.WARNING)

            logger.log(
                log_level, f"Risk Alert [{alert.level.value.upper()}] {alert.message}"
            )

        except Exception as e:
            logger.error(f"Alert storage failed: {e}")

    async def _load_risk_state(self) -> None:
        """Load existing risk state from cache"""
        try:
            # Load active positions
            for symbol in self.active_positions:
                cache_key = f"position:{symbol}"
                position_data = await self.cache.get(cache_key)
                if position_data:
                    data = position_data
                    self.active_positions[symbol] = float(data["size"])

            # Load portfolio risk metrics
            portfolio_data = await self.cache.get("portfolio_risk")
            if portfolio_data:
                data = portfolio_data
                self.portfolio_risk.total_value = float(data.get("total_value", 0))
                self.portfolio_risk.max_drawdown = float(data.get("max_drawdown", 0))

            logger.info("Risk state loaded from cache")

        except Exception as e:
            logger.error(f"Risk state loading failed: {e}")

    async def save_risk_state(self) -> None:
        """Save current risk state to cache"""
        try:
            # Save portfolio risk metrics
            await self.cache.set(
                "portfolio_risk",
                {
                    "total_value": self.portfolio_risk.total_value,
                    "max_drawdown": self.portfolio_risk.max_drawdown,
                    "last_updated": datetime.now().isoformat(),
                },
                ttl=86400,  # 24 hours
            )

            logger.info("Risk state saved to cache")

        except Exception as e:
            logger.error(f"Risk state saving failed: {e}")

    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            await self.save_risk_state()
            await self.cache.close()
            logger.info("Risk Manager cleaned up")
        except Exception as e:
            logger.error(f"Risk Manager cleanup failed: {e}")


# Utility functions for risk management
async def create_risk_manager(config: Dict[str, Any]) -> RealTimeRiskManager:
    """Create and initialize risk manager"""
    manager = RealTimeRiskManager(config)
    await manager.initialize()
    return manager


def calculate_position_size_risk(
    position_size: float, portfolio_value: float, max_risk_per_trade: float = 0.02
) -> float:
    """Calculate risk score for position size"""
    position_value = abs(position_size)
    risk_ratio = position_value / portfolio_value
    return min(1.0, risk_ratio / max_risk_per_trade)


def calculate_portfolio_heatmap(risk_scores: Dict[str, float]) -> str:
    """Generate risk heatmap visualization"""
    heatmap = []
    for metric, score in risk_scores.items():
        if score < 0.3:
            color = "🟢"  # Green - Low risk
        elif score < 0.6:
            color = "🟡"  # Yellow - Medium risk
        elif score < 0.8:
            color = "🟠"  # Orange - High risk
        else:
            color = "🔴"  # Red - Critical risk

        heatmap.append(f"{color} {metric}: {score:.2f}")

    return " | ".join(heatmap)
