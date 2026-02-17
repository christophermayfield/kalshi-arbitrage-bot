"""
Multi-Market Correlation Trading System
Advanced cross-market and cross-exchange correlation analysis with automated trading
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
import networkx as nx
import redis.asyncio as redis

from ..utils.performance_cache import PerformanceCache

logger = logging.getLogger(__name__)


class CorrelationType(Enum):
    """Types of correlations to track"""

    SPOT = "spot"
    FUTURES = "futures"
    PERPETUAL = "perpetual"
    OPTIONS = "options"
    CROSS_EXCHANGE = "cross_exchange"
    INTRAMARKET = "intramarket"
    INTERMARKET = "intermarket"


class CorrelationRegime(Enum):
    """Correlation regime states"""

    STRONG_POSITIVE = "strong_positive"  # >0.7
    MODERATE_POSITIVE = "moderate_positive"  # 0.3 to 0.7
    WEAK_POSITIVE = "weak_positive"  # 0.1 to 0.3
    NEUTRAL = "neutral"  # -0.1 to 0.1
    WEAK_NEGATIVE = "weak_negative"  # -0.3 to -0.1
    MODERATE_NEGATIVE = "moderate_negative"  # -0.7 to -0.3
    STRONG_NEGATIVE = "strong_negative"  # < -0.7
    DIVERGING = "diverging"  # Rapid correlation change


@dataclass
class CorrelationPair:
    """Correlation between two markets/assets"""

    market1: str
    asset1: str
    exchange1: str
    market2: str
    asset2: str
    exchange2: str
    correlation_type: CorrelationType

    correlation_value: float
    p_value: float
    rolling_correlation: float
    correlation_trend: str  # "increasing", "decreasing", "stable"

    spread: float
    spread_mean: float
    spread_std: float
    z_score: float

    historical_correlation: List[Tuple[datetime, float]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

    # Trading parameters
    minimum_correlation: float = 0.3
    hedge_ratio: float = 1.0

    # Performance tracking
    total_trades: int = 0
    profitable_trades: int = 0
    avg_profit: float = 0.0


@dataclass
class CorrelationSignal:
    """Trading signal based on correlation analysis"""

    signal_id: str
    correlation_pair: CorrelationPair

    signal_type: str  # "convergence", "divergence", "breakout"
    action: str  # "long", "short", "neutral"

    confidence: float
    expected_return: float
    risk_score: float

    position_sizes: List[float]
    entry_prices: List[float]
    targets: List[float]
    stops: List[float]

    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiMarketCorrelationTrader:
    """
    Advanced multi-market correlation trading system
    with automated strategy execution and risk management
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.correlation_config = config.get("correlation_trading", {})

        # Correlation tracking parameters
        self.correlation_lookback = self.correlation_config.get(
            "correlation_lookback", 30
        )  # periods
        self.min_correlation = self.correlation_config.get("min_correlation", 0.5)
        self.max_correlation_deviation = self.correlation_config.get(
            "max_correlation_deviation", 0.2
        )

        # Trading parameters
        self.min_spread_threshold = self.correlation_config.get(
            "min_spread_threshold", 0.01
        )  # 1%
        self.max_position_per_pair = self.correlation_config.get(
            "max_position_per_pair", 5000
        )
        self.risk_limit_per_trade = self.correlation_config.get(
            "risk_limit_per_trade", 0.02
        )  # 2%

        # Market universe
        self.markets = self.correlation_config.get(
            "markets", ["spot", "futures", "perpetual"]
        )
        self.exchanges = self.correlation_config.get(
            "exchanges", ["binance", "coinbase", "kraken", "bybit"]
        )
        self.assets = self.correlation_config.get("assets", ["BTC", "ETH", "BNB"])

        # Correlation tracking
        self.correlation_pairs: Dict[str, CorrelationPair] = {}
        self.correlation_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(
            list
        )

        # Trading signals
        self.active_signals: Dict[str, CorrelationSignal] = {}
        self.signal_history: List[CorrelationSignal] = []

        # Market data
        self.market_data: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(
            dict
        )  # exchange -> market -> data

        # Performance metrics
        self.performance_metrics: Dict[str, Any] = defaultdict(float)

        # Correlation network
        self.correlation_network: nx.Graph = nx.Graph()
        self.correlation_clusters: Dict[int, List[str]] = {}

        # Performance cache
        self.cache = PerformanceCache(
            redis_url=config.get("redis_url", "redis://localhost:6379"), default_ttl=30
        )

        # Active positions
        self.active_positions: Dict[str, Dict] = {}

        logger.info("Multi-Market Correlation Trader initialized")

    async def initialize(self) -> None:
        """Initialize the correlation trading system"""
        try:
            # Initialize market data structures
            await self._initialize_market_data()

            # Generate correlation pairs
            await self._generate_correlation_pairs()

            # Initialize correlation network
            await self._initialize_correlation_network()

            # Start monitoring loops
            asyncio.create_task(self._correlation_monitoring_loop())
            asyncio.create_task(self._signal_generation_loop())
            asyncio.create_task(self._position_management_loop())

            logger.info("Multi-Market Correlation Trader initialized successfully")

        except Exception as e:
            logger.error(f"Correlation Trader initialization failed: {e}")
            raise

    async def _initialize_market_data(self) -> None:
        """Initialize market data structures"""
        try:
            # Create data structures for each exchange and market
            for exchange in self.exchanges:
                for market in self.markets:
                    market_key = f"{exchange}:{market}"
                    self.market_data[exchange][market] = pd.DataFrame()

            logger.info("Market data structures initialized")

        except Exception as e:
            logger.error(f"Market data initialization failed: {e}")

    async def _generate_correlation_pairs(self) -> None:
        """Generate all possible correlation pairs"""
        try:
            pair_count = 0

            # Generate market combinations
            for exchange1 in self.exchanges:
                for market1 in self.markets:
                    # Same market, different assets
                    for asset1 in self.assets:
                        for asset2 in self.assets:
                            if asset1 >= asset2:
                                continue

                            pair_key = f"{exchange1}:{market1}:{asset1}-{asset2}"

                            self.correlation_pairs[pair_key] = CorrelationPair(
                                market1=market1,
                                asset1=asset1,
                                exchange1=exchange1,
                                market2=market1,
                                asset2=asset2,
                                exchange2=exchange1,
                                correlation_type=CorrelationType.INTRAMARKET,
                                correlation_value=0.0,
                                p_value=1.0,
                                rolling_correlation=0.0,
                                correlation_trend="stable",
                                spread=0.0,
                                spread_mean=0.0,
                                spread_std=0.0,
                                z_score=0.0,
                            )

                            pair_count += 1

                    # Cross-market, same asset
                    for asset1 in self.assets:
                        for market2 in self.markets:
                            if market1 >= market2:
                                continue

                            pair_key = f"{exchange1}:{market1}:{market2}:{asset1}"

                            self.correlation_pairs[pair_key] = CorrelationPair(
                                market1=market1,
                                asset1=asset1,
                                exchange1=exchange1,
                                market2=market2,
                                asset2=asset1,
                                exchange2=exchange1,
                                correlation_type=CorrelationType.INTERMARKET,
                                correlation_value=0.0,
                                p_value=1.0,
                                rolling_correlation=0.0,
                                correlation_trend="stable",
                                spread=0.0,
                                spread_mean=0.0,
                                spread_std=0.0,
                                z_score=0.0,
                            )

                            pair_count += 1

            # Cross-exchange, same market and asset
            for asset1 in self.assets:
                for market1 in self.markets:
                    for exchange1 in self.exchanges:
                        for exchange2 in self.exchanges:
                            if exchange1 >= exchange2:
                                continue

                            pair_key = f"{exchange1}-{exchange2}:{market1}:{asset1}"

                            self.correlation_pairs[pair_key] = CorrelationPair(
                                market1=market1,
                                asset1=asset1,
                                exchange1=exchange1,
                                market2=market1,
                                asset2=asset1,
                                exchange2=exchange2,
                                correlation_type=CorrelationType.CROSS_EXCHANGE,
                                correlation_value=0.0,
                                p_value=1.0,
                                rolling_correlation=0.0,
                                correlation_trend="stable",
                                spread=0.0,
                                spread_mean=0.0,
                                spread_std=0.0,
                                z_score=0.0,
                            )

                            pair_count += 1

            logger.info(f"Generated {pair_count} correlation pairs")

        except Exception as e:
            logger.error(f"Correlation pair generation failed: {e}")

    async def _update_correlation_data(self) -> None:
        """Update all correlation data"""
        try:
            # Update market data
            await self._update_market_data()

            # Calculate correlations for all pairs
            for pair_key, pair in self.correlation_pairs.items():
                try:
                    await self._calculate_pair_correlation(pair)
                except Exception as e:
                    logger.debug(f"Correlation calculation failed for {pair_key}: {e}")

            # Update correlation networks
            await self._update_correlation_network()

            # Cache correlation data
            await self._cache_correlation_data()

        except Exception as e:
            logger.error(f"Correlation data update failed: {e}")

    async def _update_market_data(self) -> None:
        """Update market data from all exchanges"""
        try:
            for exchange in self.exchanges:
                for market in self.markets:
                    # Simulate market data updates
                    for asset in self.assets:
                        # Generate price movement
                        base_price = np.random.uniform(100, 10000)
                        change = np.random.normal(0, 0.01)
                        price = base_price * (1 + change)

                        # Generate OHLCV data
                        timestamp = datetime.now()
                        open_price = price * np.random.uniform(0.99, 1.01)
                        high_price = price * np.random.uniform(1.0, 1.02)
                        low_price = price * np.random.uniform(0.98, 1.0)
                        close_price = price
                        volume = np.random.uniform(100000, 1000000)

                        market_key = f"{exchange}:{market}"

                        new_data = pd.DataFrame(
                            {
                                "timestamp": [timestamp],
                                "asset": [asset],
                                "open": [open_price],
                                "high": [high_price],
                                "low": [low_price],
                                "close": [close_price],
                                "volume": [volume],
                            }
                        )

                        self.market_data[exchange][market] = pd.concat(
                            [self.market_data[exchange][market], new_data],
                            ignore_index=True,
                        )

                        # Keep only last 100 data points
                        if len(self.market_data[exchange][market]) > 100:
                            self.market_data[exchange][market] = self.market_data[
                                exchange
                            ][market].iloc[-100:]

        except Exception as e:
            logger.error(f"Market data update failed: {e}")

    async def _calculate_pair_correlation(self, pair: CorrelationPair) -> None:
        """Calculate correlation for a specific pair"""
        try:
            # Get price data for both markets
            data1 = await self._get_market_data(
                pair.exchange1, pair.market1, pair.asset1
            )
            data2 = await self._get_market_data(
                pair.exchange2, pair.market2, pair.asset2
            )

            if (
                data1 is None
                or data2 is None
                or len(data1) < self.correlation_lookback
                or len(data2) < self.correlation_lookback
            ):
                return

            # Align price data
            min_length = min(len(data1), len(data2))
            prices1 = data1["close"].iloc[-min_length:].values
            prices2 = data2["close"].iloc[-min_length:].values

            # Calculate Pearson correlation
            correlation, p_value = pearsonr(prices1, prices2)

            # Calculate rolling correlation
            rolling_window = min(20, len(prices1))
            rolling_correlation = np.corrcoef(
                prices1[-rolling_window:], prices2[-rolling_window:]
            )[0, 1]

            # Calculate spread
            spread = prices1[-1] - prices2[-1]

            # Calculate spread statistics
            historical_spreads = (
                prices1[-self.correlation_lookback :]
                - prices2[-self.correlation_lookback :]
            )
            spread_mean = np.mean(historical_spreads)
            spread_std = np.std(historical_spreads)

            # Calculate Z-score
            z_score = (spread - spread_mean) / spread_std if spread_std > 0 else 0

            # Update correlation pair
            pair.correlation_value = correlation
            pair.p_value = p_value
            pair.rolling_correlation = rolling_correlation
            pair.spread = spread
            pair.spread_mean = spread_mean
            pair.spread_std = spread_std
            pair.z_score = abs(z_score)
            pair.last_updated = datetime.now()

            # Calculate correlation trend
            historical_corrs = self.correlation_history.get(
                f"{pair.market1}:{pair.asset1}:{pair.market2}:{pair.asset2}", []
            )
            if historical_corrs:
                recent_corrs = [c for _, c in historical_corrs[-5:]]
                if len(recent_corrs) >= 3:
                    if np.polyfit(range(len(recent_corrs)), recent_corrs, 1)[0] > 0.01:
                        pair.correlation_trend = "increasing"
                    elif (
                        np.polyfit(range(len(recent_corrs)), recent_corrs, 1)[0] < -0.01
                    ):
                        pair.correlation_trend = "decreasing"
                    else:
                        pair.correlation_trend = "stable"

            # Store correlation history
            history_key = f"{pair.market1}:{pair.asset1}:{pair.market2}:{pair.asset2}"
            self.correlation_history[history_key].append((datetime.now(), correlation))

            # Keep only last 100 historical correlations
            if len(self.correlation_history[history_key]) > 100:
                self.correlation_history[history_key] = self.correlation_history[
                    history_key
                ][-100:]

        except Exception as e:
            logger.error(f"Pair correlation calculation failed: {e}")

    async def _get_market_data(
        self, exchange: str, market: str, asset: str
    ) -> Optional[pd.DataFrame]:
        """Get market data for specific exchange/market/asset"""
        try:
            market_key = f"{exchange}:{market}"
            data = self.market_data.get(exchange, {}).get(market)

            if data is None or len(data) == 0:
                return None

            # Filter by asset
            asset_data = data[data["asset"] == asset]

            return asset_data if len(asset_data) > 0 else None

        except Exception as e:
            logger.error(f"Market data retrieval failed: {e}")
            return None

    async def _update_correlation_network(self) -> None:
        """Update correlation network graph"""
        try:
            # Clear existing network
            self.correlation_network.clear()

            # Add nodes for all assets
            for exchange in self.exchanges:
                for market in self.markets:
                    for asset in self.assets:
                        node_id = f"{exchange}:{market}:{asset}"
                        self.correlation_network.add_node(node_id)

            # Add edges for correlations
            for pair_key, pair in self.correlation_pairs.items():
                if abs(pair.correlation_value) >= self.min_correlation:
                    node1 = f"{pair.exchange1}:{pair.market1}:{pair.asset1}"
                    node2 = f"{pair.exchange2}:{pair.market2}:{pair.asset2}"

                    self.correlation_network.add_edge(
                        node1,
                        node2,
                        weight=abs(pair.correlation_value),
                        correlation=pair.correlation_value,
                        spread=pair.spread,
                        z_score=pair.z_score,
                    )

            # Detect clusters
            self._detect_correlation_clusters()

        except Exception as e:
            logger.error(f"Correlation network update failed: {e}")

    def _detect_correlation_clusters(self) -> None:
        """Detect correlation clusters using clustering algorithm"""
        try:
            if len(self.correlation_network.nodes()) < 3:
                return

            # Extract adjacency matrix
            adjacency = nx.to_numpy_array(self.correlation_network)

            # Use DBSCAN clustering
            try:
                clustering = DBSCAN(eps=0.5, min_samples=2, metric="precomputed")
                labels = clustering.fit_predict(adjacency)

                # Cluster nodes
                self.correlation_clusters.clear()
                for node_id, label in zip(self.correlation_network.nodes(), labels):
                    if label >= 0:
                        if label not in self.correlation_clusters:
                            self.correlation_clusters[label] = []
                        self.correlation_clusters[label].append(node_id)

            except Exception as e:
                logger.debug(
                    f"Clustering failed (might be expected for small networks): {e}"
                )

                # Fallback: use connected components
                self.correlation_clusters = {
                    i: list(component)
                    for i, component in enumerate(
                        nx.connected_components(self.correlation_network)
                    )
                }

        except Exception as e:
            logger.error(f"Cluster detection failed: {e}")

    async def _correlation_monitoring_loop(self) -> None:
        """Background loop for monitoring correlations"""
        while True:
            try:
                await self._update_correlation_data()
                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Correlation monitoring loop error: {e}")
                await asyncio.sleep(15)

    async def generate_correlation_signals(self) -> List[CorrelationSignal]:
        """Generate trading signals from correlation analysis"""
        signals = []

        try:
            for pair_key, pair in self.correlation_pairs.items():
                # Check for convergence opportunities
                if abs(pair.z_score) > self.min_spread_threshold / pair.spread_std:
                    signal = await self._generate_convergence_signal(pair)
                    if signal:
                        signals.append(signal)

                # Check for divergence opportunities
                if self._detect_correlation_divergence(pair):
                    signal = await self._generate_divergence_signal(pair)
                    if signal:
                        signals.append(signal)

                # Check for breakout opportunities
                if self._detect_correlation_breakout(pair):
                    signal = await self._generate_breakout_signal(pair)
                    if signal:
                        signals.append(signal)

            # Rank signals by quality
            filtered_signals = self._rank_correlation_signals(signals)

            # Update active signals
            for signal in filtered_signals:
                signal_key = f"{signal.signal_id}"
                self.active_signals[signal_key] = signal
                self.signal_history.append(signal)

            return filtered_signals

        except Exception as e:
            logger.error(f"Correlation signal generation failed: {e}")
            return []

    async def _generate_convergence_signal(
        self, pair: CorrelationPair
    ) -> Optional[CorrelationSignal]:
        """Generate convergence trading signal"""
        try:
            if abs(pair.correlation_value) < self.min_correlation:
                return None

            # Determine direction
            if pair.spread > 0:
                # Asset1 is overpriced - short asset1, long asset2
                action = "short_long"
                position_sizes = [
                    -self.max_position_per_pair / 2,
                    self.max_position_per_pair / 2,
                ]
            else:
                # Asset2 is overpriced - long asset1, short asset2
                action = "long_short"
                position_size = (
                    self.max_position_per_pair / 2,
                    -self.max_position_per_pair / 2,
                )

            # Expected return from spread convergence
            expected_return = abs(pair.spread - pair.spread_mean) * 0.5  # Conservative

            # Risk score based on Z-score
            risk_score = min(1.0, pair.z_score / 3.0)  # Normalize Z-score

            # Confidence based on correlation strength and Z-score
            confidence = abs(pair.correlation_value) * min(1.0, pair.z_score / 2.0)

            # Get current prices
            data1 = await self._get_market_data(
                pair.exchange1, pair.market1, pair.asset1
            )
            data2 = await self._get_market_data(
                pair.exchange2, pair.market2, pair.asset2
            )

            if data1 is None or data2 is None:
                return None

            prices = [data1["close"].iloc[-1], data2["close"].iloc[-1]]

            # Calculate targets and stops
            targets = [
                prices[0] * (1 - 0.01)
                if action.startswith("short")
                else prices[0] * (1 + 0.01),
                prices[1] * (1 + 0.01)
                if action.startswith("long")
                else prices[1] * (1 - 0.01),
            ]

            stops = [
                prices[0] * (1 + 0.02)
                if action.startswith("short")
                else prices[0] * (1 - 0.02),
                prices[1] * (1 - 0.02)
                if action.startswith("long")
                else prices[1] * (1 + 0.02),
            ]

            signal = CorrelationSignal(
                signal_id=f"conv_{pair_key}_{datetime.now().timestamp()}",
                correlation_pair=pair,
                signal_type="convergence",
                action=action,
                confidence=min(1.0, confidence),
                expected_return=expected_return,
                risk_score=min(1.0, risk_score),
                position_sizes=position_size,
                entry_prices=prices,
                targets=targets,
                stops=stops,
                metadata={
                    "spread": pair.spread,
                    "z_score": pair.z_score,
                    "correlation": pair.correlation_value,
                },
            )

            return signal if signal.confidence >= 0.6 else None

        except Exception as e:
            logger.error(f"Convergence signal generation failed: {e}")
            return None

    async def _generate_divergence_signal(
        self, pair: CorrelationPair
    ) -> Optional[CorrelationSignal]:
        """Generate divergence trading signal"""
        try:
            # Divergence occurs when correlation is breaking down
            if (
                pair.correlation_trend == "decreasing"
                and abs(pair.correlation_value) < 0.5
            ):
                # Both assets - position based on individual direction
                action = "neutral"
                position_sizes = [0, 0]

                return None  # Divergence signals are more complex, implement later

            return None

        except Exception as e:
            logger.error(f"Divergence signal generation failed: {e}")
            return None

    async def _generate_breakout_signal(
        self, pair: CorrelationPair
    ) -> Optional[CorrelationSignal]:
        """Generate breakout trading signal"""
        try:
            # Breakout occurs when spread moves beyond recent extremes
            if abs(pair.z_score) > 2.5:
                # Strong breakout - directional move
                if pair.spread > 0:
                    action = "long"
                    position_sizes = [self.max_position_per_pair, 0]
                else:
                    action = "short"
                    position_sizes = [0, -self.max_position_per_pair]

                expected_return = abs(pair.spread - pair.spread_mean) * 0.8
                confidence = min(1.0, abs(pair.z_score) / 3.0)
                risk_score = min(1.0, pair.z_score / 4.0)

                # Get current prices
                data1 = await self._get_market_data(
                    pair.exchange1, pair.market1, pair.asset1
                )
                if data1 is None:
                    return None

                price = data1["close"].iloc[-1]

                signal = CorrelationSignal(
                    signal_id=f"break_{pair_key}_{datetime.now().timestamp()}",
                    correlation_pair=pair,
                    signal_type="breakout",
                    action=action,
                    confidence=confidence,
                    expected_return=expected_return,
                    risk_score=risk_score,
                    position_sizes=position_sizes,
                    entry_prices=[price, price],
                    targets=[
                        price * (1 + 0.03) if action == "long" else price * (1 - 0.03),
                        price,
                    ],
                    stops=[
                        price * (1 - 0.02) if action == "long" else price * (1 + 0.02),
                        price,
                    ],
                    metadata={
                        "spread": pair.spread,
                        "z_score": pair.z_score,
                        "correlation": pair.correlation_value,
                    },
                )

                return signal if signal.confidence >= 0.7 else None

            return None

        except Exception as e:
            logger.error(f"Breakout signal generation failed: {e}")
            return None

    def _detect_correlation_divergence(self, pair: CorrelationPair) -> bool:
        """Detect correlation divergence"""
        try:
            # Check if correlation is rapidly changing
            history_key = f"{pair.market1}:{pair.asset1}:{pair.market2}:{pair.asset2}"
            history = self.correlation_history.get(history_key, [])

            if len(history) < 5:
                return False

            recent_corrs = [c for _, c in history[-5:]]

            # Calculate correlation change
            correlation_change = abs(recent_corrs[-1] - recent_corrs[0])

            # Check if change is significant
            if correlation_change > self.max_correlation_deviation:
                return True

            return False

        except Exception as e:
            logger.error(f"Correlation divergence detection failed: {e}")
            return False

    def _detect_correlation_breakout(self, pair: CorrelationPair) -> bool:
        """Detect correlation breakout"""
        try:
            # Check if spread has moved significantly beyond mean
            if abs(pair.z_score) > 2.0:
                return True

            return False

        except Exception as e:
            logger.error(f"Correlation breakout detection failed: {e}")
            return False

    def _rank_correlation_signals(
        self, signals: List[CorrelationSignal]
    ) -> List[CorrelationSignal]:
        """Rank signals by quality"""
        try:

            def score_signal(signal: CorrelationSignal) -> float:
                # Composite score
                confidence_weight = 0.4
                return_weight = 0.4
                risk_weight = 0.2

                return (
                    signal.confidence * confidence_weight
                    + abs(signal.expected_return) * return_weight
                    + (1 - signal.risk_score) * risk_weight
                )

            # Sort by score
            signals.sort(key=score_signal, reverse=True)

            # Return top 10 signals
            return signals[:10]

        except Exception as e:
            logger.error(f"Signal ranking failed: {e}")
            return signals

    async def _signal_generation_loop(self) -> None:
        """Background loop for generating signals"""
        while True:
            try:
                signals = await self.generate_correlation_signals()
                logger.info(f"Generated {len(signals)} correlation signals")
                await asyncio.sleep(60)  # Generate signals every minute

            except Exception as e:
                logger.error(f"Signal generation loop error: {e}")
                await asyncio.sleep(30)

    async def _position_management_loop(self) -> None:
        """Background loop for managing positions"""
        while True:
            try:
                # Check active positions
                await self._manage_active_positions()

                # Update performance metrics
                await self._update_performance_metrics()

                await asyncio.sleep(15)  # Check every 15 seconds

            except Exception as e:
                logger.error(f"Position management loop error: {e}")
                await asyncio.sleep(10)

    async def _manage_active_positions(self) -> None:
        """Manage active trading positions"""
        try:
            # Check if any active signals should be closed
            for signal_key, signal in list(self.active_signals.items()):
                should_close = False
                close_reason = ""

                # Check stop loss
                for i, (position_size, stop_price) in enumerate(
                    zip(signal.position_sizes, signal.stops)
                ):
                    if position_size != 0 and stop_price > 0:
                        # Check if stop has been hit
                        if (
                            signal.action.startswith("long")
                            and signal.entry_prices[i] < stop_price
                        ):
                            should_close = True
                            close_reason = "stop_loss"
                        elif (
                            signal.action.startswith("short")
                            and signal.entry_prices[i] > stop_price
                        ):
                            should_close = True
                            close_reason = "stop_loss"

                # Check take profit
                for i, (position_size, target_price) in enumerate(
                    zip(signal.position_sizes, signal.targets)
                ):
                    if position_size != 0 and target_price > 0:
                        if (
                            signal.action.startswith("long")
                            and signal.entry_prices[i] > target_price
                        ):
                            should_close = True
                            close_reason = "take_profit"
                        elif (
                            signal.action.startswith("short")
                            and signal.entry_prices[i] < target_price
                        ):
                            should_close = True
                            close_reason = "take_profit"

                # Check if correlation has normalized (for convergence signals)
                if signal.signal_type == "convergence":
                    pair = signal.correlation_pair
                    if abs(pair.z_score) < 0.5:  # Z-score has normalized
                        should_close = True
                        close_reason = "convergence"

                if should_close:
                    await self._close_position(signal_key, close_reason)

        except Exception as e:
            logger.error(f"Active position management failed: {e}")

    async def _close_position(self, signal_key: str, reason: str) -> None:
        """Close a trading position"""
        try:
            if signal_key in self.active_signals:
                signal = self.active_signals[signal_key]

                logger.info(f"Closing position {signal_key}: {reason}")

                # Update performance
                self.performance_metrics["positions_closed"] += 1
                self.performance_metrics[f"close_reason_{reason}"] += 1

                # Remove from active positions
                del self.active_signals[signal_key]

        except Exception as e:
            logger.error(f"Position close failed for {signal_key}: {e}")

    async def _update_performance_metrics(self) -> None:
        """Update performance metrics"""
        try:
            self.performance_metrics["active_positions"] = len(self.active_positions)
            self.performance_metrics["total_signals_generated"] = len(
                self.signal_history
            )

            # Calculate hit rate
            if self.performance_metrics["positions_closed"] > 0:
                profit_closes = self.performance_metrics.get(
                    "close_reason_take_profit", 0
                )
                self.performance_metrics["hit_rate"] = (
                    profit_closes / self.performance_metrics["positions_closed"]
                )

        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")

    async def _cache_correlation_data(self) -> None:
        """Cache correlation data for quick access"""
        try:
            correlation_data = {
                pair_key: {
                    "correlation": pair.correlation_value,
                    "rolling_correlation": pair.rolling_correlation,
                    "spread": pair.spread,
                    "z_score": pair.z_score,
                    "correlation_type": pair.correlation_type.value,
                    "last_updated": pair.last_updated.isoformat(),
                }
                for pair_key, pair in self.correlation_pairs.items()
            }

            await self.cache.set("correlation_data", correlation_data, ttl=60)

        except Exception as e:
            logger.error(f"Correlation data caching failed: {e}")

    async def get_correlation_report(self) -> Dict[str, Any]:
        """Get comprehensive correlation trading report"""
        try:
            # Calculate correlation statistics
            correlation_values = [
                p.correlation_value for p in self.correlation_pairs.values()
            ]

            return {
                "total_correlation_pairs": len(self.correlation_pairs),
                "active_signals": len(self.active_signals),
                "correlation_statistics": {
                    "mean_correlation": np.mean(correlation_values)
                    if correlation_values
                    else 0,
                    "std_correlation": np.std(correlation_values)
                    if correlation_values
                    else 0,
                    "max_correlation": np.max(correlation_values)
                    if correlation_values
                    else 0,
                    "min_correlation": np.min(correlation_values)
                    if correlation_values
                    else 0,
                },
                "strong_correlations": [
                    {
                        "pair_key": pair_key,
                        "correlation": pair.correlation_value,
                        "spread": pair.spread,
                        "z_score": pair.z_score,
                    }
                    for pair_key, pair in self.correlation_pairs.items()
                    if abs(pair.correlation_value) > 0.7
                ],
                "correlation_clusters": {
                    str(cluster_id): nodes
                    for cluster_id, nodes in self.correlation_clusters.items()
                },
                "performance_metrics": dict(self.performance_metrics),
                "recent_signals": [
                    {
                        "signal_id": s.signal_id,
                        "signal_type": s.signal_type,
                        "action": s.action,
                        "confidence": s.confidence,
                        "expected_return": s.expected_return,
                        "timestamp": s.timestamp.isoformat(),
                    }
                    for s in self.signal_history[-10:]  # Last 10 signals
                ],
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Correlation report generation failed: {e}")
            return {}


# Utility functions
async def create_correlation_trader(
    config: Dict[str, Any],
) -> MultiMarketCorrelationTrader:
    """Create and initialize correlation trader"""
    trader = MultiMarketCorrelationTrader(config)
    await trader.initialize()
    return trader


def calculate_correlation_momentum(
    correlation_history: List[Tuple[datetime, float]], window: int = 10
) -> float:
    """Calculate momentum of correlation changes"""
    if len(correlation_history) < window:
        return 0.0

    recent_correlations = [c for _, c in correlation_history[-window:]]

    # Linear regression to find trend
    x = np.arange(len(recent_correlations))
    slope = np.polyfit(x, recent_correlations, 1)[0]

    return slope


def detect_correlation_regime_change(
    current_corr: float, historical_corrs: List[float], threshold: float = 0.2
) -> bool:
    """Detect if correlation regime has changed"""
    if len(historical_corrs) < 5:
        return False

    mean_historical = np.mean(historical_corrs)

    # Check if current correlation deviates significantly from historical mean
    return abs(current_corr - mean_historical) > threshold
