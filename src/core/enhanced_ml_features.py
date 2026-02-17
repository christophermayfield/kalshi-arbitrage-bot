"""
Enhanced ML Features with Order Book Imbalance and Trade Flow Analysis
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import deque, defaultdict

from src.utils.logging_utils import get_logger
from src.core.orderbook import OrderBook, OrderBookLevel

logger = get_logger("enhanced_ml_features")


@dataclass
class TradeFlowMetrics:
    """Metrics for trade flow analysis"""

    volume_weighted_price: float = 0.0
    trade_intensity: float = 0.0
    buy_pressure: float = 0.0
    sell_pressure: float = 0.0
    flow_imbalance: float = 0.0
    aggressive_trades_ratio: float = 0.0
    large_trade_ratio: float = 0.0


@dataclass
class OrderBookImbalanceMetrics:
    """Metrics for order book imbalance analysis"""

    bid_ask_imbalance: float = 0.0
    volume_imbalance: float = 0.0
    price_level_imbalance: float = 0.0
    depth_skewness: float = 0.0
    liquidity_concentration: float = 0.0
    market_pressure: float = 0.0
    order_flow_direction: float = 0.0


class EnhancedFeatureExtractor:
    """
    Enhanced feature extractor with order book imbalance and trade flow analysis
    """

    def __init__(
        self, lookback_windows: List[int] = None, max_history_size: int = 1000
    ):
        self.lookback_windows = lookback_windows or [5, 10, 20, 50]
        self.max_history_size = max_history_size

        # Price and volume history
        self._price_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history_size)
        )
        self._volume_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history_size)
        )
        self._spread_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history_size)
        )

        # Trade flow history
        self._trade_flow_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history_size)
        )
        self._orderbook_snapshots: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )  # Last 100 snapshots

        # Market microstructure data
        self._tick_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))

        # Imbalance metrics cache
        self._imbalance_cache: Dict[str, Tuple[float, datetime]] = {}

    def update_orderbook_snapshot(self, market_id: str, orderbook: OrderBook) -> None:
        """Update orderbook snapshot for imbalance analysis"""
        try:
            snapshot = {
                "timestamp": datetime.now(),
                "bids": [
                    (level.price, level.count, level.total)
                    for level in orderbook.bids[:10]
                ],
                "asks": [
                    (level.price, level.count, level.total)
                    for level in orderbook.asks[:10]
                ],
                "mid_price": orderbook.get_mid_price() or 0,
                "spread": orderbook.get_spread() or 0,
            }

            self._orderbook_snapshots[market_id].append(snapshot)

            # Update basic price history
            mid_price = snapshot["mid_price"]
            if mid_price > 0:
                self._price_history[market_id].append(mid_price)

            # Update spread history
            spread = snapshot["spread"]
            if spread > 0:
                self._spread_history[market_id].append(spread)

        except Exception as e:
            logger.debug(f"Failed to update orderbook snapshot for {market_id}: {e}")

    def update_trade_flow(self, market_id: str, trade_data: Dict[str, Any]) -> None:
        """Update trade flow data"""
        try:
            timestamp = datetime.now()
            trade = {
                "timestamp": timestamp,
                "price": trade_data.get("price", 0),
                "quantity": trade_data.get("quantity", 0),
                "side": trade_data.get("side", "unknown"),
                "aggressive": trade_data.get("aggressive", False),
            }

            self._trade_flow_history[market_id].append(trade)

            # Update volume history
            volume = trade["quantity"]
            if volume > 0:
                self._volume_history[market_id].append(volume)

        except Exception as e:
            logger.debug(f"Failed to update trade flow for {market_id}: {e}")

    def calculate_orderbook_imbalance(
        self, market_id: str
    ) -> OrderBookImbalanceMetrics:
        """Calculate comprehensive order book imbalance metrics"""
        try:
            # Check cache
            cache_key = f"imbalance_{market_id}"
            if cache_key in self._imbalance_cache:
                cached_value, cached_time = self._imbalance_cache[cache_key]
                if datetime.now() - cached_time < timedelta(seconds=1):
                    return OrderBookImbalanceMetrics(**cached_value)

            snapshots = list(self._orderbook_snapshots[market_id])
            if not snapshots:
                return OrderBookImbalanceMetrics()

            latest_snapshot = snapshots[-1]
            bids = latest_snapshot["bids"]
            asks = latest_snapshot["asks"]

            if not bids or not asks:
                return OrderBookImbalanceMetrics()

            # Calculate bid-ask imbalance
            total_bid_volume = sum(level[2] for level in bids)  # Sum of totals
            total_ask_volume = sum(level[2] for level in asks)
            total_volume = total_bid_volume + total_ask_volume

            bid_ask_imbalance = (total_bid_volume - total_ask_volume) / max(
                1, total_volume
            )

            # Volume-weighted price imbalance
            vwap_bid = sum(level[0] * level[2] for level in bids) / max(
                1, total_bid_volume
            )
            vwap_ask = sum(level[0] * level[2] for level in asks) / max(
                1, total_ask_volume
            )
            mid_price = latest_snapshot["mid_price"]

            volume_imbalance = (
                (vwap_bid - vwap_ask) / max(1, mid_price) if mid_price > 0 else 0
            )

            # Price level imbalance (depth at different levels)
            bid_levels_volume = [level[2] for level in bids[:5]]  # Top 5 levels
            ask_levels_volume = [level[2] for level in asks[:5]]

            # Calculate weighted imbalance (closer levels get higher weight)
            level_weights = [1.0, 0.8, 0.6, 0.4, 0.2]
            weighted_bid = sum(w * v for w, v in zip(level_weights, bid_levels_volume))
            weighted_ask = sum(w * v for w, v in zip(level_weights, ask_levels_volume))
            total_weighted = weighted_bid + weighted_ask

            price_level_imbalance = (weighted_bid - weighted_ask) / max(
                1, total_weighted
            )

            # Depth skewness (distribution of liquidity)
            if len(bid_levels_volume) >= 3 and len(ask_levels_volume) >= 3:
                bid_skew = np.std(bid_levels_volume[:3]) / max(
                    1, np.mean(bid_levels_volume[:3])
                )
                ask_skew = np.std(ask_levels_volume[:3]) / max(
                    1, np.mean(ask_levels_volume[:3])
                )
                depth_skewness = (bid_skew - ask_skew) / 2
            else:
                depth_skewness = 0

            # Liquidity concentration (how concentrated liquidity is at top levels)
            top_2_bid = sum(bid_levels_volume[:2])
            top_5_bid = sum(bid_levels_volume[:5])
            top_2_ask = sum(ask_levels_volume[:2])
            top_5_ask = sum(ask_levels_volume[:5])

            concentration = (
                (top_2_bid + top_2_ask) / max(1, top_5_bid + top_5_ask)
            ) - 0.4

            # Market pressure (combination of multiple metrics)
            market_pressure = (
                bid_ask_imbalance * 0.4
                + volume_imbalance * 0.3
                + price_level_imbalance * 0.2
                + depth_skewness * 0.1
            )

            # Order flow direction (trend from recent snapshots)
            if len(snapshots) >= 3:
                recent_imbalances = []
                for snap in snapshots[-3:]:
                    recent_bids = snap["bids"]
                    recent_asks = snap["asks"]
                    if recent_bids and recent_asks:
                        bid_vol = sum(level[2] for level in recent_bids)
                        ask_vol = sum(level[2] for level in recent_asks)
                        recent_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
                        recent_imbalances.append(recent_imbalance)

                if recent_imbalances:
                    order_flow_direction = np.mean(recent_imbalances)
                else:
                    order_flow_direction = 0
            else:
                order_flow_direction = 0

            metrics = OrderBookImbalanceMetrics(
                bid_ask_imbalance=bid_ask_imbalance,
                volume_imbalance=volume_imbalance,
                price_level_imbalance=price_level_imbalance,
                depth_skewness=depth_skewness,
                liquidity_concentration=concentration,
                market_pressure=market_pressure,
                order_flow_direction=order_flow_direction,
            )

            # Cache for 1 second
            self._imbalance_cache[cache_key] = (
                {
                    "bid_ask_imbalance": bid_ask_imbalance,
                    "volume_imbalance": volume_imbalance,
                    "price_level_imbalance": price_level_imbalance,
                    "depth_skewness": depth_skewness,
                    "liquidity_concentration": concentration,
                    "market_pressure": market_pressure,
                    "order_flow_direction": order_flow_direction,
                },
                datetime.now(),
            )

            return metrics

        except Exception as e:
            logger.error(
                f"Failed to calculate orderbook imbalance for {market_id}: {e}"
            )
            return OrderBookImbalanceMetrics()

    def calculate_trade_flow_metrics(
        self, market_id: str, lookback_seconds: int = 60
    ) -> TradeFlowMetrics:
        """Calculate trade flow metrics"""
        try:
            trades = list(self._trade_flow_history[market_id])
            if not trades:
                return TradeFlowMetrics()

            # Filter by time window
            cutoff_time = datetime.now() - timedelta(seconds=lookback_seconds)
            recent_trades = [t for t in trades if t["timestamp"] >= cutoff_time]

            if not recent_trades:
                return TradeFlowMetrics()

            # Volume-weighted price
            total_volume = sum(t["quantity"] for t in recent_trades)
            if total_volume > 0:
                volume_weighted_price = (
                    sum(t["price"] * t["quantity"] for t in recent_trades)
                    / total_volume
                )
            else:
                volume_weighted_price = (
                    recent_trades[-1]["price"] if recent_trades else 0
                )

            # Trade intensity (trades per second)
            trade_intensity = len(recent_trades) / max(1, lookback_seconds)

            # Buy/sell pressure
            buy_volume = sum(t["quantity"] for t in recent_trades if t["side"] == "buy")
            sell_volume = sum(
                t["quantity"] for t in recent_trades if t["side"] == "sell"
            )

            if total_volume > 0:
                buy_pressure = buy_volume / total_volume
                sell_pressure = sell_volume / total_volume
                flow_imbalance = buy_pressure - sell_pressure
            else:
                buy_pressure = sell_pressure = flow_imbalance = 0

            # Aggressive trades ratio
            aggressive_trades = [t for t in recent_trades if t.get("aggressive", False)]
            aggressive_trades_ratio = len(aggressive_trades) / max(
                1, len(recent_trades)
            )

            # Large trade ratio (trades above average size)
            if recent_trades:
                avg_trade_size = total_volume / len(recent_trades)
                large_trades = [
                    t for t in recent_trades if t["quantity"] > avg_trade_size * 1.5
                ]
                large_trade_ratio = len(large_trades) / len(recent_trades)
            else:
                large_trade_ratio = 0

            return TradeFlowMetrics(
                volume_weighted_price=volume_weighted_price,
                trade_intensity=trade_intensity,
                buy_pressure=buy_pressure,
                sell_pressure=sell_pressure,
                flow_imbalance=flow_imbalance,
                aggressive_trades_ratio=aggressive_trades_ratio,
                large_trade_ratio=large_trade_ratio,
            )

        except Exception as e:
            logger.error(f"Failed to calculate trade flow metrics for {market_id}: {e}")
            return TradeFlowMetrics()

    def extract_enhanced_features(
        self, market_id: str, orderbook: OrderBook
    ) -> Dict[str, float]:
        """Extract enhanced features including imbalance and trade flow"""
        try:
            features = {}

            # Basic orderbook features
            features["liquidity_score"] = orderbook.get_liquidity_score()
            features["spread_percent"] = orderbook.get_spread_percent() or 0
            features["mid_price"] = orderbook.get_mid_price() or 0
            features["bid_depth"] = orderbook.get_bid_depth(3)
            features["ask_depth"] = orderbook.get_ask_depth(3)

            # Enhanced imbalance features
            imbalance_metrics = self.calculate_orderbook_imbalance(market_id)
            features["bid_ask_imbalance"] = imbalance_metrics.bid_ask_imbalance
            features["volume_imbalance"] = imbalance_metrics.volume_imbalance
            features["price_level_imbalance"] = imbalance_metrics.price_level_imbalance
            features["depth_skewness"] = imbalance_metrics.depth_skewness
            features["liquidity_concentration"] = (
                imbalance_metrics.liquidity_concentration
            )
            features["market_pressure"] = imbalance_metrics.market_pressure
            features["order_flow_direction"] = imbalance_metrics.order_flow_direction

            # Trade flow features
            trade_metrics = self.calculate_trade_flow_metrics(market_id)
            features["trade_intensity"] = trade_metrics.trade_intensity
            features["buy_pressure"] = trade_metrics.buy_pressure
            features["sell_pressure"] = trade_metrics.sell_pressure
            features["flow_imbalance"] = trade_metrics.flow_imbalance
            features["aggressive_trades_ratio"] = trade_metrics.aggressive_trades_ratio
            features["large_trade_ratio"] = trade_metrics.large_trade_ratio
            features["volume_weighted_price"] = trade_metrics.volume_weighted_price

            # Microstructure features
            features.update(self._extract_microstructure_features(market_id))

            # Temporal features
            features.update(self._extract_temporal_features(market_id))

            return features

        except Exception as e:
            logger.error(f"Failed to extract enhanced features for {market_id}: {e}")
            return {}

    def _extract_microstructure_features(self, market_id: str) -> Dict[str, float]:
        """Extract market microstructure features"""
        features = {}

        try:
            snapshots = list(self._orderbook_snapshots[market_id])
            if len(snapshots) < 2:
                return features

            # Volatility from price changes
            prices = [s["mid_price"] for s in snapshots if s["mid_price"] > 0]
            if len(prices) >= 2:
                price_changes = np.diff(prices) / prices[:-1]
                features["price_volatility"] = (
                    np.std(price_changes) if len(price_changes) > 0 else 0
                )
                features["volatility_trend"] = (
                    price_changes[-1] if len(price_changes) > 0 else 0
                )
            else:
                features["price_volatility"] = 0
                features["volatility_trend"] = 0

            # Spread dynamics
            spreads = [s["spread"] for s in snapshots if s["spread"] > 0]
            if len(spreads) >= 2:
                features["spread_volatility"] = (
                    np.std(spreads) / np.mean(spreads) if np.mean(spreads) > 0 else 0
                )
                features["spread_trend"] = (
                    (spreads[-1] - spreads[-2]) / spreads[-2]
                    if len(spreads) >= 2
                    else 0
                )
            else:
                features["spread_volatility"] = 0
                features["spread_trend"] = 0

            # Order book dynamics
            if len(snapshots) >= 5:
                recent_depth_imbalances = []
                for snap in snapshots[-5:]:
                    bids = snap["bids"]
                    asks = snap["asks"]
                    if bids and asks:
                        bid_vol = sum(level[2] for level in bids[:3])
                        ask_vol = sum(level[2] for level in asks[:3])
                        if bid_vol + ask_vol > 0:
                            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
                            recent_depth_imbalances.append(imbalance)

                if recent_depth_imbalances:
                    features["depth_imbalance_trend"] = np.mean(
                        np.diff(recent_depth_imbalances)
                    )
                    features["depth_imbalance_volatility"] = np.std(
                        recent_depth_imbalances
                    )
                else:
                    features["depth_imbalance_trend"] = 0
                    features["depth_imbalance_volatility"] = 0
            else:
                features["depth_imbalance_trend"] = 0
                features["depth_imbalance_volatility"] = 0

        except Exception as e:
            logger.debug(
                f"Failed to extract microstructure features for {market_id}: {e}"
            )

        return features

    def _extract_temporal_features(self, market_id: str) -> Dict[str, float]:
        """Extract temporal pattern features"""
        features = {}

        try:
            # Price momentum at different timeframes
            price_history = list(self._price_history[market_id])
            if len(price_history) >= 2:
                for window in [5, 10, 20, 50]:
                    if len(price_history) >= window:
                        recent_prices = price_history[-window:]
                        momentum = (recent_prices[-1] - recent_prices[0]) / max(
                            1, recent_prices[0]
                        )
                        features[f"momentum_{window}"] = momentum
                    else:
                        features[f"momentum_{window}"] = 0

                # Volatility at different timeframes
                if len(price_history) >= 10:
                    returns = np.diff(price_history) / price_history[:-1]
                    for window in [5, 10, 20]:
                        if len(returns) >= window:
                            features[f"volatility_{window}"] = np.std(returns[-window:])
                        else:
                            features[f"volatility_{window}"] = 0
                else:
                    for window in [5, 10, 20]:
                        features[f"volatility_{window}"] = 0
            else:
                for window in [5, 10, 20, 50]:
                    features[f"momentum_{window}"] = 0
                for window in [5, 10, 20]:
                    features[f"volatility_{window}"] = 0

            # Volume patterns
            volume_history = list(self._volume_history[market_id])
            if len(volume_history) >= 5:
                avg_volume = np.mean(volume_history[-5:])
                current_volume = volume_history[-1]
                features["volume_spike"] = (current_volume - avg_volume) / max(
                    1, avg_volume
                )
                features["volume_trend"] = np.mean(np.diff(volume_history[-5:]))
            else:
                features["volume_spike"] = 0
                features["volume_trend"] = 0

            # Time of day patterns (if we had timestamps)
            current_hour = datetime.now().hour
            features["hour_of_day"] = current_hour / 24.0  # Normalized
            features["is_trading_hours"] = (
                1.0 if 9 <= current_hour <= 16 else 0.0
            )  # Assuming market hours

        except Exception as e:
            logger.debug(f"Failed to extract temporal features for {market_id}: {e}")

        return features

    def extract_opportunity_enhanced_features(
        self, opportunity: Dict[str, Any], orderbook: OrderBook, market_id: str
    ) -> Dict[str, float]:
        """Extract enhanced features for an arbitrage opportunity"""
        try:
            # Base features from the opportunity
            features = {
                "confidence": opportunity.get("confidence", 0.5),
                "profit_percent": opportunity.get("profit_percent", 0),
                "net_profit_cents": opportunity.get("net_profit_cents", 0),
                "gross_profit_cents": opportunity.get("profit_cents", 0),
                "fees": opportunity.get("fees", 0),
                "quantity": opportunity.get("quantity", 0),
                "risk_level_score": {"low": 0, "medium": 0.5, "high": 1}.get(
                    opportunity.get("risk_level", "medium"), 0.5
                ),
            }

            # Enhanced orderbook features
            orderbook_features = self.extract_enhanced_features(market_id, orderbook)
            features.update(orderbook_features)

            # Opportunity-specific features
            features["execution_window"] = (
                opportunity.get("execution_window_seconds", 30) / 60.0
            )  # In minutes
            features["liquidity_requirement"] = features["quantity"] / max(
                1, features.get("liquidity_score", 1)
            )

            return features

        except Exception as e:
            logger.error(f"Failed to extract enhanced opportunity features: {e}")
            return {}

    def get_enhanced_feature_names(self) -> List[str]:
        """Get all feature names for the enhanced feature extractor"""
        base_features = [
            "confidence",
            "profit_percent",
            "net_profit_cents",
            "gross_profit_cents",
            "fees",
            "quantity",
            "risk_level_score",
            "liquidity_score",
            "spread_percent",
            "mid_price",
            "bid_depth",
            "ask_depth",
        ]

        imbalance_features = [
            "bid_ask_imbalance",
            "volume_imbalance",
            "price_level_imbalance",
            "depth_skewness",
            "liquidity_concentration",
            "market_pressure",
            "order_flow_direction",
        ]

        trade_flow_features = [
            "trade_intensity",
            "buy_pressure",
            "sell_pressure",
            "flow_imbalance",
            "aggressive_trades_ratio",
            "large_trade_ratio",
            "volume_weighted_price",
        ]

        microstructure_features = [
            "price_volatility",
            "volatility_trend",
            "spread_volatility",
            "spread_trend",
            "depth_imbalance_trend",
            "depth_imbalance_volatility",
        ]

        temporal_features = [
            "momentum_5",
            "momentum_10",
            "momentum_20",
            "momentum_50",
            "volatility_5",
            "volatility_10",
            "volatility_20",
            "volume_spike",
            "volume_trend",
            "hour_of_day",
            "is_trading_hours",
        ]

        opportunity_features = ["execution_window", "liquidity_requirement"]

        return (
            base_features
            + imbalance_features
            + trade_flow_features
            + microstructure_features
            + temporal_features
            + opportunity_features
        )


# Utility function for feature scaling
class FeatureScaler:
    """Scale features for ML models"""

    def __init__(self):
        self.scalers = {}
        self.feature_stats = {}

    def fit(self, features_df: pd.DataFrame) -> None:
        """Fit scalers on training data"""
        for column in features_df.columns:
            if column in ["opportunity_id", "label"]:
                continue

            col_data = features_df[column].dropna()
            if len(col_data) > 0:
                self.feature_stats[column] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                }

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scalers"""
        transformed = features_df.copy()

        for column, stats in self.feature_stats.items():
            if column in transformed.columns:
                # Standard scaling
                if stats["std"] > 0:
                    transformed[column] = (transformed[column] - stats["mean"]) / stats[
                        "std"
                    ]
                else:
                    transformed[column] = 0  # No variation

                # Handle extreme values (cap at 3 standard deviations)
                transformed[column] = np.clip(transformed[column], -3, 3)

        return transformed

    def fit_transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(features_df)
        return self.transform(features_df)
