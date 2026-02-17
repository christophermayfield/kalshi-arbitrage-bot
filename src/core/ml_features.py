from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger("ml_features")


class FeatureExtractor:
    def __init__(self, lookback_windows: List[int] = None):
        self.lookback_windows = lookback_windows or [5, 20, 60]
        self._price_history: Dict[str, List[float]] = {}
        self._volume_history: Dict[str, List[int]] = {}
        self._spread_history: Dict[str, List[float]] = {}

    def update_price(self, market_id: str, price: float) -> None:
        if market_id not in self._price_history:
            self._price_history[market_id] = []
        self._price_history[market_id].append(price)
        for window in self.lookback_windows:
            if len(self._price_history[market_id]) > window:
                self._price_history[market_id].pop(0)

    def update_volume(self, market_id: str, volume: int) -> None:
        if market_id not in self._volume_history:
            self._volume_history[market_id] = []
        self._volume_history[market_id].append(volume)
        for window in self.lookback_windows:
            if len(self._volume_history[market_id]) > window:
                self._volume_history[market_id].pop(0)

    def update_spread(self, market_id: str, spread: float) -> None:
        if market_id not in self._spread_history:
            self._spread_history[market_id] = []
        self._spread_history[market_id].append(spread)
        for window in self.lookback_windows:
            if len(self._spread_history[market_id]) > window:
                self._spread_history[market_id].pop(0)

    def extract_orderbook_features(self, orderbook: Any) -> Dict[str, float]:
        features = {}

        if orderbook is None:
            return {
                'liquidity_score': 0,
                'spread_percent': 0,
                'bid_depth': 0,
                'ask_depth': 0,
                'depth_imbalance': 0,
                'mid_price': 0
            }

        features['liquidity_score'] = orderbook.get_liquidity_score()
        features['spread_percent'] = orderbook.get_spread_percent() or 0
        features['mid_price'] = orderbook.get_mid_price() or 0
        features['bid_depth'] = orderbook.get_bid_depth(3)
        features['ask_depth'] = orderbook.get_ask_depth(3)

        bid_depth = features['bid_depth']
        ask_depth = features['ask_depth']
        total_depth = bid_depth + ask_depth
        if total_depth > 0:
            features['depth_imbalance'] = (bid_depth - ask_depth) / total_depth
        else:
            features['depth_imbalance'] = 0

        return features

    def extract_market_features(
        self,
        market_id: str,
        current_price: float
    ) -> Dict[str, float]:
        self.update_price(market_id, current_price)

        features = {}

        price_history = self._price_history.get(market_id, [])
        if len(price_history) >= 2:
            returns = np.diff(price_history) / price_history[:-1]
            features['momentum'] = returns[-1] if len(returns) > 0 else 0
            features['momentum_5'] = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            features['momentum_20'] = np.mean(returns[-20:]) if len(returns) >= 20 else 0

            features['volatility'] = np.std(returns) if len(returns) > 0 else 0
            features['volatility_5'] = np.std(returns[-5:]) if len(returns) >= 5 else 0

            features['price_change_5'] = (current_price - price_history[-5]) / price_history[-5] if len(price_history) >= 5 else 0
            features['price_change_20'] = (current_price - price_history[-20]) / price_history[-20] if len(price_history) >= 20 else 0

            features['rolling_mean_5'] = np.mean(price_history[-5:]) if len(price_history) >= 5 else current_price
            features['rolling_mean_20'] = np.mean(price_history[-20:]) if len(price_history) >= 20 else current_price
            features['price_to_mean_5'] = current_price / features['rolling_mean_5'] - 1 if features['rolling_mean_5'] > 0 else 0
            features['price_to_mean_20'] = current_price / features['rolling_mean_20'] - 1 if features['rolling_mean_20'] > 0 else 0
        else:
            features.update({
                'momentum': 0, 'momentum_5': 0, 'momentum_20': 0,
                'volatility': 0, 'volatility_5': 0,
                'price_change_5': 0, 'price_change_20': 0,
                'rolling_mean_5': current_price, 'rolling_mean_20': current_price,
                'price_to_mean_5': 0, 'price_to_mean_20': 0
            })

        volume_history = self._volume_history.get(market_id, [])
        if volume_history:
            features['volume'] = volume_history[-1]
            features['volume_change'] = (volume_history[-1] - volume_history[-2]) / volume_history[-2] if len(volume_history) >= 2 else 0
            features['avg_volume_5'] = np.mean(volume_history[-5:]) if len(volume_history) >= 5 else volume_history[-1]
        else:
            features.update({'volume': 0, 'volume_change': 0, 'avg_volume_5': 0})

        spread_history = self._spread_history.get(market_id, [])
        if spread_history:
            features['spread'] = spread_history[-1]
            features['avg_spread_5'] = np.mean(spread_history[-5:]) if len(spread_history) >= 5 else spread_history[-1]
        else:
            features.update({'spread': 0, 'avg_spread_5': 0})

        return features

    def extract_opportunity_features(
        self,
        opportunity: Dict[str, Any],
        orderbook: Any,
        portfolio_stats: Dict[str, Any]
    ) -> Dict[str, float]:
        features = {}

        features['confidence'] = opportunity.get('confidence', 0.5)
        features['profit_percent'] = opportunity.get('profit_percent', 0)
        features['net_profit_cents'] = opportunity.get('net_profit_cents', 0)
        features['gross_profit_cents'] = opportunity.get('profit_cents', 0)
        features['fees'] = opportunity.get('fees', 0)
        features['quantity'] = opportunity.get('quantity', 0)

        risk_level = opportunity.get('risk_level', 'medium')
        risk_map = {'low': 0, 'medium': 0.5, 'high': 1}
        features['risk_score'] = risk_map.get(risk_level, 0.5)

        orderbook_features = self.extract_orderbook_features(orderbook)
        features.update(orderbook_features)

        market_id = opportunity.get('market_id_1')
        current_price = orderbook.get_mid_price() if orderbook else 0
        if market_id and current_price:
            market_features = self.extract_market_features(market_id, current_price)
            features.update(market_features)

        features['portfolio_utilization'] = portfolio_stats.get('open_positions', 0) / max(1, portfolio_stats.get('max_positions', 50))
        features['cash_available'] = portfolio_stats.get('cash_balance', 0)
        features['position_size_ratio'] = features['quantity'] * current_price / max(1, features['cash_available']) if current_price else 0

        return features

    def extract_features_dataframe(self, opportunities: List[Dict], orderbooks: Dict, portfolios: Dict) -> pd.DataFrame:
        rows = []
        for opp in opportunities:
            market_id = opp.get('market_id_1')
            orderbook = orderbooks.get(market_id)
            portfolio = portfolios.get('current', {})

            features = self.extract_opportunity_features(opp, orderbook, portfolio)
            features['opportunity_id'] = opp.get('id')
            features['label'] = opp.get('executed', 0)
            rows.append(features)

        return pd.DataFrame(rows)

    def get_feature_names(self) -> List[str]:
        return [
            'confidence', 'profit_percent', 'net_profit_cents', 'gross_profit_cents',
            'fees', 'quantity', 'risk_score', 'liquidity_score', 'spread_percent',
            'mid_price', 'bid_depth', 'ask_depth', 'depth_imbalance',
            'momentum', 'momentum_5', 'momentum_20', 'volatility', 'volatility_5',
            'price_change_5', 'price_change_20', 'rolling_mean_5', 'rolling_mean_20',
            'price_to_mean_5', 'price_to_mean_20', 'volume', 'volume_change',
            'avg_volume_5', 'spread', 'avg_spread_5', 'portfolio_utilization',
            'cash_available', 'position_size_ratio'
        ]


class FeatureStore:
    def __init__(self, ttl_hours: int = 24):
        self.ttl_hours = ttl_hours
        self._store: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, datetime] = {}

    def save(self, key: str, features: Dict[str, float]) -> None:
        self._store[key] = features
        self._timestamps[key] = datetime.utcnow()

    def load(self, key: str) -> Optional[Dict[str, float]]:
        if key not in self._store:
            return None

        if datetime.utcnow() - self._timestamps[key] > timedelta(hours=self.ttl_hours):
            del self._store[key]
            del self._timestamps[key]
            return None

        return self._store[key]

    def cleanup(self) -> int:
        now = datetime.utcnow()
        keys_to_remove = [
            k for k, t in self._timestamps.items()
            if now - t > timedelta(hours=self.ttl_hours)
        ]
        for key in keys_to_remove:
            del self._store[key]
            del self._timestamps[key]
        return len(keys_to_remove)
