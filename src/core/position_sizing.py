from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.orderbook import OrderBook

import numpy as np


class PositionSizingStrategy(ABC):
    @abstractmethod
    def calculate_size(
        self,
        opportunity: Dict[str, Any],
        portfolio_stats: Dict[str, Any],
        orderbook: Optional[OrderBook] = None
    ) -> int:
        pass


@dataclass
class KellyConfig:
    fraction: float = 0.5
    max_position_percent: float = 0.25
    min_bet: int = 1
    max_bet: int = 1000


class KellyCriterion(PositionSizingStrategy):
    def __init__(self, config: Optional[KellyConfig] = None):
        self.config = config or KellyConfig()

    def calculate_size(
        self,
        opportunity: Dict[str, Any],
        portfolio_stats: Dict[str, Any],
        orderbook: Optional[OrderBook] = None
    ) -> int:
        confidence = opportunity.get('confidence', 0.5)
        profit_percent = opportunity.get('profit_percent', 0)
        win_rate = confidence
        win_prob = win_rate
        loss_prob = 1 - win_prob

        if win_prob == 0 or loss_prob == 0:
            return self.config.min_bet

        avg_win = profit_percent / 100
        avg_loss = 0.01

        if avg_win <= 0:
            return self.config.min_bet

        kelly_fraction = (win_prob * avg_win - loss_prob * avg_loss) / avg_win

        if kelly_fraction <= 0:
            return self.config.min_bet

        kelly_fraction *= self.config.fraction

        max_position = portfolio_stats.get('cash_balance', 10000) * self.config.max_position_percent
        position_size = int(kelly_fraction * max_position / 100)

        position_size = max(self.config.min_bet, min(position_size, self.config.max_bet))

        return position_size


@dataclass
class FixedFractionConfig:
    fraction: float = 0.1
    min_size: int = 1
    max_size: int = 100


class FixedFractionStrategy(PositionSizingStrategy):
    def __init__(self, config: Optional[FixedFractionConfig] = None):
        self.config = config or FixedFractionConfig()

    def calculate_size(
        self,
        opportunity: Dict[str, Any],
        portfolio_stats: Dict[str, Any],
        orderbook: Optional[OrderBook] = None
    ) -> int:
        portfolio_value = portfolio_stats.get('cash_balance', 10000)
        base_size = int(portfolio_value * self.config.fraction / 100)
        size = max(self.config.min_size, min(base_size, self.config.max_size))
        return size


@dataclass
class VolatilityAdjustedConfig:
    base_fraction: float = 0.1
    volatility_window: int = 20
    min_size: int = 1
    max_size: int = 100
    volatility_target: float = 0.02


class VolatilityAdjustedStrategy(PositionSizingStrategy):
    def __init__(self, config: Optional[VolatilityAdjustedConfig] = None):
        self.config = config or VolatilityAdjustedConfig()
        self.price_history: List[float] = []

    def calculate_size(
        self,
        opportunity: Dict[str, Any],
        portfolio_stats: Dict[str, Any],
        orderbook: Optional[OrderBook] = None
    ) -> int:
        if orderbook:
            mid_price = orderbook.get_mid_price()
            if mid_price:
                self.price_history.append(mid_price)
                if len(self.price_history) > self.config.volatility_window:
                    self.price_history.pop(0)

        portfolio_value = portfolio_stats.get('cash_balance', 10000)

        if len(self.price_history) >= 2:
            returns = np.diff(self.price_history) / self.price_history[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0.01

            if volatility > 0:
                size_multiplier = min(1.0, self.config.volatility_target / volatility)
            else:
                size_multiplier = 1.0
        else:
            size_multiplier = 1.0

        base_size = int(portfolio_value * self.config.base_fraction / 100 * size_multiplier)
        size = max(self.config.min_size, min(base_size, self.config.max_size))

        return size


@dataclass
class RiskParityConfig:
    max_total_exposure: float = 1.0
    max_single_position: float = 0.2
    min_size: int = 1
    max_size: int = 100


class RiskParityStrategy(PositionSizingStrategy):
    def __init__(self, config: Optional[RiskParityConfig] = None):
        self.config = config or RiskParityConfig()
        self.position_sizes: Dict[str, int] = {}

    def calculate_size(
        self,
        opportunity: Dict[str, Any],
        portfolio_stats: Dict[str, Any],
        orderbook: Optional[OrderBook] = None
    ) -> int:
        market_id = opportunity.get('market_id_1', 'unknown')
        portfolio_value = portfolio_stats.get('cash_balance', 10000)

        current_exposure = sum(self.position_sizes.values())
        available_exposure = (self.config.max_total_exposure - current_exposure / portfolio_value) * portfolio_value

        max_single = portfolio_value * self.config.max_single_position

        base_size = min(available_exposure, max_single)

        size = max(self.config.min_size, min(int(base_size), self.config.max_size))

        self.position_sizes[market_id] = self.position_sizes.get(market_id, 0) + size

        return size


class MachineLearningStrategy(PositionSizingStrategy):
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_history: List[Dict[str, float]] = []
        self.model_path = model_path

    def load_model(self) -> None:
        try:
            import joblib
            if self.model_path:
                self.model = joblib.load(self.model_path)
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")

    def extract_features(
        self,
        opportunity: Dict[str, Any],
        orderbook: Optional[OrderBook],
        portfolio_stats: Dict[str, Any]
    ) -> Dict[str, float]:
        features = {
            'confidence': opportunity.get('confidence', 0.5),
            'profit_percent': opportunity.get('profit_percent', 0),
            'net_profit_cents': opportunity.get('net_profit_cents', 0),
            'liquidity_score': 0,
            'fill_probability': 0,
            'spread_percent': 0,
            'portfolio_utilization': 0,
            'time_since_last_trade': 0
        }

        if orderbook:
            features['liquidity_score'] = orderbook.get_liquidity_score()
            features['spread_percent'] = orderbook.get_spread_percent() or 0

        features['portfolio_utilization'] = portfolio_stats.get('open_positions', 0) / 50

        return features

    def calculate_size(
        self,
        opportunity: Dict[str, Any],
        portfolio_stats: Dict[str, Any],
        orderbook: Optional[OrderBook] = None
    ) -> int:
        features = self.extract_features(opportunity, orderbook, portfolio_stats)
        self.feature_history.append(features)

        if self.model is not None:
            try:
                import numpy as np
                feature_array = np.array(list(features.values())).reshape(1, -1)
                risk_score = self.model.predict(feature_array)[0]

                base_size = portfolio_stats.get('cash_balance', 10000) * 0.1
                adjusted_size = base_size * (1 - risk_score)

                return max(1, min(int(adjusted_size), 100))
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")

        confidence = features['confidence']
        profit = features['profit_percent']

        size = int(min(100, confidence * profit * 10))

        return max(1, size)


class CompositeStrategy(PositionSizingStrategy):
    def __init__(
        self,
        strategies: List[PositionSizingStrategy],
        weights: Optional[List[float]] = None
    ):
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)

    def calculate_size(
        self,
        opportunity: Dict[str, Any],
        portfolio_stats: Dict[str, Any],
        orderbook: Optional[OrderBook] = None
    ) -> int:
        sizes = []
        for strategy in self.strategies:
            size = strategy.calculate_size(opportunity, portfolio_stats, orderbook)
            sizes.append(size)

        weighted_size = sum(s * w for s, w in zip(sizes, self.weights))
        return int(weighted_size)


def create_position_sizer(
    strategy_name: str,
    config: Optional[Dict[str, Any]] = None
) -> PositionSizingStrategy:
    strategies = {
        'kelly': lambda: KellyCriterion(KellyConfig(**config)) if config else KellyCriterion(),
        'fixed_fraction': lambda: FixedFractionStrategy(FixedFractionConfig(**config)) if config else FixedFractionStrategy(),
        'volatility': lambda: VolatilityAdjustedStrategy(VolatilityAdjustedConfig(**config)) if config else VolatilityAdjustedStrategy(),
        'risk_parity': lambda: RiskParityStrategy(RiskParityConfig(**config)) if config else RiskParityStrategy(),
        'ml': lambda: MachineLearningStrategy()
    }

    factory = strategies.get(strategy_name, strategies['fixed_fraction'])
    return factory()
