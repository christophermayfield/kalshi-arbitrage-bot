"""
Auto Pairs Discovery - Automatically find correlated markets for pairs trading.
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

from src.utils.logging_utils import get_logger

logger = get_logger("pairs_discovery")


@dataclass
class MarketPair:
    market_id_1: str
    market_id_2: str
    correlation: float
    hedge_ratio: float
    last_updated: datetime


class PairsDiscovery:
    def __init__(
        self,
        min_correlation: float = 0.7,
        lookback_period: int = 100,
        update_interval: int = 300,
    ):
        self.min_correlation = min_correlation
        self.lookback_period = lookback_period
        self.update_interval = update_interval
        self._price_history: Dict[str, List[float]] = defaultdict(list)
        self._pairs: Dict[Tuple[str, str], MarketPair] = {}
        self._last_update: Optional[datetime] = None

    def add_price(self, market_id: str, price: float) -> None:
        """Add a price observation for a market."""
        self._price_history[market_id].append(price)
        if len(self._price_history[market_id]) > self.lookback_period:
            self._price_history[market_id] = self._price_history[market_id][
                -self.lookback_period :
            ]

    def discover_pairs(self, market_ids: List[str]) -> List[MarketPair]:
        """Discover correlated market pairs."""
        discovered_pairs = []

        for i, m1 in enumerate(market_ids):
            for m2 in market_ids[i + 1 :]:
                if m1 not in self._price_history or m2 not in self._price_history:
                    continue

                prices1 = np.array(self._price_history[m1])
                prices2 = np.array(self._price_history[m2])

                if len(prices1) < 30 or len(prices2) < 30:
                    continue

                correlation = self._calculate_correlation(prices1, prices2)

                if abs(correlation) >= self.min_correlation:
                    hedge_ratio = self._calculate_hedge_ratio(prices1, prices2)

                    pair = MarketPair(
                        market_id_1=m1,
                        market_id_2=m2,
                        correlation=correlation,
                        hedge_ratio=hedge_ratio,
                        last_updated=datetime.utcnow(),
                    )
                    discovered_pairs.append(pair)
                    self._pairs[(m1, m2)] = pair

        self._last_update = datetime.utcnow()
        logger.info(f"Discovered {len(discovered_pairs)} correlated pairs")

        return discovered_pairs

    def _calculate_correlation(self, prices1: np.ndarray, prices2: np.ndarray) -> float:
        """Calculate Pearson correlation between two price series."""
        if len(prices1) != len(prices2):
            min_len = min(len(prices1), len(prices2))
            prices1 = prices1[-min_len:]
            prices2 = prices2[-min_len:]

        if np.std(prices1) == 0 or np.std(prices2) == 0:
            return 0.0

        return np.corrcoef(prices1, prices2)[0, 1]

    def _calculate_hedge_ratio(self, prices1: np.ndarray, prices2: np.ndarray) -> float:
        """Calculate hedge ratio using linear regression."""
        from sklearn.linear_model import LinearRegression

        if len(prices1) != len(prices2):
            min_len = min(len(prices1), len(prices2))
            prices1 = prices1[-min_len:]
            prices2 = prices2[-min_len:]

        if len(prices1) < 10:
            return 1.0

        X = prices1.reshape(-1, 1)
        y = prices2

        model = LinearRegression()
        model.fit(X, y)

        return model.coef_[0]

    def get_pairs_for_market(self, market_id: str) -> List[MarketPair]:
        """Get all pairs involving a specific market."""
        pairs = []
        for pair in self._pairs.values():
            if pair.market_id_1 == market_id or pair.market_id_2 == market_id:
                pairs.append(pair)
        return sorted(pairs, key=lambda p: abs(p.correlation), reverse=True)

    def get_spread(self, pair: MarketPair) -> Optional[float]:
        """Calculate the current spread between a pair."""
        if (
            pair.market_id_1 not in self._price_history
            or pair.market_id_2 not in self._price_history
        ):
            return None

        prices1 = self._price_history[pair.market_id_1]
        prices2 = self._price_history[pair.market_id_2]

        if not prices1 or not prices2:
            return None

        current1 = prices1[-1]
        current2 = prices2[-1]

        spread = current2 - pair.hedge_ratio * current1
        return spread

    def get_z_score(self, pair: MarketPair, window: int = 20) -> Optional[float]:
        """Calculate the z-score of the spread."""
        if (
            pair.market_id_1 not in self._price_history
            or pair.market_id_2 not in self._price_history
        ):
            return None

        prices1 = np.array(self._price_history[pair.market_id_1][-window:])
        prices2 = np.array(self._price_history[pair.market_id_2][-window:])

        if len(prices1) < window or len(prices2) < window:
            return None

        spread = prices2 - pair.hedge_ratio * prices1
        mean = np.mean(spread)
        std = np.std(spread)

        if std == 0:
            return 0.0

        current_spread = prices2[-1] - pair.hedge_ratio * prices1[-1]
        z_score = (current_spread - mean) / std

        return z_score
