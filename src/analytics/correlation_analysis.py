"""
Correlation Analysis - Real-time correlation matrix between markets.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from src.utils.logging_utils import get_logger

logger = get_logger("correlation")


@dataclass
class CorrelationPair:
    market_1: str
    market_2: str
    correlation: float
    hedge_ratio: float
    spread_std: float
    last_updated: datetime


class CorrelationAnalyzer:
    def __init__(
        self,
        min_correlation: float = 0.5,
        lookback_period: int = 100,
    ):
        self.min_correlation = min_correlation
        self.lookback_period = lookback_period
        self._price_history: Dict[str, List[float]] = defaultdict(list)
        self._timestamps: Dict[str, List[datetime]] = defaultdict(list)
        self._correlations: Dict[str, Dict[str, CorrelationPair]] = defaultdict(dict)

    def add_price(
        self, market_id: str, price: float, timestamp: Optional[datetime] = None
    ) -> None:
        """Add a price observation."""
        self._price_history[market_id].append(price)
        self._timestamps[market_id].append(timestamp or datetime.utcnow())

        if len(self._price_history[market_id]) > self.lookback_period:
            self._price_history[market_id] = self._price_history[market_id][
                -self.lookback_period :
            ]
            self._timestamps[market_id] = self._timestamps[market_id][
                -self.lookback_period :
            ]

    def calculate_correlation(self, market_1: str, market_2: str) -> Optional[float]:
        """Calculate Pearson correlation between two markets."""
        if market_1 not in self._price_history or market_2 not in self._price_history:
            return None

        prices1 = np.array(self._price_history[market_1])
        prices2 = np.array(self._price_history[market_2])

        min_len = min(len(prices1), len(prices2))
        if min_len < 10:
            return None

        prices1 = prices1[-min_len:]
        prices2 = prices2[-min_len:]

        if np.std(prices1) == 0 or np.std(prices2) == 0:
            return 0.0

        return np.corrcoef(prices1, prices2)[0, 1]

    def calculate_hedge_ratio(self, market_1: str, market_2: str) -> Optional[float]:
        """Calculate hedge ratio using OLS."""
        if market_1 not in self._price_history or market_2 not in self._price_history:
            return None

        prices1 = np.array(self._price_history[market_1])
        prices2 = np.array(self._price_history[market_2])

        min_len = min(len(prices1), len(prices2))
        if min_len < 10:
            return None

        prices1 = prices1[-min_len:].reshape(-1, 1)
        prices2 = prices2[-min_len:]

        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(prices1, prices2)

        return model.coef_[0]

    def calculate_spread_std(
        self, market_1: str, market_2: str, hedge_ratio: float
    ) -> Optional[float]:
        """Calculate standard deviation of the spread."""
        if market_1 not in self._price_history or market_2 not in self._price_history:
            return None

        prices1 = np.array(self._price_history[market_1])
        prices2 = np.array(self._price_history[market_2])

        min_len = min(len(prices1), len(prices2))
        if min_len < 10:
            return None

        prices1 = prices1[-min_len:]
        prices2 = prices2[-min_len:]

        spread = prices2 - hedge_ratio * prices1
        return float(np.std(spread))

    def update_correlations(self, markets: Optional[List[str]] = None) -> None:
        """Update correlation matrix for all market pairs."""
        if markets is None:
            markets = list(self._price_history.keys())

        for i, m1 in enumerate(markets):
            for m2 in markets[i + 1 :]:
                correlation = self.calculate_correlation(m1, m2)

                if correlation is None or abs(correlation) < self.min_correlation:
                    continue

                hedge_ratio = self.calculate_hedge_ratio(m1, m2) or 1.0
                spread_std = self.calculate_spread_std(m1, m2, hedge_ratio) or 0.0

                pair = CorrelationPair(
                    market_1=m1,
                    market_2=m2,
                    correlation=correlation,
                    hedge_ratio=hedge_ratio,
                    spread_std=spread_std,
                    last_updated=datetime.utcnow(),
                )

                self._correlations[m1][m2] = pair
                self._correlations[m2][m1] = pair

    def get_correlation(
        self, market_1: str, market_2: str
    ) -> Optional[CorrelationPair]:
        """Get correlation between two markets."""
        if market_1 in self._correlations and market_2 in self._correlations[market_1]:
            return self._correlations[market_1][market_2]
        return None

    def get_top_correlations(
        self, market_id: str, limit: int = 10
    ) -> List[CorrelationPair]:
        """Get top correlations for a market."""
        if market_id not in self._correlations:
            return []

        pairs = list(self._correlations[market_id].values())
        pairs.sort(key=lambda x: abs(x.correlation), reverse=True)
        return pairs[:limit]

    def get_correlation_matrix(
        self, markets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get full correlation matrix."""
        if markets is None:
            markets = list(self._price_history.keys())

        matrix = {}
        for m1 in markets:
            matrix[m1] = {}
            for m2 in markets:
                if m1 == m2:
                    matrix[m1][m2] = 1.0
                else:
                    corr = self.calculate_correlation(m1, m2)
                    matrix[m1][m2] = corr if corr is not None else 0.0

        return matrix

    def find_cointegrated_pairs(self, threshold: float = 0.7) -> List[CorrelationPair]:
        """Find pairs with high correlation (potential cointegration)."""
        pairs = []

        for m1, correlations in self._correlations.items():
            for m2, pair in correlations.items():
                if m1 < m2 and abs(pair.correlation) >= threshold:
                    pairs.append(pair)

        pairs.sort(key=lambda x: abs(x.correlation), reverse=True)
        return pairs

    def get_stats(self) -> Dict[str, Any]:
        """Get correlation analyzer stats."""
        return {
            "markets_tracked": len(self._price_history),
            "pairs_tracked": sum(len(v) for v in self._correlations.values()) // 2,
            "total_observations": sum(len(v) for v in self._price_history.values()),
        }
