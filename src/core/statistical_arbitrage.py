"""Statistical arbitrage strategies implementation.

This module implements statistical arbitrage strategies including:
- Mean reversion using z-score analysis
- Pairs trading with hedge ratios
- Statistical arbitrage opportunities detection
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from src.utils.logging_utils import get_logger
from src.core.orderbook import OrderBook, OrderBookLevel
from src.utils.database import Database

logger = get_logger("statistical_arbitrage")


class StatisticalArbitrageType(Enum):
    """Types of statistical arbitrage strategies."""

    MEAN_REVERSION = "mean_reversion"
    PAIRS_TRADING = "pairs_trading"
    CORRELATION_REVERSION = "correlation_reversion"
    MOMENTUM_REVERSAL = "momentum_reversal"


@dataclass
class StatisticalArbitrageOpportunity:
    """Statistical arbitrage opportunity with full metadata."""

    id: str
    type: StatisticalArbitrageType
    market_id_1: str
    market_id_2: Optional[str] = None
    strategy_signal: str = ""

    # Price and ratio information
    current_price_1: float = 0.0
    current_price_2: float = 0.0
    expected_price_1: float = 0.0
    expected_price_2: float = 0.0
    hedge_ratio: float = 1.0

    # Statistical metrics
    z_score: float = 0.0
    correlation: float = 0.0
    mean_reversion_target: float = 0.0
    confidence: float = 0.0

    # Trade parameters
    quantity_1: int = 0
    quantity_2: int = 0
    entry_price_1: int = 0
    entry_price_2: int = 0
    target_price_1: int = 0
    target_price_2: int = 0

    # Risk and profit
    expected_profit_cents: int = 0
    risk_score: float = 0.0
    max_loss_cents: int = 0
    holding_period_hours: int = 24

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    market_data_points: int = 0
    statistical_p_value: float = 0.0

    @property
    def is_profitable(self) -> bool:
        return self.expected_profit_cents > 0

    @property
    def profit_margin_percent(self) -> float:
        if self.entry_price_1 > 0:
            return (
                self.expected_profit_cents / (self.entry_price_1 * self.quantity_1)
            ) * 100
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "type": self.type.value,
            "market_id_1": self.market_id_1,
            "market_id_2": self.market_id_2,
            "strategy_signal": self.strategy_signal,
            "current_price_1": self.current_price_1,
            "current_price_2": self.current_price_2,
            "expected_price_1": self.expected_price_1,
            "expected_price_2": self.expected_price_2,
            "hedge_ratio": self.hedge_ratio,
            "z_score": self.z_score,
            "correlation": self.correlation,
            "mean_reversion_target": self.mean_reversion_target,
            "confidence": self.confidence,
            "quantity_1": self.quantity_1,
            "quantity_2": self.quantity_2,
            "entry_price_1": self.entry_price_1,
            "entry_price_2": self.entry_price_2,
            "target_price_1": self.target_price_1,
            "target_price_2": self.target_price_2,
            "expected_profit_cents": self.expected_profit_cents,
            "risk_score": self.risk_score,
            "max_loss_cents": self.max_loss_cents,
            "holding_period_hours": self.holding_period_hours,
            "timestamp": self.timestamp,
            "market_data_points": self.market_data_points,
            "statistical_p_value": self.statistical_p_value,
            "profit_margin_percent": self.profit_margin_percent,
        }


class StatisticalArbitrageBase:
    """Base class for statistical arbitrage strategies."""

    def __init__(
        self,
        lookback_period_days: int = 30,
        min_confidence: float = 0.7,
        max_positions: int = 10,
        risk_free_rate: float = 0.02,
        db: Optional[Database] = None,
    ):
        self.lookback_period_days = lookback_period_days
        self.min_confidence = min_confidence
        self.max_positions = max_positions
        self.risk_free_rate = risk_free_rate
        self.db = db or Database()

        # Price history storage
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.volume_history: Dict[str, List[Tuple[datetime, int]]] = {}

        # Statistical parameters
        self.correlations: Dict[Tuple[str, str], float] = {}
        self.hedge_ratios: Dict[Tuple[str, str], float] = {}
        self.volatility: Dict[str, float] = {}

        logger.info(
            f"Initialized StatisticalArbitrageBase with lookback={lookback_period_days} days"
        )

    def update_price_history(
        self, market_id: str, price: float, volume: int = 0
    ) -> None:
        """Update price history for a market."""
        timestamp = datetime.utcnow()

        if market_id not in self.price_history:
            self.price_history[market_id] = []
            self.volume_history[market_id] = []

        self.price_history[market_id].append((timestamp, price))
        self.volume_history[market_id].append((timestamp, volume))

        # Keep only data within lookback period
        cutoff_time = timestamp - timedelta(days=self.lookback_period_days)
        self.price_history[market_id] = [
            (ts, p) for ts, p in self.price_history[market_id] if ts >= cutoff_time
        ]
        self.volume_history[market_id] = [
            (ts, v) for ts, v in self.volume_history[market_id] if ts >= cutoff_time
        ]

    def calculate_returns(self, market_id: str) -> List[float]:
        """Calculate historical returns for a market."""
        if (
            market_id not in self.price_history
            or len(self.price_history[market_id]) < 2
        ):
            return []

        prices = [p for _, p in self.price_history[market_id]]
        returns = []

        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i - 1]) / prices[i - 1]
            returns.append(ret)

        return returns

    def calculate_volatility(self, market_id: str) -> float:
        """Calculate annualized volatility for a market."""
        returns = self.calculate_returns(market_id)
        if not returns:
            return 0.0

        daily_vol = np.std(returns)
        annualized_vol = daily_vol * np.sqrt(252)  # Trading days per year
        return annualized_vol

    def calculate_correlation(self, market_id_1: str, market_id_2: str) -> float:
        """Calculate correlation between two markets."""
        returns_1 = self.calculate_returns(market_id_1)
        returns_2 = self.calculate_returns(market_id_2)

        if len(returns_1) != len(returns_2) or len(returns_1) < 10:
            return 0.0

        correlation = np.corrcoef(returns_1, returns_2)[0, 1]
        key = (market_id_1, market_id_2)
        self.correlations[key] = correlation
        self.correlations[(market_id_2, market_id_1)] = correlation

        return correlation

    def calculate_hedge_ratio(self, market_id_1: str, market_id_2: str) -> float:
        """Calculate optimal hedge ratio for pairs trading."""
        returns_1 = self.calculate_returns(market_id_1)
        returns_2 = self.calculate_returns(market_id_2)

        if len(returns_1) != len(returns_2) or len(returns_1) < 20:
            return 1.0

        # Use linear regression to find hedge ratio
        x = np.array(returns_1).reshape(-1, 1)
        y = np.array(returns_2)

        # Calculate hedge ratio as slope of regression
        hedge_ratio = np.linalg.lstsq(x, y, rcond=None)[0][0]

        key = (market_id_1, market_id_2)
        self.hedge_ratios[key] = hedge_ratio

        return hedge_ratio

    def get_latest_price(self, market_id: str) -> Optional[float]:
        """Get the latest price for a market."""
        if market_id not in self.price_history or not self.price_history[market_id]:
            return None
        return self.price_history[market_id][-1][
            1
        ]  # Return the price part of (timestamp, price)

    def calculate_z_score(self, market_id: str, price: float) -> float:
        """Calculate z-score for current price vs historical mean."""
        if (
            market_id not in self.price_history
            or len(self.price_history[market_id]) < 10
        ):
            return 0.0

        prices = [p for _, p in self.price_history[market_id]]
        mean_price = np.mean(prices)
        std_price = np.std(prices)

        if std_price == 0:
            return 0.0

        z_score = (price - mean_price) / std_price
        return z_score

    def get_market_data_points(self, market_id: str) -> int:
        """Get number of data points for a market."""
        return len(self.price_history.get(market_id, []))

    def calculate_confidence(
        self, z_score: float, correlation: float, data_points: int, volatility: float
    ) -> float:
        """Calculate confidence score for statistical arbitrage."""
        # Base confidence from data quality
        data_confidence = min(
            1.0, data_points / 30.0
        )  # Full confidence at 30+ data points

        # Confidence from statistical significance
        z_confidence = min(
            1.0, abs(z_score) / 2.0
        )  # Higher z-score = higher confidence

        # Confidence from correlation strength
        corr_confidence = abs(correlation)

        # Confidence from stability (inverse of volatility)
        vol_confidence = max(0.1, 1.0 - min(1.0, volatility * 50))

        # Combine confidences
        combined_confidence = (
            data_confidence * 0.3
            + z_confidence * 0.3
            + corr_confidence * 0.2
            + vol_confidence * 0.2
        )

        return min(1.0, max(0.0, combined_confidence))


class MeanReversionStrategy(StatisticalArbitrageBase):
    """Mean reversion strategy using z-score analysis."""

    def __init__(self, z_score_threshold: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.z_score_threshold = z_score_threshold
        self.mean_reversion_window = 20  # Window for mean calculation

    def find_opportunities(
        self, orderbooks: Dict[str, OrderBook], min_profit_cents: int = 10
    ) -> List[StatisticalArbitrageOpportunity]:
        """Find mean reversion opportunities across all markets."""
        opportunities = []

        for market_id, orderbook in orderbooks.items():
            if not orderbook or not orderbook.mid_price:
                continue

            current_price = orderbook.get_mid_price()
            z_score = self.calculate_z_score(market_id, current_price)

            # Check if price deviates significantly from mean
            if abs(z_score) >= self.z_score_threshold:
                opp = self._create_mean_reversion_opportunity(
                    market_id, current_price, z_score
                )

                if opp and opp.expected_profit_cents >= min_profit_cents:
                    opportunities.append(opp)

        logger.info(f"Mean reversion: Found {len(opportunities)} opportunities")
        return opportunities

    def _create_mean_reversion_opportunity(
        self, market_id: str, current_price: float, z_score: float
    ) -> Optional[StatisticalArbitrageOpportunity]:
        """Create a mean reversion opportunity."""

        # Get historical prices for target calculation
        if market_id not in self.price_history:
            return None

        prices = [p for _, p in self.price_history[market_id]]
        if len(prices) < self.mean_reversion_window:
            return None

        # Calculate target (reversion to recent mean)
        recent_prices = prices[-self.mean_reversion_window :]
        target_price = np.mean(recent_prices)

        # Determine trade direction
        if z_score > 0:  # Price is overvalued -> short
            entry_price = int(current_price * 100)  # Convert to cents
            target_price = int(target_price * 100)
            quantity = 100  # Default position size
            expected_profit = (entry_price - target_price) * quantity
            signal = "overvalued_short"
        else:  # Price is undervalued -> long
            entry_price = int(current_price * 100)
            target_price = int(target_price * 100)
            quantity = 100
            expected_profit = (target_price - entry_price) * quantity
            signal = "undervalued_long"

        # Calculate risk
        volatility = self.calculate_volatility(market_id)
        max_loss = int(abs(expected_profit) * 2)  # Risk 2x expected loss

        confidence = self.calculate_confidence(
            abs(z_score), 1.0, len(prices), volatility
        )

        return StatisticalArbitrageOpportunity(
            id=f"mr_{market_id}_{int(datetime.utcnow().timestamp())}",
            type=StatisticalArbitrageType.MEAN_REVERSION,
            market_id_1=market_id,
            strategy_signal=signal,
            current_price_1=current_price,
            expected_price_1=target_price,
            z_score=z_score,
            mean_reversion_target=target_price,
            confidence=confidence,
            quantity_1=quantity,
            entry_price_1=entry_price,
            target_price_1=target_price,
            expected_profit_cents=expected_profit,
            max_loss_cents=max_loss,
            risk_score=volatility,
            market_data_points=len(prices),
            statistical_p_value=max(
                0.001, (1 - abs(z_score) * 0.1)
            ),  # Approximate p-value
        )


class PairsTradingStrategy(StatisticalArbitrageBase):
    """Pairs trading strategy with statistical hedge ratios."""

    def __init__(self, min_correlation: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.min_correlation = min_correlation
        self.pair_universe = self._create_pair_universe()

    def _create_pair_universe(self) -> List[Tuple[str, str]]:
        """Create pairs universe based on market relationships."""
        # This should be customized based on actual market relationships
        # For now, return empty - will be populated dynamically
        return []

    def find_opportunities(
        self, orderbooks: Dict[str, OrderBook], min_profit_cents: int = 10
    ) -> List[StatisticalArbitrageOpportunity]:
        """Find pairs trading opportunities."""
        opportunities = []

        # Update price history from orderbooks
        for market_id, orderbook in orderbooks.items():
            if orderbook and orderbook.mid_price:
                mid_price = orderbook.get_mid_price()
                # Estimate volume from orderbook depth
                volume = self._estimate_volume_from_orderbook(orderbook)
                self.update_price_history(market_id, mid_price, volume)

        # Find opportunities for each pair
        for market_id_1, market_id_2 in self.pair_universe:
            opp = self._analyze_pair(market_id_1, market_id_2, orderbooks)

            if opp and opp.expected_profit_cents >= min_profit_cents:
                opportunities.append(opp)

        logger.info(f"Pairs trading: Found {len(opportunities)} opportunities")
        return opportunities

    def _estimate_volume_from_orderbook(self, orderbook: OrderBook) -> int:
        """Estimate volume from orderbook depth."""
        if not orderbook.bids or not orderbook.asks:
            return 0

        bid_volume = sum(level.count for level in orderbook.bids[:5])
        ask_volume = sum(level.count for level in orderbook.asks[:5])
        return (bid_volume + ask_volume) // 2

    def _analyze_pair(
        self, market_id_1: str, market_id_2: str, orderbooks: Dict[str, OrderBook]
    ) -> Optional[StatisticalArbitrageOpportunity]:
        """Analyze a specific pair for arbitrage opportunity."""

        if (
            market_id_1 not in orderbooks
            or market_id_2 not in orderbooks
            or not orderbooks[market_id_1]
            or not orderbooks[market_id_2]
        ):
            return None

        # Calculate statistics
        correlation = self.calculate_correlation(market_id_1, market_id_2)
        hedge_ratio = self.calculate_hedge_ratio(market_id_1, market_id_2)

        if abs(correlation) < self.min_correlation:
            return None

        # Get current prices
        price_1 = orderbooks[market_id_1].get_mid_price()
        price_2 = orderbooks[market_id_2].get_mid_price()

        if not price_1 or not price_2:
            return None

        # Calculate expected prices based on historical relationship
        recent_prices_1 = [p for _, p in self.price_history.get(market_id_1, [])[-10:]]
        recent_prices_2 = [p for _, p in self.price_history.get(market_id_2, [])[-10:]]

        if len(recent_prices_1) < 5 or len(recent_prices_2) < 5:
            return None

        # Calculate spread deviation
        current_spread = price_1 - (price_2 * hedge_ratio)
        historical_spreads = [
            p1 - (p2 * hedge_ratio) for p1, p2 in zip(recent_prices_1, recent_prices_2)
        ]

        mean_spread = np.mean(historical_spreads)
        std_spread = np.std(historical_spreads)

        # Check if current spread deviates significantly
        if std_spread == 0:
            return None

        spread_z = (current_spread - mean_spread) / std_spread

        if abs(spread_z) < 1.5:  # Threshold for spread deviation
            return None

        # Create opportunity
        expected_spread = mean_spread  # Expected reversion
        quantity_1 = 100
        quantity_2 = int(quantity_1 * hedge_ratio)

        # Expected profit when spread reverts
        expected_profit = int(abs(current_spread - expected_spread) * quantity_1 * 100)

        # Risk assessment
        vol_1 = self.calculate_volatility(market_id_1)
        vol_2 = self.calculate_volatility(market_id_2)
        avg_volatility = (vol_1 + vol_2) / 2

        confidence = self.calculate_confidence(
            abs(spread_z),
            abs(correlation),
            min(len(recent_prices_1), len(recent_prices_2)),
            avg_volatility,
        )

        signal = (
            "spread_widening"
            if current_spread > expected_spread
            else "spread_narrowing"
        )

        return StatisticalArbitrageOpportunity(
            id=f"pair_{market_id_1}_{market_id_2}_{int(datetime.utcnow().timestamp())}",
            type=StatisticalArbitrageType.PAIRS_TRADING,
            market_id_1=market_id_1,
            market_id_2=market_id_2,
            strategy_signal=signal,
            current_price_1=price_1,
            current_price_2=price_2,
            expected_price_1=price_1 - (current_spread - expected_spread),
            expected_price_2=price_2 + (current_spread - expected_spread) / hedge_ratio,
            hedge_ratio=hedge_ratio,
            correlation=correlation,
            z_score=spread_z,
            confidence=confidence,
            quantity_1=quantity_1,
            quantity_2=quantity_2,
            entry_price_1=int(price_1 * 100),
            entry_price_2=int(price_2 * 100),
            expected_profit_cents=expected_profit,
            risk_score=avg_volatility,
            max_loss_cents=expected_profit * 2,
            market_data_points=min(len(recent_prices_1), len(recent_prices_2)),
            statistical_p_value=max(0.001, (1 - abs(spread_z) * 0.1)),
        )


class StatisticalArbitrageDetector:
    """Main detector for statistical arbitrage opportunities."""

    def __init__(
        self,
        strategies: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or {}

        # Initialize enabled strategies
        self.strategies = {}
        if not strategies:
            strategies = ["mean_reversion", "pairs_trading"]

        if "mean_reversion" in strategies:
            z_threshold = self.config.get("statistical.mean_reversion.z_threshold", 2.0)
            self.strategies["mean_reversion"] = MeanReversionStrategy(
                z_score_threshold=z_threshold, **self.config.get("statistical", {})
            )

        if "pairs_trading" in strategies:
            min_corr = self.config.get("statistical.pairs_trading.min_correlation", 0.7)
            self.strategies["pairs_trading"] = PairsTradingStrategy(
                min_correlation=min_corr, **self.config.get("statistical", {})
            )

        logger.info(
            f"Initialized statistical arbitrage with strategies: {list(self.strategies.keys())}"
        )

    def update_price_history(self, orderbooks: Dict[str, OrderBook]) -> None:
        """Update price history for all markets."""
        for strategy in self.strategies.values():
            for market_id, orderbook in orderbooks.items():
                if orderbook and orderbook.mid_price:
                    mid_price = orderbook.get_mid_price()
                    volume = self._estimate_volume(orderbook)
                    strategy.update_price_history(market_id, mid_price, volume)

    def _estimate_volume(self, orderbook: OrderBook) -> int:
        """Estimate trading volume from orderbook."""
        if not orderbook.bids or not orderbook.asks:
            return 0
        bid_volume = sum(level.count for level in orderbook.bids[:3])
        ask_volume = sum(level.count for level in orderbook.asks[:3])
        return (bid_volume + ask_volume) // 2

    def find_opportunities(
        self, orderbooks: Dict[str, OrderBook], min_profit_cents: int = 10
    ) -> List[StatisticalArbitrageOpportunity]:
        """Find statistical arbitrage opportunities."""
        self.update_price_history(orderbooks)

        all_opportunities = []
        for strategy_name, strategy in self.strategies.items():
            try:
                opportunities = strategy.find_opportunities(
                    orderbooks, min_profit_cents
                )
                all_opportunities.extend(opportunities)
                logger.info(
                    f"{strategy_name}: Found {len(opportunities)} opportunities"
                )
            except Exception as e:
                logger.error(f"Error in {strategy_name} strategy: {e}")

        # Rank by confidence and expected profit
        all_opportunities.sort(
            key=lambda x: (x.confidence * x.expected_profit_cents), reverse=True
        )

        logger.info(
            f"Statistical arbitrage: Found {len(all_opportunities)} total opportunities"
        )
        return all_opportunities[:20]  # Return top 20 opportunities

    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get statistics for each strategy."""
        stats = {}
        for name, strategy in self.strategies.items():
            stats[name] = {
                "price_history_markets": len(strategy.price_history),
                "total_data_points": sum(
                    len(prices) for prices in strategy.price_history.values()
                ),
                "correlations_calculated": len(strategy.correlations),
                "hedge_ratios_calculated": len(strategy.hedge_ratios),
            }
        return stats
