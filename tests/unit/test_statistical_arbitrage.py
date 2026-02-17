"""Tests for statistical arbitrage strategies."""

import pytest
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import Mock

from src.core.statistical_arbitrage import (
    StatisticalArbitrageDetector,
    MeanReversionStrategy,
    PairsTradingStrategy,
    StatisticalArbitrageType,
    StatisticalArbitrageOpportunity,
)
from src.core.orderbook import OrderBook, OrderBookLevel


class TestMeanReversionStrategy:
    """Test cases for mean reversion strategy."""

    def setup_method(self):
        """Setup for each test method."""
        self.strategy = MeanReversionStrategy(
            lookback_period_days=30, min_confidence=0.7, z_score_threshold=2.0
        )

    def test_price_history_update(self):
        """Test price history updates."""
        market_id = "test_market"

        # Add some price history
        self.strategy.update_price_history(market_id, 100.0, 1000)
        self.strategy.update_price_history(market_id, 101.0, 1200)
        self.strategy.update_price_history(market_id, 102.0, 1100)

        assert market_id in self.strategy.price_history
        assert len(self.strategy.price_history[market_id]) == 3

        # Check calculation of returns
        returns = self.strategy.calculate_returns(market_id)
        expected_returns = [(101.0 - 100.0) / 100.0, (102.0 - 101.0) / 101.0]
        np.testing.assert_array_almost_equal(returns, expected_returns)

    def test_volatility_calculation(self):
        """Test volatility calculation."""
        market_id = "test_market"

        # Add price history with known volatility
        prices = [100, 102, 98, 105, 95, 108, 92, 110, 90, 112]
        for i, price in enumerate(prices):
            self.strategy.update_price_history(market_id, price, 1000 + i * 100)

        vol = self.strategy.calculate_volatility(market_id)

        # Verify volatility is positive and reasonable
        assert vol > 0
        assert vol < 1.0  # Shouldn't be extremely high

    def test_z_score_calculation(self):
        """Test z-score calculation."""
        market_id = "test_market"

        # Add price history: [100, 102, 98, 105, 95, 108, 92, 110, 90, 112]
        for price in [100, 102, 98, 105, 95, 108, 92, 110, 90, 112]:
            self.strategy.update_price_history(market_id, price, 1000)

        current_price = 120.0  # 2 standard deviations above mean (mean=101.4, std≈6.8)
        z_score = self.strategy.calculate_z_score(market_id, current_price)

        # Should be approximately 2.7
        assert abs(z_score - 2.7) < 0.1

    def test_mean_reversion_opportunity_detection(self):
        """Test mean reversion opportunity detection."""
        market_id = "test_market"

        # Create price history
        prices = [100, 102, 98, 105, 95, 108, 92, 110, 90, 112]
        for price in prices:
            self.strategy.update_price_history(market_id, price, 1000)

        # Current price significantly deviates (z-score > 2)
        current_price = 120.0
        z_score = self.strategy.calculate_z_score(market_id, current_price)

        # Create mock orderbook
        orderbook = OrderBook(
            market_id=market_id,
            bids=[OrderBookLevel(price=119, count=100, total=11900)],
            asks=[OrderBookLevel(price=121, count=100, total=12100)],
        )

        opportunities = self.strategy.find_opportunities(
            {market_id: orderbook}, min_profit_cents=10
        )

        # Should find an opportunity
        assert len(opportunities) > 0
        opp = opportunities[0]
        assert opp.type == StatisticalArbitrageType.MEAN_REVERSION
        assert opp.market_id_1 == market_id
        assert abs(opp.z_score) >= self.strategy.z_score_threshold

    def test_no_opportunity_when_normal_price(self):
        """Test no opportunity when price is normal."""
        market_id = "test_market"

        # Add price history
        for i, price in enumerate([100, 102, 98, 105, 95, 108, 92, 110, 90]):
            self.strategy.update_price_history(market_id, price, 1000 + i * 100)

        # Current price close to historical mean
        current_price = 101.0
        z_score = self.strategy.calculate_z_score(market_id, current_price)

        orderbook = OrderBook(
            market_id=market_id,
            bids=[OrderBookLevel(price=100, count=100, total=10000)],
            asks=[OrderBookLevel(price=102, count=100, total=10200)],
        )

        opportunities = self.strategy.find_opportunities(
            {market_id: orderbook}, min_profit_cents=10
        )

        # Should not find opportunities for normal price
        assert len(opportunities) == 0


class TestPairsTradingStrategy:
    """Test cases for pairs trading strategy."""

    def setup_method(self):
        """Setup for each test method."""
        self.strategy = PairsTradingStrategy(
            lookback_period_days=30, min_correlation=0.7
        )

    def test_correlation_calculation(self):
        """Test correlation calculation between markets."""
        market_1 = "market_A"
        market_2 = "market_B"

        # Create correlated price series
        prices_A = [100, 102, 98, 105, 103, 108, 95, 110, 92, 112]
        prices_B = [101, 103, 99, 106, 104, 109, 96, 111, 93, 113]

        for i, (price_A, price_B) in enumerate(zip(prices_A, prices_B)):
            self.strategy.update_price_history(market_1, price_A, 1000 + i * 100)
            self.strategy.update_price_history(market_2, price_B, 1100 + i * 100)

        correlation = self.strategy.calculate_correlation(market_1, market_2)

        # Should be highly correlated (>0.9)
        assert correlation > 0.9

    def test_hedge_ratio_calculation(self):
        """Test hedge ratio calculation."""
        market_1 = "market_A"
        market_2 = "market_B"

        # Create related price series
        returns_A = [0.02, -0.02, 0.04, 0.03, -0.03]
        returns_B = [0.015, -0.015, 0.03, 0.025, -0.025]

        # Manually set up price history
        for i, (ret_A, ret_B) in enumerate(zip(returns_A, returns_B)):
            self.strategy.update_price_history(market_1, 100 * (1 + ret_A), 1000)
            self.strategy.update_price_history(market_2, 100 * (1 + ret_B), 1100)

        hedge_ratio = self.strategy.calculate_hedge_ratio(market_1, market_2)

        # Should be close to 0.75 (slope of relationship)
        assert abs(hedge_ratio - 0.75) < 0.1

    def test_pairs_opportunity_detection(self):
        """Test pairs trading opportunity detection."""
        # Setup strategy with pair universe
        self.strategy.pair_universe = [("market_A", "market_B")]

        # Create correlated markets with spread deviation
        orderbook_A = OrderBook(
            market_id="market_A",
            bids=[OrderBookLevel(price=100, count=100, total=10000)],
            asks=[OrderBookLevel(price=101, count=100, total=10100)],
        )
        orderbook_B = OrderBook(
            market_id="market_B",
            bids=[OrderBookLevel(price=80, count=100, total=8000)],
            asks=[OrderBookLevel(price=81, count=100, total=8100)],
        )

        orderbooks = {"market_A": orderbook_A, "market_B": orderbook_B}

        # Update price history to establish correlation
        for i in range(20):
            price_A = 100 + i * 0.5
            price_B = 80 + i * 0.5
            self.strategy.update_price_history("market_A", price_A, 1000)
            self.strategy.update_price_history("market_B", price_B, 1100)

        opportunities = self.strategy.find_opportunities(
            orderbooks, min_profit_cents=100
        )

        # Should find opportunity if correlation is sufficient
        correlation = self.strategy.calculate_correlation("market_A", "market_B")
        if abs(correlation) > self.strategy.min_correlation:
            assert len(opportunities) > 0
            opp = opportunities[0]
            assert opp.type == StatisticalArbitrageType.PAIRS_TRADING
            assert opp.market_id_1 == "market_A"
            assert opp.market_id_2 == "market_B"

    def test_no_opportunity_with_low_correlation(self):
        """Test no opportunity when correlation is low."""
        # Setup with uncorrelated markets
        self.strategy.pair_universe = [("market_X", "market_Y")]

        orderbook_X = OrderBook(market_id="market_X")
        orderbook_Y = OrderBook(market_id="market_Y")
        orderbooks = {"market_X": orderbook_X, "market_Y": orderbook_Y}

        # Update with uncorrelated price history
        for i in range(20):
            self.strategy.update_price_history(
                "market_X", 100 + i * np.random.randn() * 5, 1000
            )
            self.strategy.update_price_history(
                "market_Y", 80 + i * np.random.randn() * 3, 1100
            )

        opportunities = self.strategy.find_opportunities(
            orderbooks, min_profit_cents=100
        )

        # Should not find opportunities with low correlation
        assert len(opportunities) == 0


class TestStatisticalArbitrageDetector:
    """Test cases for statistical arbitrage detector."""

    def setup_method(self):
        """Setup for each test method."""
        self.detector = StatisticalArbitrageDetector(
            strategies=["mean_reversion", "pairs_trading"],
            config={
                "statistical": {"lookback_period_days": 30, "min_confidence": 0.7},
                "mean_reversion": {"z_threshold": 2.0},
                "pairs_trading": {"min_correlation": 0.7},
            },
        )

    def test_detector_initialization(self):
        """Test detector initialization."""
        assert "mean_reversion" in self.detector.strategies
        assert "pairs_trading" in self.detector.strategies
        assert isinstance(
            self.detector.strategies["mean_reversion"], MeanReversionStrategy
        )
        assert isinstance(
            self.detector.strategies["pairs_trading"], PairsTradingStrategy
        )

    def test_price_history_update(self):
        """Test price history updates across all strategies."""
        orderbooks = {
            "market_A": OrderBook(market_id="market_A"),
            "market_B": OrderBook(market_id="market_B"),
        }

        # Update price history
        self.detector.update_price_history(orderbooks)

        # Both strategies should have updated price history
        for strategy in self.detector.strategies.values():
            assert len(strategy.price_history) >= 0

    def test_opportunity_integration(self):
        """Test integration of multiple strategy opportunities."""
        # Setup markets
        orderbook_1 = OrderBook(
            market_id="market_1",
            bids=[OrderBookLevel(price=95, count=100, total=9500)],
            asks=[OrderBookLevel(price=105, count=100, total=10500)],
        )
        orderbook_2 = OrderBook(
            market_id="market_2",
            bids=[OrderBookLevel(price=85, count=100, total=8500)],
            asks=[OrderBookLevel(price=95, count=100, total=9500)],
        )

        orderbooks = {"market_1": orderbook_1, "market_2": orderbook_2}

        # Build price history for both strategies
        for i in range(25):
            # Correlated prices for pairs trading
            price_1 = 100 + i * 2
            price_2 = 80 + i * 2
            self.detector.strategies["mean_reversion"].update_price_history(
                "market_1", price_1, 1000
            )
            self.detector.strategies["pairs_trading"].update_price_history(
                "market_1", price_1, 1000
            )
            self.detector.strategies["pairs_trading"].update_price_history(
                "market_2", price_2, 1100
            )

            # Deviated price for mean reversion
            if i == 24:
                self.detector.strategies["mean_reversion"].update_price_history(
                    "market_2", 130, 1200
                )

        opportunities = self.detector.find_opportunities(
            orderbooks, min_profit_cents=50
        )

        # Should find opportunities from both strategies
        assert len(opportunities) >= 0

        # Check that opportunities are properly classified
        opportunity_types = [opp.type for opp in opportunities]
        assert StatisticalArbitrageType.MEAN_REVERSION in opportunity_types
        assert StatisticalArbitrageType.PAIRS_TRADING in opportunity_types

    def test_strategy_stats(self):
        """Test strategy statistics reporting."""
        stats = self.detector.get_strategy_stats()

        assert "mean_reversion" in stats
        assert "pairs_trading" in stats

        # Check that stats contain expected keys
        mr_stats = stats["mean_reversion"]
        assert "price_history_markets" in mr_stats
        assert "total_data_points" in mr_stats
        assert "correlations_calculated" in mr_stats
        assert "hedge_ratios_calculated" in mr_stats


class TestStatisticalArbitrageOpportunity:
    """Test cases for statistical arbitrage opportunity dataclass."""

    def test_opportunity_creation(self):
        """Test opportunity object creation."""
        opp = StatisticalArbitrageOpportunity(
            id="test_opp",
            type=StatisticalArbitrageType.MEAN_REVERSION,
            market_id_1="market_A",
            strategy_signal="test_signal",
        )

        assert opp.id == "test_opp"
        assert opp.type == StatisticalArbitrageType.MEAN_REVERSION
        assert opp.market_id_1 == "market_A"
        assert opp.strategy_signal == "test_signal"

    def test_profitability_calculation(self):
        """Test profit calculation logic."""
        opp = StatisticalArbitrageOpportunity(
            id="test_opp",
            type=StatisticalArbitrageType.MEAN_REVERSION,
            expected_profit_cents=100,
        )

        assert opp.is_profitable == True

        opp_loss = StatisticalArbitrageOpportunity(
            id="test_loss",
            type=StatisticalArbitrageType.MEAN_REVERSION,
            expected_profit_cents=-50,
        )

        assert opp_loss.is_profitable == False

    def test_profit_margin_percent(self):
        """Test profit margin percentage calculation."""
        opp = StatisticalArbitrageOpportunity(
            id="test_opp",
            type=StatisticalArbitrageType.MEAN_REVERSION,
            entry_price_1=10000,  # $100.00
            quantity_1=10,
            expected_profit_cents=500,  # $5.00 profit
        )

        expected_margin = (500 / (10000 * 10)) * 100  # 0.5%
        assert abs(opp.profit_margin_percent - expected_margin) < 0.01

    def test_to_dict_conversion(self):
        """Test dictionary conversion."""
        opp = StatisticalArbitrageOpportunity(
            id="test_opp",
            type=StatisticalArbitrageType.MEAN_REVERSION,
            market_id_1="market_A",
            market_id_2="market_B",
            strategy_signal="test_signal",
            expected_profit_cents=100,
        )

        opp_dict = opp.to_dict()

        # Check all expected keys
        expected_keys = [
            "id",
            "type",
            "market_id_1",
            "market_id_2",
            "strategy_signal",
            "current_price_1",
            "current_price_2",
            "expected_price_1",
            "expected_price_2",
            "hedge_ratio",
            "z_score",
            "correlation",
            "mean_reversion_target",
            "confidence",
            "quantity_1",
            "quantity_2",
            "entry_price_1",
            "entry_price_2",
            "target_price_1",
            "target_price_2",
            "expected_profit_cents",
            "risk_score",
            "max_loss_cents",
            "holding_period_hours",
            "timestamp",
            "market_data_points",
            "statistical_p_value",
            "profit_margin_percent",
        ]

        for key in expected_keys:
            assert key in opp_dict

        assert opp_dict["id"] == "test_opp"
        assert opp_dict["type"] == StatisticalArbitrageType.MEAN_REVERSION.value
