import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.arbitrage import ArbitrageDetector, ArbitrageOpportunity, ArbitrageType
from src.core.orderbook import OrderBook, OrderBookLevel


class TestArbitrageDetector:
    @pytest.mark.asyncio
    async def test_detect_internal_arbitrage_exists(self):
        orderbook = OrderBook(
            market_id="test",
            bids=[OrderBookLevel(price=60, count=10, total=600)],
            asks=[OrderBookLevel(price=55, count=10, total=550)],
        )

        detector = ArbitrageDetector(min_profit_cents=10)
        opportunities = detector.detect_internal_arbitrage(orderbook, "test")

        assert len(opportunities) == 1
        assert opportunities[0].type == ArbitrageType.INTERNAL
        assert opportunities[0].buy_price == 55
        assert opportunities[0].sell_price == 60

    @pytest.mark.asyncio
    async def test_detect_internal_arbitrage_not_exists(self):
        orderbook = OrderBook(
            market_id="test",
            bids=[OrderBookLevel(price=55, count=10, total=550)],
            asks=[OrderBookLevel(price=60, count=10, total=600)],
        )

        detector = ArbitrageDetector(min_profit_cents=10)
        opportunities = detector.detect_internal_arbitrage(orderbook, "test")

        assert len(opportunities) == 0

    @pytest.mark.asyncio
    async def test_detect_cross_market_arbitrage(self):
        market_1 = OrderBook(
            market_id="market-1",
            bids=[OrderBookLevel(price=65, count=10, total=650)],
            asks=[OrderBookLevel(price=55, count=10, total=550)],
        )
        market_2 = OrderBook(
            market_id="market-2",
            bids=[OrderBookLevel(price=50, count=10, total=500)],
            asks=[OrderBookLevel(price=60, count=10, total=600)],
        )

        detector = ArbitrageDetector(min_profit_cents=10)
        opportunities = await detector.detect_cross_market_arbitrage(market_1, market_2)

        assert len(opportunities) >= 1
        assert any(o.type == ArbitrageType.CROSS_MARKET for o in opportunities)

    @pytest.mark.asyncio
    async def test_scan_for_opportunities(self):
        orderbooks = {
            "market-1": OrderBook(
                market_id="market-1",
                bids=[OrderBookLevel(price=60, count=10, total=600)],
                asks=[OrderBookLevel(price=55, count=10, total=550)],
            ),
            "market-2": OrderBook(
                market_id="market-2",
                bids=[OrderBookLevel(price=50, count=10, total=500)],
                asks=[OrderBookLevel(price=65, count=10, total=650)],
            ),
        }

        detector = ArbitrageDetector(min_profit_cents=1)
        opportunities = await detector.scan_for_opportunities(orderbooks)

        assert len(opportunities) > 0

    def test_filter_by_threshold(self):
        opportunities = [
            ArbitrageOpportunity(
                id="test-1",
                type=ArbitrageType.INTERNAL,
                market_id_1="test",
                net_profit_cents=5,
                confidence=0.9,
            ),
            ArbitrageOpportunity(
                id="test-2",
                type=ArbitrageType.INTERNAL,
                market_id_1="test",
                net_profit_cents=50,
                confidence=0.5,
            ),
            ArbitrageOpportunity(
                id="test-3",
                type=ArbitrageType.INTERNAL,
                market_id_1="test",
                net_profit_cents=100,
                confidence=0.9,
            ),
        ]

        detector = ArbitrageDetector(min_profit_cents=10, min_confidence=0.8)
        filtered = detector.filter_by_threshold(opportunities)

        assert len(filtered) == 1
        assert filtered[0].id == "test-3"

    def test_opportunity_is_profitable(self):
        opp = ArbitrageOpportunity(
            id="test",
            type=ArbitrageType.INTERNAL,
            market_id_1="test",
            net_profit_cents=100,
        )

        assert opp.is_profitable

    def test_opportunity_is_not_profitable(self):
        opp = ArbitrageOpportunity(
            id="test",
            type=ArbitrageType.INTERNAL,
            market_id_1="test",
            net_profit_cents=-50,
        )

        assert not opp.is_profitable

    def test_opportunity_to_dict(self):
        opp = ArbitrageOpportunity(
            id="test-123",
            type=ArbitrageType.CROSS_MARKET,
            market_id_1="market-1",
            market_id_2="market-2",
            buy_market_id="market-2",
            sell_market_id="market-1",
            buy_price=50,
            sell_price=60,
            quantity=10,
            profit_cents=100,
            profit_percent=20.0,
            confidence=0.9,
        )

        data = opp.to_dict()

        assert data["id"] == "test-123"
        assert data["type"] == "cross_market"
        assert data["buy_price"] == 50
        assert data["sell_price"] == 60
