import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.orderbook import OrderBook, OrderBookLevel, OrderSide


class TestOrderBook:
    def test_from_api_response(self):
        data = {
            'market_id': 'test-market',
            'bids': [
                {'price': '95', 'count': 10, 'total': 950},
                {'price': '94', 'count': 20, 'total': 1880}
            ],
            'asks': [
                {'price': '96', 'count': 15, 'total': 1440},
                {'price': '97', 'count': 25, 'total': 2425}
            ],
            'ts': '2024-01-01T00:00:00Z',
            'seq': 123
        }

        orderbook = OrderBook.from_api_response(data)

        assert orderbook.market_id == 'test-market'
        assert len(orderbook.bids) == 2
        assert len(orderbook.asks) == 2
        assert orderbook.bids[0].price == 95
        assert orderbook.asks[0].price == 96

    def test_get_best_bid(self):
        orderbook = OrderBook(
            market_id='test',
            bids=[
                OrderBookLevel(price=95, count=10, total=950),
                OrderBookLevel(price=94, count=20, total=1880)
            ],
            asks=[
                OrderBookLevel(price=96, count=15, total=1440)
            ]
        )

        best_bid = orderbook.get_best_bid()
        assert best_bid is not None
        assert best_bid.price == 95

    def test_get_best_ask(self):
        orderbook = OrderBook(
            market_id='test',
            bids=[OrderBookLevel(price=95, count=10, total=950)],
            asks=[
                OrderBookLevel(price=96, count=15, total=1440),
                OrderBookLevel(price=97, count=25, total=2425)
            ]
        )

        best_ask = orderbook.get_best_ask()
        assert best_ask is not None
        assert best_ask.price == 96

    def test_get_mid_price(self):
        orderbook = OrderBook(
            market_id='test',
            bids=[OrderBookLevel(price=95, count=10, total=950)],
            asks=[OrderBookLevel(price=96, count=15, total=1440)]
        )

        mid = orderbook.get_mid_price()
        assert mid == 95.5

    def test_get_spread(self):
        orderbook = OrderBook(
            market_id='test',
            bids=[OrderBookLevel(price=95, count=10, total=950)],
            asks=[OrderBookLevel(price=96, count=15, total=1440)]
        )

        spread = orderbook.get_spread()
        assert spread == 1

    def test_get_spread_percent(self):
        orderbook = OrderBook(
            market_id='test',
            bids=[OrderBookLevel(price=95, count=10, total=950)],
            asks=[OrderBookLevel(price=96, count=15, total=1440)]
        )

        spread_pct = orderbook.get_spread_percent()
        assert spread_pct == pytest.approx(1.047, rel=0.01)

    def test_empty_orderbook(self):
        orderbook = OrderBook(market_id='test')

        assert orderbook.get_best_bid() is None
        assert orderbook.get_best_ask() is None
        assert orderbook.get_mid_price() is None

    def test_estimate_slippage_buy(self):
        orderbook = OrderBook(
            market_id='test',
            bids=[OrderBookLevel(price=99, count=5, total=495)],
            asks=[
                OrderBookLevel(price=51, count=10, total=510),
                OrderBookLevel(price=52, count=10, total=520)
            ]
        )

        cost, slippage = orderbook.estimate_slippage(OrderSide.BUY, 15)

        assert cost == 510 + 260
        assert slippage > 0

    def test_estimate_slippage_sell(self):
        orderbook = OrderBook(
            market_id='test',
            bids=[
                OrderBookLevel(price=49, count=10, total=490),
                OrderBookLevel(price=48, count=10, total=480)
            ],
            asks=[OrderBookLevel(price=51, count=5, total=255)]
        )

        cost, slippage = orderbook.estimate_slippage(OrderSide.SELL, 15)

        assert cost == 490 + 240
        assert slippage > 0

    def test_get_liquidity_score(self):
        orderbook = OrderBook(
            market_id='test',
            bids=[
                OrderBookLevel(price=50, count=20, total=1000),
                OrderBookLevel(price=51, count=15, total=765)
            ],
            asks=[
                OrderBookLevel(price=52, count=25, total=1300),
                OrderBookLevel(price=53, count=10, total=530)
            ]
        )

        score = orderbook.get_liquidity_score()
        assert 0 < score <= 100

    def test_is_healthy(self):
        orderbook = OrderBook(
            market_id='test',
            bids=[OrderBookLevel(price=95, count=10, total=950)],
            asks=[OrderBookLevel(price=96, count=15, total=1440)]
        )

        healthy, reason = orderbook.is_healthy(
            min_liquidity=50,
            max_spread_percent=5.0
        )

        assert healthy
        assert reason == "Healthy"

    def test_is_healthy_fails_wide_spread(self):
        orderbook = OrderBook(
            market_id='test',
            bids=[OrderBookLevel(price=50, count=10, total=500)],
            asks=[OrderBookLevel(price=80, count=15, total=1200)]
        )

        healthy, reason = orderbook.is_healthy(max_spread_percent=5.0)

        assert not healthy
        assert "Spread too wide" in reason

    def test_get_fill_probability(self):
        orderbook = OrderBook(
            market_id='test',
            bids=[OrderBookLevel(price=99, count=10, total=990)],
            asks=[
                OrderBookLevel(price=51, count=5, total=255),
                OrderBookLevel(price=52, count=10, total=520)
            ]
        )

        prob = orderbook.get_fill_probability(OrderSide.BUY, 5, 51)
        assert prob == 1.0

        prob = orderbook.get_fill_probability(OrderSide.BUY, 10, 51)
        assert prob == 0.5
