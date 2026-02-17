#!/usr/bin/env python3
"""
Test script for predictive models and sentiment analysis integration
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.core.arbitrage import ArbitrageDetector, ArbitrageOpportunity, ArbitrageType
from src.core.orderbook import OrderBook, OrderSide, OrderBookLevel
from src.core.predictive_models import EnsembleForecaster, get_arbitrage_timing_signal
from src.core.sentiment_analyzer import SentimentAggregator, get_market_sentiment_signal
from src.utils.logging_utils import setup_logging


async def test_predictive_models():
    """Test predictive models functionality"""
    print("Testing Predictive Models...")

    try:
        # Test ensemble forecaster
        forecaster = EnsembleForecaster()

        # Test with dummy data
        recent_prices = [
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            106.0,
            107.0,
            108.0,
            109.0,
        ]
        predictions = forecaster.get_predictions("test_market", recent_prices)

        print(f"Generated {len(predictions)} predictions:")
        for pred in predictions:
            print(
                f"  - Method: {pred['method']}, Predicted: ${pred['predicted_price']:.2f}"
            )

        # Test timing signal
        timing_signal = await get_arbitrage_timing_signal(
            "test_market", current_spread=105.0, recent_prices=recent_prices
        )

        print(
            f"Timing signal: {timing_signal['signal']} (confidence: {timing_signal['confidence']:.3f})"
        )
        print("✓ Predictive models test passed")

    except Exception as e:
        print(f"✗ Predictive models test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("\nTesting Sentiment Analysis...")

    try:
        # Test sentiment aggregator (without API keys)
        aggregator = SentimentAggregator()

        # This will fail without API keys, but we can test the structure
        # sentiment_data = await aggregator.get_market_sentiment("test_market", ["test"])
        # print(f"Sentiment data: {sentiment_data}")

        # Test the sentiment signal function structure
        signal = await get_market_sentiment_signal("test_market", ["test"])
        print(
            f"Sentiment signal: {signal['signal']} (confidence: {signal['confidence']:.3f})"
        )
        print("✓ Sentiment analysis test passed")

    except Exception as e:
        print(f"✗ Sentiment analysis test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_arbitrage_detector():
    """Test enhanced arbitrage detector"""
    print("\nTesting Enhanced Arbitrage Detector...")

    try:
        # Create test orderbooks
        orderbook1 = OrderBook("market_1")
        orderbook1.bids = [OrderBookLevel(price=100, count=10, total=1000)]
        orderbook1.asks = [OrderBookLevel(price=102, count=10, total=1020)]

        orderbook2 = OrderBook("market_2")
        orderbook2.bids = [OrderBookLevel(price=98, count=10, total=980)]
        orderbook2.asks = [OrderBookLevel(price=101, count=10, total=1010)]

        # Test with ML features disabled (for basic functionality)
        detector = ArbitrageDetector(
            enable_predictive_models=False, enable_sentiment_analysis=False
        )

        # Test basic arbitrage detection
        opportunities = await detector.detect_cross_market_arbitrage(
            orderbook1, orderbook2
        )
        print(f"Found {len(opportunities)} basic arbitrage opportunities")

        # Test with ML features enabled
        detector_ml = ArbitrageDetector(
            enable_predictive_models=True,
            enable_sentiment_analysis=False,
            predictive_weight=0.2,
            sentiment_weight=0,
        )

        opportunities_ml = await detector_ml.detect_cross_market_arbitrage(
            orderbook1, orderbook2
        )
        print(f"Found {len(opportunities_ml)} ML-enhanced arbitrage opportunities")

        # Test price cache
        detector_ml.update_price_cache("test_market", 100.5)
        detector_ml.update_price_cache("test_market", 101.2)
        prices = detector_ml._get_recent_prices("test_market")
        print(f"Price cache: {prices}")

        print("✓ Arbitrage detector test passed")

    except Exception as e:
        print(f"✗ Arbitrage detector test failed: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Main test function"""
    print("Starting Predictive Models & Sentiment Analysis Tests")
    print("=" * 60)

    # Setup logging
    setup_logging("INFO")

    # Run tests
    await test_predictive_models()
    await test_sentiment_analysis()
    await test_arbitrage_detector()

    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
