#!/usr/bin/env python3
"""Run arbitrage bot with mock API for testing (no real credentials needed)."""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests', 'fixtures'))

from mock_kalshi_api import MockKalshiAPI
from src.core.advanced_risk_manager import AdvancedRiskManager
from src.utils.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger("mock_bot")


class MockBotDemo:
    """Demo arbitrage bot using mock API."""

    def __init__(self):
        self.api = MockKalshiAPI()
        self.risk_manager = AdvancedRiskManager(
            initial_balance_cents=100000,  # $1,000
            max_daily_loss_cents=7500,     # $75
        )
        self.max_open_positions = 10
        self.trades_count = 0
        self.profitable_trades = 0
        self.total_profit = 0

    async def run_demo(self, duration_seconds=30):
        """Run bot demo for specified duration."""
        logger.info("🚀 Starting Mock API Arbitrage Bot Demo")
        logger.info(f"📊 Initial Balance: $100.00")
        logger.info(f"⚠️  Daily Loss Limit: $75.00")
        logger.info(f"📈 Duration: {duration_seconds} seconds\n")

        start_time = datetime.now(timezone.utc)
        cycle = 0

        while True:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed > duration_seconds:
                break

            cycle += 1
            logger.info(f"\n=== Scan Cycle {cycle} ===")

            # Get markets
            markets = await self.api.get_markets(status="open")
            logger.info(f"📍 Found {len(markets)} open markets")

            if not markets:
                await asyncio.sleep(2)
                continue

            # Simulate detecting arbitrage opportunity
            market = markets[0]
            market_id = market["market_id"]

            # Get orderbook
            orderbook = await self.api.get_market_orderbook(market_id)
            if not orderbook:
                await asyncio.sleep(2)
                continue

            best_bid = orderbook.get("best_bid", 5000)
            best_ask = orderbook.get("best_ask", 5100)
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100

            logger.info(f"💰 Market {market_id}")
            logger.info(f"   Bid: {best_bid/100:.2f} | Ask: {best_ask/100:.2f} | Spread: {spread_pct:.2f}%")

            # Check if profitable opportunity
            if spread_pct >= 0.5:  # 0.5% spread threshold
                quantity = 10

                # Check if we can open position
                allowed, reason = self.risk_manager.can_open_position(
                    market_id=market_id,
                    entry_price_cents=best_bid,
                    quantity=quantity,
                    current_balance_cents=self.risk_manager.equity_curve.balances[-1]
                    if self.risk_manager.equity_curve.balances
                    else 100000,
                )

                if allowed:
                    # Buy
                    buy_order = await self.api.create_order(
                        market_id=market_id,
                        side="BUY",
                        order_type="MARKET",
                        price=best_bid,
                        count=quantity,
                    )

                    if buy_order["status"] == "FILLED":
                        # Sell
                        sell_order = await self.api.create_order(
                            market_id=market_id,
                            side="SELL",
                            order_type="MARKET",
                            price=best_ask,
                            count=quantity,
                        )

                        if sell_order["status"] == "FILLED":
                            # Calculate profit
                            profit = quantity * (best_ask - best_bid)
                            self.trades_count += 1
                            self.total_profit += profit

                            if profit > 0:
                                self.profitable_trades += 1

                            logger.info(f"✅ Trade Executed!")
                            logger.info(f"   Buy:  {quantity} @ {best_bid/100:.2f}")
                            logger.info(f"   Sell: {quantity} @ {best_ask/100:.2f}")
                            logger.info(f"   Profit: ${profit/100:.2f}")

                            # Update risk manager
                            portfolio = await self.api.get_portfolio()
                            new_balance = portfolio["balance_cents"]
                            self.risk_manager.update_equity(new_balance)
                else:
                    logger.info(f"⏭️  Skipped: {reason}")
            else:
                logger.info(f"⏭️  Spread too small ({spread_pct:.2f}%)")

            await asyncio.sleep(2)

        # Print summary
        await self.print_summary()

    async def print_summary(self):
        """Print trading summary."""
        logger.info("\n" + "=" * 50)
        logger.info("📊 TRADING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total Trades: {self.trades_count}")
        logger.info(f"Profitable: {self.profitable_trades}")
        if self.trades_count > 0:
            logger.info(f"Win Rate: {(self.profitable_trades/self.trades_count)*100:.1f}%")
        logger.info(f"Total Profit: ${self.total_profit/100:.2f}")
        logger.info("=" * 50)


async def main():
    """Main entry point."""
    bot = MockBotDemo()
    await bot.run_demo(duration_seconds=30)


if __name__ == "__main__":
    asyncio.run(main())
