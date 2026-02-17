"""
Paper Trading Demo Script

This script demonstrates the arbitrage bot in paper trading mode.
It shows:
- How opportunities are detected
- How trades are executed (with fake money)
- Real-time P&L tracking
- What the dashboard looks like

Run this to see the bot in action before using real money!
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.utils.logging_utils import setup_logging, get_logger
from src.execution.paper_trading import PaperTradingSimulator
from src.core.orderbook import OrderBook
from src.core.arbitrage import ArbitrageDetector, ArbitrageOpportunity
from src.core.portfolio import PortfolioManager
from src.core.kill_switch import KillSwitch, KillLevel, KillReason

# Setup logging
setup_logging(level="INFO")
logger = get_logger("demo")


class PaperTradingDemo:
    """
    Interactive demo of paper trading mode

    Shows the bot finding and executing arbitrage opportunities
    with fake money so you can see how it works.
    """

    def __init__(self):
        self.config = Config()
        self.simulator = PaperTradingSimulator(
            initial_balance=10000,  # $100 starting balance
            slippage_model="fixed",
            slippage_rate=0.001,
            fill_probability=0.95,
            commission_rate=0.01,
        )
        self.portfolio = PortfolioManager(
            max_daily_loss=5000,  # $50 daily loss limit
            max_open_positions=10,
        )
        self.detector = ArbitrageDetector(
            min_profit_cents=10,  # 10 cents minimum profit
            fee_rate=0.01,
            min_confidence=0.7,
        )
        self.kill_switch = KillSwitch()

        # Demo orderbooks
        self.orderbooks = {}
        self.running = False
        self.trades_executed = 0
        self.trades_won = 0
        self.trades_lost = 0

    async def run_demo(self, duration_minutes: int = 5):
        """
        Run the paper trading demo

        Args:
            duration_minutes: How long to run the demo
        """
        print("\n" + "=" * 70)
        print("🎮 PAPER TRADING DEMO")
        print("=" * 70)
        print("\nThis demo shows how the arbitrage bot finds and executes trades.")
        print("All trades use FAKE money - no real funds at risk!")
        print("\nPress Ctrl+C to stop the demo early\n")

        self.running = True
        start_time = datetime.now()

        # Setup signal handler
        self.kill_switch.setup_signal_handlers()

        # Create demo markets
        await self._setup_demo_markets()

        print(f"✓ Demo started at {start_time.strftime('%H:%M:%S')}")
        print(f"✓ Initial balance: ${self.simulator.balance / 100:.2f}")
        print(f"✓ Running for {duration_minutes} minutes...\n")

        try:
            while self.running:
                # Check if demo should end
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                if elapsed >= duration_minutes:
                    print("\n⏰ Demo time limit reached")
                    break

                # Check kill switch
                if self.kill_switch.is_killed:
                    print("\n🛑 Kill switch activated - stopping demo")
                    break

                # Simulate market movement
                await self._simulate_market_movement()

                # Scan for opportunities
                opportunities = await self.detector.scan_for_opportunities(
                    self.orderbooks
                )
                opportunities = self.detector.filter_by_threshold(opportunities)

                if opportunities:
                    print(f"\n🔍 Found {len(opportunities)} arbitrage opportunities!")

                    for opp in opportunities[:3]:  # Show top 3
                        await self._execute_demo_trade(opp)

                # Show status every 30 seconds
                if int(elapsed * 60) % 30 == 0:
                    self._show_status()

                # Wait before next scan
                await asyncio.sleep(5)

        except KeyboardInterrupt:
            print("\n\n👋 Demo stopped by user")

        # Show final results
        await self._show_final_results()

    async def _setup_demo_markets(self):
        """Create demo markets with simulated orderbooks"""

        # Market 1: Will it rain tomorrow?
        market1 = OrderBook(market_id="RAIN-NYC-2024-01-15")
        market1.update_bid(45, 100)  # Someone willing to buy at 45 cents
        market1.update_ask(47, 100)  # Someone willing to sell at 47 cents
        self.orderbooks["RAIN-NYC-2024-01-15"] = market1

        # Market 2: Will stock market go up?
        market2 = OrderBook(market_id="SP500-UP-2024-01-15")
        market2.update_bid(52, 150)
        market2.update_ask(54, 150)
        self.orderbooks["SP500-UP-2024-01-15"] = market2

        # Market 3: Will Bitcoin be above $50k?
        market3 = OrderBook(market_id="BTC-50K-2024-01-15")
        market3.update_bid(38, 200)
        market3.update_ask(40, 200)
        self.orderbooks["BTC-50K-2024-01-15"] = market3

        print("✓ Setup 3 demo markets:")
        print("  1. RAIN-NYC-2024-01-15 - Will it rain in NYC?")
        print("  2. SP500-UP-2024-01-15 - Will S&P 500 go up?")
        print("  3. BTC-50K-2024-01-15 - Will Bitcoin be >$50k?")

    async def _simulate_market_movement(self):
        """Simulate realistic market price movements"""
        import random

        for market_id, orderbook in self.orderbooks.items():
            # Randomly adjust prices slightly to simulate market movement
            if random.random() < 0.3:  # 30% chance of movement
                bid = orderbook.get_best_bid()
                ask = orderbook.get_best_ask()

                if bid and ask:
                    # Small random movement (-2 to +2 cents)
                    movement = random.randint(-2, 2)

                    new_bid = max(1, min(99, bid[0] + movement))
                    new_ask = max(1, min(99, ask[0] + movement))

                    # Ensure ask > bid
                    if new_ask <= new_bid:
                        new_ask = new_bid + 1

                    orderbook.update_bid(new_bid, 100)
                    orderbook.update_ask(new_ask, 100)

    async def _execute_demo_trade(self, opportunity: ArbitrageOpportunity):
        """Execute a demo trade and show what happens"""

        print(f"\n💡 Opportunity Found:")
        print(f"   Market: {opportunity.market_id_1}")
        print(
            f"   Buy at: {opportunity.buy_price}¢ → Sell at: {opportunity.sell_price}¢"
        )
        print(
            f"   Spread: {opportunity.sell_price - opportunity.buy_price}¢ ({opportunity.profit_percent:.1f}%)"
        )
        print(f"   Quantity: {opportunity.quantity} contracts")
        print(f"   Expected Profit: ${opportunity.net_profit_cents / 100:.2f}")

        # Check if we can afford it
        trade_cost = opportunity.quantity * opportunity.buy_price
        if trade_cost > self.simulator.balance:
            print(
                f"   ❌ Insufficient funds (need ${trade_cost / 100:.2f}, have ${self.simulator.balance / 100:.2f})"
            )
            return

        # Execute the trade
        print(f"   🚀 Executing trade...")

        try:
            # Simulate buy order
            buy_order = await self.simulator.create_order(
                market_id=opportunity.buy_market_id or opportunity.market_id_1,
                side="buy",
                order_type="limit",
                price=opportunity.buy_price,
                quantity=opportunity.quantity,
            )

            await asyncio.sleep(0.5)  # Simulate network delay

            if buy_order.status == "filled":
                # Simulate sell order
                sell_order = await self.simulator.create_order(
                    market_id=opportunity.sell_market_id
                    or opportunity.market_id_2
                    or opportunity.market_id_1,
                    side="sell",
                    order_type="limit",
                    price=opportunity.sell_price,
                    quantity=opportunity.quantity,
                )

                await asyncio.sleep(0.5)

                # Calculate actual profit
                buy_cost = buy_order.fill_price * opportunity.quantity
                sell_revenue = sell_order.fill_price * opportunity.quantity
                gross_profit = sell_revenue - buy_cost

                # Apply fees (Kalshi style: 7% of expected earnings)
                fee_rate = 0.07
                price_dollars = opportunity.buy_price / 100
                fee_per_side = (
                    fee_rate
                    * opportunity.quantity
                    * price_dollars
                    * (1 - price_dollars)
                )
                total_fees = int((fee_per_side * 2) * 100) + 1

                net_profit = gross_profit - total_fees

                self.trades_executed += 1

                if net_profit > 0:
                    self.trades_won += 1
                    print(f"   ✅ TRADE SUCCESS!")
                    print(f"      Gross Profit: ${gross_profit / 100:.2f}")
                    print(f"      Fees: ${total_fees / 100:.2f}")
                    print(f"      Net Profit: ${net_profit / 100:.2f}")
                else:
                    self.trades_lost += 1
                    print(f"   ⚠️  TRADE LOSS")
                    print(f"      Loss: ${abs(net_profit) / 100:.2f}")

                # Update balance
                self.simulator.balance += net_profit

            else:
                print(f"   ❌ Buy order not filled: {buy_order.status}")

        except Exception as e:
            print(f"   ❌ Trade failed: {e}")

    def _show_status(self):
        """Show current status"""
        stats = self.simulator.get_stats()

        print(f"\n📊 Status Update ({datetime.now().strftime('%H:%M:%S')})")
        print(f"   Balance: ${stats['balance'] / 100:.2f}")
        print(
            f"   Trades: {self.trades_executed} ({self.trades_won} wins, {self.trades_lost} losses)"
        )

        if self.trades_executed > 0:
            win_rate = (self.trades_won / self.trades_executed) * 100
            print(f"   Win Rate: {win_rate:.1f}%")

    async def _show_final_results(self):
        """Show final demo results"""
        stats = self.simulator.get_stats()
        initial = 10000  # $100
        final = stats["balance"]
        pnl = final - initial

        print("\n" + "=" * 70)
        print("📈 DEMO RESULTS")
        print("=" * 70)
        print(f"\nInitial Balance: ${initial / 100:.2f}")
        print(f"Final Balance:   ${final / 100:.2f}")
        print(f"P&L:             ${pnl / 100:.2f} ({(pnl / initial) * 100:+.1f}%)")
        print(f"\nTrades Executed: {self.trades_executed}")
        print(f"Winning Trades:  {self.trades_won}")
        print(f"Losing Trades:   {self.trades_lost}")

        if self.trades_executed > 0:
            win_rate = (self.trades_won / self.trades_executed) * 100
            print(f"Win Rate:        {win_rate:.1f}%")
            avg_pnl = pnl / self.trades_executed
            print(f"Avg P&L/Trade:   ${avg_pnl / 100:.2f}")

        print("\n" + "=" * 70)
        print("Demo Complete!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("• This is how the bot finds and executes trades")
        print("• All numbers are simulated - no real money used")
        print("• Win rate and P&L will vary in live trading")
        print("• Ready to try with real money? Proceed to live trading mode.")
        print("=" * 70 + "\n")


async def main():
    """Main entry point for demo"""
    print("\n" + "=" * 70)
    print("KALSHI ARBITRAGE BOT - PAPER TRADING DEMO")
    print("=" * 70)
    print("\nThis demo uses FAKE money to show you how the bot works.")
    print("You can safely watch it trade without any financial risk.\n")

    # Ask user how long to run
    try:
        minutes = input("How many minutes to run demo? (default: 5): ").strip()
        minutes = int(minutes) if minutes else 5
    except ValueError:
        minutes = 5

    print(f"\nStarting {minutes}-minute demo...\n")

    # Run demo
    demo = PaperTradingDemo()
    await demo.run_demo(duration_minutes=minutes)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
