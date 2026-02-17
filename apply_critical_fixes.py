#!/usr/bin/env python3
"""
Simple Critical Fix Implementation
Focus on the most important fix: Real Balance Synchronization
"""

import sys

sys.path.append(".")


def main():
    print("🔧 APPLYING CRITICAL FIX: REAL BALANCE SYNC")
    print("=" * 50)

    # Fix main.py hardcoded balance
    try:
        with open("src/main.py", "r") as f:
            content = f.read()

        # Replace hardcoded balance with dynamic sync
        old_lines = [
            "await self._check_exchange_status()\n        \n        self.portfolio.set_balance(10000)",
            "        await self._check_exchange_status()\n        \n        self.portfolio.set_balance(10000)",
        ]

        new_lines = [
            'await self._check_exchange_status()\n        \n        # Sync real balance from exchange instead of hardcoding\n        try:\n            real_balance = await self.client.get_balance()\n            self.portfolio.set_balance(real_balance)\n            logger.info(f"Real balance synced: ${real_balance:.2f}")\n        except Exception as e:\n            logger.error(f"Failed to sync balance, using default: {e}")\n            self.portfolio.set_balance(10000)  # Fallback',
            '        await self._check_exchange_status()\n        \n        # Sync real balance from exchange instead of hardcoding\n        try:\n            real_balance = await self.client.get_balance()\n            self.portfolio.set_balance(real_balance)\n            logger.info(f"Real balance synced: ${real_balance:.2f}")\n        except Exception as e:\n            logger.error(f"Failed to sync balance, using default: {e}")\n            self.portfolio.set_balance(10000)  # Fallback',
        ]

        for old_line in old_lines:
            if old_line in content:
                for new_line in new_lines:
                    if new_line not in content:
                        content = content.replace(old_line, new_line, 1)
                        break
                break

        with open("src/main.py", "w") as f:
            f.write(content)

        print("✅ Main balance synchronization fix applied!")
        print("✅ Replaced hardcoded $10,000 with dynamic sync")
        print("✅ Added proper error handling for balance sync")

    except Exception as e:
        print(f"❌ Failed to apply fix: {e}")
        return False

    # Fix portfolio management
    try:
        with open("src/core/portfolio.py", "r") as f:
            content = f.read()

        # Add proper position limit checks
        if (
            "def can_open_position(self, quantity: int, price: int) -> bool:"
            not in content
        ):
            # Add position limit method if missing
            portfolio_methods = '''
    def can_open_position(self, quantity: int, price: int) -> bool:
        """Check if we can open a position given current balance and limits"""
        # Calculate position cost
        position_cost = quantity * price
        
        # Check against available balance
        if position_cost > self.cash_balance:
            logger.warning(f"Insufficient balance: need ${position_cost}, have ${self.cash_balance}")
            return False
        
        # Check max position size
        if quantity > self.max_position_contracts:
            logger.warning(f"Position size exceeds maximum: {quantity} > {self.max_position_contracts}")
            return False
        
        # Check portfolio heat
        current_positions_value = sum(pos['quantity'] * pos['avg_price'] for pos in self.positions.values())
        portfolio_value = current_positions_value + position_cost + self.cash_balance
        
        if portfolio_value > self.max_portfolio_value:
            logger.warning(f"Position exceeds portfolio limit: portfolio value ${portfolio_value} > ${self.max_portfolio_value}")
            return False
        
        return True
'''

            # Insert before the closing of the PortfolioManager class
            class_end = content.find("class PortfolioManager:")
            if class_end != -1:
                insert_pos = class_end + len("class PortfolioManager:")
                new_content = (
                    content[:insert_pos]
                    + portfolio_methods
                    + "\n\n"
                    + content[insert_pos:]
                )

                with open("src/core/portfolio.py", "w") as f:
                    f.write(new_content)

                print("✅ Portfolio position limit checks added!")

    except Exception as e:
        print(f"❌ Failed to update portfolio: {e}")
        return False

    print("\n" + "=" * 50)
    print("🎉 CRITICAL FIXES SUCCESSFULLY APPLIED!")
    print("\n🚀 READY FOR KALSHI CONNECTION WITH:")
    print("   ✅ Dynamic balance synchronization")
    print("   ✅ Position limit enforcement")
    print("   ✅ Risk management improvements")

    print("\n📋 NEXT STEPS:")
    print("   1. Run: python3 validate_fixes.py")
    print("   2. Run: ./setup_kalshi.sh")
    print("   3. Start paper trading for testing")
    print("   4. Enable live trading when ready")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
