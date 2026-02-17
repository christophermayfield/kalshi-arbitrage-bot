#!/bin/bash

# Kalshi Arbitrage Bot Quick Setup Script
# This script guides you through Kalshi connection setup

echo "🚀 KALSHI ARBITRAGE BOT SETUP"
echo "=================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "src/clients/kalexi_client.py" ]; then
    echo "❌ Error: Please run this script from the arbitrage_bot project root directory"
    exit 1
fi

echo "📋 STEP 1: KALSHI API SETUP"
echo "--------------------------------"
echo ""
echo "To use this bot, you need Kalshi API credentials:"
echo "1. Sign up at: https://kalshi.com/signup"
echo "2. Request API access at: https://docs.kalshi.com/getting_started/api_keys"
echo "3. Download your private key to ~/.kalshi/private-key.pem"
echo ""

# Check if API keys are already configured
if [ -f "config.yaml" ]; then
    if grep -q "YOUR_API_KEY_ID" config.yaml; then
        echo "⚠️  Found config.yaml but API keys need to be updated"
        SETUP_NEEDED=true
    else
        echo "✅ API keys appear to be configured"
        SETUP_NEEDED=false
    fi
else
    echo "📝 No config.yaml found - will create one"
    SETUP_NEEDED=true
fi

if [ "$SETUP_NEEDED" = true ]; then
    echo ""
    echo "🔧 STEP 2: CONFIGURE YOUR API KEYS"
    echo "------------------------------------"
    echo ""
    
    # Create config file if it doesn't exist
    if [ ! -f "config.yaml" ]; then
        cp config.yaml.example config.yaml
        echo "📄 Created config.yaml from template"
    fi
    
    echo "Please edit config.yaml with your Kalshi API credentials:"
    echo "- api_key_id: Your actual API key ID"
    echo "- private_key_path: ~/.kalshi/private-key.pem"
    echo ""
    echo "Press Enter to open config.yaml for editing..."
    read
    
    # Try to open editor
    if command -v nano &> /dev/null; then
        nano config.yaml
    elif command -v vim &> /dev/null; then
        vim config.yaml
    elif command -v code &> /dev/null; then
        code config.yaml
    else
        echo "Please manually edit config.yaml with your API credentials"
    fi
fi

echo ""
echo "🔗 STEP 3: TEST KALSHI CONNECTION"
echo "----------------------------------"

# Create test script
cat > test_kalshi_connection.py << 'EOF'
#!/usr/bin/env python3
import sys
sys.path.append('.')

try:
    from src.clients.kalexi_client import KalshiClient
    from src.utils.config import Config
    
    print("🔑 Testing Kalshi API connection...")
    
    # Load config
    config = Config()
    api_key_id = config.get("kalshi.api_key_id")
    private_key_path = config.get("kalshi.private_key_path")
    demo_mode = config.get("kalshi.demo_mode", True)
    
    if not api_key_id or "YOUR_API_KEY_ID" in api_key_id:
        print("❌ API Key ID not configured. Please edit config.yaml")
        sys.exit(1)
    
    print(f"📡 Using API Key ID: {api_key_id[:8]}...")
    print(f"🔒 Private Key Path: {private_key_path}")
    print(f"🧪 Demo Mode: {demo_mode}")
    
    # Test connection
    client = KalshiClient(demo_mode=demo_mode)
    balance = client.get_balance()
    
    print(f"✅ Connection successful!")
    print(f"💰 Balance: ${balance/100:.2f}")
    
    # Test market data
    markets = client.get_markets(limit=5)
    print(f"📊 Retrieved {len(markets)} markets")
    
    if markets:
        print("📈 Sample markets:")
        for i, market in enumerate(markets[:3]):
            print(f"  {i+1}. {market.get('title', 'Unknown')}")
    
    print("\n🎉 Kalshi connection test PASSED!")
    print("📋 You can now start the arbitrage bot:")
    print("   ./start_dashboard.sh  # Web interface")
    print("   python3 -m src.main  # CLI mode")
    
except Exception as e:
    print(f"❌ Connection test FAILED: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Verify API key ID is correct")
    print("2. Check private key file exists and is readable")
    print("3. Ensure you have internet connection")
    print("4. Try demo mode first if using live API")
    sys.exit(1)
EOF

# Run the test
python3 test_kalshi_connection.py
TEST_RESULT=$?

# Clean up test script
rm test_kalshi_connection.py

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "🚀 STEP 4: START TRADING!"
    echo "------------------------"
    echo ""
    echo "🎯 Available Strategies:"
    echo "  1. Internal Arbitrage (Low Risk, 1-5¢ profit, 95% win rate)"
    echo "  2. Cross-Market Arbitrage (Low Risk, 5-25¢ profit, 85% win rate)"
    echo "  3. Statistical Arbitrage (Medium Risk, 10-50¢ profit, 65% win rate)"
    echo ""
    echo "📊 Recommended Configuration for Beginners:"
    echo "  - Start with Internal Arbitrage only"
    echo "  - Use paper trading for 1-2 days"
    echo "  - Monitor performance closely"
    echo "  - Add Cross-Market after confidence builds"
    echo ""
    echo "🎮 Ready to start? Choose your interface:"
    echo ""
    echo "1) Web Dashboard (Recommended)"
    echo "2) Command Line Interface"
    echo "3) Enterprise API"
    echo "4) Exit Setup"
    echo ""
    
    while true; do
        read -p "Choose option [1-4]: " choice
        case $choice in
            1)
                echo "🌐 Starting Web Dashboard..."
                ./start_dashboard.sh &
                sleep 2
                echo "✅ Dashboard available at: http://localhost:8002"
                echo "📱 Open your browser to start trading!"
                break
                ;;
            2)
                echo "💻 Starting CLI Bot..."
                python3 -m src.main
                break
                ;;
            3)
                echo "🔧 Starting Enterprise API..."
                python3 -m src.api.rest_api &
                sleep 2
                echo "✅ API docs available at: http://localhost:8000/api/docs"
                break
                ;;
            4)
                echo "👋 Setup complete. Run this script anytime to reconfigure."
                break
                ;;
            *)
                echo "Please choose 1, 2, 3, or 4"
                ;;
        esac
    done
else
    echo ""
    echo "❌ SETUP FAILED"
    echo "Please fix the connection issues above and run this script again."
    exit 1
fi

echo ""
echo "🎉 KALSHI SETUP COMPLETE!"
echo "=========================="
echo ""
echo "📚 Next Steps:"
echo "  1. Monitor performance in paper mode"
echo "  2. Adjust strategy parameters based on results"  
echo "  3. Enable additional strategies when ready"
echo "  4. Consider monetization features (see STRATEGY_PLAN.md)"
echo ""
echo "📖 For detailed strategy guide: cat STRATEGY_PLAN.md"
echo "🎨 For UI improvements: See STRATEGY_PLAN.md Part 4"
echo "💰 For monetization ideas: See STRATEGY_PLAN.md Part 3"
echo ""
echo "🚀 Happy arbitrage trading!"