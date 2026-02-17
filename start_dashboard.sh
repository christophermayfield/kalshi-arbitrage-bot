#!/bin/bash

# Arbitrage Bot API Server Launcher
# This script starts the API server with the web dashboard

echo "🚀 Starting Arbitrage Bot API Server with Dashboard..."
echo "📊 Dashboard will be available at: http://localhost:8002"
echo "🔗 API endpoints available at: http://localhost:8002"
echo ""

# Check if we're in the correct directory
if [ ! -f "src/api/main_api.py" ]; then
    echo "❌ Error: Please run this script from the arbitrage_bot project root directory"
    exit 1
fi

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected. Consider using a venv."
fi

# Check if dependencies are installed
python3 -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Missing dependencies. Please run: pip install -r requirements.txt"
    exit 1
fi

# Start the API server
echo "✅ Starting server... (Press Ctrl+C to stop)"
python3 -m src.api.main_api