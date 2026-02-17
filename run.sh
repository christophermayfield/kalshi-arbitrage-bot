#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running tests..."
python3 -m pytest tests/unit/ -v

echo "Starting arbitrage bot..."
python3 -m src.main
