#!/bin/bash

# IBKR Trading Application Launch Script
# This script helps you run the trading application with proper setup

set -e

echo "🚀 IBKR Trend Reversal Trading Application"
echo "=========================================="
echo

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: Rust/Cargo not found. Please install Rust from https://rustup.rs/"
    exit 1
fi

# Check if config file exists
if [ ! -f "config.toml" ]; then
    echo "⚠️  Configuration file not found. Creating default config.toml..."
    echo "   Please edit config.toml with your IBKR settings before running."
    echo
    cargo run --release --quiet
    echo "📝 Default configuration created. Exiting."
    echo "   Edit config.toml and run this script again."
    exit 0
fi

# Validate IBKR Gateway connectivity (basic check)
echo "🔍 Checking configuration..."
IBKR_HOST=$(grep '^host' config.toml | cut -d'"' -f2)
IBKR_PORT=$(grep '^port' config.toml | cut -d'=' -f2 | tr -d ' ')

if [ -n "$IBKR_HOST" ] && [ -n "$IBKR_PORT" ]; then
    echo "   IBKR Gateway: $IBKR_HOST:$IBKR_PORT"
    
    # Test connectivity
    if timeout 3 bash -c "</dev/tcp/$IBKR_HOST/$IBKR_PORT" 2>/dev/null; then
        echo "✅ IBKR Gateway connection successful"
    else
        echo "⚠️  Warning: Cannot connect to IBKR Gateway at $IBKR_HOST:$IBKR_PORT"
        echo "   Make sure IBKR Gateway/TWS is running and API is enabled"
        echo
        read -p "Continue anyway? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Exiting. Please start IBKR Gateway first."
            exit 1
        fi
    fi
else
    echo "⚠️  Could not parse IBKR configuration"
fi

# Set logging level if not already set
if [ -z "$RUST_LOG" ]; then
    export RUST_LOG=info
    echo "📊 Setting log level to: $RUST_LOG"
fi

# Check for paper trading mode
if grep -q 'paper_trading = true' config.toml; then
    echo "🧪 Paper Trading Mode: ENABLED (no real money at risk)"
else
    echo "💰 LIVE TRADING MODE: REAL MONEY AT RISK!"
    echo "⚠️  Please ensure you understand the risks before proceeding"
    echo
    read -p "Are you sure you want to continue with live trading? (type 'YES' to confirm): " -r
    if [[ $REPLY != "YES" ]]; then
        echo "Exiting for safety. Edit config.toml to enable paper_trading = true for testing."
        exit 1
    fi
fi

echo
echo "🏁 Starting trading application..."
echo "   Press Ctrl+C to stop gracefully"
echo

# Build and run the application
cargo build --release

if [ $? -eq 0 ]; then
    echo "✅ Build successful, launching application..."
    echo
    cargo run --release
else
    echo "❌ Build failed. Please check the error messages above."
    exit 1
fi