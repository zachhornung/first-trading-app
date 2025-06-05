# IBKR Trend Reversal Trading App

A Rust-based algorithmic trading application that implements a trend reversal strategy using Exponential Moving Averages (EMAs) as noise filters, designed to work with Interactive Brokers (IBKR) using the rust-ibapi crate.

## Overview

This trading application implements a sophisticated trend reversal strategy that:

- Uses EMAs as noise filters rather than direct signals
- Analyzes the gap between current price and EMA values
- Identifies potential trend reversals with confirmation periods
- Manages risk through position sizing and stop-loss mechanisms
- Integrates with Interactive Brokers for live trading

## Strategy Explanation

### Core Concept
The strategy focuses on identifying trend reversal points by analyzing the relationship between current price and Exponential Moving Averages. Instead of using EMA crossovers, it examines the "gap" between price and EMA to filter out market noise and identify genuine trend changes.

### Key Components
1. **EMA Noise Filter**: Uses multiple EMAs (9, 21, 50 periods) to reduce false signals
2. **Gap Analysis**: Measures the distance between current price and primary EMA
3. **Trend Direction**: Determines current trend based on EMA alignment
4. **Reversal Probability**: Calculates likelihood of trend reversal using multiple factors
5. **Confirmation Periods**: Requires multiple periods of confirmation before signaling

### Trading Logic
- **Buy Signal**: Generated when downward trend shows large positive gap (price above EMA)
- **Sell Signal**: Generated when upward trend shows large negative gap (price below EMA)
- **Confidence Scoring**: Each signal includes confidence level based on gap size, noise level, and trend age

## Features

- **Real-time Market Data**: Connects to IBKR TWS/Gateway for live market data via rust-ibapi
- **Automated Trading**: Places orders based on strategy signals through native Rust API
- **Portfolio Management**: Tracks positions, P&L, and risk metrics
- **Risk Management**: Implements position sizing, stop-loss, and daily loss limits
- **Configurable Strategy**: All parameters can be adjusted via configuration file
- **Paper Trading**: Supports both paper and live trading modes
- **Native Rust Integration**: Uses rust-ibapi crate for direct TWS API communication

## Prerequisites

1. **Rust**: Install Rust from [rustup.rs](https://rustup.rs/)
2. **Interactive Brokers Account**: Either paper trading or live account
3. **IBKR TWS or Gateway**: Download and install from IBKR
4. **API Access**: Enable API access in your IBKR account settings
5. **rust-ibapi**: Automatically included as a dependency

## Setup Instructions

### 1. Clone and Build
```bash
git clone <repository-url>
cd first-trading-app
cargo build --release
```

### 2. Configure IBKR TWS/Gateway
1. Download and install IBKR TWS or Gateway from Interactive Brokers
2. Log in with your IBKR credentials
3. Enable API access:
   - Go to Global Configuration → API → Settings
   - Enable "Enable ActiveX and Socket Clients"
   - Set API port (7497 for paper, 7496 for live)
   - Add your machine's IP to trusted IPs (use 127.0.0.1, not localhost)
4. Start TWS/Gateway and keep it running

### 3. Configuration
Edit `config.toml` to match your setup:

```toml
[ibkr]
host = "127.0.0.1"
port = 7497  # Paper trading port
client_id = 1
account_id = "YOUR_PAPER_ACCOUNT_ID"  # Replace with your account ID
paper_trading = true

[strategy]
ema_periods = [9, 21, 50]
gap_threshold = 0.02  # 2% gap for reversal signals
min_confidence_threshold = 0.6

[trading]
symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
position_size_pct = 0.1  # 10% of portfolio per position

[risk]
max_daily_loss_pct = 0.02  # 2% maximum daily loss
stop_loss_pct = 0.05  # 5% stop loss
```

### 4. Run the Application
```bash
# Set log level (optional)
export RUST_LOG=info

# Run the trading application
cargo run --release
```

## Configuration Reference

### IBKR Settings
- `host`: IBKR TWS/Gateway host (use 127.0.0.1, not localhost)
- `port`: API port (7497 for paper, 7496 for live)
- `client_id`: Unique client identifier (different for each connection)
- `account_id`: Your IBKR account ID
- `paper_trading`: true for paper trading, false for live

### Strategy Parameters
- `ema_periods`: EMA periods for trend analysis [fast, medium, slow]
- `noise_filter_threshold`: Threshold for noise filtering (0.0-1.0)
- `gap_threshold`: Minimum gap percentage for signal generation
- `reversal_confirmation_periods`: Number of periods to confirm reversal
- `min_confidence_threshold`: Minimum confidence for trade execution
- `lookback_periods`: Historical periods to analyze

### Trading Settings
- `symbols`: List of symbols to trade
- `timeframe`: Data timeframe (1min, 5min, 1h, etc.)
- `max_positions`: Maximum number of concurrent positions
- `position_size_pct`: Position size as percentage of portfolio

### Risk Management
- `max_daily_loss_pct`: Maximum daily loss percentage
- `max_position_size_pct`: Maximum individual position size
- `stop_loss_pct`: Stop loss percentage
- `take_profit_pct`: Take profit percentage

## Usage

### Starting the Application
1. Ensure IBKR TWS/Gateway is running and configured
2. Update `config.toml` with your settings
3. Run `cargo run --release` or use `./run.sh`
4. Monitor logs for connection status and trading activity

### Monitoring
The application logs important events including:
- Connection status to IBKR
- Market data subscriptions
- Trading signals generated
- Orders placed and executed
- Portfolio updates
- Risk limit violations

### Stopping the Application
- Use Ctrl+C to gracefully stop the application
- The application will attempt to close connections cleanly

## Architecture

### Main Components
- **Strategy Engine**: Implements the trend reversal algorithm
- **IBKR Client**: Native Rust client using rust-ibapi for TWS communication
- **Market Data Manager**: Processes and stores market data
- **Portfolio Manager**: Tracks positions and calculates metrics
- **Risk Manager**: Enforces risk limits and position sizing

### Data Flow
1. Market data received from IBKR
2. Data processed and stored by Market Data Manager
3. Strategy Engine analyzes data and generates signals
4. Portfolio Manager validates trades against risk limits
5. Orders sent to IBKR for execution
6. Portfolio updated with trade results

## Safety Features

### Paper Trading
- Always start with paper trading to test your configuration
- Paper trading uses the same API but with simulated money
- No risk of financial loss during testing

### Risk Limits
- Daily loss limits prevent excessive losses
- Position size limits control concentration risk
- Stop-loss orders limit per-trade losses
- Correlation checks prevent over-concentration

### Error Handling
- Robust error handling for network issues
- Automatic reconnection to IBKR Gateway
- Graceful handling of invalid orders
- Detailed logging for troubleshooting

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Verify IBKR TWS/Gateway is running
   - Check port configuration matches TWS/Gateway settings
   - Use 127.0.0.1 instead of localhost
   - Ensure API access is enabled in IBKR
   - Make sure client_id is unique and not in use

2. **Authentication Errors**
   - Verify account ID is correct
   - Check if API access is enabled for your account
   - Ensure client ID is not in use by another application
   - Restart TWS/Gateway if connection issues persist

3. **No Market Data**
   - Verify market data subscriptions in IBKR
   - Check if markets are open
   - Ensure symbols are valid and tradeable
   - Check market data permissions in your account

4. **Orders Rejected**
   - Check account permissions for the instrument
   - Verify sufficient buying power
   - Check if outside trading hours
   - Ensure order parameters are valid

### Logs
Enable detailed logging:
```bash
export RUST_LOG=debug
cargo run --release
```

## Testing

Run the test suite:
```bash
# Run all tests
cargo test

# Run specific module tests
cargo test strategy
cargo test portfolio
cargo test ibkr
```

## Disclaimer

⚠️ **Important Risk Warning** ⚠️

This software is for educational and research purposes. Algorithmic trading involves substantial risk of loss and is not suitable for all investors. 

- Always test thoroughly with paper trading first
- Never risk more than you can afford to lose
- Past performance does not guarantee future results
- The developers are not responsible for any financial losses

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review IBKR API documentation
3. Check rust-ibapi documentation: https://docs.rs/ibapi/
4. Open an issue on GitHub with detailed logs and configuration