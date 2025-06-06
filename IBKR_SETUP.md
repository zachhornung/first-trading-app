# IBKR Integration Setup Guide

This guide will help you set up Interactive Brokers (IBKR) integration for historical backtesting and live trading with your trading application.

## Prerequisites

1. **Interactive Brokers Account**
   - Paper trading account (recommended for testing)
   - Live trading account (for production)
   - Account must have API access enabled

2. **IBKR Software**
   - TWS (Trader Workstation) - Full desktop application
   - IB Gateway - Lightweight API-only application (recommended for backtesting)

## Step 1: Download and Install IBKR Software

### Option A: TWS (Trader Workstation)
1. Go to [Interactive Brokers Downloads](https://www.interactivebrokers.com/en/trading/tws.php)
2. Download TWS for your operating system
3. Install and run TWS
4. Log in with your IBKR credentials

### Option B: IB Gateway (Recommended for API)
1. Go to [IB Gateway Downloads](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
2. Download IB Gateway for your operating system
3. Install and run IB Gateway
4. Log in with your IBKR credentials

## Step 2: Enable API Access

### In TWS:
1. Go to **File** → **Global Configuration** → **API** → **Settings**
2. Check **Enable ActiveX and Socket Clients**
3. Set **Socket port**: `7497` (paper trading) or `7496` (live trading)
4. **Trusted IPs**: Add `127.0.0.1` (localhost)
5. Uncheck **Read-Only API** if you want to place orders
6. Click **OK** and restart TWS

### In IB Gateway:
1. Go to **Configure** → **Settings** → **API**
2. Check **Enable ActiveX and Socket Clients**
3. Set **Socket port**: `7497` (paper trading) or `7496` (live trading)
4. **Trusted IPs**: Add `127.0.0.1`
5. Uncheck **Read-Only API** for order placement
6. Click **OK** and restart Gateway

## Step 3: Configure Your Application

Update your `config.toml` file:

```toml
[ibkr]
host = "127.0.0.1"
port = 7497              # Paper trading (use 7496 for live)
client_id = 1            # Unique client ID
account_id = "DU123456"  # Your paper trading account ID
paper_trading = true     # Set to false for live trading

[historical_backtesting]
enabled = true
symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
start_date = "2023-01-01"
end_date = "2023-12-31"
bar_size = "1 day"       # Options: "1 day", "1 hour", "30 mins", "15 mins", "5 mins", "1 min"
initial_capital = 100000.0
commission_per_trade = 1.0
slippage_pct = 0.001
what_to_show = "TRADES"  # Options: "TRADES", "MIDPOINT", "BID", "ASK"
use_rth = true           # Regular Trading Hours only
benchmark_symbol = "SPY"
save_results = true
output_directory = "backtest_results"
```

## Step 4: Find Your Account ID

### Paper Trading Account:
1. In TWS/Gateway, go to **Account** → **Account Window**
2. Look for account starting with "DU" (e.g., "DU123456")
3. This is your paper trading account ID

### Live Trading Account:
1. In TWS/Gateway, go to **Account** → **Account Window**
2. Look for your actual account number (usually starts with "U")
3. **WARNING**: Only use live accounts after thorough testing

## Step 5: Test Connection

Run a simple connection test:

```bash
# Test basic connection
cargo run --bin backtest -- --symbols AAPL --start-date 2023-06-01 --end-date 2023-06-30 --capital 10000
```

If successful, you should see:
```
✓ Successfully connected to IBKR TWS/Gateway
✓ Requesting historical data from IBKR for AAPL
✓ Retrieved X historical bars for AAPL
```

## Step 6: Run Historical Backtesting

### Basic Backtest:
```bash
cargo run --bin backtest -- \
  --symbols AAPL,MSFT,GOOGL \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --capital 100000 \
  --bar-size "1 day"
```

### Advanced Backtest with Logging:
```bash
RUST_LOG=info cargo run --bin backtest -- \
  --symbols AAPL,MSFT,GOOGL,TSLA,NVDA \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --capital 100000 \
  --bar-size "1 hour" \
  --output backtest_results \
  --verbose
```

## Troubleshooting

### Connection Issues

**Error: "failed to fill whole buffer"**
- Ensure TWS/Gateway is running
- Check API settings are enabled
- Verify port number (7497 for paper, 7496 for live)
- Add `127.0.0.1` to trusted IPs

**Error: "Connection refused"**
- TWS/Gateway not running
- Wrong port number in config
- Firewall blocking connection

**Error: "Authentication failed"**
- Check username/password
- Account may be locked
- API access not enabled

### Data Issues

**Error: "No historical data received"**
- Symbol might not exist or be invalid
- Date range might be too old
- Market might be closed
- Check `what_to_show` parameter

**Error: "Invalid contract"**
- Verify symbol spelling
- Some symbols require exchange specification
- Try different contract types

### Performance Issues

**Slow data retrieval:**
- IBKR has rate limits
- Use larger bar sizes for longer periods
- Consider splitting large requests

**Memory usage:**
- Large datasets can consume significant memory
- Use shorter date ranges for testing
- Monitor system resources

## API Limits and Best Practices

### Rate Limits:
- **Historical Data**: ~60 requests per 10 minutes
- **Market Data**: Limited concurrent subscriptions
- **Orders**: Reasonable rate for retail accounts

### Best Practices:
1. **Start with paper trading** - Never test with real money
2. **Use appropriate bar sizes** - Daily for long backtests, intraday for detailed analysis
3. **Cache data** - Save historical data to avoid repeated API calls
4. **Monitor logs** - Enable logging to debug issues
5. **Test thoroughly** - Validate strategy logic before live trading

## Security Considerations

1. **Never hardcode credentials** - Use environment variables or config files
2. **Secure config files** - Don't commit sensitive data to version control
3. **Use paper trading first** - Thoroughly test before going live
4. **Monitor positions** - Keep track of all open positions
5. **Set limits** - Configure maximum daily loss and position sizes

## Environment Variables (Optional)

Create a `.env` file for sensitive data:

```bash
IBKR_ACCOUNT_ID=DU123456
IBKR_USERNAME=your_username
IBKR_PASSWORD=your_password
```

Then update your application to read from environment variables.

## Market Data Subscriptions

For live trading, you may need market data subscriptions:

1. **US Stocks**: Usually included with account
2. **Options**: May require additional subscription
3. **Futures**: Additional subscription required
4. **Forex**: Usually included
5. **International**: Additional subscriptions required

Check your IBKR account for available market data.

## Next Steps

1. **Validate backtesting results** - Compare with known benchmarks
2. **Optimize strategies** - Use backtest results to refine parameters
3. **Implement risk management** - Set appropriate stop losses and position sizes
4. **Paper trade live** - Test with real-time data before going live
5. **Monitor performance** - Track actual vs expected results

## Support

- **IBKR API Documentation**: https://interactivebrokers.github.io/tws-api/
- **IBKR Support**: Contact through your account portal
- **Community Forums**: Reddit r/algotrading, Elite Trader forums

---

**⚠️ IMPORTANT DISCLAIMER**: 
This software is for educational purposes. Trading involves substantial risk of loss. Always test thoroughly with paper trading before using real money. The developers are not responsible for any financial losses.