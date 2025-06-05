# IBKR Algorithmic Trading Platform

A comprehensive Rust-based algorithmic trading application that implements multiple trading strategies and technical indicators, designed to work with Interactive Brokers (IBKR) using native Rust APIs.

## Overview

This trading platform provides a sophisticated framework for algorithmic trading with:

- **Multiple Trading Strategies**: Trend reversal, mean reversion, momentum, and multi-indicator strategies
- **Comprehensive Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R, CCI, ATR, SMA, EMA, Volume indicators, and Support/Resistance
- **Advanced Risk Management**: Position sizing, stop-loss, correlation analysis, and portfolio-wide risk controls
- **Real-time Market Data**: Native integration with Interactive Brokers TWS/Gateway
- **Backtesting Engine**: Historical strategy testing with performance analytics
- **Paper Trading Support**: Safe testing environment before live trading

## 🚀 Quick Start

### Prerequisites

1. **Rust** (latest stable): Install from [rustup.rs](https://rustup.rs/)
2. **Interactive Brokers Account**: Paper or live trading account
3. **IBKR TWS/Gateway**: Download from Interactive Brokers
4. **API Access**: Enable in your IBKR account settings

### Installation

```bash
git clone <repository-url>
cd first-trading-app
cargo build --release
```

### Basic Setup

1. **Configure IBKR TWS/Gateway**:
   - Enable API access in Global Configuration → API → Settings
   - Set API port (7497 for paper, 7496 for live)
   - Add 127.0.0.1 to trusted IPs

2. **Edit Configuration** (`config.toml`):
   ```toml
   [ibkr]
   host = "127.0.0.1"
   port = 7497  # Paper trading
   account_id = "YOUR_ACCOUNT_ID"
   paper_trading = true

   [trading]
   symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
   position_size_pct = 0.1  # 10% per position
   ```

3. **Run the Application**:
   ```bash
   cargo run --release
   ```

## 📊 Available Strategies

### 1. Trend Reversal Strategy

**Purpose**: Identifies potential trend reversals using EMA gap analysis.

**How it Works**:
- Uses multiple EMAs (9, 21, 50 periods) as noise filters
- Analyzes price gaps relative to EMAs
- Generates signals when trends show exhaustion
- Requires confirmation periods to reduce false signals

**Configuration Example**:
```toml
[strategy]
ema_periods = [9, 21, 50]
gap_threshold = 0.02  # 2% gap for reversal
min_confidence_threshold = 0.6
reversal_confirmation_periods = 3
```

**Best For**: Range-bound markets, counter-trend trading

### 2. Enhanced Multi-Indicator Strategy

**Purpose**: Combines multiple technical indicators for robust signal generation.

**Indicators Used**:
- RSI for momentum
- MACD for trend changes
- Bollinger Bands for volatility
- Volume analysis for confirmation
- Support/Resistance levels

**Configuration Example**:
```toml
[enhanced_strategy]
strategy_type = "Combined"

[enhanced_strategy.indicators]
rsi_enabled = true
rsi_period = 14
rsi_weight = 1.0

macd_enabled = true
macd_fast_period = 12
macd_slow_period = 26
macd_signal_period = 9
macd_weight = 1.0

bollinger_enabled = true
bollinger_period = 20
bollinger_std_dev = 2.0
bollinger_weight = 0.8
```

**Best For**: Trending markets, confirmation-based trading

## 🔧 Technical Indicators

### Momentum Indicators

#### RSI (Relative Strength Index)
- **Purpose**: Measures overbought/oversold conditions
- **Range**: 0-100
- **Signals**: >70 overbought, <30 oversold
- **Configuration**:
  ```toml
  rsi_enabled = true
  rsi_period = 14
  rsi_weight = 1.0
  ```

#### Stochastic Oscillator
- **Purpose**: Compares closing price to price range
- **Range**: 0-100
- **Signals**: >80 overbought, <20 oversold
- **Configuration**:
  ```toml
  stochastic_enabled = true
  stochastic_k_period = 14
  stochastic_d_period = 3
  stochastic_weight = 0.7
  ```

#### Williams %R
- **Purpose**: Momentum indicator similar to Stochastic
- **Range**: -100 to 0
- **Signals**: >-20 overbought, <-80 oversold
- **Configuration**:
  ```toml
  williams_r_enabled = true
  williams_r_period = 14
  williams_r_weight = 0.6
  ```

#### CCI (Commodity Channel Index)
- **Purpose**: Identifies cyclical trends
- **Range**: Unbounded (typically ±100)
- **Signals**: >100 overbought, <-100 oversold
- **Configuration**:
  ```toml
  cci_enabled = true
  cci_period = 20
  cci_weight = 0.7
  ```

### Trend Indicators

#### MACD (Moving Average Convergence Divergence)
- **Purpose**: Trend following momentum indicator
- **Components**: MACD line, Signal line, Histogram
- **Signals**: Line crossovers, histogram divergence
- **Configuration**:
  ```toml
  macd_enabled = true
  macd_fast_period = 12
  macd_slow_period = 26
  macd_signal_period = 9
  macd_weight = 1.0
  ```

#### EMA (Exponential Moving Average)
- **Purpose**: Trend direction and support/resistance
- **Responsive**: More weight to recent prices
- **Multiple Periods**: Fast (9), Medium (21), Slow (50)
- **Configuration**:
  ```toml
  ema_periods = [9, 21, 50]
  ```

#### SMA (Simple Moving Average)
- **Purpose**: Smooth price action, trend identification
- **Equal Weight**: All periods weighted equally
- **Usage**: Trend direction, crossover signals

### Volatility Indicators

#### Bollinger Bands
- **Purpose**: Measure volatility and identify overbought/oversold
- **Components**: Upper band, Middle (SMA), Lower band
- **Signals**: Price touching bands, band squeezes
- **Configuration**:
  ```toml
  bollinger_enabled = true
  bollinger_period = 20
  bollinger_std_dev = 2.0
  bollinger_weight = 0.8
  ```

#### ATR (Average True Range)
- **Purpose**: Measure market volatility
- **Usage**: Position sizing, stop-loss placement
- **Not Directional**: Only measures volatility magnitude
- **Configuration**:
  ```toml
  atr_enabled = true
  atr_period = 14
  atr_weight = 0.5
  ```

### Volume Indicators

#### Volume Analysis
- **Purpose**: Confirm price movements with volume
- **Metrics**: Volume ratio, average volume
- **Signals**: High volume on breakouts
- **Configuration**:
  ```toml
  volume_enabled = true
  volume_period = 20
  volume_weight = 0.6
  ```

### Support/Resistance

#### Dynamic Support/Resistance
- **Purpose**: Identify key price levels
- **Method**: Historical high/low analysis
- **Usage**: Entry/exit points, risk management
- **Configuration**:
  ```toml
  support_resistance_enabled = true
  support_resistance_lookback = 50
  support_resistance_weight = 0.8
  ```

## ⚙️ Configuration Guide

### Complete Configuration Example

```toml
[ibkr]
host = "127.0.0.1"
port = 7497
client_id = 1
account_id = "DU123456"
paper_trading = true

[strategy]
ema_periods = [9, 21, 50]
noise_filter_threshold = 0.5
gap_threshold = 0.02
reversal_confirmation_periods = 3
min_confidence_threshold = 0.6
lookback_periods = 100

[enhanced_strategy]
strategy_type = "Combined"

[enhanced_strategy.indicators]
# Momentum Indicators
rsi_enabled = true
rsi_period = 14
rsi_weight = 1.0

stochastic_enabled = false
stochastic_k_period = 14
stochastic_d_period = 3
stochastic_weight = 0.7

williams_r_enabled = false
williams_r_period = 14
williams_r_weight = 0.6

cci_enabled = false
cci_period = 20
cci_weight = 0.7

# Trend Indicators
macd_enabled = true
macd_fast_period = 12
macd_slow_period = 26
macd_signal_period = 9
macd_weight = 1.0

# Volatility Indicators
bollinger_enabled = true
bollinger_period = 20
bollinger_std_dev = 2.0
bollinger_weight = 0.8

atr_enabled = true
atr_period = 14
atr_weight = 0.5

# Volume Indicators
volume_enabled = true
volume_period = 20
volume_weight = 0.6

# Support/Resistance
support_resistance_enabled = true
support_resistance_lookback = 50
support_resistance_weight = 0.8

[enhanced_strategy.signal_weights]
trend_following = 1.0
mean_reversion = 0.8
momentum = 0.9
volume_confirmation = 0.6
volatility_adjustment = 0.5

[enhanced_strategy.risk_parameters]
min_confidence_threshold = 0.6
max_position_size = 0.1
stop_loss_pct = 0.05
take_profit_pct = 0.10
max_drawdown_pct = 0.15
correlation_threshold = 0.7

[enhanced_strategy.backtesting]
enabled = false
initial_capital = 100000.0
commission_per_trade = 1.0
slippage_pct = 0.001
benchmark_symbol = "SPY"

[trading]
symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
timeframe = "1min"
max_positions = 5
position_size_pct = 0.1

[trading.trading_hours]
start = "09:30"
end = "16:00"
timezone = "US/Eastern"

[risk]
max_daily_loss_pct = 0.02
max_position_size_pct = 0.15
stop_loss_pct = 0.05
take_profit_pct = 0.10
max_correlation = 0.7
var_limit = 0.03
```

## 🧪 Testing Strategies and Indicators

### Running Strategy Tests

```bash
# Test specific strategy
cargo run --example strategy_test

# Run all unit tests
cargo test

# Test specific indicator
cargo test indicators::rsi
cargo test indicators::macd
```

### Backtesting

Enable backtesting in configuration:
```toml
[enhanced_strategy.backtesting]
enabled = true
start_date = "2023-01-01"
end_date = "2023-12-31"
initial_capital = 100000.0
benchmark_symbol = "SPY"
```

### Paper Trading Test

1. Configure for paper trading:
   ```toml
   [ibkr]
   port = 7497
   paper_trading = true
   ```

2. Start with small position sizes:
   ```toml
   [trading]
   position_size_pct = 0.05  # Start with 5%
   ```

3. Monitor logs:
   ```bash
   export RUST_LOG=info
   cargo run --release
   ```

## 📈 Strategy Customization

### Creating Custom Indicator Combinations

```toml
# Conservative Setup (Lower Risk)
[enhanced_strategy.indicators]
rsi_enabled = true
rsi_weight = 0.8
bollinger_enabled = true
bollinger_weight = 0.9
volume_enabled = true
volume_weight = 0.7

# Aggressive Setup (Higher Frequency)
[enhanced_strategy.indicators]
macd_enabled = true
macd_weight = 1.2
stochastic_enabled = true
stochastic_weight = 1.0
williams_r_enabled = true
williams_r_weight = 0.8
```

### Adjusting Signal Weights

```toml
[enhanced_strategy.signal_weights]
# For Trending Markets
trend_following = 1.5
mean_reversion = 0.5

# For Range-Bound Markets
trend_following = 0.5
mean_reversion = 1.5

# For High Volatility
volatility_adjustment = 1.2
momentum = 1.1
```

## 🛡️ Risk Management

### Position Sizing
- **Fixed Percentage**: Set percentage of portfolio per position
- **Volatility-Based**: Adjust size based on ATR
- **Correlation-Aware**: Reduce size for correlated positions

### Stop-Loss Strategies
- **Fixed Percentage**: Static percentage below entry
- **ATR-Based**: Dynamic based on volatility
- **Support/Resistance**: Technical level-based

### Portfolio-Level Limits
- **Daily Loss Limit**: Maximum daily portfolio loss
- **Drawdown Limit**: Maximum peak-to-trough decline
- **Correlation Limit**: Maximum correlation between positions

## 📊 Performance Monitoring

### Key Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Calmar Ratio**: Annual return / Maximum drawdown

### Real-Time Monitoring
```bash
# Enable detailed logging
export RUST_LOG=debug

# Monitor specific components
export RUST_LOG=first_trading_app::strategy=debug
export RUST_LOG=first_trading_app::indicators=info
```

## 🔧 Advanced Usage

### Custom Timeframes
```toml
[trading]
timeframe = "5min"  # 1min, 5min, 15min, 1h, 1d
```

### Multiple Symbol Sets
```toml
[trading]
# Tech stocks
symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]

# Or ETFs
symbols = ["SPY", "QQQ", "IWM", "EFA", "EEM"]

# Or forex (if available)
symbols = ["EUR.USD", "GBP.USD", "USD.JPY"]
```

### Dynamic Parameter Adjustment
```rust
// Example: Adjust parameters based on market conditions
if volatility > 0.02 {
    config.risk.stop_loss_pct = 0.03;  // Tighter stops in volatile markets
}
```

## 🚨 Safety and Disclaimers

### ⚠️ Important Warnings

1. **Always Start with Paper Trading**: Test thoroughly before using real money
2. **Risk Management is Critical**: Never risk more than you can afford to lose
3. **Market Conditions Change**: Strategies that work in one market may not work in another
4. **No Guarantees**: Past performance does not guarantee future results

### Best Practices

1. **Start Small**: Begin with small position sizes
2. **Diversify**: Don't put all capital in one strategy
3. **Monitor Continuously**: Keep track of performance and adjust as needed
4. **Regular Reviews**: Periodically review and update strategies
5. **Stay Informed**: Keep up with market conditions and news

### Legal Disclaimer

This software is for educational and research purposes only. The developers are not responsible for any financial losses. Always consult with financial professionals before making investment decisions.

## 🤝 Contributing

We welcome contributions! Please see [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development guidelines.

### Areas for Contribution
- New technical indicators
- Additional trading strategies
- Performance optimizations
- Documentation improvements
- Test coverage expansion

## 📚 Additional Resources

- [DEVELOPMENT.md](DEVELOPMENT.md) - Developer guide
- [Interactive Brokers API Documentation](https://interactivebrokers.github.io/tws-api/)
- [Technical Analysis Concepts](https://www.investopedia.com/technical-analysis-4689657)
- [Risk Management in Trading](https://www.investopedia.com/articles/trading/09/risk-management.asp)

## 📞 Support

For support and questions:
1. Check the documentation and examples
2. Review the troubleshooting section in DEVELOPMENT.md
3. Open an issue on GitHub with detailed information
4. Include logs, configuration, and error messages

---

**Happy Trading! 📈**

Remember: Successful algorithmic trading requires careful strategy development, rigorous testing, and disciplined risk management.