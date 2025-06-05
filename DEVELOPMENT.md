# Development Guide - IBKR Trend Reversal Trading App

## Overview

This guide provides detailed information for developers working on the IBKR trading application. The application implements a trend reversal strategy using Exponential Moving Averages (EMAs) as noise filters.

## Architecture

### Core Components

1. **Strategy Engine** (`src/strategy.rs`)
   - Implements the trend reversal algorithm
   - Calculates EMAs and analyzes price gaps
   - Generates trading signals with confidence scores

2. **IBKR Client** (`src/ibkr.rs`)
   - Handles communication with Interactive Brokers API
   - Manages market data subscriptions
   - Executes orders and manages positions

3. **Market Data Manager** (`src/data.rs`)
   - Processes and stores real-time market data
   - Maintains price history and tick data
   - Provides statistical analysis capabilities

4. **Portfolio Manager** (`src/portfolio.rs`)
   - Tracks positions and calculates P&L
   - Implements risk management rules
   - Provides portfolio analytics

5. **Configuration** (`src/config.rs`)
   - Manages application configuration
   - Provides default settings and validation

## Development Setup

### Prerequisites

1. **Rust** (latest stable)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **IBKR Gateway/TWS**
   - Download from Interactive Brokers
   - Set up paper trading account
   - Enable API access

### Environment Setup

1. Clone and build:
   ```bash
   git clone <repository>
   cd first-trading-app
   cargo build
   ```

2. Run tests:
   ```bash
   cargo test
   ```

3. Run example:
   ```bash
   cargo run --example strategy_test
   ```

## Strategy Implementation Details

### EMA-Based Noise Filtering

The strategy uses multiple EMAs to filter market noise:

- **Fast EMA (9 periods)**: Responds quickly to price changes
- **Medium EMA (21 periods)**: Balances responsiveness and stability
- **Slow EMA (50 periods)**: Provides long-term trend context

### Gap Analysis

The core of the strategy analyzes the "gap" between current price and the primary EMA:

```rust
let ema_gap = (current_price - primary_ema) / primary_ema;
```

### Signal Generation Logic

Signals are generated when:
1. Gap exceeds threshold (`gap_threshold`)
2. Trend direction conflicts with gap direction
3. Confidence score exceeds minimum threshold
4. Noise level is acceptable

### Confidence Scoring

The confidence score combines multiple factors:
- Gap magnitude (40% weight)
- Trend exhaustion (30% weight)
- Noise level adjustment (20% weight)
- Volume confirmation (10% weight)

## Adding New Features

### Adding a New Strategy

1. Create new strategy module in `src/strategy/`
2. Implement the `Strategy` trait
3. Add configuration options to `StrategyConfig`
4. Register in main application loop

### Adding New Data Sources

1. Extend `MarketDataManager` with new data types
2. Add parsing logic for new data formats
3. Update storage and retrieval methods
4. Add tests for new functionality

### Adding New Risk Metrics

1. Extend `RiskMetrics` struct in `portfolio.rs`
2. Implement calculation logic in `update_risk_metrics()`
3. Add validation in `check_risk_limits()`
4. Update portfolio summary display

## Testing

### Unit Tests

Run all unit tests:
```bash
cargo test --lib
```

Run specific module tests:
```bash
cargo test strategy
cargo test portfolio
cargo test data
```

### Integration Tests

Test with paper trading:
```bash
# Ensure IBKR Gateway is running in paper mode
cargo run --release
```

### Strategy Backtesting

Use the strategy test example:
```bash
cargo run --example strategy_test
```

## Configuration Reference

### Strategy Parameters

- `ema_periods`: EMA periods for trend analysis
- `noise_filter_threshold`: Maximum acceptable noise level (0.0-1.0)
- `gap_threshold`: Minimum gap percentage for signal generation
- `reversal_confirmation_periods`: Number of periods for trend confirmation
- `min_confidence_threshold`: Minimum signal confidence (0.0-1.0)
- `lookback_periods`: Historical periods to analyze

### Risk Management

- `max_daily_loss_pct`: Maximum daily portfolio loss
- `max_position_size_pct`: Maximum individual position size
- `stop_loss_pct`: Automatic stop loss percentage
- `take_profit_pct`: Automatic take profit percentage
- `var_limit`: Value at Risk limit

## Debugging

### Logging

Set log level for detailed output:
```bash
export RUST_LOG=debug
cargo run --release
```

Available log levels:
- `error`: Only errors
- `warn`: Warnings and errors
- `info`: General information (default)
- `debug`: Detailed debugging
- `trace`: Very verbose output

### Common Issues

1. **Connection Failed**
   - Check IBKR Gateway is running
   - Verify port configuration
   - Ensure API access is enabled

2. **No Market Data**
   - Check market hours
   - Verify symbol subscriptions
   - Check data permissions

3. **Orders Rejected**
   - Verify account permissions
   - Check buying power
   - Validate order parameters

## Performance Optimization

### Memory Management

- Use `VecDeque` for bounded historical data
- Implement proper cleanup in data structures
- Monitor memory usage during extended runs

### CPU Optimization

- Minimize EMA recalculations
- Use efficient data structures for lookups
- Implement lazy evaluation where possible

### Network Optimization

- Batch API requests when possible
- Implement connection pooling
- Handle rate limiting gracefully

## Security Considerations

### API Keys

- Never hardcode API keys
- Use environment variables or secure storage
- Rotate keys regularly

### Risk Controls

- Implement circuit breakers for large losses
- Add manual override capabilities
- Log all trading decisions

### Data Protection

- Encrypt sensitive configuration data
- Secure network communications
- Implement audit trails

## Deployment

### Production Checklist

- [ ] Test with paper trading extensively
- [ ] Validate all configuration parameters
- [ ] Set up monitoring and alerting
- [ ] Implement backup and recovery
- [ ] Document operational procedures

### Monitoring

Key metrics to monitor:
- Connection status to IBKR
- Signal generation frequency
- Portfolio performance
- Risk metrics compliance
- System resource usage

## Contributing

### Code Style

- Use `rustfmt` for formatting
- Run `clippy` for lint checks
- Follow Rust naming conventions
- Add comprehensive documentation

### Pull Request Process

1. Create feature branch
2. Add tests for new functionality
3. Ensure all tests pass
4. Update documentation
5. Submit pull request with clear description

### Testing Requirements

- Unit tests for all new functions
- Integration tests for API changes
- Performance tests for critical paths
- Documentation tests for examples

## Additional Resources

- [Interactive Brokers API Documentation](https://interactivebrokers.github.io/tws-api/)
- [Rust Documentation](https://doc.rust-lang.org/)
- [Tokio Async Runtime](https://tokio.rs/)
- [Serde Serialization](https://serde.rs/)

## Support

For issues and questions:
1. Check this development guide
2. Review unit tests for usage examples
3. Consult IBKR API documentation
4. Open GitHub issue with detailed information