use anyhow::Result;
use chrono::Utc;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;
// Include the modules from the main application
use first_trading_app::config::StrategyConfig;
use first_trading_app::data::{PriceData, MarketDataManager, OHLCV};
use first_trading_app::strategy::TrendReversalStrategy;
use first_trading_app::indicators::{RSI, MACDIndicator, BollingerBandsIndicator};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    println!("🚀 IBKR Trading Platform - Strategy & Indicator Testing");
    println!("======================================================\n");

    // Run all tests
    test_individual_indicators().await?;
    test_trend_reversal_strategy().await?;
    test_strategy_comparison().await?;

    println!("\n✅ All tests completed successfully!");
    Ok(())
}

/// Test individual technical indicators
async fn test_individual_indicators() -> Result<()> {
    println!("📊 Testing Individual Technical Indicators");
    println!("==========================================\n");

    // Test RSI
    println!("Testing RSI (Relative Strength Index):");
    let mut rsi = RSI::new(14);
    let rsi_test_prices = vec![44.0, 44.25, 44.5, 43.75, 44.0, 44.25, 45.0, 47.0, 46.75, 46.5, 46.25, 47.75, 47.5, 47.25, 48.0];
    
    for (i, price) in rsi_test_prices.iter().enumerate() {
        if let Some(rsi_value) = rsi.update(*price) {
            let signal = rsi.generate_signal(rsi_value);
            println!("  Day {}: Price=${:.2}, RSI={:.2}, Signal={:?}", i+1, price, rsi_value, signal.signal_type);
        }
    }

    // Test MACD
    println!("\nTesting MACD (Moving Average Convergence Divergence):");
    let mut macd = MACDIndicator::new(12, 26, 9);
    let macd_test_prices = generate_trending_prices(50, 100.0, 0.02)?;
    
    for (i, price) in macd_test_prices.iter().enumerate().take(10) {
        if let Some(macd_data) = macd.update(*price) {
            let signal = macd.generate_signal(&macd_data);
            println!("  Day {}: Price=${:.2}, MACD={:.3}, Signal={:.3}, Histogram={:.3}, Signal={:?}", 
                     i+1, price, macd_data.macd_line, macd_data.signal_line, macd_data.histogram, signal.signal_type);
        }
    }

    // Test Bollinger Bands
    println!("\nTesting Bollinger Bands:");
    let mut bb = BollingerBandsIndicator::new(20, 2.0);
    let bb_test_prices = generate_volatile_prices(30, 100.0, 0.03)?;
    
    for (i, price) in bb_test_prices.iter().enumerate().skip(19).take(10) {
        if let Some(bands) = bb.update(*price) {
            let signal = bb.generate_signal(&bands, *price);
            println!("  Day {}: Price=${:.2}, Upper=${:.2}, Middle=${:.2}, Lower=${:.2}, Signal={:?}", 
                     i+1, price, bands.upper, bands.middle, bands.lower, signal.signal_type);
        }
    }

    println!();
    Ok(())
}

/// Test the trend reversal strategy
async fn test_trend_reversal_strategy() -> Result<()> {
    println!("📈 Testing Trend Reversal Strategy");
    println!("=================================\n");

    // Create strategy configuration
    let strategy_config = StrategyConfig {
        ema_periods: vec![9, 21, 50],
        noise_filter_threshold: 0.3,
        gap_threshold: 0.025, // 2.5% gap threshold
        reversal_confirmation_periods: 3,
        min_confidence_threshold: 0.65,
        lookback_periods: 100,
    };

    // Initialize strategy and data manager
    let mut strategy = TrendReversalStrategy::new(&strategy_config)?;
    let data_manager = MarketDataManager::new();

    // Test symbol
    let symbol = "AAPL";
    println!("Testing strategy on symbol: {}", symbol);

    // Generate sample price data to simulate market conditions
    let test_data = generate_sample_data(symbol, 150)?;
    
    println!("Generated {} price points for backtesting", test_data.len());

    // Process each price point through the strategy
    let mut signals_generated = 0;
    let mut buy_signals = 0;
    let mut sell_signals = 0;

    for (i, price_data) in test_data.iter().enumerate() {
        // Add price data to our data manager
        let ohlcv = OHLCV {
            timestamp: price_data.timestamp,
            open: price_data.open,
            high: price_data.high,
            low: price_data.low,
            close: price_data.close,
            volume: price_data.volume,
        };
        
        data_manager.add_ohlcv_data(symbol, ohlcv).await?;

        // Analyze with strategy
        let signals = strategy.analyze(symbol, price_data).await?;

        // Process any signals generated
        for signal in signals {
            signals_generated += 1;
            
            match signal.action {
                first_trading_app::strategy::Action::Buy => {
                    buy_signals += 1;
                    println!("📈 BUY Signal #{} at ${:.2} (Confidence: {:.1}%, Gap: {:.3}%)", 
                             signals_generated, signal.price, signal.confidence * 100.0, signal.ema_gap * 100.0);
                    println!("   Reasoning: {}", signal.reasoning);
                }
                first_trading_app::strategy::Action::Sell => {
                    sell_signals += 1;
                    println!("📉 SELL Signal #{} at ${:.2} (Confidence: {:.1}%, Gap: {:.3}%)", 
                             signals_generated, signal.price, signal.confidence * 100.0, signal.ema_gap * 100.0);
                    println!("   Reasoning: {}", signal.reasoning);
                }
                first_trading_app::strategy::Action::Hold => {
                    // Hold signals are not typically displayed
                }
            }
        }

        // Show progress every 50 data points
        if (i + 1) % 50 == 0 {
            println!("Processed {} price points...", i + 1);
        }
    }

    // Summary
    println!("\n{}", "=".repeat(50));
    println!("TREND REVERSAL STRATEGY SUMMARY");
    println!("{}", "=".repeat(50));
    println!("Total price points processed: {}", test_data.len());
    println!("Total signals generated: {}", signals_generated);
    println!("  - Buy signals: {}", buy_signals);
    println!("  - Sell signals: {}", sell_signals);
    println!("Signal frequency: {:.2}%", (signals_generated as f64 / test_data.len() as f64) * 100.0);
    
    // Calculate signal ratio
    if signals_generated > 0 {
        println!("Buy/Sell ratio: {:.1}:{:.1}", buy_signals as f64, sell_signals as f64);
    }
    
    Ok(())
}



/// Compare different strategies side by side
async fn test_strategy_comparison() -> Result<()> {
    println!("\n⚖️  Strategy Comparison Test");
    println!("===========================\n");
    
    let symbol = "TSLA";
    let comparison_data = generate_sample_data(symbol, 100)?;
    
    // Test basic trend reversal
    let basic_config = StrategyConfig {
        ema_periods: vec![9, 21, 50],
        noise_filter_threshold: 0.3,
        gap_threshold: 0.02,
        reversal_confirmation_periods: 2,
        min_confidence_threshold: 0.5,
        lookback_periods: 50,
    };
    
    let mut basic_strategy = TrendReversalStrategy::new(&basic_config)?;
    let mut basic_signals = 0;
    
    println!("Running basic trend reversal strategy...");
    for data_point in &comparison_data {
        let signals = basic_strategy.analyze(symbol, data_point).await?;
        basic_signals += signals.len();
    }
    
    // Test conservative settings
    let conservative_config = StrategyConfig {
        ema_periods: vec![20, 50, 100],
        noise_filter_threshold: 0.2,
        gap_threshold: 0.035, // Higher threshold
        reversal_confirmation_periods: 5,
        min_confidence_threshold: 0.8, // Higher confidence required
        lookback_periods: 100,
    };
    
    let mut conservative_strategy = TrendReversalStrategy::new(&conservative_config)?;
    let mut conservative_signals = 0;
    
    println!("Running conservative trend reversal strategy...");
    for data_point in &comparison_data {
        let signals = conservative_strategy.analyze(symbol, data_point).await?;
        conservative_signals += signals.len();
    }
    
    // Test aggressive settings
    let aggressive_config = StrategyConfig {
        ema_periods: vec![5, 10, 20],
        noise_filter_threshold: 0.6,
        gap_threshold: 0.01, // Lower threshold
        reversal_confirmation_periods: 1,
        min_confidence_threshold: 0.4, // Lower confidence required
        lookback_periods: 30,
    };
    
    let mut aggressive_strategy = TrendReversalStrategy::new(&aggressive_config)?;
    let mut aggressive_signals = 0;
    
    println!("Running aggressive trend reversal strategy...");
    for data_point in &comparison_data {
        let signals = aggressive_strategy.analyze(symbol, data_point).await?;
        aggressive_signals += signals.len();
    }
    
    println!("\n{}", "=".repeat(50));
    println!("STRATEGY COMPARISON RESULTS");
    println!("{}", "=".repeat(50));
    println!("Data points analyzed: {}", comparison_data.len());
    println!("Basic Strategy signals: {}", basic_signals);
    println!("Conservative Strategy signals: {}", conservative_signals);
    println!("Aggressive Strategy signals: {}", aggressive_signals);
    
    println!("\nSignal frequency comparison:");
    println!("  Basic: {:.2}%", (basic_signals as f64 / comparison_data.len() as f64) * 100.0);
    println!("  Conservative: {:.2}%", (conservative_signals as f64 / comparison_data.len() as f64) * 100.0);
    println!("  Aggressive: {:.2}%", (aggressive_signals as f64 / comparison_data.len() as f64) * 100.0);
    
    Ok(())
}

/// Generate trending prices for testing
fn generate_trending_prices(count: usize, start_price: f64, trend_rate: f64) -> Result<Vec<f64>> {
    let mut prices = Vec::new();
    let mut current_price = start_price;
    
    for i in 0..count {
        // Add trend
        current_price *= 1.0 + trend_rate;
        
        // Add some noise
        let noise = (i as f64 * 0.01).sin() * 0.5;
        prices.push(current_price + noise);
    }
    
    Ok(prices)
}

/// Generate volatile prices for testing
fn generate_volatile_prices(count: usize, base_price: f64, volatility: f64) -> Result<Vec<f64>> {
    let mut prices = Vec::new();
    let mut current_price = base_price;
    
    for i in 0..count {
        // Add volatility
        let change_pct = (i as f64 * 0.1).sin() * volatility;
        current_price *= 1.0 + change_pct;
        prices.push(current_price);
    }
    
    Ok(prices)
}

/// Generate complex market data patterns for testing


fn generate_sample_data(symbol: &str, count: usize) -> Result<Vec<PriceData>> {
    let mut data: Vec<PriceData> = Vec::new();
    let mut price = dec!(150.0); // Starting price
    let mut timestamp = Utc::now() - chrono::Duration::minutes(count as i64);

    for i in 0..count {
        // Create some realistic price movement
        let trend_factor = if i < count / 3 {
            0.002 // Uptrend
        } else if i < 2 * count / 3 {
            -0.001 // Downtrend
        } else {
            0.0015 // Recovery
        };

        // Add some randomness
        let random_factor = (i as f64 * 17.0).sin() * 0.01;
        let price_change = (trend_factor + random_factor) * price.to_f64().unwrap_or(150.0);
        
        price += Decimal::from_f64_retain(price_change).unwrap_or(dec!(0.0));
        
        // Ensure price doesn't go negative
        if price < dec!(1.0) {
            price = dec!(1.0);
        }

        // Create OHLC data (simplified)
        let high = price * dec!(1.005);
        let low = price * dec!(0.995);
        let open = if i == 0 { price } else { data[i-1].close };

        data.push(PriceData {
            symbol: symbol.to_string(),
            timestamp,
            open,
            high,
            low,
            close: price,
            volume: 100000 + (i * 1000) as u64,
        });

        timestamp += chrono::Duration::minutes(1);
    }

    Ok(data)
}