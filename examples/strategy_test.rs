use anyhow::Result;
use chrono::Utc;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;

// Include the modules from the main application
use first_trading_app::config::StrategyConfig;
use first_trading_app::data::{PriceData, MarketDataManager, OHLCV};
use first_trading_app::strategy::TrendReversalStrategy;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    println!("Strategy Test Example - IBKR Trend Reversal");
    println!("============================================\n");

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
    
    println!("Generated {} price points for backtesting\n", test_data.len());

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

        // Show progress every 25 data points
        if (i + 1) % 25 == 0 {
            println!("Processed {} price points...", i + 1);
        }
    }

    // Summary
    println!("\n{}", "=".repeat(50));
    println!("STRATEGY TEST SUMMARY");
    println!("{}", "=".repeat(50));
    println!("Total price points processed: {}", test_data.len());
    println!("Total signals generated: {}", signals_generated);
    println!("  - Buy signals: {}", buy_signals);
    println!("  - Sell signals: {}", sell_signals);
    println!("Signal frequency: {:.2}%", (signals_generated as f64 / test_data.len() as f64) * 100.0);

    if signals_generated > 0 {
        println!("\nStrategy Configuration Used:");
        println!("  - EMA Periods: {:?}", strategy_config.ema_periods);
        println!("  - Gap Threshold: {:.1}%", strategy_config.gap_threshold * 100.0);
        println!("  - Min Confidence: {:.1}%", strategy_config.min_confidence_threshold * 100.0);
        println!("  - Noise Filter: {:.1}", strategy_config.noise_filter_threshold);
    } else {
        println!("\nNo signals generated. Consider adjusting strategy parameters:");
        println!("  - Lower gap_threshold (currently {:.1}%)", strategy_config.gap_threshold * 100.0);
        println!("  - Lower min_confidence_threshold (currently {:.1}%)", strategy_config.min_confidence_threshold * 100.0);
        println!("  - Increase noise_filter_threshold (currently {:.1})", strategy_config.noise_filter_threshold);
    }

    // Test specific market conditions
    println!("\n{}", "=".repeat(50));
    println!("TESTING SPECIFIC MARKET CONDITIONS");
    println!("{}", "=".repeat(50));
    
    test_trending_market(&mut strategy, "Uptrend Test").await?;
    test_reversal_scenario(&mut strategy, "Reversal Test").await?;

    println!("\nStrategy test completed successfully!");
    Ok(())
}

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

async fn test_trending_market(strategy: &mut TrendReversalStrategy, test_name: &str) -> Result<()> {
    println!("\n{}", test_name);
    println!("{}", "-".repeat(test_name.len()));
    
    // Create strong uptrend followed by potential reversal
    let mut price = dec!(100.0);
    let symbol = "TEST_TREND";
    
    // Strong uptrend (should not generate sell signals)
    for i in 0..10 {
        price += dec!(2.0); // Strong upward movement
        
        let price_data = PriceData {
            symbol: symbol.to_string(),
            timestamp: Utc::now() + chrono::Duration::minutes(i),
            open: price - dec!(0.5),
            high: price + dec!(0.5),
            low: price - dec!(1.0),
            close: price,
            volume: 50000,
        };

        let signals = strategy.analyze(symbol, &price_data).await?;
        if !signals.is_empty() {
            println!("Signal during uptrend at ${:.2}: {:?}", price, signals[0].action);
        }
    }

    // Sharp reversal (should generate signals)
    for i in 0..5 {
        price -= dec!(3.0); // Sharp downward movement
        
        let price_data = PriceData {
            symbol: symbol.to_string(),
            timestamp: Utc::now() + chrono::Duration::minutes(10 + i),
            open: price + dec!(1.0),
            high: price + dec!(2.0),
            low: price - dec!(0.5),
            close: price,
            volume: 75000,
        };

        let signals = strategy.analyze(symbol, &price_data).await?;
        for signal in signals {
            println!("Reversal signal at ${:.2}: {:?} (Confidence: {:.1}%)", 
                     price, signal.action, signal.confidence * 100.0);
        }
    }

    Ok(())
}

async fn test_reversal_scenario(strategy: &mut TrendReversalStrategy, test_name: &str) -> Result<()> {
    println!("\n{}", test_name);
    println!("{}", "-".repeat(test_name.len()));
    
    let symbol = "TEST_REV";
    let _base_price = dec!(200.0);
    
    // Create scenario: price moves away from EMA, then reverses
    let scenarios = vec![
        (dec!(200.0), "Baseline"),
        (dec!(195.0), "Small drop"),
        (dec!(190.0), "Medium drop"),
        (dec!(185.0), "Large drop - potential reversal zone"),
        (dec!(188.0), "Bounce back"),
        (dec!(192.0), "Continued recovery"),
    ];

    for (i, (price, description)) in scenarios.iter().enumerate() {
        let price_data = PriceData {
            symbol: symbol.to_string(),
            timestamp: Utc::now() + chrono::Duration::minutes(i as i64),
            open: *price - dec!(0.5),
            high: *price + dec!(1.0),
            low: *price - dec!(1.5),
            close: *price,
            volume: 60000,
        };

        let signals = strategy.analyze(symbol, &price_data).await?;
        
        println!("{}: ${:.2}", description, price);
        for signal in signals {
            println!("  -> {:?} signal (Confidence: {:.1}%, Gap: {:.3}%)", 
                     signal.action, signal.confidence * 100.0, signal.ema_gap * 100.0);
        }
    }

    Ok(())
}