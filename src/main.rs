use anyhow::Result;
use log::{info, error, warn};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::io::{self, Write};
use tokio::signal;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

mod config;
mod ibkr;
mod strategy;
mod data;
mod portfolio;
mod indicators;
mod strategy_engine;
mod backtesting;
mod orders;

use config::Config;
use ibkr::IBKRClient;
use strategy::TrendReversalStrategy;
use data::MarketDataManager;
use portfolio::PortfolioManager;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    // Show startup banner
    show_banner();
    
    // Load configuration
    let config = match Config::load() {
        Ok(config) => {
            info!("Configuration loaded successfully");
            config
        }
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            warn!("Creating default configuration file...");
            let default_config = Config::default();
            if let Err(save_err) = default_config.save() {
                error!("Failed to save default configuration: {}", save_err);
            } else {
                info!("Default configuration saved to config.toml");
                info!("Please update the configuration and restart the application");
            }
            return Ok(());
        }
    };

    // Show configuration summary
    show_config_summary(&config);

    // Confirm before starting (in paper trading mode)
    if config.ibkr.paper_trading {
        info!("Paper trading mode enabled - no real money at risk");
    } else {
        warn!("LIVE TRADING MODE - REAL MONEY AT RISK!");
        if !confirm_live_trading() {
            info!("Live trading cancelled by user");
            return Ok(());
        }
    }

    // Initialize components
    let ibkr_client = Arc::new(Mutex::new(IBKRClient::new(&config.ibkr)?));
    let market_data_manager = Arc::new(Mutex::new(MarketDataManager::new()));
    let portfolio_manager = Arc::new(Mutex::new(PortfolioManager::new()));
    let strategy = Arc::new(Mutex::new(TrendReversalStrategy::new(&config.strategy)?));

    // Connect to IBKR with retry logic
    let max_retries = 3;
    let mut connected = false;
    for attempt in 1..=max_retries {
        match connect_with_retry(ibkr_client.clone(), attempt).await {
            Ok(()) => {
                connected = true;
                break;
            }
            Err(e) => {
                error!("Connection attempt {} failed: {}", attempt, e);
                if attempt < max_retries {
                    warn!("Retrying connection in 5 seconds...");
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                }
            }
        }
    }

    if !connected {
        error!("Failed to connect to IBKR after {} attempts", max_retries);
        error!("Please check that IBKR TWS/Gateway is running and properly configured");
        return Err(anyhow::anyhow!("Could not establish IBKR connection"));
    }

    // Create shutdown signal handler
    let shutdown_signal = setup_shutdown_handler();

    // Start market data collection in the current task
    let symbols = config.trading.symbols.clone();
    start_market_data_collection(
        market_data_manager.clone(),
        ibkr_client.clone(),
        symbols,
    ).await?;

    // Run the trading loop directly in the main task
    let config_clone = config.clone();
    
    tokio::select! {
        _ = shutdown_signal => {
            info!("Shutdown signal received, stopping application...");
        }
        result = trading_loop(
            strategy,
            market_data_manager,
            portfolio_manager,
            ibkr_client,
            config_clone,
        ) => {
            match result {
                Ok(()) => info!("Trading application completed successfully"),
                Err(e) => error!("Trading application error: {}", e),
            }
        }
    }

    // Cleanup
    info!("Performing cleanup...");
    info!("Application stopped");

    Ok(())
}

fn show_banner() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║               IBKR Trend Reversal Trading Bot                ║");
    println!("║                                                              ║");
    println!("║  Strategy: EMA-based trend reversal with noise filtering    ║");
    println!("║  Risk Warning: Trading involves substantial risk of loss    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}

fn show_config_summary(config: &Config) {
    info!("Configuration Summary:");
    info!("  IBKR Host: {}:{}", config.ibkr.host, config.ibkr.port);
    info!("  Paper Trading: {}", config.ibkr.paper_trading);
    info!("  Symbols: {:?}", config.trading.symbols);
    info!("  Max Positions: {}", config.trading.max_positions);
    info!("  Position Size: {:.1}%", config.trading.position_size_pct * 100.0);
    info!("  Daily Loss Limit: {:.1}%", config.risk.max_daily_loss_pct * 100.0);
    info!("  EMA Periods: {:?}", config.strategy.ema_periods);
}

fn confirm_live_trading() -> bool {
    print!("You are about to start LIVE TRADING with real money. Continue? (yes/no): ");
    io::stdout().flush().unwrap();
    
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    
    let input = input.trim().to_lowercase();
    input == "yes" || input == "y"
}

async fn connect_with_retry(ibkr_client: Arc<Mutex<IBKRClient>>, attempt: u32) -> Result<()> {
    info!("Connecting to IBKR TWS/Gateway (attempt {})...", attempt);
    let mut client = ibkr_client.lock().await;
    client.connect().await?;
    info!("Successfully connected to IBKR TWS/Gateway");
    Ok(())
}

async fn setup_shutdown_handler() {
    match signal::ctrl_c().await {
        Ok(()) => {}
        Err(err) => {
            error!("Unable to listen for shutdown signal: {}", err);
        }
    }
}

async fn start_market_data_collection(
    data_manager: Arc<Mutex<MarketDataManager>>,
    ibkr_client: Arc<Mutex<IBKRClient>>,
    symbols: Vec<String>,
) -> Result<()> {
    info!("Starting market data collection for symbols: {:?}", symbols);
    
    // Subscribe to market data for all symbols
    for symbol in &symbols {
        let mut client = ibkr_client.lock().await;
        client.subscribe_market_data(symbol).await?;
        info!("Subscribed to market data for {}", symbol);
    }

    // Start market data handler
    {
        let client = ibkr_client.lock().await;
        let data_manager_clone = data_manager.clone();
        
        client.start_market_data_handler(move |market_data| {
            let data_manager = data_manager_clone.clone();
            tokio::spawn(async move {
                let data_manager = data_manager.lock().await;
                if let Err(e) = data_manager.add_market_data(market_data).await {
                    warn!("Failed to add market data: {}", e);
                }
            });
        }).await?;
    }

    info!("Market data collection setup completed");
    Ok(())
}

async fn trading_loop(
    strategy: Arc<Mutex<TrendReversalStrategy>>,
    market_data_manager: Arc<Mutex<MarketDataManager>>,
    portfolio_manager: Arc<Mutex<PortfolioManager>>,
    ibkr_client: Arc<Mutex<IBKRClient>>,
    config: Config,
) -> Result<()> {
    info!("Starting trading loop");
    let mut iteration_count = 0;
    let mut last_portfolio_summary = std::time::Instant::now();
    
    loop {
        iteration_count += 1;
        
        // Check connection health periodically
        if iteration_count % 100 == 0 {
            let client = ibkr_client.lock().await;
            if !client.is_connected().await {
                error!("Lost connection to IBKR, attempting to reconnect...");
                drop(client);
                if let Err(e) = connect_with_retry(ibkr_client.clone(), 1).await {
                    error!("Failed to reconnect: {}", e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
                    continue;
                }
            }
        }

        // Get latest market data
        let market_data = {
            let data_manager = market_data_manager.lock().await;
            match data_manager.get_latest_data().await {
                Ok(data) => data,
                Err(e) => {
                    warn!("Failed to get market data: {}", e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    continue;
                }
            }
        };

        // Skip if no market data available
        if market_data.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            continue;
        }

        // Process each symbol
        for (symbol, price_data) in market_data {
            // Check risk limits before processing
            {
                let portfolio = portfolio_manager.lock().await;
                let violations = portfolio.check_risk_limits(
                    Decimal::from_f64_retain(config.risk.max_daily_loss_pct).unwrap_or(dec!(0.02)),
                    Decimal::from_f64_retain(config.risk.max_position_size_pct).unwrap_or(dec!(0.15)),
                );
                
                if !violations.is_empty() {
                    warn!("Risk limit violations detected for {}: {:?}", symbol, violations);
                    continue;
                }
            }

            // Run strategy analysis
            let signals = {
                let mut strat = strategy.lock().await;
                match strat.analyze(&symbol, &price_data).await {
                    Ok(signals) => signals,
                    Err(e) => {
                        warn!("Strategy analysis failed for {}: {}", symbol, e);
                        continue;
                    }
                }
            };

            // Execute trades based on signals
            for signal in signals {
                let mut portfolio = portfolio_manager.lock().await;
                let current_position = portfolio.get_position(&symbol)?;
                
                // Additional validation before trade execution
                let can_trade = portfolio.can_take_position(
                    &symbol,
                    match signal.action {
                        strategy::Action::Buy => 100,
                        strategy::Action::Sell => -100,
                        strategy::Action::Hold => 0,
                    },
                    signal.price,
                    Decimal::from_f64_retain(config.risk.max_position_size_pct).unwrap_or(dec!(0.15)),
                )?;

                if should_execute_trade(&signal, &current_position) && can_trade {
                    let mut client = ibkr_client.lock().await;
                    match client.place_order(&signal.to_order(&symbol)?).await {
                        Ok(order_id) => {
                            info!("Order placed successfully: {} for {} (confidence: {:.2}, gap: {:.4})", 
                                  order_id, symbol, signal.confidence, signal.ema_gap);
                            portfolio.update_position(&symbol, &signal)?;
                        }
                        Err(e) => {
                            error!("Failed to place order for {}: {}", symbol, e);
                        }
                    }
                } else if !can_trade {
                    warn!("Trade blocked by risk limits for {}: {:?}", symbol, signal.action);
                }
            }
        }

        // Update market prices in portfolio
        {
            let mut portfolio = portfolio_manager.lock().await;
            let mut current_prices = std::collections::HashMap::new();
            
            let data_manager = market_data_manager.lock().await;
            for symbol in &config.trading.symbols {
                if let Some(price) = data_manager.get_latest_price(symbol).await {
                    current_prices.insert(symbol.clone(), price);
                }
            }
            
            if !current_prices.is_empty() {
                if let Err(e) = portfolio.update_market_prices(&current_prices) {
                    warn!("Failed to update portfolio prices: {}", e);
                }
            }
        }

        // Show portfolio summary periodically
        if last_portfolio_summary.elapsed().as_secs() >= 60 {
            let portfolio = portfolio_manager.lock().await;
            let summary = portfolio.get_portfolio_summary();
            info!("Portfolio Summary - Total: ${:.2}, P&L: ${:.2} ({:.2}%), Positions: {}", 
                  summary.total_value, summary.daily_pnl, summary.daily_pnl_pct, summary.number_of_positions);
            last_portfolio_summary = std::time::Instant::now();
        }

        // Sleep before next iteration
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
}

fn should_execute_trade(signal: &strategy::TradingSignal, current_position: &portfolio::Position) -> bool {
    // Implement trade execution logic based on:
    // - Signal strength
    // - Current position
    // - Risk management rules
    // - Portfolio constraints
    
    match signal.action {
        strategy::Action::Buy => {
            current_position.quantity <= 0 && signal.confidence > 0.7
        }
        strategy::Action::Sell => {
            current_position.quantity >= 0 && signal.confidence > 0.7
        }
        strategy::Action::Hold => false,
    }
}