use anyhow::Result;
use chrono::{DateTime, Utc, NaiveDate, TimeZone};
use log::{info, error, warn};
use std::sync::Arc;
use tokio::sync::Mutex;
use clap::{Arg, Command};

use first_trading_app::config::Config;
use first_trading_app::ibkr::IBKRClient;
use first_trading_app::historical_backtesting::{HistoricalBacktestEngine, HistoricalBacktestConfig};
use rust_decimal::Decimal;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    // Load environment variables
    dotenv::dotenv().ok();
    
    // Parse command line arguments
    let matches = Command::new("Historical Backtest")
        .about("Run historical backtests using IBKR data")
        .arg(Arg::new("symbols")
            .short('s')
            .long("symbols")
            .value_name("SYMBOLS")
            .help("Comma-separated list of symbols to backtest")
            .takes_value(true))
        .arg(Arg::new("start-date")
            .long("start-date")
            .value_name("YYYY-MM-DD")
            .help("Start date for backtest")
            .takes_value(true))
        .arg(Arg::new("end-date")
            .long("end-date")
            .value_name("YYYY-MM-DD")
            .help("End date for backtest")
            .takes_value(true))
        .arg(Arg::new("bar-size")
            .long("bar-size")
            .value_name("SIZE")
            .help("Bar size: 1 day, 1 hour, 30 mins, 15 mins, 5 mins, 1 min")
            .takes_value(true))
        .arg(Arg::new("capital")
            .long("capital")
            .value_name("AMOUNT")
            .help("Initial capital for backtest")
            .takes_value(true))
        .arg(Arg::new("config")
            .short('c')
            .long("config")
            .value_name("FILE")
            .help("Configuration file path")
            .takes_value(true))
        .arg(Arg::new("output")
            .short('o')
            .long("output")
            .value_name("DIR")
            .help("Output directory for results")
            .takes_value(true))
        .arg(Arg::new("verbose")
            .short('v')
            .long("verbose")
            .help("Enable verbose logging")
            .takes_value(false))
        .get_matches();

    // Show banner
    show_banner();

    // Load configuration
    let mut config = Config::load().map_err(|e| {
        error!("Failed to load configuration: {}", e);
        e
    })?;

    // Override configuration with command line arguments
    if let Some(mut backtest_config) = config.historical_backtesting.clone() {
        // Override symbols
        if let Some(symbols_str) = matches.value_of("symbols") {
            backtest_config.symbols = symbols_str
                .split(',')
                .map(|s| s.trim().to_uppercase())
                .collect();
        }

        // Override dates
        if let Some(start_date) = matches.value_of("start-date") {
            backtest_config.start_date = start_date.to_string();
        }
        if let Some(end_date) = matches.value_of("end-date") {
            backtest_config.end_date = end_date.to_string();
        }

        // Override bar size
        if let Some(bar_size) = matches.value_of("bar-size") {
            backtest_config.bar_size = bar_size.to_string();
        }

        // Override capital
        if let Some(capital_str) = matches.value_of("capital") {
            if let Ok(capital) = capital_str.parse::<f64>() {
                backtest_config.initial_capital = capital;
            }
        }

        // Override output directory
        if let Some(output_dir) = matches.value_of("output") {
            backtest_config.output_directory = output_dir.to_string();
        }

        // Enable detailed logging if verbose
        if matches.is_present("verbose") {
            backtest_config.detailed_logging = true;
        }

        config.historical_backtesting = Some(backtest_config);
    } else {
        error!("No historical backtesting configuration found");
        return Err(anyhow::anyhow!("Historical backtesting not configured"));
    }

    let backtest_config = config.historical_backtesting.unwrap();

    // Validate configuration
    validate_backtest_config(&backtest_config)?;

    // Show configuration summary
    show_config_summary(&backtest_config);

    // Confirm before starting
    if !confirm_backtest(&backtest_config) {
        info!("Backtest cancelled by user");
        return Ok(());
    }

    // Create IBKR client
    let ibkr_client = Arc::new(Mutex::new(IBKRClient::new(&config.ibkr)?));

    // Skip IBKR connection for mock data mode
    info!("Using mock data mode - skipping IBKR connection");

    // Create backtest configuration with proper date parsing
    let historical_config = create_historical_config(&backtest_config)?;

    // Create and run backtest engine
    let strategy_config = config.enhanced_strategy.unwrap_or_default();
    let mut backtest_engine = HistoricalBacktestEngine::new(
        historical_config,
        ibkr_client.clone(),
        &strategy_config,
    )?;

    info!("Starting historical backtest...");
    let performance = backtest_engine.run_backtest().await?;

    // Display results
    backtest_engine.print_summary(&performance);

    // Print detailed results if verbose
    if backtest_config.detailed_logging {
        print_detailed_results(&performance);
    }

    info!("Backtest completed successfully!");

    Ok(())
}

fn show_banner() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                 HISTORICAL BACKTEST ENGINE                  ║");
    println!("║                                                              ║");
    println!("║  Test your trading strategies against historical IBKR data  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}

fn validate_backtest_config(config: &first_trading_app::config::HistoricalBacktestConfig) -> Result<()> {
    // Validate symbols
    if config.symbols.is_empty() {
        return Err(anyhow::anyhow!("No symbols specified for backtesting"));
    }

    // Validate dates
    let start_date = NaiveDate::parse_from_str(&config.start_date, "%Y-%m-%d")
        .map_err(|_| anyhow::anyhow!("Invalid start date format. Use YYYY-MM-DD"))?;
    let end_date = NaiveDate::parse_from_str(&config.end_date, "%Y-%m-%d")
        .map_err(|_| anyhow::anyhow!("Invalid end date format. Use YYYY-MM-DD"))?;

    if start_date >= end_date {
        return Err(anyhow::anyhow!("Start date must be before end date"));
    }

    // Validate bar size
    let valid_bar_sizes = ["1 day", "1 hour", "30 mins", "15 mins", "5 mins", "1 min"];
    if !valid_bar_sizes.contains(&config.bar_size.as_str()) {
        return Err(anyhow::anyhow!("Invalid bar size. Valid options: {:?}", valid_bar_sizes));
    }

    // Validate capital
    if config.initial_capital <= 0.0 {
        return Err(anyhow::anyhow!("Initial capital must be positive"));
    }

    Ok(())
}

fn show_config_summary(config: &first_trading_app::config::HistoricalBacktestConfig) {
    info!("Backtest Configuration:");
    info!("  Symbols: {:?}", config.symbols);
    info!("  Period: {} to {}", config.start_date, config.end_date);
    info!("  Bar Size: {}", config.bar_size);
    info!("  Initial Capital: ${:.2}", config.initial_capital);
    info!("  Commission: ${:.2} per trade", config.commission_per_trade);
    info!("  Slippage: {:.3}%", config.slippage_pct * 100.0);
    info!("  Data Type: {}", config.what_to_show);
    info!("  Regular Hours Only: {}", config.use_rth);
    if let Some(benchmark) = &config.benchmark_symbol {
        info!("  Benchmark: {}", benchmark);
    }
    info!("  Output Directory: {}", config.output_directory);
    println!();
}

fn confirm_backtest(config: &first_trading_app::config::HistoricalBacktestConfig) -> bool {
    use std::io::{self, Write};

    // Calculate approximate duration
    let start_date = NaiveDate::parse_from_str(&config.start_date, "%Y-%m-%d").unwrap();
    let end_date = NaiveDate::parse_from_str(&config.end_date, "%Y-%m-%d").unwrap();
    let duration_days = (end_date - start_date).num_days();

    println!("You are about to run a backtest with the following parameters:");
    println!("  • {} symbols over {} days", config.symbols.len(), duration_days);
    println!("  • Bar size: {}", config.bar_size);
    println!("  • This will use mock historical data for testing");
    
    print!("Continue with backtest? (yes/no): ");
    io::stdout().flush().unwrap();
    
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    
    let input = input.trim().to_lowercase();
    input == "yes" || input == "y"
}

fn create_historical_config(
    config: &first_trading_app::config::HistoricalBacktestConfig
) -> Result<first_trading_app::historical_backtesting::HistoricalBacktestConfig> {
    // Parse dates
    let start_date = NaiveDate::parse_from_str(&config.start_date, "%Y-%m-%d")?;
    let end_date = NaiveDate::parse_from_str(&config.end_date, "%Y-%m-%d")?;

    let start_datetime = Utc.from_utc_datetime(&start_date.and_hms_opt(0, 0, 0).unwrap());
    let end_datetime = Utc.from_utc_datetime(&end_date.and_hms_opt(23, 59, 59).unwrap());

    Ok(first_trading_app::historical_backtesting::HistoricalBacktestConfig {
        enabled: true,
        symbols: config.symbols.clone(),
        start_date: start_datetime,
        end_date: end_datetime,
        bar_size: config.bar_size.clone(),
        initial_capital: Decimal::from_f64_retain(config.initial_capital).unwrap(),
        commission_per_trade: Decimal::from_f64_retain(config.commission_per_trade).unwrap(),
        slippage_pct: config.slippage_pct,
        what_to_show: config.what_to_show.clone(),
        use_rth: config.use_rth,
        benchmark_symbol: config.benchmark_symbol.clone(),
        save_results: config.save_results,
        output_directory: config.output_directory.clone(),
        generate_charts: config.generate_charts,
        detailed_logging: config.detailed_logging,
    })
}

fn print_detailed_results(performance: &first_trading_app::historical_backtesting::BacktestPerformance) {
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                      DETAILED RESULTS");
    println!("═══════════════════════════════════════════════════════════════");

    // Risk Metrics
    println!("\nRisk Metrics:");
    println!("  Value at Risk (95%):     {:.2}%", performance.risk_metrics.value_at_risk_95 * 100.0);
    println!("  Value at Risk (99%):     {:.2}%", performance.risk_metrics.value_at_risk_99 * 100.0);
    println!("  Conditional VaR (95%):   {:.2}%", performance.risk_metrics.conditional_var_95 * 100.0);
    println!("  Downside Deviation:      {:.2}%", performance.risk_metrics.downside_deviation * 100.0);
    println!("  Upside Deviation:        {:.2}%", performance.risk_metrics.upside_deviation * 100.0);
    println!("  Pain Index:              {:.2}%", performance.risk_metrics.pain_index * 100.0);
    println!("  Ulcer Index:             {:.2}%", performance.risk_metrics.ulcer_index * 100.0);

    // Symbol Performance
    if !performance.symbol_performance.is_empty() {
        println!("\nSymbol Performance:");
        let mut sorted_symbols: Vec<_> = performance.symbol_performance.values().collect();
        sorted_symbols.sort_by(|a, b| b.total_return.partial_cmp(&a.total_return).unwrap());
        
        for symbol_perf in sorted_symbols {
            println!("  {}: Return {:.2}%, Win Rate {:.1}%, Trades {}, Avg Hold {:.1}h",
                     symbol_perf.symbol,
                     symbol_perf.total_return * 100.0,
                     symbol_perf.win_rate * 100.0,
                     symbol_perf.total_trades,
                     symbol_perf.avg_hold_time_hours);
        }
    }

    // Monthly Returns
    if !performance.monthly_returns.is_empty() {
        println!("\nMonthly Returns (last 12 months):");
        for monthly in performance.monthly_returns.iter().rev().take(12).rev() {
            println!("  {}-{:02}: {:.2}% ({} trades, {:.1}% win rate)",
                     monthly.year, monthly.month,
                     monthly.return_pct,
                     monthly.trades,
                     monthly.win_rate * 100.0);
        }
    }

    // Yearly Returns
    if !performance.yearly_returns.is_empty() {
        println!("\nYearly Returns:");
        for yearly in &performance.yearly_returns {
            println!("  {}: {:.2}% ({} trades)",
                     yearly.year,
                     yearly.return_pct,
                     yearly.trades);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════\n");
}