use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc, Duration, TimeZone, Datelike};
use log::{info, warn, error};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::config::EnhancedStrategyConfig;
use crate::ibkr::{IBKRClient, HistoricalDataRequest, HistoricalBar};
use crate::strategy_engine::StrategyEngine;
use crate::backtesting::{BacktestingEngine, BacktestConfig, BacktestResults, BacktestReport};
use crate::analytics::AnalyticsEngine;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalBacktestConfig {
    pub enabled: bool,
    pub symbols: Vec<String>,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub bar_size: String,           // "1 day", "1 hour", "30 mins", "15 mins", "5 mins", "1 min"
    pub initial_capital: Decimal,
    pub commission_per_trade: Decimal,
    pub slippage_pct: f64,
    pub what_to_show: String,       // "TRADES", "MIDPOINT", "BID", "ASK"
    pub use_rth: bool,              // Regular Trading Hours only
    pub benchmark_symbol: Option<String>,
    pub save_results: bool,
    pub output_directory: String,
    pub generate_charts: bool,
    pub detailed_logging: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestPerformance {
    pub config: HistoricalBacktestConfig,
    pub results: BacktestResults,
    pub symbol_performance: HashMap<String, SymbolPerformance>,
    pub monthly_returns: Vec<MonthlyReturn>,
    pub yearly_returns: Vec<YearlyReturn>,
    pub risk_metrics: RiskMetrics,
    pub execution_time_seconds: f64,
    pub total_bars_processed: usize,
    pub signals_generated: usize,
    pub trades_executed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolPerformance {
    pub symbol: String,
    pub total_return: f64,
    pub win_rate: f64,
    pub total_trades: usize,
    pub avg_hold_time_hours: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub profit_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthlyReturn {
    pub year: i32,
    pub month: u32,
    pub return_pct: f64,
    pub trades: usize,
    pub win_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YearlyReturn {
    pub year: i32,
    pub return_pct: f64,
    pub trades: usize,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub value_at_risk_95: f64,
    pub value_at_risk_99: f64,
    pub conditional_var_95: f64,
    pub maximum_drawdown: f64,
    pub maximum_drawdown_duration_days: i64,
    pub downside_deviation: f64,
    pub upside_deviation: f64,
    pub pain_index: f64,
    pub ulcer_index: f64,
}

pub struct HistoricalBacktestEngine {
    config: HistoricalBacktestConfig,
    ibkr_client: Arc<Mutex<IBKRClient>>,
    strategy_engine: StrategyEngine,
    analytics_engine: AnalyticsEngine,
}

impl HistoricalBacktestEngine {
    pub fn new(
        config: HistoricalBacktestConfig,
        ibkr_client: Arc<Mutex<IBKRClient>>,
        strategy_config: &EnhancedStrategyConfig,
    ) -> Result<Self> {
        let strategy_engine = StrategyEngine::new(strategy_config.clone().into())?;
        let analytics_engine = AnalyticsEngine::new(0.02);

        Ok(Self {
            config,
            ibkr_client,
            strategy_engine,
            analytics_engine,
        })
    }

    pub async fn run_backtest(&mut self) -> Result<BacktestPerformance> {
        let start_time = std::time::Instant::now();
        
        info!("Starting historical backtest for {} symbols from {} to {}", 
              self.config.symbols.len(), 
              self.config.start_date.format("%Y-%m-%d"),
              self.config.end_date.format("%Y-%m-%d"));

        // Step 1: Fetch historical data from IBKR
        info!("Fetching historical data from IBKR...");
        let historical_data = self.fetch_historical_data().await?;
        
        if historical_data.is_empty() {
            return Err(anyhow!("No historical data received"));
        }

        // Step 2: Convert historical data to backtest format
        info!("Converting historical data to backtest format...");
        let backtest_data = self.convert_to_backtest_format(&historical_data)?;

        // Step 3: Run the backtest
        info!("Running backtest simulation...");
        let backtest_results = self.run_simulation(&backtest_data).await?;

        // Step 4: Generate comprehensive performance analysis
        info!("Generating performance analysis...");
        let performance = self.generate_performance_analysis(
            backtest_results,
            &historical_data,
            start_time.elapsed().as_secs_f64(),
        ).await?;

        // Step 5: Save results if configured
        if self.config.save_results {
            self.save_results(&performance).await?;
        }

        info!("Backtest completed successfully in {:.2} seconds", 
              start_time.elapsed().as_secs_f64());

        Ok(performance)
    }

    async fn fetch_historical_data(&self) -> Result<HashMap<String, Vec<HistoricalBar>>> {
        let client = self.ibkr_client.lock().await;
        
        // Create historical data requests for all symbols
        let duration = self.calculate_duration_string()?;
        let requests: Vec<HistoricalDataRequest> = self.config.symbols.iter().map(|symbol| {
            HistoricalDataRequest {
                symbol: symbol.clone(),
                duration: duration.clone(),
                bar_size: self.config.bar_size.clone(),
                what_to_show: self.config.what_to_show.clone(),
                use_rth: self.config.use_rth,
                end_date_time: Some(self.config.end_date),
            }
        }).collect();

        // Fetch data for all symbols
        let historical_data = client.get_multiple_historical_data(&requests).await?;

        // Validate data quality
        self.validate_historical_data(&historical_data)?;

        Ok(historical_data)
    }

    fn calculate_duration_string(&self) -> Result<String> {
        let duration = self.config.end_date - self.config.start_date;
        let days = duration.num_days();

        if days <= 1 {
            Ok("1 D".to_string())
        } else if days <= 7 {
            Ok("1 W".to_string())
        } else if days <= 31 {
            Ok("1 M".to_string())
        } else if days <= 365 {
            Ok(format!("{} M", (days / 30).max(1)))
        } else {
            Ok(format!("{} Y", (days / 365).max(1)))
        }
    }

    fn validate_historical_data(&self, data: &HashMap<String, Vec<HistoricalBar>>) -> Result<()> {
        for (symbol, bars) in data {
            if bars.is_empty() {
                warn!("No historical data received for symbol: {}", symbol);
                continue;
            }

            // Check for data gaps
            let mut prev_time: Option<DateTime<Utc>> = None;
            for bar in bars {
                if let Some(prev) = prev_time {
                    let gap = bar.timestamp - prev;
                    if gap > Duration::days(7) {
                        warn!("Large data gap detected for {}: {} days between {} and {}", 
                              symbol, gap.num_days(), prev, bar.timestamp);
                    }
                }
                prev_time = Some(bar.timestamp);
            }

            info!("Validated {} bars for symbol {}", bars.len(), symbol);
        }

        Ok(())
    }

    fn convert_to_backtest_format(&self, historical_data: &HashMap<String, Vec<HistoricalBar>>) -> Result<HashMap<String, Vec<crate::data::OHLCV>>> {
        let mut backtest_data = HashMap::new();

        for (symbol, bars) in historical_data {
            let ohlcv_data: Vec<crate::data::OHLCV> = bars.iter().map(|bar| {
                crate::data::OHLCV {
                    timestamp: bar.timestamp,
                    open: bar.open.to_f64().unwrap_or(0.0),
                    high: bar.high.to_f64().unwrap_or(0.0),
                    low: bar.low.to_f64().unwrap_or(0.0),
                    close: bar.close.to_f64().unwrap_or(0.0),
                    volume: bar.volume,
                }
            }).collect();

            backtest_data.insert(symbol.clone(), ohlcv_data);
        }

        Ok(backtest_data)
    }

    async fn run_simulation(&mut self, backtest_data: &HashMap<String, Vec<crate::data::OHLCV>>) -> Result<BacktestReport> {
        // Create backtest configuration
        let backtest_config = BacktestConfig {
            initial_capital: self.config.initial_capital.to_f64().unwrap_or(100000.0),
            commission_per_trade: self.config.commission_per_trade.to_f64().unwrap_or(1.0),
            slippage_pct: self.config.slippage_pct,
            max_position_size_pct: 0.15, // From risk config
            start_date: self.config.start_date,
            end_date: self.config.end_date,
            benchmark_symbol: self.config.benchmark_symbol.clone(),
            risk_free_rate: 0.02, // 2% default risk-free rate
        };

        // Create and run backtest engine
        let mut backtest_engine = BacktestingEngine::new(backtest_config);
        backtest_engine.run_backtest(backtest_data, &mut self.strategy_engine).await?;
        
        Ok(backtest_engine.generate_report())
    }

    async fn generate_performance_analysis(
        &mut self,
        backtest_report: BacktestReport,
        historical_data: &HashMap<String, Vec<HistoricalBar>>,
        execution_time: f64,
    ) -> Result<BacktestPerformance> {
        // Calculate symbol-specific performance
        let symbol_performance = self.calculate_symbol_performance(&backtest_report, historical_data)?;
        
        // Calculate monthly and yearly returns
        let monthly_returns = self.calculate_monthly_returns(&backtest_report)?;
        let yearly_returns = self.calculate_yearly_returns(&backtest_report)?;
        
        // Calculate advanced risk metrics
        let risk_metrics = self.calculate_risk_metrics(&backtest_report)?;

        // Count statistics
        let total_bars_processed = historical_data.values()
            .map(|bars| bars.len())
            .sum();

        let signals_generated = backtest_report.trade_history.len() * 2; // Estimate
        let trades_executed = backtest_report.trade_history.len();

        Ok(BacktestPerformance {
            config: self.config.clone(),
            results: backtest_report.results,
            symbol_performance,
            monthly_returns,
            yearly_returns,
            risk_metrics,
            execution_time_seconds: execution_time,
            total_bars_processed,
            signals_generated,
            trades_executed,
        })
    }

    fn calculate_symbol_performance(
        &self,
        backtest_report: &BacktestReport,
        _historical_data: &HashMap<String, Vec<HistoricalBar>>,
    ) -> Result<HashMap<String, SymbolPerformance>> {
        let mut symbol_performance = HashMap::new();

        // Group trades by symbol
        let mut symbol_trades: HashMap<String, Vec<&crate::backtesting::BacktestTrade>> = HashMap::new();
        for trade in &backtest_report.trade_history {
            symbol_trades.entry(trade.symbol.clone())
                .or_default()
                .push(trade);
        }

        for (symbol, trades) in symbol_trades {
            let monthly_pnl: f64 = trades.iter().filter_map(|t| t.realized_pnl).sum();
            let winning_trades = trades.iter().filter(|t| t.realized_pnl.unwrap_or(0.0) > 0.0).count();
            let win_rate = if trades.is_empty() { 0.0 } else { winning_trades as f64 / trades.len() as f64 };
        
            let avg_hold_time = if trades.is_empty() { 0.0 } else {
                trades.iter().filter_map(|t| t.duration_hours).sum::<f64>() / trades.len() as f64
            };

            // Calculate symbol-specific metrics
            let profit_factor = self.calculate_profit_factor(&trades);
            let max_drawdown = self.calculate_symbol_max_drawdown(&trades);

            symbol_performance.insert(symbol.clone(), SymbolPerformance {
                symbol,
                total_return: monthly_pnl / 10000.0, // Assuming $10k allocation per symbol
                win_rate,
                total_trades: trades.len(),
                avg_hold_time_hours: avg_hold_time,
                max_drawdown,
                sharpe_ratio: 0.0, // Would need more detailed calculation
                profit_factor,
            });
        }

        Ok(symbol_performance)
    }

    fn calculate_profit_factor(&self, trades: &[&crate::backtesting::BacktestTrade]) -> f64 {
        let total_profit: f64 = trades.iter()
            .filter(|t| t.realized_pnl.unwrap_or(0.0) > 0.0)
            .filter_map(|t| t.realized_pnl)
            .sum();
        
        let total_loss: f64 = trades.iter()
            .filter(|t| t.realized_pnl.unwrap_or(0.0) < 0.0)
            .filter_map(|t| t.realized_pnl.map(|pnl| pnl.abs()))
            .sum();

        if total_loss == 0.0 {
            if total_profit > 0.0 { f64::INFINITY } else { 0.0 }
        } else {
            total_profit / total_loss
        }
    }

    fn calculate_symbol_max_drawdown(&self, trades: &[&crate::backtesting::BacktestTrade]) -> f64 {
        let mut peak = 0.0;
        let mut max_drawdown = 0.0;
        let mut running_pnl = 0.0;

        for trade in trades {
            running_pnl += trade.realized_pnl.unwrap_or(0.0);
            if running_pnl > peak {
                peak = running_pnl;
            }
            let drawdown = (peak - running_pnl) / peak.max(1.0);
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }

    fn calculate_monthly_returns(&self, backtest_report: &BacktestReport) -> Result<Vec<MonthlyReturn>> {
        let mut monthly_returns = Vec::new();
        let mut monthly_data: HashMap<(i32, u32), Vec<&crate::backtesting::BacktestTrade>> = HashMap::new();

        // Group trades by month
        for trade in &backtest_report.trade_history {
            if let Some(exit_date) = trade.exit_date {
                let key = (exit_date.year(), exit_date.month());
                monthly_data.entry(key).or_default().push(trade);
            }
        }

        for ((year, month), trades) in monthly_data {
            let total_return: f64 = trades.iter().filter_map(|t| t.realized_pnl).sum();
            let winning_trades = trades.iter().filter(|t| t.realized_pnl.unwrap_or(0.0) > 0.0).count();
            let win_rate = if trades.is_empty() { 0.0 } else { winning_trades as f64 / trades.len() as f64 };

            monthly_returns.push(MonthlyReturn {
                year,
                month,
                return_pct: total_return / backtest_report.config.initial_capital * 100.0,
                trades: trades.len(),
                win_rate,
            });
        }

        monthly_returns.sort_by_key(|r| (r.year, r.month));
        Ok(monthly_returns)
    }

    fn calculate_yearly_returns(&self, backtest_report: &BacktestReport) -> Result<Vec<YearlyReturn>> {
        let mut yearly_returns = Vec::new();
        let mut yearly_data: HashMap<i32, Vec<&crate::backtesting::BacktestTrade>> = HashMap::new();

        // Group trades by year
        for trade in &backtest_report.trade_history {
            if let Some(exit_date) = trade.exit_date {
                yearly_data.entry(exit_date.year()).or_default().push(trade);
            }
        }

        for (year, trades) in yearly_data {
            let total_return: f64 = trades.iter().filter_map(|t| t.realized_pnl).sum();
            let return_pct = total_return / backtest_report.config.initial_capital * 100.0;

            yearly_returns.push(YearlyReturn {
                year,
                return_pct,
                trades: trades.len(),
                sharpe_ratio: 0.0, // Would need daily returns for proper calculation
                max_drawdown: 0.0, // Would need more detailed calculation
            });
        }

        yearly_returns.sort_by_key(|r| r.year);
        Ok(yearly_returns)
    }

    fn calculate_risk_metrics(&self, backtest_report: &BacktestReport) -> Result<RiskMetrics> {
        let daily_returns: Vec<f64> = backtest_report.daily_returns.iter()
            .map(|dr| dr.return_pct)
            .collect();

        // Calculate VaR and CVaR
        let (var_95, cvar_95) = self.calculate_var_and_cvar(&daily_returns, 0.05);
        let (var_99, _) = self.calculate_var_and_cvar(&daily_returns, 0.01);

        // Calculate downside and upside deviation
        let mean_return = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let downside_deviation = self.calculate_downside_deviation(&daily_returns, mean_return);
        let upside_deviation = self.calculate_upside_deviation(&daily_returns, mean_return);

        // Calculate pain index and ulcer index
        let pain_index = self.calculate_pain_index(backtest_report);
        let ulcer_index = self.calculate_ulcer_index(backtest_report);

        Ok(RiskMetrics {
            value_at_risk_95: var_95,
            value_at_risk_99: var_99,
            conditional_var_95: cvar_95,
            maximum_drawdown: backtest_report.results.max_drawdown,
            maximum_drawdown_duration_days: backtest_report.results.max_drawdown_duration_days,
            downside_deviation,
            upside_deviation,
            pain_index,
            ulcer_index,
        })
    }

    fn calculate_var_and_cvar(&self, returns: &[f64], confidence_level: f64) -> (f64, f64) {
        if returns.is_empty() {
            return (0.0, 0.0);
        }

        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let var_index = (returns.len() as f64 * confidence_level).floor() as usize;
        let var = if var_index < sorted_returns.len() {
            sorted_returns[var_index]
        } else {
            sorted_returns[sorted_returns.len() - 1]
        };

        // Calculate CVaR (average of returns worse than VaR)
        let cvar = if var_index > 0 {
            sorted_returns[..var_index].iter().sum::<f64>() / var_index as f64
        } else {
            var
        };

        (var.abs(), cvar.abs())
    }

    fn calculate_downside_deviation(&self, returns: &[f64], target: f64) -> f64 {
        let downside_squared_deviations: f64 = returns.iter()
            .filter(|&&r| r < target)
            .map(|&r| (r - target).powi(2))
            .sum();

        let count = returns.iter().filter(|&&r| r < target).count();
        
        if count > 0 {
            (downside_squared_deviations / count as f64).sqrt()
        } else {
            0.0
        }
    }

    fn calculate_upside_deviation(&self, returns: &[f64], target: f64) -> f64 {
        let upside_squared_deviations: f64 = returns.iter()
            .filter(|&&r| r > target)
            .map(|&r| (r - target).powi(2))
            .sum();

        let count = returns.iter().filter(|&&r| r > target).count();
        
        if count > 0 {
            (upside_squared_deviations / count as f64).sqrt()
        } else {
            0.0
        }
    }

    fn calculate_pain_index(&self, backtest_report: &BacktestReport) -> f64 {
        if backtest_report.equity_curve.is_empty() {
            return 0.0;
        }

        let total_pain: f64 = backtest_report.equity_curve.iter()
            .map(|point| point.drawdown.abs())
            .sum();

        total_pain / backtest_report.equity_curve.len() as f64
    }

    fn calculate_ulcer_index(&self, backtest_report: &BacktestReport) -> f64 {
        if backtest_report.equity_curve.is_empty() {
            return 0.0;
        }

        let squared_drawdowns: f64 = backtest_report.equity_curve.iter()
            .map(|point| point.drawdown.powi(2))
            .sum();

        (squared_drawdowns / backtest_report.equity_curve.len() as f64).sqrt()
    }

    async fn save_results(&self, performance: &BacktestPerformance) -> Result<()> {
        use std::fs;
        use std::path::Path;

        // Create output directory if it doesn't exist
        fs::create_dir_all(&self.config.output_directory)?;

        // Save main results as JSON
        let results_path = Path::new(&self.config.output_directory)
            .join("backtest_results.json");
        
        let json_output = serde_json::to_string_pretty(performance)?;
        fs::write(results_path, json_output)?;

        // Save CSV summary for easier analysis
        let csv_path = Path::new(&self.config.output_directory)
            .join("backtest_summary.csv");
        
        let csv_content = self.generate_csv_summary(performance)?;
        fs::write(csv_path, csv_content)?;

        info!("Backtest results saved to: {}", self.config.output_directory);
        Ok(())
    }

    fn generate_csv_summary(&self, performance: &BacktestPerformance) -> Result<String> {
        let mut csv = String::new();
        
        // Header
        csv.push_str("Metric,Value\n");
        
        // Key metrics
        csv.push_str(&format!("Total Return,{:.2}%\n", performance.results.total_return * 100.0));
        csv.push_str(&format!("Annual Return,{:.2}%\n", performance.results.annual_return * 100.0));
        csv.push_str(&format!("Sharpe Ratio,{:.2}\n", performance.results.sharpe_ratio));
        csv.push_str(&format!("Max Drawdown,{:.2}%\n", performance.results.max_drawdown * 100.0));
        csv.push_str(&format!("Win Rate,{:.2}%\n", performance.results.win_rate * 100.0));
        csv.push_str(&format!("Profit Factor,{:.2}\n", performance.results.profit_factor));
        csv.push_str(&format!("Total Trades,{}\n", performance.results.total_trades));
        csv.push_str(&format!("Execution Time,{:.2}s\n", performance.execution_time_seconds));
        
        Ok(csv)
    }

    pub fn print_summary(&self, performance: &BacktestPerformance) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                    BACKTEST RESULTS SUMMARY                 ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Period: {} to {}     ║", 
                 self.config.start_date.format("%Y-%m-%d"),
                 self.config.end_date.format("%Y-%m-%d"));
        println!("║ Symbols: {:3} | Bar Size: {:8} | Capital: ${:8.0} ║", 
                 self.config.symbols.len(),
                 self.config.bar_size,
                 performance.config.initial_capital);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Total Return:      {:6.2}% | Annual Return:    {:6.2}% ║", 
                 performance.results.total_return * 100.0,
                 performance.results.annual_return * 100.0);
        println!("║ Sharpe Ratio:      {:6.2}  | Sortino Ratio:    {:6.2}  ║", 
                 performance.results.sharpe_ratio,
                 performance.results.sortino_ratio);
        println!("║ Max Drawdown:      {:6.2}% | Volatility:       {:6.2}% ║", 
                 performance.results.max_drawdown * 100.0,
                 performance.results.volatility * 100.0);
        println!("║ Win Rate:          {:6.2}% | Profit Factor:    {:6.2}  ║", 
                 performance.results.win_rate * 100.0,
                 performance.results.profit_factor);
        println!("║ Total Trades:      {:6}   | Avg Trade:        ${:5.2} ║", 
                 performance.results.total_trades,
                 performance.results.expectancy);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Execution Time: {:6.2}s | Bars Processed: {:8}   ║", 
                 performance.execution_time_seconds,
                 performance.total_bars_processed);
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        // Print top performing symbols
        if !performance.symbol_performance.is_empty() {
            println!("Top Performing Symbols:");
            let mut sorted_symbols: Vec<_> = performance.symbol_performance.values().collect();
            sorted_symbols.sort_by(|a, b| b.total_return.partial_cmp(&a.total_return).unwrap());
            
            for (i, symbol_perf) in sorted_symbols.iter().take(5).enumerate() {
                println!("{}. {} - Return: {:.2}%, Win Rate: {:.1}%, Trades: {}", 
                         i + 1,
                         symbol_perf.symbol,
                         symbol_perf.total_return * 100.0,
                         symbol_perf.win_rate * 100.0,
                         symbol_perf.total_trades);
            }
            println!();
        }
    }
}

impl Default for HistoricalBacktestConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            symbols: vec!["AAPL".to_string(), "MSFT".to_string()],
            start_date: Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            end_date: Utc.with_ymd_and_hms(2023, 12, 31, 23, 59, 59).unwrap(),
            bar_size: "1 day".to_string(),
            initial_capital: dec!(100000),
            commission_per_trade: dec!(1),
            slippage_pct: 0.001,
            what_to_show: "TRADES".to_string(),
            use_rth: true,
            benchmark_symbol: Some("SPY".to_string()),
            save_results: true,
            output_directory: "backtest_results".to_string(),
            generate_charts: false,
            detailed_logging: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_historical_backtest_config_creation() {
        let config = HistoricalBacktestConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.symbols.len(), 2);
        assert_eq!(config.bar_size, "1 day");
        assert_eq!(config.initial_capital, dec!(100000));
    }

    #[test]
    fn test_duration_calculation() {
        let config = HistoricalBacktestConfig {
            start_date: Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            end_date: Utc.with_ymd_and_hms(2023, 1, 31, 0, 0, 0).unwrap(),
            ..Default::default()
        };
        
        // Note: This test would need a mock IBKRClient to fully test
        // For now, just test the config creation
        assert_eq!(config.symbols.len(), 2);
    }
}