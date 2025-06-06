use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use rust_decimal::{Decimal, prelude::ToPrimitive};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use log::info;

use crate::data::{PriceData, OHLCV};
use crate::strategy_engine::{StrategyEngine, EnhancedTradingSignal, Action};


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestingEngine {
    pub config: BacktestConfig,
    pub results: BacktestResults,
    pub portfolio: BacktestPortfolio,
    pub trade_history: Vec<BacktestTrade>,
    pub daily_returns: Vec<DailyReturn>,
    pub drawdowns: Vec<DrawdownPeriod>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub commission_per_trade: f64,
    pub slippage_pct: f64,
    pub max_position_size_pct: f64,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub benchmark_symbol: Option<String>,
    pub risk_free_rate: f64, // Annual risk-free rate for Sharpe calculation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    pub total_return: f64,
    pub annual_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration_days: i64,
    pub volatility: f64,
    pub beta: f64,
    pub alpha: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub expectancy: f64,
    pub calmar_ratio: f64,
    pub information_ratio: f64,
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub average_win: f64,
    pub average_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub average_trade_duration_hours: f64,
    pub final_portfolio_value: f64,
    pub benchmark_return: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestPortfolio {
    pub cash: f64,
    pub total_value: f64,
    pub positions: HashMap<String, BacktestPosition>,
    pub equity_curve: Vec<EquityPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestPosition {
    pub symbol: String,
    pub quantity: f64,
    pub average_price: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub entry_date: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestTrade {
    pub id: u32,
    pub symbol: String,
    pub action: Action,
    pub entry_date: DateTime<Utc>,
    pub exit_date: Option<DateTime<Utc>>,
    pub entry_price: f64,
    pub exit_price: Option<f64>,
    pub quantity: f64,
    pub commission: f64,
    pub slippage: f64,
    pub realized_pnl: Option<f64>,
    pub duration_hours: Option<f64>,
    pub strategy_signal: String,
    pub max_favorable_excursion: f64,
    pub max_adverse_excursion: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    pub date: DateTime<Utc>,
    pub portfolio_value: f64,
    pub cash: f64,
    pub positions_value: f64,
    pub daily_return: f64,
    pub cumulative_return: f64,
    pub drawdown: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyReturn {
    pub date: DateTime<Utc>,
    pub return_pct: f64,
    pub portfolio_value: f64,
    pub benchmark_return: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownPeriod {
    pub start_date: DateTime<Utc>,
    pub end_date: Option<DateTime<Utc>>,
    pub peak_value: f64,
    pub trough_value: f64,
    pub drawdown_pct: f64,
    pub duration_days: i64,
    pub recovery_date: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardAnalysis {
    pub window_size_days: i64,
    pub step_size_days: i64,
    pub optimization_period_days: i64,
    pub test_period_days: i64,
    pub results: Vec<WalkForwardPeriod>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardPeriod {
    pub optimization_start: DateTime<Utc>,
    pub optimization_end: DateTime<Utc>,
    pub test_start: DateTime<Utc>,
    pub test_end: DateTime<Utc>,
    pub optimization_return: f64,
    pub test_return: f64,
    pub best_parameters: HashMap<String, f64>,
}

impl BacktestingEngine {
    pub fn new(config: BacktestConfig) -> Self {
        let portfolio = BacktestPortfolio {
            cash: config.initial_capital,
            total_value: config.initial_capital,
            positions: HashMap::new(),
            equity_curve: Vec::new(),
        };

        Self {
            config,
            results: BacktestResults::default(),
            portfolio,
            trade_history: Vec::new(),
            daily_returns: Vec::new(),
            drawdowns: Vec::new(),
        }
    }

    pub async fn run_backtest(
        &mut self,
        price_data: &HashMap<String, Vec<OHLCV>>,
        strategy_engine: &mut StrategyEngine,
    ) -> Result<()> {
        info!("Starting backtest from {} to {}", self.config.start_date, self.config.end_date);

        // Create a timeline of all price data points
        let timeline = self.create_timeline(price_data)?;
        
        let mut trade_id_counter = 1u32;
        let mut last_portfolio_value = self.config.initial_capital;

        for (timestamp, symbol_prices) in timeline {
            // Update portfolio values with current prices
            self.update_portfolio_values(&symbol_prices);

            // Process signals for each symbol
            for (symbol, ohlcv) in &symbol_prices {
                let price_data = PriceData {
                    symbol: symbol.clone(),
                    timestamp,
                    open: Decimal::from_f64_retain(ohlcv.open).unwrap_or_default(),
                    high: Decimal::from_f64_retain(ohlcv.high).unwrap_or_default(),
                    low: Decimal::from_f64_retain(ohlcv.low).unwrap_or_default(),
                    close: Decimal::from_f64_retain(ohlcv.close).unwrap_or_default(),
                    volume: ohlcv.volume,
                };

                // Get strategy signals
                if let Ok(signals) = strategy_engine.analyze(symbol, &price_data).await {
                    for signal in signals {
                        if signal.action != Action::Hold {
                            self.execute_signal(&signal, &mut trade_id_counter)?;
                        }
                    }
                }
            }

            // Record daily performance
            let current_value = self.portfolio.total_value;
            let daily_return = (current_value - last_portfolio_value) / last_portfolio_value;
            
            self.daily_returns.push(DailyReturn {
                date: timestamp,
                return_pct: daily_return,
                portfolio_value: current_value,
                benchmark_return: None, // TODO: Calculate benchmark return
            });

            // Update equity curve
            let cumulative_return = (current_value - self.config.initial_capital) / self.config.initial_capital;
            let drawdown = self.calculate_current_drawdown(current_value);

            self.portfolio.equity_curve.push(EquityPoint {
                date: timestamp,
                portfolio_value: current_value,
                cash: self.portfolio.cash,
                positions_value: current_value - self.portfolio.cash,
                daily_return,
                cumulative_return,
                drawdown,
            });

            last_portfolio_value = current_value;
        }

        // Calculate final results
        self.calculate_results()?;
        
        info!("Backtest completed. Total return: {:.2}%, Sharpe ratio: {:.2}", 
              self.results.total_return * 100.0, self.results.sharpe_ratio);

        Ok(())
    }

    fn create_timeline(&self, price_data: &HashMap<String, Vec<OHLCV>>) -> Result<Vec<(DateTime<Utc>, HashMap<String, OHLCV>)>> {
        let mut timeline: HashMap<DateTime<Utc>, HashMap<String, OHLCV>> = HashMap::new();

        for (symbol, ohlcv_data) in price_data {
            for ohlcv in ohlcv_data {
                if ohlcv.timestamp >= self.config.start_date && ohlcv.timestamp <= self.config.end_date {
                    timeline.entry(ohlcv.timestamp)
                        .or_insert_with(HashMap::new)
                        .insert(symbol.clone(), ohlcv.clone());
                }
            }
        }

        let mut sorted_timeline: Vec<_> = timeline.into_iter().collect();
        sorted_timeline.sort_by_key(|(timestamp, _)| *timestamp);

        Ok(sorted_timeline)
    }

    fn update_portfolio_values(&mut self, current_prices: &HashMap<String, OHLCV>) {
        let mut total_positions_value = 0.0;

        for (symbol, position) in &mut self.portfolio.positions {
            if let Some(ohlcv) = current_prices.get(symbol) {
                position.market_value = position.quantity * ohlcv.close.to_f64().unwrap_or(0.0);
                position.unrealized_pnl = position.market_value - (position.quantity * position.average_price);
                total_positions_value += position.market_value;
            }
        }

        self.portfolio.total_value = self.portfolio.cash + total_positions_value;
    }

    fn execute_signal(&mut self, signal: &EnhancedTradingSignal, trade_id: &mut u32) -> Result<()> {
        let price = signal.price.to_f64().unwrap_or(0.0);
        let adjusted_price = self.apply_slippage(price, &signal.action);
        
        match signal.action {
            Action::Buy | Action::StrongBuy => {
                self.execute_buy_order(signal, adjusted_price, trade_id)?;
            }
            Action::Sell | Action::StrongSell => {
                self.execute_sell_order(signal, adjusted_price, trade_id)?;
            }
            Action::Hold => {}
        }

        Ok(())
    }

    fn execute_buy_order(&mut self, signal: &EnhancedTradingSignal, price: f64, trade_id: &mut u32) -> Result<()> {
        let max_position_value = self.portfolio.total_value * self.config.max_position_size_pct;
        let available_cash = self.portfolio.cash - self.config.commission_per_trade;
        
        if available_cash <= 0.0 {
            return Ok(()); // Not enough cash
        }

        let target_value = max_position_value.min(available_cash);
        let quantity = (target_value / price).floor();
        
        if quantity > 0.0 {
            let total_cost = quantity * price + self.config.commission_per_trade;
            
            if total_cost <= self.portfolio.cash {
                // Execute the trade
                self.portfolio.cash -= total_cost;
                
                let position = self.portfolio.positions.entry(signal.symbol.clone())
                    .or_insert(BacktestPosition {
                        symbol: signal.symbol.clone(),
                        quantity: 0.0,
                        average_price: 0.0,
                        market_value: 0.0,
                        unrealized_pnl: 0.0,
                        entry_date: signal.timestamp,
                    });

                // Update position
                let new_total_quantity = position.quantity + quantity;
                let new_total_cost = (position.quantity * position.average_price) + (quantity * price);
                position.average_price = new_total_cost / new_total_quantity;
                position.quantity = new_total_quantity;
                position.market_value = position.quantity * price;

                // Record trade
                let trade = BacktestTrade {
                    id: *trade_id,
                    symbol: signal.symbol.clone(),
                    action: signal.action.clone(),
                    entry_date: signal.timestamp,
                    exit_date: None,
                    entry_price: price,
                    exit_price: None,
                    quantity,
                    commission: self.config.commission_per_trade,
                    slippage: price - signal.price.to_f64().unwrap_or(0.0),
                    realized_pnl: None,
                    duration_hours: None,
                    strategy_signal: signal.reasoning.clone(),
                    max_favorable_excursion: 0.0,
                    max_adverse_excursion: 0.0,
                };

                self.trade_history.push(trade);
                *trade_id += 1;
            }
        }

        Ok(())
    }

    fn execute_sell_order(&mut self, signal: &EnhancedTradingSignal, price: f64, trade_id: &mut u32) -> Result<()> {
        if let Some(position) = self.portfolio.positions.get_mut(&signal.symbol) {
            if position.quantity > 0.0 {
                let quantity_to_sell = position.quantity; // Sell entire position for simplicity
                let gross_proceeds = quantity_to_sell * price;
                let net_proceeds = gross_proceeds - self.config.commission_per_trade;
                
                // Calculate realized P&L
                let cost_basis = quantity_to_sell * position.average_price;
                let realized_pnl = gross_proceeds - cost_basis - self.config.commission_per_trade;

                // Update portfolio
                self.portfolio.cash += net_proceeds;
                
                // Find and update the corresponding buy trade
                if let Some(buy_trade) = self.trade_history.iter_mut()
                    .filter(|t| t.symbol == signal.symbol && t.exit_date.is_none())
                    .last() {
                    
                    buy_trade.exit_date = Some(signal.timestamp);
                    buy_trade.exit_price = Some(price);
                    buy_trade.realized_pnl = Some(realized_pnl);
                    
                    if let Some(duration) = signal.timestamp.signed_duration_since(buy_trade.entry_date).num_hours().to_f64() {
                        buy_trade.duration_hours = Some(duration);
                    }
                }

                // Record sell trade
                let sell_trade = BacktestTrade {
                    id: *trade_id,
                    symbol: signal.symbol.clone(),
                    action: signal.action.clone(),
                    entry_date: signal.timestamp,
                    exit_date: Some(signal.timestamp),
                    entry_price: price,
                    exit_price: Some(price),
                    quantity: -quantity_to_sell,
                    commission: self.config.commission_per_trade,
                    slippage: price - signal.price.to_f64().unwrap_or(0.0),
                    realized_pnl: Some(realized_pnl),
                    duration_hours: Some(0.0),
                    strategy_signal: signal.reasoning.clone(),
                    max_favorable_excursion: 0.0,
                    max_adverse_excursion: 0.0,
                };

                self.trade_history.push(sell_trade);
                *trade_id += 1;

                // Remove position
                self.portfolio.positions.remove(&signal.symbol);
            }
        }

        Ok(())
    }

    fn apply_slippage(&self, price: f64, action: &Action) -> f64 {
        match action {
            Action::Buy | Action::StrongBuy => price * (1.0 + self.config.slippage_pct),
            Action::Sell | Action::StrongSell => price * (1.0 - self.config.slippage_pct),
            Action::Hold => price,
        }
    }

    fn calculate_current_drawdown(&self, current_value: f64) -> f64 {
        if self.portfolio.equity_curve.is_empty() {
            return 0.0;
        }

        let peak_value = self.portfolio.equity_curve.iter()
            .map(|point| point.portfolio_value)
            .fold(0.0f64, f64::max);

        if peak_value > 0.0 {
            (peak_value - current_value) / peak_value
        } else {
            0.0
        }
    }

    fn calculate_results(&mut self) -> Result<()> {
        if self.daily_returns.is_empty() {
            return Ok(());
        }

        let initial_value = self.config.initial_capital;
        let final_value = self.portfolio.total_value;
        
        // Basic returns
        self.results.total_return = (final_value - initial_value) / initial_value;
        
        let days = (self.config.end_date - self.config.start_date).num_days() as f64;
        let years = days / 365.25;
        
        if years > 0.0 {
            self.results.annual_return = (1.0 + self.results.total_return).powf(1.0 / years) - 1.0;
        }

        // Calculate volatility
        let returns: Vec<f64> = self.daily_returns.iter().map(|r| r.return_pct).collect();
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        self.results.volatility = variance.sqrt() * (252.0f64).sqrt(); // Annualized

        // Sharpe ratio
        if self.results.volatility > 0.0 {
            self.results.sharpe_ratio = (self.results.annual_return - self.config.risk_free_rate) / self.results.volatility;
        }

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();
        
        if !downside_returns.is_empty() {
            let downside_variance = downside_returns.iter()
                .map(|r| r.powi(2))
                .sum::<f64>() / downside_returns.len() as f64;
            let downside_deviation = downside_variance.sqrt() * (252.0f64).sqrt();
            
            if downside_deviation > 0.0 {
                self.results.sortino_ratio = (self.results.annual_return - self.config.risk_free_rate) / downside_deviation;
            }
        }

        // Maximum drawdown
        let mut peak = initial_value;
        let mut max_dd = 0.0;
        let mut current_dd_start: Option<DateTime<Utc>> = None;
        let mut max_dd_duration = 0i64;

        for point in &self.portfolio.equity_curve {
            if point.portfolio_value > peak {
                peak = point.portfolio_value;
                current_dd_start = None;
            } else {
                let dd = (peak - point.portfolio_value) / peak;
                if dd > max_dd {
                    max_dd = dd;
                }
                
                if current_dd_start.is_none() && dd > 0.0 {
                    current_dd_start = Some(point.date);
                }
                
                if let Some(start_date) = current_dd_start {
                    let duration = (point.date - start_date).num_days();
                    if duration > max_dd_duration {
                        max_dd_duration = duration;
                    }
                }
            }
        }

        self.results.max_drawdown = max_dd;
        self.results.max_drawdown_duration_days = max_dd_duration;

        // Calmar ratio
        if self.results.max_drawdown > 0.0 {
            self.results.calmar_ratio = self.results.annual_return / self.results.max_drawdown;
        }

        // Trade statistics
        let completed_trades: Vec<&BacktestTrade> = self.trade_history.iter()
            .filter(|t| t.realized_pnl.is_some())
            .collect();

        self.results.total_trades = completed_trades.len() as u32;
        
        if !completed_trades.is_empty() {
            let winning_trades: Vec<&BacktestTrade> = completed_trades.iter()
                .filter(|t| t.realized_pnl.unwrap_or(0.0) > 0.0)
                .cloned()
                .collect();
            
            let losing_trades: Vec<&BacktestTrade> = completed_trades.iter()
                .filter(|t| t.realized_pnl.unwrap_or(0.0) <= 0.0)
                .cloned()
                .collect();

            self.results.winning_trades = winning_trades.len() as u32;
            self.results.losing_trades = losing_trades.len() as u32;
            self.results.win_rate = self.results.winning_trades as f64 / self.results.total_trades as f64;

            if !winning_trades.is_empty() {
                self.results.average_win = winning_trades.iter()
                    .map(|t| t.realized_pnl.unwrap_or(0.0))
                    .sum::<f64>() / winning_trades.len() as f64;
                
                self.results.largest_win = winning_trades.iter()
                    .map(|t| t.realized_pnl.unwrap_or(0.0))
                    .fold(0.0f64, f64::max);
            }

            if !losing_trades.is_empty() {
                self.results.average_loss = losing_trades.iter()
                    .map(|t| t.realized_pnl.unwrap_or(0.0).abs())
                    .sum::<f64>() / losing_trades.len() as f64;
                
                self.results.largest_loss = losing_trades.iter()
                    .map(|t| t.realized_pnl.unwrap_or(0.0))
                    .fold(0.0f64, f64::min);
            }

            // Profit factor
            let gross_profit: f64 = winning_trades.iter()
                .map(|t| t.realized_pnl.unwrap_or(0.0))
                .sum();
            let gross_loss: f64 = losing_trades.iter()
                .map(|t| t.realized_pnl.unwrap_or(0.0).abs())
                .sum();

            if gross_loss > 0.0 {
                self.results.profit_factor = gross_profit / gross_loss;
            }

            // Expectancy
            self.results.expectancy = (self.results.win_rate * self.results.average_win) - 
                                    ((1.0 - self.results.win_rate) * self.results.average_loss);

            // Average trade duration
            let durations: Vec<f64> = completed_trades.iter()
                .filter_map(|t| t.duration_hours)
                .collect();
            
            if !durations.is_empty() {
                self.results.average_trade_duration_hours = durations.iter().sum::<f64>() / durations.len() as f64;
            }
        }

        self.results.final_portfolio_value = final_value;

        Ok(())
    }

    pub fn generate_report(&self) -> BacktestReport {
        BacktestReport {
            config: self.config.clone(),
            results: self.results.clone(),
            equity_curve: self.portfolio.equity_curve.clone(),
            trade_history: self.trade_history.clone(),
            daily_returns: self.daily_returns.clone(),
            drawdowns: self.drawdowns.clone(),
        }
    }

    pub async fn run_walk_forward_analysis(
        &mut self,
        _price_data: &HashMap<String, Vec<OHLCV>>,
        analysis_config: WalkForwardAnalysis,
    ) -> Result<WalkForwardAnalysis> {
        let mut results = Vec::new();
        let mut current_date = self.config.start_date;

        while current_date + Duration::days(analysis_config.optimization_period_days + analysis_config.test_period_days) <= self.config.end_date {
            let optimization_end = current_date + Duration::days(analysis_config.optimization_period_days);
            let test_start = optimization_end;
            let test_end = test_start + Duration::days(analysis_config.test_period_days);

            // Run optimization period (simplified - in practice you'd optimize parameters)
            let opt_engine = Self::new(BacktestConfig {
                start_date: current_date,
                end_date: optimization_end,
                ..self.config.clone()
            });

            // Run test period
            let test_engine = Self::new(BacktestConfig {
                start_date: test_start,
                end_date: test_end,
                ..self.config.clone()
            });

            let period = WalkForwardPeriod {
                optimization_start: current_date,
                optimization_end,
                test_start,
                test_end,
                optimization_return: opt_engine.results.total_return,
                test_return: test_engine.results.total_return,
                best_parameters: HashMap::new(), // TODO: Implement parameter optimization
            };

            results.push(period);
            current_date += Duration::days(analysis_config.step_size_days);
        }

        Ok(WalkForwardAnalysis {
            results,
            ..analysis_config
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    pub config: BacktestConfig,
    pub results: BacktestResults,
    pub equity_curve: Vec<EquityPoint>,
    pub trade_history: Vec<BacktestTrade>,
    pub daily_returns: Vec<DailyReturn>,
    pub drawdowns: Vec<DrawdownPeriod>,
}

impl Default for BacktestResults {
    fn default() -> Self {
        Self {
            total_return: 0.0,
            annual_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            max_drawdown_duration_days: 0,
            volatility: 0.0,
            beta: 0.0,
            alpha: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            expectancy: 0.0,
            calmar_ratio: 0.0,
            information_ratio: 0.0,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            average_win: 0.0,
            average_loss: 0.0,
            largest_win: 0.0,
            largest_loss: 0.0,
            average_trade_duration_hours: 0.0,
            final_portfolio_value: 0.0,
            benchmark_return: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_engine_creation() {
        let config = BacktestConfig {
            initial_capital: 100000.0,
            commission_per_trade: 1.0,
            slippage_pct: 0.001,
            max_position_size_pct: 0.1,
            start_date: Utc::now() - Duration::days(365),
            end_date: Utc::now(),
            benchmark_symbol: None,
            risk_free_rate: 0.02,
        };

        let engine = BacktestingEngine::new(config);
        assert_eq!(engine.portfolio.cash, 100000.0);
        assert_eq!(engine.portfolio.total_value, 100000.0);
    }

    #[test]
    fn test_slippage_calculation() {
        let config = BacktestConfig {
            initial_capital: 100000.0,
            commission_per_trade: 1.0,
            slippage_pct: 0.001, // 0.1%
            max_position_size_pct: 0.1,
            start_date: Utc::now() - Duration::days(365),
            end_date: Utc::now(),
            benchmark_symbol: None,
            risk_free_rate: 0.02,
        };

        let engine = BacktestingEngine::new(config);
        
        let buy_price = engine.apply_slippage(100.0, &Action::Buy);
        let sell_price = engine.apply_slippage(100.0, &Action::Sell);
        
        assert_eq!(buy_price, 100.1); // 100 * (1 + 0.001)
        assert_eq!(sell_price, 99.9);  // 100 * (1 - 0.001)
    }
}