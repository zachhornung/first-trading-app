use anyhow::Result;
use chrono::{DateTime, Utc, Datelike};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::backtesting::{BacktestTrade, DailyReturn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalytics {
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub total_return: f64,
    pub annual_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: DrawdownAnalysis,
    pub trade_analysis: TradeAnalysis,
    pub risk_metrics: RiskMetrics,
    pub benchmark_comparison: Option<BenchmarkComparison>,
    pub monthly_returns: Vec<MonthlyReturn>,
    pub daily_statistics: DailyStatistics,
    pub rolling_metrics: RollingMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownAnalysis {
    pub max_drawdown_pct: f64,
    pub max_drawdown_duration_days: i64,
    pub current_drawdown_pct: f64,
    pub current_drawdown_duration_days: i64,
    pub drawdown_periods: Vec<DrawdownPeriod>,
    pub recovery_factor: f64,
    pub ulcer_index: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownPeriod {
    pub start_date: DateTime<Utc>,
    pub end_date: Option<DateTime<Utc>>,
    pub recovery_date: Option<DateTime<Utc>>,
    pub peak_value: f64,
    pub trough_value: f64,
    pub drawdown_pct: f64,
    pub duration_days: i64,
    pub recovery_days: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeAnalysis {
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub average_win: f64,
    pub average_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub average_trade_duration_hours: f64,
    pub expectancy: f64,
    pub kelly_criterion: f64,
    pub consecutive_wins: ConsecutiveStats,
    pub consecutive_losses: ConsecutiveStats,
    pub trade_distribution: TradeDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsecutiveStats {
    pub max_consecutive: u32,
    pub current_consecutive: u32,
    pub average_consecutive: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeDistribution {
    pub win_buckets: HashMap<String, u32>, // e.g., "0-1%", "1-2%", etc.
    pub loss_buckets: HashMap<String, u32>,
    pub duration_buckets: HashMap<String, u32>, // e.g., "0-1h", "1-4h", etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub value_at_risk_95: f64,
    pub value_at_risk_99: f64,
    pub conditional_var_95: f64,
    pub conditional_var_99: f64,
    pub beta: Option<f64>,
    pub alpha: Option<f64>,
    pub information_ratio: Option<f64>,
    pub tracking_error: Option<f64>,
    pub correlation_with_benchmark: Option<f64>,
    pub downside_deviation: f64,
    pub up_capture_ratio: Option<f64>,
    pub down_capture_ratio: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub benchmark_symbol: String,
    pub benchmark_return: f64,
    pub excess_return: f64,
    pub beta: f64,
    pub alpha: f64,
    pub correlation: f64,
    pub tracking_error: f64,
    pub information_ratio: f64,
    pub up_capture_ratio: f64,
    pub down_capture_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthlyReturn {
    pub year: i32,
    pub month: u32,
    pub return_pct: f64,
    pub benchmark_return_pct: Option<f64>,
    pub excess_return_pct: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyStatistics {
    pub mean_daily_return: f64,
    pub median_daily_return: f64,
    pub std_dev_daily_return: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub positive_days: u32,
    pub negative_days: u32,
    pub zero_days: u32,
    pub best_day: f64,
    pub worst_day: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingMetrics {
    pub rolling_sharpe_30d: Vec<RollingPoint>,
    pub rolling_volatility_30d: Vec<RollingPoint>,
    pub rolling_return_30d: Vec<RollingPoint>,
    pub rolling_max_drawdown_30d: Vec<RollingPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingPoint {
    pub date: DateTime<Utc>,
    pub value: f64,
}

pub struct AnalyticsEngine {
    risk_free_rate: f64,
}

impl AnalyticsEngine {
    pub fn new(risk_free_rate: f64) -> Self {
        Self { risk_free_rate }
    }

    pub fn analyze_performance(
        &self,
        daily_returns: &[DailyReturn],
        trades: &[BacktestTrade],
        benchmark_returns: Option<&[f64]>,
        benchmark_symbol: Option<String>,
    ) -> Result<PerformanceAnalytics> {
        if daily_returns.is_empty() {
            return Err(anyhow::anyhow!("No daily returns data provided"));
        }

        let period_start = daily_returns.first().unwrap().date;
        let period_end = daily_returns.last().unwrap().date;
        
        // Calculate basic returns
        let returns: Vec<f64> = daily_returns.iter().map(|r| r.return_pct).collect();
        let total_return = self.calculate_total_return(&returns);
        let annual_return = self.calculate_annual_return(&returns, period_start, period_end);
        let volatility = self.calculate_volatility(&returns);
        
        // Calculate risk-adjusted metrics
        let sharpe_ratio = self.calculate_sharpe_ratio(annual_return, volatility);
        let sortino_ratio = self.calculate_sortino_ratio(&returns, annual_return);
        let calmar_ratio = self.calculate_calmar_ratio(annual_return, &returns);
        
        // Drawdown analysis
        let portfolio_values: Vec<f64> = daily_returns.iter().map(|r| r.portfolio_value).collect();
        let max_drawdown = self.analyze_drawdowns(&portfolio_values, daily_returns);
        
        // Trade analysis
        let trade_analysis = self.analyze_trades(trades);
        
        // Risk metrics
        let risk_metrics = self.calculate_risk_metrics(&returns, benchmark_returns);
        
        // Benchmark comparison
        let benchmark_comparison = if let (Some(benchmark_rets), Some(symbol)) = (benchmark_returns, benchmark_symbol) {
            Some(self.compare_with_benchmark(&returns, benchmark_rets, symbol)?)
        } else {
            None
        };
        
        // Monthly returns
        let monthly_returns = self.calculate_monthly_returns(daily_returns, benchmark_returns);
        
        // Daily statistics
        let daily_statistics = self.calculate_daily_statistics(&returns);
        
        // Rolling metrics
        let rolling_metrics = self.calculate_rolling_metrics(daily_returns);

        Ok(PerformanceAnalytics {
            period_start,
            period_end,
            total_return,
            annual_return,
            volatility,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            max_drawdown,
            trade_analysis,
            risk_metrics,
            benchmark_comparison,
            monthly_returns,
            daily_statistics,
            rolling_metrics,
        })
    }

    fn calculate_total_return(&self, returns: &[f64]) -> f64 {
        returns.iter().fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0
    }

    fn calculate_annual_return(&self, returns: &[f64], start: DateTime<Utc>, end: DateTime<Utc>) -> f64 {
        let total_return = self.calculate_total_return(returns);
        let days = (end - start).num_days() as f64;
        let years = days / 365.25;
        
        if years > 0.0 {
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            total_return
        }
    }

    fn calculate_volatility(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        
        variance.sqrt() * (252.0f64).sqrt() // Annualized
    }

    fn calculate_sharpe_ratio(&self, annual_return: f64, volatility: f64) -> f64 {
        if volatility > 0.0 {
            (annual_return - self.risk_free_rate) / volatility
        } else {
            0.0
        }
    }

    fn calculate_sortino_ratio(&self, returns: &[f64], annual_return: f64) -> f64 {
        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();
        
        if downside_returns.is_empty() {
            return f64::INFINITY;
        }

        let downside_variance = downside_returns.iter()
            .map(|r| r.powi(2))
            .sum::<f64>() / downside_returns.len() as f64;
        let downside_deviation = downside_variance.sqrt() * (252.0f64).sqrt();
        
        if downside_deviation > 0.0 {
            (annual_return - self.risk_free_rate) / downside_deviation
        } else {
            0.0
        }
    }

    fn calculate_calmar_ratio(&self, annual_return: f64, returns: &[f64]) -> f64 {
        let max_dd = self.calculate_max_drawdown_simple(returns);
        if max_dd > 0.0 {
            annual_return / max_dd
        } else {
            0.0
        }
    }

    fn calculate_max_drawdown_simple(&self, returns: &[f64]) -> f64 {
        let mut cumulative = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;

        for &ret in returns {
            cumulative *= 1.0 + ret;
            if cumulative > peak {
                peak = cumulative;
            }
            let dd = (peak - cumulative) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }

    fn analyze_drawdowns(&self, portfolio_values: &[f64], daily_returns: &[DailyReturn]) -> DrawdownAnalysis {
        let mut drawdown_periods = Vec::new();
        let mut peak = portfolio_values[0];
        let mut _peak_date = daily_returns[0].date;
        let mut in_drawdown = false;
        let mut drawdown_start: Option<DateTime<Utc>> = None;
        let mut max_drawdown_pct = 0.0;
        let mut max_drawdown_duration = 0i64;
        let mut trough_value = peak;
        let mut ulcer_squared_sum = 0.0;

        for (i, &value) in portfolio_values.iter().enumerate() {
            let date = daily_returns[i].date;
            
            if value > peak {
                // New peak
                if in_drawdown {
                    // End of drawdown period
                    let start_date = drawdown_start.unwrap();
                    let duration = (date - start_date).num_days();
                    let dd_pct = (peak - trough_value) / peak;
                    
                    drawdown_periods.push(DrawdownPeriod {
                        start_date,
                        end_date: Some(date),
                        recovery_date: Some(date),
                        peak_value: peak,
                        trough_value,
                        drawdown_pct: dd_pct,
                        duration_days: duration,
                        recovery_days: Some(duration),
                    });
                    
                    in_drawdown = false;
                }
                peak = value;
                _peak_date = date;
                trough_value = value;
            } else {
                // Below peak
                if !in_drawdown {
                    in_drawdown = true;
                    drawdown_start = Some(date);
                }
                
                if value < trough_value {
                    trough_value = value;
                }
                
                let current_dd = (peak - value) / peak;
                if current_dd > max_drawdown_pct {
                    max_drawdown_pct = current_dd;
                }
                
                let duration = (date - drawdown_start.unwrap()).num_days();
                if duration > max_drawdown_duration {
                    max_drawdown_duration = duration;
                }
            }
            
            // Calculate Ulcer Index component
            let dd_pct = (peak - value) / peak;
            ulcer_squared_sum += dd_pct * dd_pct;
        }

        // Handle ongoing drawdown
        let (current_drawdown_pct, current_drawdown_duration) = if in_drawdown {
            let current_dd = (peak - portfolio_values.last().unwrap()) / peak;
            let duration = (daily_returns.last().unwrap().date - drawdown_start.unwrap()).num_days();
            (current_dd, duration)
        } else {
            (0.0, 0)
        };

        let ulcer_index = (ulcer_squared_sum / portfolio_values.len() as f64).sqrt();
        let recovery_factor = if max_drawdown_pct > 0.0 {
            self.calculate_total_return(&daily_returns.iter().map(|r| r.return_pct).collect::<Vec<_>>()) / max_drawdown_pct
        } else {
            0.0
        };

        DrawdownAnalysis {
            max_drawdown_pct,
            max_drawdown_duration_days: max_drawdown_duration,
            current_drawdown_pct,
            current_drawdown_duration_days: current_drawdown_duration,
            drawdown_periods,
            recovery_factor,
            ulcer_index,
        }
    }

    fn analyze_trades(&self, trades: &[BacktestTrade]) -> TradeAnalysis {
        let completed_trades: Vec<&BacktestTrade> = trades.iter()
            .filter(|t| t.realized_pnl.is_some())
            .collect();

        if completed_trades.is_empty() {
            return TradeAnalysis::default();
        }

        let winning_trades: Vec<&BacktestTrade> = completed_trades.iter()
            .filter(|t| t.realized_pnl.unwrap_or(0.0) > 0.0)
            .cloned()
            .collect();

        let losing_trades: Vec<&BacktestTrade> = completed_trades.iter()
            .filter(|t| t.realized_pnl.unwrap_or(0.0) <= 0.0)
            .cloned()
            .collect();

        let total_trades = completed_trades.len() as u32;
        let winning_count = winning_trades.len() as u32;
        let losing_count = losing_trades.len() as u32;
        let win_rate = winning_count as f64 / total_trades as f64;

        let average_win = if winning_count > 0 {
            winning_trades.iter().map(|t| t.realized_pnl.unwrap_or(0.0)).sum::<f64>() / winning_count as f64
        } else {
            0.0
        };

        let average_loss = if losing_count > 0 {
            losing_trades.iter().map(|t| t.realized_pnl.unwrap_or(0.0).abs()).sum::<f64>() / losing_count as f64
        } else {
            0.0
        };

        let largest_win = winning_trades.iter()
            .map(|t| t.realized_pnl.unwrap_or(0.0))
            .fold(0.0f64, f64::max);

        let largest_loss = losing_trades.iter()
            .map(|t| t.realized_pnl.unwrap_or(0.0))
            .fold(0.0f64, f64::min);

        let profit_factor = if average_loss > 0.0 {
            (average_win * winning_count as f64) / (average_loss * losing_count as f64)
        } else {
            f64::INFINITY
        };

        let expectancy = (win_rate * average_win) - ((1.0 - win_rate) * average_loss);

        let kelly_criterion = if average_loss > 0.0 {
            win_rate - ((1.0 - win_rate) * average_win / average_loss)
        } else {
            0.0
        };

        let average_duration = if total_trades > 0 {
            completed_trades.iter()
                .filter_map(|t| t.duration_hours)
                .sum::<f64>() / total_trades as f64
        } else {
            0.0
        };

        // Calculate consecutive statistics
        let consecutive_wins = self.calculate_consecutive_stats(&completed_trades, true);
        let consecutive_losses = self.calculate_consecutive_stats(&completed_trades, false);

        // Calculate trade distribution
        let trade_distribution = self.calculate_trade_distribution(&completed_trades);

        TradeAnalysis {
            total_trades,
            winning_trades: winning_count,
            losing_trades: losing_count,
            win_rate,
            profit_factor,
            average_win,
            average_loss,
            largest_win,
            largest_loss,
            average_trade_duration_hours: average_duration,
            expectancy,
            kelly_criterion,
            consecutive_wins,
            consecutive_losses,
            trade_distribution,
        }
    }

    fn calculate_consecutive_stats(&self, trades: &[&BacktestTrade], wins: bool) -> ConsecutiveStats {
        let mut max_consecutive = 0u32;
        let mut current_consecutive = 0u32;
        let mut streaks = Vec::new();
        
        for trade in trades {
            let is_win = trade.realized_pnl.unwrap_or(0.0) > 0.0;
            
            if is_win == wins {
                current_consecutive += 1;
            } else {
                if current_consecutive > 0 {
                    streaks.push(current_consecutive);
                    if current_consecutive > max_consecutive {
                        max_consecutive = current_consecutive;
                    }
                }
                current_consecutive = 0;
            }
        }
        
        // Don't forget the last streak
        if current_consecutive > 0 {
            streaks.push(current_consecutive);
            if current_consecutive > max_consecutive {
                max_consecutive = current_consecutive;
            }
        }

        let average_consecutive = if streaks.is_empty() {
            0.0
        } else {
            streaks.iter().sum::<u32>() as f64 / streaks.len() as f64
        };

        ConsecutiveStats {
            max_consecutive,
            current_consecutive,
            average_consecutive,
        }
    }

    fn calculate_trade_distribution(&self, trades: &[&BacktestTrade]) -> TradeDistribution {
        let mut win_buckets = HashMap::new();
        let mut loss_buckets = HashMap::new();
        let mut duration_buckets = HashMap::new();

        // Initialize buckets
        let win_bucket_ranges = vec!["0-1%", "1-2%", "2-5%", "5-10%", "10%+"];
        let loss_bucket_ranges = vec!["0-1%", "1-2%", "2-5%", "5-10%", "10%+"];
        let duration_ranges = vec!["0-1h", "1-4h", "4-12h", "12-24h", "1d+"];

        for range in &win_bucket_ranges {
            win_buckets.insert(range.to_string(), 0);
        }
        for range in &loss_bucket_ranges {
            loss_buckets.insert(range.to_string(), 0);
        }
        for range in &duration_ranges {
            duration_buckets.insert(range.to_string(), 0);
        }

        for trade in trades {
            let pnl_pct = trade.realized_pnl.unwrap_or(0.0) / (trade.entry_price * trade.quantity);
            
            // Categorize wins and losses
            if pnl_pct > 0.0 {
                let bucket = if pnl_pct <= 0.01 { "0-1%" }
                else if pnl_pct <= 0.02 { "1-2%" }
                else if pnl_pct <= 0.05 { "2-5%" }
                else if pnl_pct <= 0.10 { "5-10%" }
                else { "10%+" };
                
                *win_buckets.get_mut(bucket).unwrap() += 1;
            } else if pnl_pct < 0.0 {
                let loss_pct = pnl_pct.abs();
                let bucket = if loss_pct <= 0.01 { "0-1%" }
                else if loss_pct <= 0.02 { "1-2%" }
                else if loss_pct <= 0.05 { "2-5%" }
                else if loss_pct <= 0.10 { "5-10%" }
                else { "10%+" };
                
                *loss_buckets.get_mut(bucket).unwrap() += 1;
            }

            // Categorize duration
            if let Some(duration_hours) = trade.duration_hours {
                let bucket = if duration_hours <= 1.0 { "0-1h" }
                else if duration_hours <= 4.0 { "1-4h" }
                else if duration_hours <= 12.0 { "4-12h" }
                else if duration_hours <= 24.0 { "12-24h" }
                else { "1d+" };
                
                *duration_buckets.get_mut(bucket).unwrap() += 1;
            }
        }

        TradeDistribution {
            win_buckets,
            loss_buckets,
            duration_buckets,
        }
    }

    fn calculate_risk_metrics(&self, returns: &[f64], benchmark_returns: Option<&[f64]>) -> RiskMetrics {
        let sorted_returns = {
            let mut returns = returns.to_vec();
            returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
            returns
        };

        let var_95 = self.calculate_var(&sorted_returns, 0.05);
        let var_99 = self.calculate_var(&sorted_returns, 0.01);
        let cvar_95 = self.calculate_conditional_var(&sorted_returns, 0.05);
        let cvar_99 = self.calculate_conditional_var(&sorted_returns, 0.01);

        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();

        let downside_deviation = if !downside_returns.is_empty() {
            let variance = downside_returns.iter()
                .map(|r| r.powi(2))
                .sum::<f64>() / downside_returns.len() as f64;
            variance.sqrt() * (252.0f64).sqrt()
        } else {
            0.0
        };

        let (beta, alpha, correlation, tracking_error, information_ratio, up_capture, down_capture) = 
            if let Some(benchmark) = benchmark_returns {
                let beta = self.calculate_beta(returns, benchmark);
                let alpha = self.calculate_alpha(returns, benchmark, beta);
                let correlation = self.calculate_correlation(returns, benchmark);
                let tracking_error = self.calculate_tracking_error(returns, benchmark);
                let information_ratio = if tracking_error > 0.0 {
                    alpha / tracking_error
                } else {
                    0.0
                };
                let (up_cap, down_cap) = self.calculate_capture_ratios(returns, benchmark);
                
                (Some(beta), Some(alpha), Some(correlation), Some(tracking_error), 
                 Some(information_ratio), Some(up_cap), Some(down_cap))
            } else {
                (None, None, None, None, None, None, None)
            };

        RiskMetrics {
            value_at_risk_95: var_95,
            value_at_risk_99: var_99,
            conditional_var_95: cvar_95,
            conditional_var_99: cvar_99,
            beta,
            alpha,
            information_ratio,
            tracking_error,
            correlation_with_benchmark: correlation,
            downside_deviation,
            up_capture_ratio: up_capture,
            down_capture_ratio: down_capture,
        }
    }

    fn calculate_var(&self, sorted_returns: &[f64], confidence_level: f64) -> f64 {
        let index = (confidence_level * sorted_returns.len() as f64) as usize;
        if index < sorted_returns.len() {
            -sorted_returns[index] // VaR is typically expressed as positive number
        } else {
            0.0
        }
    }

    fn calculate_conditional_var(&self, sorted_returns: &[f64], confidence_level: f64) -> f64 {
        let index = (confidence_level * sorted_returns.len() as f64) as usize;
        if index < sorted_returns.len() {
            let tail_returns = &sorted_returns[..index];
            if !tail_returns.is_empty() {
                -(tail_returns.iter().sum::<f64>() / tail_returns.len() as f64)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    fn calculate_beta(&self, returns: &[f64], benchmark_returns: &[f64]) -> f64 {
        if returns.len() != benchmark_returns.len() || returns.len() < 2 {
            return 1.0; // Default beta
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let mean_benchmark = benchmark_returns.iter().sum::<f64>() / benchmark_returns.len() as f64;

        let covariance = returns.iter().zip(benchmark_returns.iter())
            .map(|(&r, &b)| (r - mean_return) * (b - mean_benchmark))
            .sum::<f64>() / (returns.len() - 1) as f64;

        let benchmark_variance = benchmark_returns.iter()
            .map(|&b| (b - mean_benchmark).powi(2))
            .sum::<f64>() / (benchmark_returns.len() - 1) as f64;

        if benchmark_variance > 0.0 {
            covariance / benchmark_variance
        } else {
            1.0
        }
    }

    fn calculate_alpha(&self, returns: &[f64], benchmark_returns: &[f64], beta: f64) -> f64 {
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let mean_benchmark = benchmark_returns.iter().sum::<f64>() / benchmark_returns.len() as f64;
        
        (mean_return - self.risk_free_rate / 252.0) - beta * (mean_benchmark - self.risk_free_rate / 252.0)
    }

    fn calculate_correlation(&self, returns: &[f64], benchmark_returns: &[f64]) -> f64 {
        if returns.len() != benchmark_returns.len() || returns.len() < 2 {
            return 0.0;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let mean_benchmark = benchmark_returns.iter().sum::<f64>() / benchmark_returns.len() as f64;

        let covariance = returns.iter().zip(benchmark_returns.iter())
            .map(|(&r, &b)| (r - mean_return) * (b - mean_benchmark))
            .sum::<f64>() / (returns.len() - 1) as f64;

        let return_std = (returns.iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64).sqrt();

        let benchmark_std = (benchmark_returns.iter()
            .map(|&b| (b - mean_benchmark).powi(2))
            .sum::<f64>() / (benchmark_returns.len() - 1) as f64).sqrt();

        if return_std > 0.0 && benchmark_std > 0.0 {
            covariance / (return_std * benchmark_std)
        } else {
            0.0
        }
    }

    fn calculate_tracking_error(&self, returns: &[f64], benchmark_returns: &[f64]) -> f64 {
        if returns.len() != benchmark_returns.len() {
            return 0.0;
        }

        let excess_returns: Vec<f64> = returns.iter()
            .zip(benchmark_returns.iter())
            .map(|(&r, &b)| r - b)
            .collect();

        if excess_returns.is_empty() {
            return 0.0;
        }

        let mean_excess = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
        let variance = excess_returns.iter()
            .map(|&r| (r - mean_excess).powi(2))
            .sum::<f64>() / (excess_returns.len() - 1) as f64;

        variance.sqrt() * (252.0f64).sqrt()
    }

    fn calculate_capture_ratios(&self, returns: &[f64], benchmark_returns: &[f64]) -> (f64, f64) {
        if returns.len() != benchmark_returns.len() {
            return (1.0, 1.0);
        }

        let up_periods: Vec<(f64, f64)> = returns.iter()
            .zip(benchmark_returns.iter())
            .filter(|(&_r, &b)| b > 0.0)
            .map(|(&r, &b)| (r, b))
            .collect();

        let down_periods: Vec<(f64, f64)> = returns.iter()
            .zip(benchmark_returns.iter())
            .filter(|(&_r, &b)| b < 0.0)
            .map(|(&r, &b)| (r, b))
            .collect();

        let up_capture = if !up_periods.is_empty() {
            let strategy_up = up_periods.iter().map(|(r, _)| r).sum::<f64>() / up_periods.len() as f64;
            let benchmark_up = up_periods.iter().map(|(_, b)| b).sum::<f64>() / up_periods.len() as f64;
            if benchmark_up != 0.0 { strategy_up / benchmark_up } else { 1.0 }
        } else {
            1.0
        };

        let down_capture = if !down_periods.is_empty() {
            let strategy_down = down_periods.iter().map(|(r, _)| r).sum::<f64>() / down_periods.len() as f64;
            let benchmark_down = down_periods.iter().map(|(_, b)| b).sum::<f64>() / down_periods.len() as f64;
            if benchmark_down != 0.0 { strategy_down / benchmark_down } else { 1.0 }
        } else {
            1.0
        };

        (up_capture, down_capture)
    }

    fn compare_with_benchmark(
        &self,
        returns: &[f64],
        benchmark_returns: &[f64],
        benchmark_symbol: String,
    ) -> Result<BenchmarkComparison> {
        if returns.len() != benchmark_returns.len() {
            return Err(anyhow::anyhow!("Returns and benchmark data length mismatch"));
        }

        let strategy_total_return = self.calculate_total_return(returns);
        let benchmark_total_return = self.calculate_total_return(benchmark_returns);
        let excess_return = strategy_total_return - benchmark_total_return;

        let beta = self.calculate_beta(returns, benchmark_returns);
        let alpha = self.calculate_alpha(returns, benchmark_returns, beta);
        let correlation = self.calculate_correlation(returns, benchmark_returns);
        let tracking_error = self.calculate_tracking_error(returns, benchmark_returns);
        let information_ratio = if tracking_error > 0.0 { alpha / tracking_error } else { 0.0 };
        let (up_capture_ratio, down_capture_ratio) = self.calculate_capture_ratios(returns, benchmark_returns);

        Ok(BenchmarkComparison {
            benchmark_symbol,
            benchmark_return: benchmark_total_return,
            excess_return,
            beta,
            alpha,
            correlation,
            tracking_error,
            information_ratio,
            up_capture_ratio,
            down_capture_ratio,
        })
    }

    fn calculate_monthly_returns(
        &self,
        daily_returns: &[DailyReturn],
        _benchmark_returns: Option<&[f64]>,
    ) -> Vec<MonthlyReturn> {
        let mut monthly_returns = Vec::new();
        let mut current_month_data: Vec<&DailyReturn> = Vec::new();
        let mut current_month = 0u32;
        let mut current_year = 0i32;

        for daily_return in daily_returns {
            let month = daily_return.date.month();
            let year = daily_return.date.year();

            if current_month == 0 {
                current_month = month;
                current_year = year;
            }

            if month != current_month || year != current_year {
                // Process previous month
                if !current_month_data.is_empty() {
                    let month_return = self.calculate_period_return(&current_month_data);
                    monthly_returns.push(MonthlyReturn {
                        year: current_year,
                        month: current_month,
                        return_pct: month_return,
                        benchmark_return_pct: None, // TODO: Calculate benchmark monthly return
                        excess_return_pct: None,
                    });
                }

                current_month_data.clear();
                current_month = month;
                current_year = year;
            }

            current_month_data.push(daily_return);
        }

        // Don't forget the last month
        if !current_month_data.is_empty() {
            let month_return = self.calculate_period_return(&current_month_data);
            monthly_returns.push(MonthlyReturn {
                year: current_year,
                month: current_month,
                return_pct: month_return,
                benchmark_return_pct: None,
                excess_return_pct: None,
            });
        }

        monthly_returns
    }

    fn calculate_period_return(&self, period_data: &[&DailyReturn]) -> f64 {
        let returns: Vec<f64> = period_data.iter().map(|r| r.return_pct).collect();
        self.calculate_total_return(&returns)
    }

    fn calculate_daily_statistics(&self, returns: &[f64]) -> DailyStatistics {
        if returns.is_empty() {
            return DailyStatistics::default();
        }

        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let median = if sorted_returns.len() % 2 == 0 {
            let mid = sorted_returns.len() / 2;
            (sorted_returns[mid - 1] + sorted_returns[mid]) / 2.0
        } else {
            sorted_returns[sorted_returns.len() / 2]
        };

        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        let std_dev = variance.sqrt();

        // Calculate skewness and kurtosis
        let skewness = if std_dev > 0.0 {
            let skew_sum = returns.iter()
                .map(|r| ((r - mean) / std_dev).powi(3))
                .sum::<f64>();
            skew_sum / returns.len() as f64
        } else {
            0.0
        };

        let kurtosis = if std_dev > 0.0 {
            let kurt_sum = returns.iter()
                .map(|r| ((r - mean) / std_dev).powi(4))
                .sum::<f64>();
            (kurt_sum / returns.len() as f64) - 3.0 // Excess kurtosis
        } else {
            0.0
        };

        let positive_days = returns.iter().filter(|&&r| r > 0.0).count() as u32;
        let negative_days = returns.iter().filter(|&&r| r < 0.0).count() as u32;
        let zero_days = returns.iter().filter(|&&r| r == 0.0).count() as u32;

        let best_day = sorted_returns.last().cloned().unwrap_or(0.0);
        let worst_day = sorted_returns.first().cloned().unwrap_or(0.0);

        DailyStatistics {
            mean_daily_return: mean,
            median_daily_return: median,
            std_dev_daily_return: std_dev,
            skewness,
            kurtosis,
            positive_days,
            negative_days,
            zero_days,
            best_day,
            worst_day,
        }
    }

    fn calculate_rolling_metrics(&self, daily_returns: &[DailyReturn]) -> RollingMetrics {
        let window_size = 30; // 30-day rolling window
        let mut rolling_sharpe = Vec::new();
        let mut rolling_volatility = Vec::new();
        let mut rolling_return = Vec::new();
        let mut rolling_max_drawdown = Vec::new();

        if daily_returns.len() < window_size {
            return RollingMetrics {
                rolling_sharpe_30d: rolling_sharpe,
                rolling_volatility_30d: rolling_volatility,
                rolling_return_30d: rolling_return,
                rolling_max_drawdown_30d: rolling_max_drawdown,
            };
        }

        for i in window_size..=daily_returns.len() {
            let window = &daily_returns[i - window_size..i];
            let returns: Vec<f64> = window.iter().map(|r| r.return_pct).collect();
            let date = window.last().unwrap().date;

            // Rolling return
            let period_return = self.calculate_total_return(&returns);
            rolling_return.push(RollingPoint {
                date,
                value: period_return,
            });

            // Rolling volatility
            let volatility = self.calculate_volatility(&returns);
            rolling_volatility.push(RollingPoint {
                date,
                value: volatility,
            });

            // Rolling Sharpe ratio
            let annual_return = self.calculate_annual_return(&returns, window[0].date, date);
            let sharpe = self.calculate_sharpe_ratio(annual_return, volatility);
            rolling_sharpe.push(RollingPoint {
                date,
                value: sharpe,
            });

            // Rolling max drawdown
            let max_dd = self.calculate_max_drawdown_simple(&returns);
            rolling_max_drawdown.push(RollingPoint {
                date,
                value: max_dd,
            });
        }

        RollingMetrics {
            rolling_sharpe_30d: rolling_sharpe,
            rolling_volatility_30d: rolling_volatility,
            rolling_return_30d: rolling_return,
            rolling_max_drawdown_30d: rolling_max_drawdown,
        }
    }

    pub fn generate_performance_report(&self, analytics: &PerformanceAnalytics) -> String {
        format!(
            r#"
PERFORMANCE ANALYTICS REPORT
============================
Period: {} to {}

RETURNS
-------
Total Return: {:.2}%
Annual Return: {:.2}%
Volatility: {:.2}%

RISK-ADJUSTED METRICS
--------------------
Sharpe Ratio: {:.2}
Sortino Ratio: {:.2}
Calmar Ratio: {:.2}

DRAWDOWN ANALYSIS
----------------
Max Drawdown: {:.2}%
Max Drawdown Duration: {} days
Current Drawdown: {:.2}%
Ulcer Index: {:.4}

TRADE ANALYSIS
-------------
Total Trades: {}
Win Rate: {:.1}%
Profit Factor: {:.2}
Average Win: {:.2}%
Average Loss: {:.2}%
Expectancy: {:.4}
Kelly Criterion: {:.2}%

RISK METRICS
-----------
95% VaR: {:.2}%
99% VaR: {:.2}%
Downside Deviation: {:.2}%

DAILY STATISTICS
---------------
Mean Daily Return: {:.4}%
Median Daily Return: {:.4}%
Best Day: {:.2}%
Worst Day: {:.2}%
Positive Days: {}
Negative Days: {}
"#,
            analytics.period_start.format("%Y-%m-%d"),
            analytics.period_end.format("%Y-%m-%d"),
            analytics.total_return * 100.0,
            analytics.annual_return * 100.0,
            analytics.volatility * 100.0,
            analytics.sharpe_ratio,
            analytics.sortino_ratio,
            analytics.calmar_ratio,
            analytics.max_drawdown.max_drawdown_pct * 100.0,
            analytics.max_drawdown.max_drawdown_duration_days,
            analytics.max_drawdown.current_drawdown_pct * 100.0,
            analytics.max_drawdown.ulcer_index,
            analytics.trade_analysis.total_trades,
            analytics.trade_analysis.win_rate * 100.0,
            analytics.trade_analysis.profit_factor,
            analytics.trade_analysis.average_win * 100.0,
            analytics.trade_analysis.average_loss * 100.0,
            analytics.trade_analysis.expectancy,
            analytics.trade_analysis.kelly_criterion * 100.0,
            analytics.risk_metrics.value_at_risk_95 * 100.0,
            analytics.risk_metrics.value_at_risk_99 * 100.0,
            analytics.risk_metrics.downside_deviation * 100.0,
            analytics.daily_statistics.mean_daily_return * 100.0,
            analytics.daily_statistics.median_daily_return * 100.0,
            analytics.daily_statistics.best_day * 100.0,
            analytics.daily_statistics.worst_day * 100.0,
            analytics.daily_statistics.positive_days,
            analytics.daily_statistics.negative_days,
        )
    }
}

impl Default for TradeAnalysis {
    fn default() -> Self {
        Self {
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: 0.0,
            profit_factor: 0.0,
            average_win: 0.0,
            average_loss: 0.0,
            largest_win: 0.0,
            largest_loss: 0.0,
            average_trade_duration_hours: 0.0,
            expectancy: 0.0,
            kelly_criterion: 0.0,
            consecutive_wins: ConsecutiveStats {
                max_consecutive: 0,
                current_consecutive: 0,
                average_consecutive: 0.0,
            },
            consecutive_losses: ConsecutiveStats {
                max_consecutive: 0,
                current_consecutive: 0,
                average_consecutive: 0.0,
            },
            trade_distribution: TradeDistribution {
                win_buckets: HashMap::new(),
                loss_buckets: HashMap::new(),
                duration_buckets: HashMap::new(),
            },
        }
    }
}

impl Default for DailyStatistics {
    fn default() -> Self {
        Self {
            mean_daily_return: 0.0,
            median_daily_return: 0.0,
            std_dev_daily_return: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            positive_days: 0,
            negative_days: 0,
            zero_days: 0,
            best_day: 0.0,
            worst_day: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_engine_creation() {
        let engine = AnalyticsEngine::new(0.02);
        assert_eq!(engine.risk_free_rate, 0.02);
    }

    #[test]
    fn test_total_return_calculation() {
        let engine = AnalyticsEngine::new(0.02);
        let returns = vec![0.01, 0.02, -0.01, 0.03];
        let total_return = engine.calculate_total_return(&returns);
        
        // (1.01 * 1.02 * 0.99 * 1.03) - 1 = 0.051...
        assert!((total_return - 0.0510).abs() < 0.001);
    }

    #[test]
    fn test_volatility_calculation() {
        let engine = AnalyticsEngine::new(0.02);
        let returns = vec![0.01, 0.02, -0.01, 0.03, 0.005];
        let volatility = engine.calculate_volatility(&returns);
        
        assert!(volatility > 0.0);
        assert!(volatility < 1.0); // Should be reasonable
    }

    #[test]
    fn test_sharpe_ratio_calculation() {
        let engine = AnalyticsEngine::new(0.02);
        let annual_return = 0.12; // 12%
        let volatility = 0.15; // 15%
        let sharpe = engine.calculate_sharpe_ratio(annual_return, volatility);
        
        // (0.12 - 0.02) / 0.15 = 0.667
        assert!((sharpe - 0.667).abs() < 0.001);
    }

    #[test]
    fn test_max_drawdown_calculation() {
        let engine = AnalyticsEngine::new(0.02);
        let returns = vec![0.10, -0.05, -0.03, 0.08, -0.12, 0.06];
        let max_dd = engine.calculate_max_drawdown_simple(&returns);
        
        assert!(max_dd > 0.0);
        assert!(max_dd < 1.0);
    }

    #[test]
    fn test_var_calculation() {
        let engine = AnalyticsEngine::new(0.02);
        let mut returns = vec![0.01, 0.02, -0.05, 0.03, -0.02, -0.08, 0.04, -0.01];
        returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let var_95 = engine.calculate_var(&returns, 0.05);
        assert!(var_95 > 0.0);
    }
}