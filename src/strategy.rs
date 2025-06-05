use anyhow::Result;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

use crate::config::StrategyConfig;
use crate::data::PriceData;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendReversalStrategy {
    config: StrategyConfig,
    ema_calculators: HashMap<String, EMACalculator>,
    price_history: HashMap<String, VecDeque<PricePoint>>,
    signals_history: HashMap<String, VecDeque<TradingSignal>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePoint {
    pub timestamp: DateTime<Utc>,
    pub price: Decimal,
    pub volume: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub symbol: String,
    pub action: Action,
    pub confidence: f64,
    pub timestamp: DateTime<Utc>,
    pub price: Decimal,
    pub reasoning: String,
    pub ema_gap: f64,
    pub trend_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Action {
    Buy,
    Sell,
    Hold,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EMACalculator {
    periods: Vec<u32>,
    ema_values: HashMap<u32, f64>,
    smoothing_factors: HashMap<u32, f64>,
    initialized: HashMap<u32, bool>,
}

#[derive(Debug, Clone)]
struct TrendAnalysis {
    ema_gap: f64,
    trend_direction: TrendDirection,
    reversal_probability: f64,
    noise_level: f64,
    confirmation_count: u32,
}

#[derive(Debug, Clone, PartialEq)]
enum TrendDirection {
    Upward,
    Downward,
    Sideways,
}

impl TrendReversalStrategy {
    pub fn new(config: &StrategyConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            ema_calculators: HashMap::new(),
            price_history: HashMap::new(),
            signals_history: HashMap::new(),
        })
    }

    pub async fn analyze(&mut self, symbol: &str, price_data: &PriceData) -> Result<Vec<TradingSignal>> {
        // Initialize EMA calculator for symbol if not exists
        if !self.ema_calculators.contains_key(symbol) {
            self.ema_calculators.insert(
                symbol.to_string(),
                EMACalculator::new(&self.config.ema_periods),
            );
        }

        // Initialize price history for symbol if not exists
        if !self.price_history.contains_key(symbol) {
            self.price_history.insert(symbol.to_string(), VecDeque::new());
        }

        if !self.signals_history.contains_key(symbol) {
            self.signals_history.insert(symbol.to_string(), VecDeque::new());
        }

        // Add new price point
        let price_point = PricePoint {
            timestamp: price_data.timestamp,
            price: price_data.close,
            volume: price_data.volume,
        };

        let price_history = self.price_history.get_mut(symbol).unwrap();
        price_history.push_back(price_point.clone());

        // Keep only necessary history
        while price_history.len() > self.config.lookback_periods as usize {
            price_history.pop_front();
        }

        // Update EMA calculations
        let ema_calculator = self.ema_calculators.get_mut(symbol).unwrap();
        ema_calculator.update(price_data.close.to_f64().unwrap_or(0.0));

        // Generate signals if we have enough data
        if price_history.len() >= self.config.reversal_confirmation_periods as usize {
            let analysis = self.analyze_trend(symbol, &price_point)?;
            let signals = self.generate_signals(symbol, &price_point, &analysis)?;
            
            // Store signals in history
            let signals_history = self.signals_history.get_mut(symbol).unwrap();
            for signal in &signals {
                signals_history.push_back(signal.clone());
                // Keep only recent signals
                while signals_history.len() > 50 {
                    signals_history.pop_front();
                }
            }
            
            Ok(signals)
        } else {
            Ok(vec![])
        }
    }

    fn analyze_trend(&self, symbol: &str, current_price: &PricePoint) -> Result<TrendAnalysis> {
        let ema_calculator = self.ema_calculators.get(symbol).unwrap();
        let price_history = self.price_history.get(symbol).unwrap();
        
        let current_price_f64 = current_price.price.to_f64().unwrap_or(0.0);
        
        // Calculate EMA gap (distance between price and primary EMA)
        let primary_ema = ema_calculator.get_ema(self.config.ema_periods[0]).unwrap_or(current_price_f64);
        let ema_gap = (current_price_f64 - primary_ema) / primary_ema;
        
        // Determine trend direction based on EMA alignment
        let trend_direction = self.determine_trend_direction(ema_calculator, current_price_f64);
        
        // Calculate noise level
        let noise_level = self.calculate_noise_level(price_history);
        
        // Calculate reversal probability
        let reversal_probability = self.calculate_reversal_probability(
            ema_gap,
            &trend_direction,
            noise_level,
            price_history,
        );
        
        // Count confirmation periods
        let confirmation_count = self.count_confirmation_periods(price_history, ema_gap);

        Ok(TrendAnalysis {
            ema_gap,
            trend_direction,
            reversal_probability,
            noise_level,
            confirmation_count,
        })
    }

    fn determine_trend_direction(&self, ema_calculator: &EMACalculator, current_price: f64) -> TrendDirection {
        let emas: Vec<f64> = self.config.ema_periods
            .iter()
            .filter_map(|&period| ema_calculator.get_ema(period))
            .collect();

        if emas.len() < 2 {
            return TrendDirection::Sideways;
        }

        let fast_ema = emas[0];
        let slow_ema = emas[emas.len() - 1];

        if current_price > fast_ema && fast_ema > slow_ema {
            TrendDirection::Upward
        } else if current_price < fast_ema && fast_ema < slow_ema {
            TrendDirection::Downward
        } else {
            TrendDirection::Sideways
        }
    }

    fn calculate_noise_level(&self, price_history: &VecDeque<PricePoint>) -> f64 {
        if price_history.len() < 10 {
            return 1.0; // High noise when insufficient data
        }

        let prices: Vec<f64> = price_history
            .iter()
            .take(20) // Use last 20 points
            .map(|p| p.price.to_f64().unwrap_or(0.0))
            .collect();

        if prices.is_empty() {
            return 1.0;
        }

        // Calculate coefficient of variation as noise measure
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();

        if mean == 0.0 {
            1.0
        } else {
            std_dev / mean
        }
    }

    fn calculate_reversal_probability(
        &self,
        ema_gap: f64,
        trend_direction: &TrendDirection,
        noise_level: f64,
        price_history: &VecDeque<PricePoint>,
    ) -> f64 {
        let mut probability: f64 = 0.0;

        // Gap-based probability
        let gap_magnitude = ema_gap.abs();
        if gap_magnitude > self.config.gap_threshold {
            probability += 0.4; // Strong gap indicates potential reversal
        }

        // Trend exhaustion probability
        if matches!(trend_direction, TrendDirection::Upward | TrendDirection::Downward) {
            let trend_age = self.calculate_trend_age(price_history, trend_direction);
            if trend_age > 10 {
                probability += 0.3; // Old trends more likely to reverse
            }
        }

        // Noise filter adjustment
        if noise_level < self.config.noise_filter_threshold {
            probability += 0.2; // Low noise increases confidence
        } else {
            probability -= 0.1; // High noise decreases confidence
        }

        // Volume confirmation (if available)
        if let Some(recent_volume) = price_history.back().map(|p| p.volume) {
            let avg_volume = self.calculate_average_volume(price_history);
            if recent_volume > (avg_volume as f64 * 1.5) as u64 {
                probability += 0.1; // High volume supports reversal
            }
        }

        probability.max(0.0_f64).min(1.0_f64)
    }

    fn calculate_trend_age(&self, price_history: &VecDeque<PricePoint>, current_trend: &TrendDirection) -> u32 {
        let mut age = 0;
        let prices: Vec<f64> = price_history
            .iter()
            .rev()
            .map(|p| p.price.to_f64().unwrap_or(0.0))
            .collect();

        for i in 1..prices.len().min(20) {
            let trend_consistent = match current_trend {
                TrendDirection::Upward => prices[i-1] > prices[i],
                TrendDirection::Downward => prices[i-1] < prices[i],
                TrendDirection::Sideways => return 0,
            };

            if trend_consistent {
                age += 1;
            } else {
                break;
            }
        }

        age
    }

    fn calculate_average_volume(&self, price_history: &VecDeque<PricePoint>) -> u64 {
        let volumes: Vec<u64> = price_history
            .iter()
            .take(20)
            .map(|p| p.volume)
            .collect();

        if volumes.is_empty() {
            0
        } else {
            volumes.iter().sum::<u64>() / volumes.len() as u64
        }
    }

    fn count_confirmation_periods(&self, price_history: &VecDeque<PricePoint>, current_gap: f64) -> u32 {
        let mut count = 0;
        let threshold = self.config.gap_threshold;

        for _point in price_history.iter().rev().take(self.config.reversal_confirmation_periods as usize) {
            // This is a simplified confirmation check
            // In practice, you'd calculate the gap for each historical point
            if current_gap.abs() > threshold {
                count += 1;
            }
        }

        count
    }

    fn generate_signals(
        &self,
        symbol: &str,
        current_price: &PricePoint,
        analysis: &TrendAnalysis,
    ) -> Result<Vec<TradingSignal>> {
        let mut signals = Vec::new();

        if analysis.reversal_probability < self.config.min_confidence_threshold {
            return Ok(signals);
        }

        let action = match analysis.trend_direction {
            TrendDirection::Upward if analysis.ema_gap < -self.config.gap_threshold => {
                // Upward trend with negative gap suggests potential reversal to downside
                Action::Sell
            }
            TrendDirection::Downward if analysis.ema_gap > self.config.gap_threshold => {
                // Downward trend with positive gap suggests potential reversal to upside
                Action::Buy
            }
            _ => Action::Hold,
        };

        if action != Action::Hold {
            let reasoning = format!(
                "Trend reversal detected: direction={:?}, gap={:.4}, noise={:.4}, confirmation={}",
                analysis.trend_direction,
                analysis.ema_gap,
                analysis.noise_level,
                analysis.confirmation_count
            );

            signals.push(TradingSignal {
                symbol: symbol.to_string(),
                action,
                confidence: analysis.reversal_probability,
                timestamp: current_price.timestamp,
                price: current_price.price,
                reasoning,
                ema_gap: analysis.ema_gap,
                trend_strength: 1.0 - analysis.noise_level,
            });
        }

        Ok(signals)
    }


}

impl EMACalculator {
    fn new(periods: &[u32]) -> Self {
        let mut smoothing_factors = HashMap::new();
        let mut ema_values = HashMap::new();
        let mut initialized = HashMap::new();

        for &period in periods {
            smoothing_factors.insert(period, 2.0 / (period as f64 + 1.0));
            ema_values.insert(period, 0.0);
            initialized.insert(period, false);
        }

        Self {
            periods: periods.to_vec(),
            ema_values,
            smoothing_factors,
            initialized,
        }
    }

    fn update(&mut self, price: f64) {
        for &period in &self.periods {
            let smoothing = self.smoothing_factors[&period];
            
            if !self.initialized[&period] {
                self.ema_values.insert(period, price);
                self.initialized.insert(period, true);
            } else {
                let current_ema = self.ema_values[&period];
                let new_ema = (price * smoothing) + (current_ema * (1.0 - smoothing));
                self.ema_values.insert(period, new_ema);
            }
        }
    }

    fn get_ema(&self, period: u32) -> Option<f64> {
        if *self.initialized.get(&period).unwrap_or(&false) {
            self.ema_values.get(&period).copied()
        } else {
            None
        }
    }
}

impl TradingSignal {
    pub fn to_order(&self, symbol: &str) -> Result<crate::ibkr::Order> {
        let quantity = match self.action {
            Action::Buy => 100, // Default position size
            Action::Sell => -100,
            Action::Hold => return Err(anyhow::anyhow!("Cannot create order for Hold action")),
        };

        Ok(crate::ibkr::Order {
            symbol: symbol.to_string(),
            quantity,
            order_type: crate::ibkr::OrderType::Market,
            price: None,
            time_in_force: crate::ibkr::TimeInForce::Day,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StrategyConfig;

    #[test]
    fn test_ema_calculator() {
        let mut calc = EMACalculator::new(&[9, 21]);
        
        calc.update(100.0);
        assert_eq!(calc.get_ema(9), Some(100.0));
        
        calc.update(110.0);
        let ema9 = calc.get_ema(9).unwrap();
        assert!(ema9 > 100.0 && ema9 < 110.0);
    }

    #[test]
    fn test_trend_direction() {
        let config = StrategyConfig {
            ema_periods: vec![9, 21],
            noise_filter_threshold: 0.5,
            gap_threshold: 0.02,
            reversal_confirmation_periods: 3,
            min_confidence_threshold: 0.6,
            lookback_periods: 100,
        };

        let strategy = TrendReversalStrategy::new(&config).unwrap();
        let mut calc = EMACalculator::new(&[9, 21]);
        
        // Simulate upward trend
        calc.update(100.0);
        calc.update(105.0);
        calc.update(110.0);
        
        let direction = strategy.determine_trend_direction(&calc, 115.0);
        assert_eq!(direction, TrendDirection::Upward);
    }
}