use anyhow::Result;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::data::PriceData;
use crate::indicators::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyEngineConfig {
    pub base_strategy: BaseStrategyType,
    pub indicators: IndicatorConfig,
    pub signal_weights: SignalWeights,
    pub risk_parameters: RiskParameters,
    pub backtesting: BacktestingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaseStrategyType {
    TrendReversal,
    MeanReversion,
    Momentum,
    Combined,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorConfig {
    pub rsi: Option<RSIConfig>,
    pub macd: Option<MACDConfig>,
    pub bollinger_bands: Option<BollingerBandsConfig>,
    pub stochastic: Option<StochasticConfig>,
    pub williams_r: Option<WilliamsRConfig>,
    pub cci: Option<CCIConfig>,
    pub atr: Option<ATRConfig>,
    pub volume: Option<VolumeConfig>,
    pub support_resistance: Option<SupportResistanceConfig>,
    pub ema: Option<EMAConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RSIConfig {
    pub enabled: bool,
    pub period: usize,
    pub overbought_threshold: f64,
    pub oversold_threshold: f64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MACDConfig {
    pub enabled: bool,
    pub fast_period: usize,
    pub slow_period: usize,
    pub signal_period: usize,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BollingerBandsConfig {
    pub enabled: bool,
    pub period: usize,
    pub std_dev_multiplier: f64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticConfig {
    pub enabled: bool,
    pub k_period: usize,
    pub d_period: usize,
    pub overbought_threshold: f64,
    pub oversold_threshold: f64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WilliamsRConfig {
    pub enabled: bool,
    pub period: usize,
    pub overbought_threshold: f64,
    pub oversold_threshold: f64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CCIConfig {
    pub enabled: bool,
    pub period: usize,
    pub overbought_threshold: f64,
    pub oversold_threshold: f64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATRConfig {
    pub enabled: bool,
    pub period: usize,
    pub volatility_threshold: f64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeConfig {
    pub enabled: bool,
    pub period: usize,
    pub volume_spike_threshold: f64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportResistanceConfig {
    pub enabled: bool,
    pub lookback_period: usize,
    pub proximity_threshold: f64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EMAConfig {
    pub enabled: bool,
    pub periods: Vec<usize>,
    pub gap_threshold: f64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalWeights {
    pub trend_following: f64,
    pub mean_reversion: f64,
    pub momentum: f64,
    pub volume_confirmation: f64,
    pub volatility_adjustment: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskParameters {
    pub min_confidence_threshold: f64,
    pub max_position_size: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    pub max_drawdown_pct: f64,
    pub correlation_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestingConfig {
    pub enabled: bool,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
    pub initial_capital: f64,
    pub commission_per_trade: f64,
    pub slippage_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedTradingSignal {
    pub symbol: String,
    pub action: Action,
    pub confidence: f64,
    pub timestamp: DateTime<Utc>,
    pub price: Decimal,
    pub reasoning: String,
    pub indicator_signals: HashMap<String, IndicatorSignal>,
    pub combined_score: f64,
    pub risk_score: f64,
    pub expected_return: f64,
    pub stop_loss: Option<Decimal>,
    pub take_profit: Option<Decimal>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Action {
    Buy,
    Sell,
    Hold,
    StrongBuy,
    StrongSell,
}

pub struct StrategyEngine {
    config: StrategyEngineConfig,
    indicators: IndicatorCollection,
    signal_history: HashMap<String, Vec<EnhancedTradingSignal>>,
    performance_metrics: PerformanceMetrics,
}

struct IndicatorCollection {
    rsi: Option<RSI>,
    macd: Option<MACDIndicator>,
    bollinger_bands: Option<BollingerBandsIndicator>,
    stochastic: Option<StochasticIndicator>,
    williams_r: Option<WilliamsR>,
    cci: Option<CCI>,
    atr: Option<ATR>,
    volume: Option<VolumeIndicators>,
    support_resistance: Option<SupportResistance>,
    _emas: Vec<EMA>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub profit_factor: f64,
    pub average_win: f64,
    pub average_loss: f64,
    pub win_rate: f64,
    pub expectancy: f64,
}

impl StrategyEngine {
    pub fn new(config: StrategyEngineConfig) -> Result<Self> {
        let indicators = IndicatorCollection::new(&config.indicators)?;
        
        Ok(Self {
            config,
            indicators,
            signal_history: HashMap::new(),
            performance_metrics: PerformanceMetrics::default(),
        })
    }

    pub async fn analyze(&mut self, symbol: &str, price_data: &PriceData) -> Result<Vec<EnhancedTradingSignal>> {
        let price = price_data.close.to_f64().unwrap_or(0.0);
        let high = price_data.high.to_f64().unwrap_or(0.0);
        let low = price_data.low.to_f64().unwrap_or(0.0);
        let volume = price_data.volume as f64;

        // Collect signals from all enabled indicators
        let mut indicator_signals = HashMap::new();

        // RSI
        if let Some(ref mut rsi) = self.indicators.rsi {
            if let Some(rsi_value) = rsi.update(price) {
                let signal = rsi.generate_signal(rsi_value);
                indicator_signals.insert("RSI".to_string(), signal);
            }
        }

        // MACD
        if let Some(ref mut macd) = self.indicators.macd {
            if let Some(macd_data) = macd.update(price) {
                let signal = macd.generate_signal(&macd_data);
                indicator_signals.insert("MACD".to_string(), signal);
            }
        }

        // Bollinger Bands
        if let Some(ref mut bb) = self.indicators.bollinger_bands {
            if let Some(bands) = bb.update(price) {
                let signal = bb.generate_signal(&bands, price);
                indicator_signals.insert("BollingerBands".to_string(), signal);
            }
        }

        // Stochastic
        if let Some(ref mut stoch) = self.indicators.stochastic {
            if let Some(stoch_data) = stoch.update(high, low, price) {
                let signal = stoch.generate_signal(&stoch_data);
                indicator_signals.insert("Stochastic".to_string(), signal);
            }
        }

        // Williams %R
        if let Some(ref mut williams) = self.indicators.williams_r {
            if let Some(wr_value) = williams.update(high, low, price) {
                let signal = williams.generate_signal(wr_value);
                indicator_signals.insert("WilliamsR".to_string(), signal);
            }
        }

        // CCI
        if let Some(ref mut cci) = self.indicators.cci {
            if let Some(cci_value) = cci.update(high, low, price) {
                let signal = cci.generate_signal(cci_value);
                indicator_signals.insert("CCI".to_string(), signal);
            }
        }

        // ATR (for volatility context)
        if let Some(ref mut atr) = self.indicators.atr {
            if let Some(_atr_value) = atr.update(high, low, price) {
                // ATR used for volatility context, not direct signals
            }
        }

        // Volume
        if let Some(ref mut vol) = self.indicators.volume {
            if let Some(_avg_vol) = vol.update(volume) {
                if let Some(vol_ratio) = vol.volume_ratio(volume) {
                    let signal = vol.generate_signal(vol_ratio);
                    indicator_signals.insert("Volume".to_string(), signal);
                }
            }
        }

        // Support/Resistance
        if let Some(ref mut sr) = self.indicators.support_resistance {
            sr.update(high, low, price);
            let signal = sr.generate_signal(price);
            indicator_signals.insert("SupportResistance".to_string(), signal);
        }

        // Generate combined signal
        let signals = self.generate_combined_signals(symbol, price_data, indicator_signals)?;
        
        // Store signals in history
        self.signal_history.entry(symbol.to_string())
            .or_insert_with(Vec::new)
            .extend(signals.clone());

        // Keep only recent signals
        if let Some(history) = self.signal_history.get_mut(symbol) {
            if history.len() > 1000 {
                history.drain(0..500);
            }
        }

        Ok(signals)
    }

    fn generate_combined_signals(
        &self,
        symbol: &str,
        price_data: &PriceData,
        indicator_signals: HashMap<String, IndicatorSignal>,
    ) -> Result<Vec<EnhancedTradingSignal>> {
        if indicator_signals.is_empty() {
            return Ok(vec![]);
        }

        let mut signals = Vec::new();
        
        // Calculate weighted scores
        let (bullish_score, bearish_score) = self.calculate_weighted_scores(&indicator_signals);
        
        // Determine primary action
        let (action, confidence) = self.determine_action(bullish_score, bearish_score);
        
        if action != Action::Hold && confidence >= self.config.risk_parameters.min_confidence_threshold {
            let combined_score = bullish_score - bearish_score;
            let risk_score = self.calculate_risk_score(&indicator_signals);
            let expected_return = self.calculate_expected_return(combined_score, risk_score);
            
            // Calculate stop loss and take profit levels
            let (stop_loss, take_profit) = self.calculate_stop_take_levels(price_data, &action);
            
            let reasoning = self.build_reasoning_string(&indicator_signals, combined_score);

            let signal = EnhancedTradingSignal {
                symbol: symbol.to_string(),
                action,
                confidence,
                timestamp: price_data.timestamp,
                price: price_data.close,
                reasoning,
                indicator_signals,
                combined_score,
                risk_score,
                expected_return,
                stop_loss,
                take_profit,
            };

            signals.push(signal);
        }

        Ok(signals)
    }

    fn calculate_weighted_scores(&self, signals: &HashMap<String, IndicatorSignal>) -> (f64, f64) {
        let mut bullish_score = 0.0;
        let mut bearish_score = 0.0;

        for (indicator_name, signal) in signals {
            let weight = self.get_indicator_weight(indicator_name);
            let weighted_strength = signal.strength * weight;

            match signal.signal_type {
                SignalType::Buy | SignalType::Bullish | SignalType::Oversold => {
                    bullish_score += weighted_strength;
                }
                SignalType::Sell | SignalType::Bearish | SignalType::Overbought => {
                    bearish_score += weighted_strength;
                }
                SignalType::Neutral => {
                    // Neutral signals don't contribute to either score
                }
            }
        }

        (bullish_score, bearish_score)
    }

    fn get_indicator_weight(&self, indicator_name: &str) -> f64 {
        match indicator_name {
            "RSI" => self.config.indicators.rsi.as_ref().map(|c| c.weight).unwrap_or(0.0),
            "MACD" => self.config.indicators.macd.as_ref().map(|c| c.weight).unwrap_or(0.0),
            "BollingerBands" => self.config.indicators.bollinger_bands.as_ref().map(|c| c.weight).unwrap_or(0.0),
            "Stochastic" => self.config.indicators.stochastic.as_ref().map(|c| c.weight).unwrap_or(0.0),
            "WilliamsR" => self.config.indicators.williams_r.as_ref().map(|c| c.weight).unwrap_or(0.0),
            "CCI" => self.config.indicators.cci.as_ref().map(|c| c.weight).unwrap_or(0.0),
            "Volume" => self.config.indicators.volume.as_ref().map(|c| c.weight).unwrap_or(0.0),
            "SupportResistance" => self.config.indicators.support_resistance.as_ref().map(|c| c.weight).unwrap_or(0.0),
            _ => 1.0,
        }
    }

    fn determine_action(&self, bullish_score: f64, bearish_score: f64) -> (Action, f64) {
        let net_score = bullish_score - bearish_score;
        let total_score = bullish_score + bearish_score;
        
        if total_score == 0.0 {
            return (Action::Hold, 0.0);
        }

        let confidence = net_score.abs() / total_score;
        
        let action = match net_score {
            x if x > 2.0 => Action::StrongBuy,
            x if x > 0.5 => Action::Buy,
            x if x < -2.0 => Action::StrongSell,
            x if x < -0.5 => Action::Sell,
            _ => Action::Hold,
        };

        (action, confidence)
    }

    fn calculate_risk_score(&self, signals: &HashMap<String, IndicatorSignal>) -> f64 {
        let mut risk_factors = 0.0;
        let mut total_factors = 0.0;

        // High volatility increases risk
        if let Some(atr_signal) = signals.get("ATR") {
            risk_factors += atr_signal.value * 0.3;
            total_factors += 1.0;
        }

        // Overbought/oversold conditions increase risk
        for signal in signals.values() {
            match signal.signal_type {
                SignalType::Overbought | SignalType::Oversold => {
                    risk_factors += signal.strength * 0.2;
                    total_factors += 1.0;
                }
                _ => {}
            }
        }

        if total_factors > 0.0 {
            (risk_factors / total_factors).min(1.0)
        } else {
            0.5 // Default moderate risk
        }
    }

    fn calculate_expected_return(&self, combined_score: f64, risk_score: f64) -> f64 {
        // Simple expected return calculation
        // In practice, this would be based on historical performance
        let base_return = combined_score * 0.02; // 2% per unit of combined score
        let risk_adjustment = risk_score * 0.5; // Reduce expected return by risk
        
        (base_return - risk_adjustment).max(-0.1).min(0.1) // Cap at +/-10%
    }

    fn calculate_stop_take_levels(&self, price_data: &PriceData, action: &Action) -> (Option<Decimal>, Option<Decimal>) {
        let price = price_data.close;
        let stop_loss_pct = Decimal::from_f64(self.config.risk_parameters.stop_loss_pct).unwrap_or_default();
        let take_profit_pct = Decimal::from_f64(self.config.risk_parameters.take_profit_pct).unwrap_or_default();

        match action {
            Action::Buy | Action::StrongBuy => {
                let stop_loss = price * (Decimal::ONE - stop_loss_pct);
                let take_profit = price * (Decimal::ONE + take_profit_pct);
                (Some(stop_loss), Some(take_profit))
            }
            Action::Sell | Action::StrongSell => {
                let stop_loss = price * (Decimal::ONE + stop_loss_pct);
                let take_profit = price * (Decimal::ONE - take_profit_pct);
                (Some(stop_loss), Some(take_profit))
            }
            Action::Hold => (None, None),
        }
    }

    fn build_reasoning_string(&self, signals: &HashMap<String, IndicatorSignal>, combined_score: f64) -> String {
        let mut reasoning_parts = Vec::new();
        
        for (indicator, signal) in signals {
            if signal.strength > 0.1 {
                reasoning_parts.push(format!(
                    "{}: {:?} (strength: {:.2})",
                    indicator, signal.signal_type, signal.strength
                ));
            }
        }

        let base_reasoning = format!("Combined score: {:.2}", combined_score);
        
        if reasoning_parts.is_empty() {
            base_reasoning
        } else {
            format!("{} | {}", base_reasoning, reasoning_parts.join(", "))
        }
    }

    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }

    pub fn update_performance_metrics(&mut self, trade_result: &TradeResult) {
        self.performance_metrics.total_trades += 1;
        
        if trade_result.profit > 0.0 {
            self.performance_metrics.winning_trades += 1;
            self.performance_metrics.average_win = 
                (self.performance_metrics.average_win * (self.performance_metrics.winning_trades - 1) as f64 + trade_result.profit) 
                / self.performance_metrics.winning_trades as f64;
        } else {
            self.performance_metrics.losing_trades += 1;
            self.performance_metrics.average_loss = 
                (self.performance_metrics.average_loss * (self.performance_metrics.losing_trades - 1) as f64 + trade_result.profit.abs()) 
                / self.performance_metrics.losing_trades as f64;
        }

        self.performance_metrics.total_return += trade_result.profit;
        self.performance_metrics.win_rate = self.performance_metrics.winning_trades as f64 / self.performance_metrics.total_trades as f64;
        
        if self.performance_metrics.average_loss > 0.0 {
            self.performance_metrics.profit_factor = 
                (self.performance_metrics.average_win * self.performance_metrics.winning_trades as f64) /
                (self.performance_metrics.average_loss * self.performance_metrics.losing_trades as f64);
        }

        self.performance_metrics.expectancy = 
            (self.performance_metrics.win_rate * self.performance_metrics.average_win) -
            ((1.0 - self.performance_metrics.win_rate) * self.performance_metrics.average_loss);
    }

    pub fn get_signal_history(&self, symbol: &str) -> Option<&Vec<EnhancedTradingSignal>> {
        self.signal_history.get(symbol)
    }
}

impl IndicatorCollection {
    fn new(config: &IndicatorConfig) -> Result<Self> {
        Ok(Self {
            rsi: config.rsi.as_ref().filter(|c| c.enabled).map(|c| RSI::new(c.period)),
            macd: config.macd.as_ref().filter(|c| c.enabled).map(|c| MACDIndicator::new(c.fast_period, c.slow_period, c.signal_period)),
            bollinger_bands: config.bollinger_bands.as_ref().filter(|c| c.enabled).map(|c| BollingerBandsIndicator::new(c.period, c.std_dev_multiplier)),
            stochastic: config.stochastic.as_ref().filter(|c| c.enabled).map(|c| StochasticIndicator::new(c.k_period, c.d_period)),
            williams_r: config.williams_r.as_ref().filter(|c| c.enabled).map(|c| WilliamsR::new(c.period)),
            cci: config.cci.as_ref().filter(|c| c.enabled).map(|c| CCI::new(c.period)),
            atr: config.atr.as_ref().filter(|c| c.enabled).map(|c| ATR::new(c.period)),
            volume: config.volume.as_ref().filter(|c| c.enabled).map(|c| VolumeIndicators::new(c.period)),
            support_resistance: config.support_resistance.as_ref().filter(|c| c.enabled).map(|c| SupportResistance::new(c.lookback_period)),
            _emas: config.ema.as_ref()
                .filter(|c| c.enabled)
                .map(|c| c.periods.iter().map(|&period| EMA::new(period)).collect())
                .unwrap_or_default(),
        })
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            total_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            profit_factor: 0.0,
            average_win: 0.0,
            average_loss: 0.0,
            win_rate: 0.0,
            expectancy: 0.0,
        }
    }
}

impl Default for StrategyEngineConfig {
    fn default() -> Self {
        Self {
            base_strategy: BaseStrategyType::Combined,
            indicators: IndicatorConfig::default(),
            signal_weights: SignalWeights::default(),
            risk_parameters: RiskParameters::default(),
            backtesting: BacktestingConfig::default(),
        }
    }
}

impl Default for IndicatorConfig {
    fn default() -> Self {
        Self {
            rsi: Some(RSIConfig {
                enabled: true,
                period: 14,
                overbought_threshold: 70.0,
                oversold_threshold: 30.0,
                weight: 1.0,
            }),
            macd: Some(MACDConfig {
                enabled: true,
                fast_period: 12,
                slow_period: 26,
                signal_period: 9,
                weight: 1.0,
            }),
            bollinger_bands: Some(BollingerBandsConfig {
                enabled: true,
                period: 20,
                std_dev_multiplier: 2.0,
                weight: 0.8,
            }),
            stochastic: Some(StochasticConfig {
                enabled: false, // Disabled by default to avoid redundancy with RSI
                k_period: 14,
                d_period: 3,
                overbought_threshold: 80.0,
                oversold_threshold: 20.0,
                weight: 0.7,
            }),
            williams_r: Some(WilliamsRConfig {
                enabled: false,
                period: 14,
                overbought_threshold: -20.0,
                oversold_threshold: -80.0,
                weight: 0.6,
            }),
            cci: Some(CCIConfig {
                enabled: false,
                period: 20,
                overbought_threshold: 100.0,
                oversold_threshold: -100.0,
                weight: 0.7,
            }),
            atr: Some(ATRConfig {
                enabled: true,
                period: 14,
                volatility_threshold: 2.0,
                weight: 0.5,
            }),
            volume: Some(VolumeConfig {
                enabled: true,
                period: 20,
                volume_spike_threshold: 1.5,
                weight: 0.6,
            }),
            support_resistance: Some(SupportResistanceConfig {
                enabled: true,
                lookback_period: 50,
                proximity_threshold: 0.01,
                weight: 0.8,
            }),
            ema: Some(EMAConfig {
                enabled: true,
                periods: vec![9, 21, 50],
                gap_threshold: 0.02,
                weight: 1.0,
            }),
        }
    }
}

impl Default for SignalWeights {
    fn default() -> Self {
        Self {
            trend_following: 1.0,
            mean_reversion: 0.8,
            momentum: 0.9,
            volume_confirmation: 0.6,
            volatility_adjustment: 0.5,
        }
    }
}

impl Default for RiskParameters {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.6,
            max_position_size: 0.1, // 10%
            stop_loss_pct: 0.05, // 5%
            take_profit_pct: 0.10, // 10%
            max_drawdown_pct: 0.15, // 15%
            correlation_threshold: 0.7,
        }
    }
}

impl Default for BacktestingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            start_date: None,
            end_date: None,
            initial_capital: 100000.0,
            commission_per_trade: 1.0,
            slippage_pct: 0.001, // 0.1%
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeResult {
    pub symbol: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub profit: f64,
    pub duration_minutes: i64,
    pub strategy_used: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_engine_creation() {
        let config = StrategyEngineConfig::default();
        let engine = StrategyEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_indicator_weights() {
        let config = StrategyEngineConfig::default();
        let engine = StrategyEngine::new(config).unwrap();
        
        assert_eq!(engine.get_indicator_weight("RSI"), 1.0);
        assert_eq!(engine.get_indicator_weight("MACD"), 1.0);
        assert_eq!(engine.get_indicator_weight("NonExistent"), 1.0);
    }

    #[test]
    fn test_action_determination() {
        let config = StrategyEngineConfig::default();
        let engine = StrategyEngine::new(config).unwrap();
        
        let (action, confidence) = engine.determine_action(3.0, 1.0);
        assert!(matches!(action, Action::StrongBuy));
        assert!(confidence > 0.0);
    }
}