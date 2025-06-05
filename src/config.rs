use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub ibkr: IBKRConfig,
    pub strategy: StrategyConfig,
    pub enhanced_strategy: Option<EnhancedStrategyConfig>,
    pub trading: TradingConfig,
    pub risk: RiskConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IBKRConfig {
    pub host: String,
    pub port: u16,
    pub client_id: i32,
    pub account_id: String,
    pub paper_trading: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    pub ema_periods: Vec<u32>,
    pub noise_filter_threshold: f64,
    pub gap_threshold: f64,
    pub reversal_confirmation_periods: u32,
    pub min_confidence_threshold: f64,
    pub lookback_periods: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedStrategyConfig {
    pub strategy_type: String,
    pub indicators: IndicatorSettings,
    pub signal_weights: SignalWeightSettings,
    pub risk_parameters: RiskParameterSettings,
    pub backtesting: BacktestingSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorSettings {
    pub rsi_enabled: bool,
    pub rsi_period: u32,
    pub rsi_weight: f64,
    pub macd_enabled: bool,
    pub macd_fast_period: u32,
    pub macd_slow_period: u32,
    pub macd_signal_period: u32,
    pub macd_weight: f64,
    pub bollinger_enabled: bool,
    pub bollinger_period: u32,
    pub bollinger_std_dev: f64,
    pub bollinger_weight: f64,
    pub stochastic_enabled: bool,
    pub stochastic_k_period: u32,
    pub stochastic_d_period: u32,
    pub stochastic_weight: f64,
    pub williams_r_enabled: bool,
    pub williams_r_period: u32,
    pub williams_r_weight: f64,
    pub cci_enabled: bool,
    pub cci_period: u32,
    pub cci_weight: f64,
    pub atr_enabled: bool,
    pub atr_period: u32,
    pub atr_weight: f64,
    pub volume_enabled: bool,
    pub volume_period: u32,
    pub volume_weight: f64,
    pub support_resistance_enabled: bool,
    pub support_resistance_lookback: u32,
    pub support_resistance_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalWeightSettings {
    pub trend_following: f64,
    pub mean_reversion: f64,
    pub momentum: f64,
    pub volume_confirmation: f64,
    pub volatility_adjustment: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskParameterSettings {
    pub min_confidence_threshold: f64,
    pub max_position_size: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    pub max_drawdown_pct: f64,
    pub correlation_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestingSettings {
    pub enabled: bool,
    pub start_date: Option<String>,
    pub end_date: Option<String>,
    pub initial_capital: f64,
    pub commission_per_trade: f64,
    pub slippage_pct: f64,
    pub benchmark_symbol: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    pub symbols: Vec<String>,
    pub timeframe: String,
    pub max_positions: u32,
    pub position_size_pct: f64,
    pub trading_hours: TradingHours,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingHours {
    pub start: String, // "09:30"
    pub end: String,   // "16:00"
    pub timezone: String, // "US/Eastern"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    pub max_daily_loss_pct: f64,
    pub max_position_size_pct: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    pub max_correlation: f64,
    pub var_limit: f64,
}

impl Config {
    pub fn load() -> Result<Self> {
        // Try to load from config file first
        if let Ok(contents) = fs::read_to_string("config.toml") {
            let config: Config = toml::from_str(&contents)?;
            return Ok(config);
        }

        // Fall back to default configuration
        Ok(Self::default())
    }

    pub fn save(&self) -> Result<()> {
        let contents = toml::to_string_pretty(self)?;
        fs::write("config.toml", contents)?;
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ibkr: IBKRConfig {
                host: "127.0.0.1".to_string(),
                port: 7497, // Paper trading port (7496 for live)
                client_id: 1,
                account_id: "DU123456".to_string(), // Default paper trading account
                paper_trading: true,
            },
            strategy: StrategyConfig {
                ema_periods: vec![9, 21, 50], // Fast, medium, slow EMAs
                noise_filter_threshold: 0.5,
                gap_threshold: 0.02, // 2% gap threshold for reversal signals
                reversal_confirmation_periods: 3,
                min_confidence_threshold: 0.6,
                lookback_periods: 100,
            },
            enhanced_strategy: Some(EnhancedStrategyConfig::default()),
            trading: TradingConfig {
                symbols: vec![
                    "AAPL".to_string(),
                    "MSFT".to_string(),
                    "GOOGL".to_string(),
                    "TSLA".to_string(),
                ],
                timeframe: "1min".to_string(),
                max_positions: 5,
                position_size_pct: 0.1, // 10% of portfolio per position
                trading_hours: TradingHours {
                    start: "09:30".to_string(),
                    end: "16:00".to_string(),
                    timezone: "US/Eastern".to_string(),
                },
            },
            risk: RiskConfig {
                max_daily_loss_pct: 0.02, // 2%
                max_position_size_pct: 0.15, // 15%
                stop_loss_pct: 0.05, // 5%
                take_profit_pct: 0.10, // 10%
                max_correlation: 0.7,
                var_limit: 0.03, // 3% VaR limit
            },
        }
    }
}

impl Default for EnhancedStrategyConfig {
    fn default() -> Self {
        Self {
            strategy_type: "Combined".to_string(),
            indicators: IndicatorSettings::default(),
            signal_weights: SignalWeightSettings::default(),
            risk_parameters: RiskParameterSettings::default(),
            backtesting: BacktestingSettings::default(),
        }
    }
}

impl Default for IndicatorSettings {
    fn default() -> Self {
        Self {
            rsi_enabled: true,
            rsi_period: 14,
            rsi_weight: 1.0,
            macd_enabled: true,
            macd_fast_period: 12,
            macd_slow_period: 26,
            macd_signal_period: 9,
            macd_weight: 1.0,
            bollinger_enabled: true,
            bollinger_period: 20,
            bollinger_std_dev: 2.0,
            bollinger_weight: 0.8,
            stochastic_enabled: false,
            stochastic_k_period: 14,
            stochastic_d_period: 3,
            stochastic_weight: 0.7,
            williams_r_enabled: false,
            williams_r_period: 14,
            williams_r_weight: 0.6,
            cci_enabled: false,
            cci_period: 20,
            cci_weight: 0.7,
            atr_enabled: true,
            atr_period: 14,
            atr_weight: 0.5,
            volume_enabled: true,
            volume_period: 20,
            volume_weight: 0.6,
            support_resistance_enabled: true,
            support_resistance_lookback: 50,
            support_resistance_weight: 0.8,
        }
    }
}

impl Default for SignalWeightSettings {
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

impl Default for RiskParameterSettings {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.6,
            max_position_size: 0.1,
            stop_loss_pct: 0.05,
            take_profit_pct: 0.10,
            max_drawdown_pct: 0.15,
            correlation_threshold: 0.7,
        }
    }
}

impl Default for BacktestingSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            start_date: None,
            end_date: None,
            initial_capital: 100000.0,
            commission_per_trade: 1.0,
            slippage_pct: 0.001,
            benchmark_symbol: Some("SPY".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.ibkr.host, "127.0.0.1");
        assert_eq!(config.ibkr.port, 7497);
        assert!(config.ibkr.paper_trading);
        assert!(!config.strategy.ema_periods.is_empty());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        let deserialized: Config = toml::from_str(&toml_str).unwrap();
        
        assert_eq!(config.ibkr.host, deserialized.ibkr.host);
        assert_eq!(config.strategy.ema_periods, deserialized.strategy.ema_periods);
    }
}