use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::{env, fs};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub ibkr: IBKRConfig,
    pub strategy: StrategyConfig,
    pub enhanced_strategy: Option<EnhancedStrategyConfig>,
    pub trading: TradingConfig,
    pub risk: RiskConfig,
    pub historical_backtesting: Option<HistoricalBacktestConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IBKRConfig {
    pub host: String,
    pub port: u16,
    pub client_id: i32,
    #[serde(skip)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalBacktestConfig {
    pub enabled: bool,
    pub symbols: Vec<String>,
    pub start_date: String,           // "2023-01-01"
    pub end_date: String,             // "2023-12-31"
    pub bar_size: String,             // "1 day", "1 hour", "30 mins", "15 mins", "5 mins", "1 min"
    pub initial_capital: f64,
    pub commission_per_trade: f64,
    pub slippage_pct: f64,
    pub what_to_show: String,         // "TRADES", "MIDPOINT", "BID", "ASK"
    pub use_rth: bool,                // Regular Trading Hours only
    pub benchmark_symbol: Option<String>,
    pub save_results: bool,
    pub output_directory: String,
    pub generate_charts: bool,
    pub detailed_logging: bool,
}

impl Config {
    pub fn load() -> Result<Self> {
        // Try to load from config file first
        let mut config = if let Ok(contents) = fs::read_to_string("config.toml") {
            toml::from_str::<Config>(&contents)?
        } else {
            // Fall back to default configuration
            Self::default()
        };

        // Override sensitive fields with environment variables
        config.load_from_env()?;
        
        Ok(config)
    }

    /// Load sensitive configuration from environment variables
    fn load_from_env(&mut self) -> Result<()> {
        // IBKR Account ID - required
        self.ibkr.account_id = env::var("IBKR_ACCOUNT_ID")
            .map_err(|_| anyhow!("IBKR_ACCOUNT_ID environment variable is required"))?;

        // Optional overrides for other IBKR settings
        if let Ok(host) = env::var("IBKR_HOST") {
            self.ibkr.host = host;
        }

        if let Ok(port_str) = env::var("IBKR_PORT") {
            self.ibkr.port = port_str.parse()
                .map_err(|_| anyhow!("Invalid IBKR_PORT value: {}", port_str))?;
        }

        if let Ok(client_id_str) = env::var("IBKR_CLIENT_ID") {
            self.ibkr.client_id = client_id_str.parse()
                .map_err(|_| anyhow!("Invalid IBKR_CLIENT_ID value: {}", client_id_str))?;
        }

        if let Ok(paper_trading_str) = env::var("IBKR_PAPER_TRADING") {
            self.ibkr.paper_trading = paper_trading_str.to_lowercase() == "true";
        }

        Ok(())
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
                account_id: String::new(), // Will be loaded from environment
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
            historical_backtesting: Some(HistoricalBacktestConfig::default()),
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

impl Default for HistoricalBacktestConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            symbols: vec!["AAPL".to_string(), "MSFT".to_string(), "GOOGL".to_string()],
            start_date: "2023-01-01".to_string(),
            end_date: "2023-12-31".to_string(),
            bar_size: "1 day".to_string(),
            initial_capital: 100000.0,
            commission_per_trade: 1.0,
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

// Conversion from EnhancedStrategyConfig to StrategyEngineConfig
impl From<EnhancedStrategyConfig> for crate::strategy_engine::StrategyEngineConfig {
    fn from(enhanced: EnhancedStrategyConfig) -> Self {
        use crate::strategy_engine::*;
        
        let base_strategy = match enhanced.strategy_type.as_str() {
            "TrendReversal" => BaseStrategyType::TrendReversal,
            "MeanReversion" => BaseStrategyType::MeanReversion,
            "Momentum" => BaseStrategyType::Momentum,
            "Combined" => BaseStrategyType::Combined,
            _ => BaseStrategyType::Combined,
        };
        
        let indicators = IndicatorConfig {
            rsi: if enhanced.indicators.rsi_enabled {
                Some(RSIConfig {
                    enabled: true,
                    period: enhanced.indicators.rsi_period as usize,
                    overbought_threshold: 70.0,
                    oversold_threshold: 30.0,
                    weight: enhanced.indicators.rsi_weight,
                })
            } else { None },
            macd: if enhanced.indicators.macd_enabled {
                Some(MACDConfig {
                    enabled: true,
                    fast_period: enhanced.indicators.macd_fast_period as usize,
                    slow_period: enhanced.indicators.macd_slow_period as usize,
                    signal_period: enhanced.indicators.macd_signal_period as usize,
                    weight: enhanced.indicators.macd_weight,
                })
            } else { None },
            bollinger_bands: if enhanced.indicators.bollinger_enabled {
                Some(BollingerBandsConfig {
                    enabled: true,
                    period: enhanced.indicators.bollinger_period as usize,
                    std_dev_multiplier: enhanced.indicators.bollinger_std_dev,
                    weight: enhanced.indicators.bollinger_weight,
                })
            } else { None },
            stochastic: if enhanced.indicators.stochastic_enabled {
                Some(StochasticConfig {
                    enabled: true,
                    k_period: enhanced.indicators.stochastic_k_period as usize,
                    d_period: enhanced.indicators.stochastic_d_period as usize,
                    overbought_threshold: 80.0,
                    oversold_threshold: 20.0,
                    weight: enhanced.indicators.stochastic_weight,
                })
            } else { None },
            williams_r: if enhanced.indicators.williams_r_enabled {
                Some(WilliamsRConfig {
                    enabled: true,
                    period: enhanced.indicators.williams_r_period as usize,
                    overbought_threshold: -20.0,
                    oversold_threshold: -80.0,
                    weight: enhanced.indicators.williams_r_weight,
                })
            } else { None },
            cci: if enhanced.indicators.cci_enabled {
                Some(CCIConfig {
                    enabled: true,
                    period: enhanced.indicators.cci_period as usize,
                    overbought_threshold: 100.0,
                    oversold_threshold: -100.0,
                    weight: enhanced.indicators.cci_weight,
                })
            } else { None },
            atr: if enhanced.indicators.atr_enabled {
                Some(ATRConfig {
                    enabled: true,
                    period: enhanced.indicators.atr_period as usize,
                    volatility_threshold: 1.5,
                    weight: enhanced.indicators.atr_weight,
                })
            } else { None },
            volume: if enhanced.indicators.volume_enabled {
                Some(VolumeConfig {
                    enabled: true,
                    period: enhanced.indicators.volume_period as usize,
                    volume_spike_threshold: 2.0,
                    weight: enhanced.indicators.volume_weight,
                })
            } else { None },
            support_resistance: if enhanced.indicators.support_resistance_enabled {
                Some(SupportResistanceConfig {
                    enabled: true,
                    lookback_period: enhanced.indicators.support_resistance_lookback as usize,
                    proximity_threshold: 0.02,
                    weight: enhanced.indicators.support_resistance_weight,
                })
            } else { None },
            ema: Some(EMAConfig {
                enabled: true,
                periods: vec![9, 21, 50],
                gap_threshold: 0.02,
                weight: 1.0,
            }),
        };
        
        let signal_weights = SignalWeights {
            trend_following: enhanced.signal_weights.trend_following,
            mean_reversion: enhanced.signal_weights.mean_reversion,
            momentum: enhanced.signal_weights.momentum,
            volume_confirmation: enhanced.signal_weights.volume_confirmation,
            volatility_adjustment: enhanced.signal_weights.volatility_adjustment,
        };
        
        let risk_parameters = RiskParameters {
            min_confidence_threshold: enhanced.risk_parameters.min_confidence_threshold,
            max_position_size: enhanced.risk_parameters.max_position_size,
            stop_loss_pct: enhanced.risk_parameters.stop_loss_pct,
            take_profit_pct: enhanced.risk_parameters.take_profit_pct,
            max_drawdown_pct: enhanced.risk_parameters.max_drawdown_pct,
            correlation_threshold: enhanced.risk_parameters.correlation_threshold,
        };
        
        let backtesting = BacktestingConfig {
            enabled: enhanced.backtesting.enabled,
            initial_capital: enhanced.backtesting.initial_capital,
            commission_per_trade: enhanced.backtesting.commission_per_trade,
            slippage_pct: enhanced.backtesting.slippage_pct,
            start_date: None,
            end_date: None,
        };
        
        Self {
            base_strategy,
            indicators,
            signal_weights,
            risk_parameters,
            backtesting,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

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

    #[test]
    fn test_config_load_with_env_vars() {
        // Store original values if they exist
        let original_account_id = env::var("IBKR_ACCOUNT_ID").ok();
        let original_host = env::var("IBKR_HOST").ok();
        let original_port = env::var("IBKR_PORT").ok();
        let original_client_id = env::var("IBKR_CLIENT_ID").ok();
        let original_paper_trading = env::var("IBKR_PAPER_TRADING").ok();

        // Set test environment variables
        env::set_var("IBKR_ACCOUNT_ID", "TEST123456");
        env::set_var("IBKR_HOST", "test-host");
        env::set_var("IBKR_PORT", "8888");
        env::set_var("IBKR_CLIENT_ID", "99");
        env::set_var("IBKR_PAPER_TRADING", "false");

        let mut config = Config::default();
        config.load_from_env().unwrap();

        assert_eq!(config.ibkr.account_id, "TEST123456");
        assert_eq!(config.ibkr.host, "test-host");
        assert_eq!(config.ibkr.port, 8888);
        assert_eq!(config.ibkr.client_id, 99);
        assert!(!config.ibkr.paper_trading);

        // Restore original values or remove if they didn't exist
        if let Some(value) = original_account_id {
            env::set_var("IBKR_ACCOUNT_ID", value);
        } else {
            env::remove_var("IBKR_ACCOUNT_ID");
        }
        if let Some(value) = original_host {
            env::set_var("IBKR_HOST", value);
        } else {
            env::remove_var("IBKR_HOST");
        }
        if let Some(value) = original_port {
            env::set_var("IBKR_PORT", value);
        } else {
            env::remove_var("IBKR_PORT");
        }
        if let Some(value) = original_client_id {
            env::set_var("IBKR_CLIENT_ID", value);
        } else {
            env::remove_var("IBKR_CLIENT_ID");
        }
        if let Some(value) = original_paper_trading {
            env::set_var("IBKR_PAPER_TRADING", value);
        } else {
            env::remove_var("IBKR_PAPER_TRADING");
        }
    }

    #[test]
    fn test_config_load_missing_required_env() {
        // Store original value if it exists
        let original_value = env::var("IBKR_ACCOUNT_ID").ok();
        
        // Ensure IBKR_ACCOUNT_ID is not set
        env::remove_var("IBKR_ACCOUNT_ID");

        let mut config = Config::default();
        let result = config.load_from_env();

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("IBKR_ACCOUNT_ID"));
        
        // Restore original value if it existed
        if let Some(value) = original_value {
            env::set_var("IBKR_ACCOUNT_ID", value);
        }
    }
}