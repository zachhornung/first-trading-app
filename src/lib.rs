pub mod config;
pub mod ibkr;
pub mod strategy;
pub mod data;
pub mod portfolio;
pub mod indicators;
pub mod strategy_engine;
pub mod backtesting;
pub mod historical_backtesting;
pub mod orders;
pub mod analytics;

pub use config::Config;
pub use ibkr::{IBKRClient, Order, OrderType, TimeInForce, Position, MarketData};
pub use strategy::{TrendReversalStrategy, TradingSignal, Action};
pub use data::{PriceData, MarketDataManager, OHLCV, TickData, TickType};
pub use portfolio::{PortfolioManager, Position as PortfolioPosition, Trade, RiskMetrics};
pub use indicators::*;
pub use strategy_engine::{StrategyEngine, StrategyEngineConfig, EnhancedTradingSignal, Action as EngineAction};
pub use backtesting::{BacktestingEngine, BacktestConfig, BacktestResults};
pub use orders::{OrderManager, AdvancedOrder, BracketOrder, TrailingStopOrder, OrderType as AdvancedOrderType};
pub use analytics::{AnalyticsEngine, PerformanceAnalytics, TradeAnalysis, RiskMetrics as AnalyticsRiskMetrics};

// Re-export commonly used types
pub use anyhow::Result;
pub use chrono::{DateTime, Utc};
pub use rust_decimal::Decimal;