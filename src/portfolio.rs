use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use log::info;

use crate::strategy::{TradingSignal, Action};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioManager {
    pub positions: HashMap<String, Position>,
    pub cash_balance: Decimal,
    pub total_equity: Decimal,
    pub daily_pnl: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub margin_used: Decimal,
    pub margin_available: Decimal,
    pub trades: Vec<Trade>,
    pub risk_metrics: RiskMetrics,
    pub last_update: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Position {
    pub symbol: String,
    pub quantity: i32,
    pub avg_cost: Decimal,
    pub market_value: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl_today: Decimal,
    pub last_price: Decimal,
    pub entry_time: DateTime<Utc>,
    pub last_update: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: String,
    pub symbol: String,
    pub action: Action,
    pub quantity: i32,
    pub price: Decimal,
    pub timestamp: DateTime<Utc>,
    pub commission: Decimal,
    pub realized_pnl: Decimal,
    pub strategy_signal: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RiskMetrics {
    pub var_1d: Decimal,           // 1-day Value at Risk
    pub max_drawdown: Decimal,
    pub sharpe_ratio: f64,
    pub beta: f64,
    pub correlation_risk: HashMap<String, f64>,
    pub concentration_risk: Decimal,
    pub leverage_ratio: Decimal,
    pub daily_loss_pct: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSummary {
    pub total_value: Decimal,
    pub cash: Decimal,
    pub positions_value: Decimal,
    pub daily_pnl: Decimal,
    pub daily_pnl_pct: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub number_of_positions: usize,
    pub largest_position_pct: Decimal,
    pub risk_metrics: RiskMetrics,
}

impl PortfolioManager {
    pub fn new() -> Self {
        Self {
            positions: HashMap::new(),
            cash_balance: dec!(100000.0), // Start with $100,000
            total_equity: dec!(100000.0),
            daily_pnl: dec!(0.0),
            unrealized_pnl: dec!(0.0),
            realized_pnl: dec!(0.0),
            margin_used: dec!(0.0),
            margin_available: dec!(100000.0),
            trades: Vec::new(),
            risk_metrics: RiskMetrics::default(),
            last_update: Utc::now(),
        }
    }

    pub fn with_initial_cash(initial_cash: Decimal) -> Self {
        Self {
            positions: HashMap::new(),
            cash_balance: initial_cash,
            total_equity: initial_cash,
            daily_pnl: dec!(0.0),
            unrealized_pnl: dec!(0.0),
            realized_pnl: dec!(0.0),
            margin_used: dec!(0.0),
            margin_available: initial_cash,
            trades: Vec::new(),
            risk_metrics: RiskMetrics::default(),
            last_update: Utc::now(),
        }
    }

    pub fn get_position(&self, symbol: &str) -> Result<Position> {
        Ok(self.positions
            .get(symbol)
            .cloned()
            .unwrap_or_else(|| Position {
                symbol: symbol.to_string(),
                quantity: 0,
                avg_cost: dec!(0.0),
                market_value: dec!(0.0),
                unrealized_pnl: dec!(0.0),
                realized_pnl_today: dec!(0.0),
                last_price: dec!(0.0),
                entry_time: Utc::now(),
                last_update: Utc::now(),
            }))
    }

    pub fn update_position(&mut self, symbol: &str, signal: &TradingSignal) -> Result<()> {
        let trade_quantity: i32 = match signal.action {
            Action::Buy => 100,   // Default buy quantity
            Action::Sell => -100, // Default sell quantity
            Action::Hold => return Ok(()), // No position change for hold
        };

        let trade_value = signal.price * Decimal::from(trade_quantity.abs());
        
        // Check if we have enough cash for buy orders
        if trade_quantity > 0 && trade_value > self.cash_balance {
            return Err(anyhow!("Insufficient cash for trade. Required: {}, Available: {}", 
                trade_value, self.cash_balance));
        }

        // Get or create position
        let mut position = self.get_position(symbol)?;
        let old_quantity = position.quantity;
        
        // Calculate new position
        let new_quantity = position.quantity + trade_quantity;
        let mut realized_pnl = dec!(0.0);

        if old_quantity != 0 && (old_quantity > 0) != (trade_quantity > 0) {
            // Closing or reducing position - calculate realized PnL
            let closing_quantity = if trade_quantity.abs() > old_quantity.abs() {
                old_quantity
            } else {
                -trade_quantity
            };
            
            realized_pnl = (signal.price - position.avg_cost) * Decimal::from(closing_quantity);
            info!("Realized PnL for {}: ${}", symbol, realized_pnl);
        }

        // Update average cost for position increases
        if new_quantity != 0 && (old_quantity == 0 || (old_quantity > 0) == (trade_quantity > 0)) {
            let old_value = position.avg_cost * Decimal::from(old_quantity.abs());
            let new_trade_value = signal.price * Decimal::from(trade_quantity.abs());
            let total_value = old_value + new_trade_value;
            let total_quantity = Decimal::from((old_quantity + trade_quantity).abs());
            
            if total_quantity > dec!(0.0) {
                position.avg_cost = total_value / total_quantity;
            }
        }

        // Update position fields
        position.quantity = new_quantity;
        position.last_price = signal.price;
        position.last_update = signal.timestamp;
        position.realized_pnl_today += realized_pnl;

        if old_quantity == 0 {
            position.entry_time = signal.timestamp;
        }

        // Calculate market value and unrealized PnL
        if position.quantity != 0 {
            position.market_value = signal.price * Decimal::from(position.quantity.abs());
            position.unrealized_pnl = (signal.price - position.avg_cost) * Decimal::from(position.quantity);
        } else {
            position.market_value = dec!(0.0);
            position.unrealized_pnl = dec!(0.0);
        }

        // Update cash balance
        self.cash_balance -= signal.price * Decimal::from(trade_quantity);
        self.realized_pnl += realized_pnl;

        // Store or remove position
        if position.quantity == 0 {
            self.positions.remove(symbol);
            info!("Closed position in {}", symbol);
        } else {
            self.positions.insert(symbol.to_string(), position);
            info!("Updated position in {}: {} shares @ ${}", symbol, new_quantity, signal.price);
        }

        // Record the trade
        let trade = Trade {
            id: uuid::Uuid::new_v4().to_string(),
            symbol: symbol.to_string(),
            action: signal.action.clone(),
            quantity: trade_quantity,
            price: signal.price,
            timestamp: signal.timestamp,
            commission: dec!(1.0), // Default commission
            realized_pnl,
            strategy_signal: Some(signal.reasoning.clone()),
        };

        self.trades.push(trade);
        self.last_update = Utc::now();

        // Update portfolio metrics
        self.update_portfolio_metrics()?;

        Ok(())
    }

    pub fn update_market_prices(&mut self, prices: &HashMap<String, Decimal>) -> Result<()> {
        let mut total_unrealized_pnl = dec!(0.0);
        let mut total_market_value = dec!(0.0);

        for (symbol, position) in self.positions.iter_mut() {
            if let Some(&current_price) = prices.get(symbol) {
                position.last_price = current_price;
                position.market_value = current_price * Decimal::from(position.quantity.abs());
                position.unrealized_pnl = (current_price - position.avg_cost) * Decimal::from(position.quantity);
                position.last_update = Utc::now();

                total_unrealized_pnl += position.unrealized_pnl;
                total_market_value += position.market_value;
            }
        }

        self.unrealized_pnl = total_unrealized_pnl;
        self.total_equity = self.cash_balance + total_market_value;
        self.last_update = Utc::now();

        // Update risk metrics
        self.update_risk_metrics()?;

        Ok(())
    }

    fn update_portfolio_metrics(&mut self) -> Result<()> {
        let mut total_market_value = dec!(0.0);
        let mut total_unrealized_pnl = dec!(0.0);

        for position in self.positions.values() {
            total_market_value += position.market_value;
            total_unrealized_pnl += position.unrealized_pnl;
        }

        self.unrealized_pnl = total_unrealized_pnl;
        self.total_equity = self.cash_balance + total_market_value;
        
        // Calculate daily PnL
        self.daily_pnl = self.unrealized_pnl + self.realized_pnl;

        Ok(())
    }

    fn update_risk_metrics(&mut self) -> Result<()> {
        // Calculate concentration risk (largest position as % of portfolio)
        let mut max_position_value = dec!(0.0);
        for position in self.positions.values() {
            max_position_value = max_position_value.max(position.market_value.abs());
        }

        self.risk_metrics.concentration_risk = if self.total_equity > dec!(0.0) {
            max_position_value / self.total_equity
        } else {
            dec!(0.0)
        };

        // Calculate leverage ratio
        let total_position_value: Decimal = self.positions.values()
            .map(|p| p.market_value.abs())
            .sum();

        self.risk_metrics.leverage_ratio = if self.total_equity > dec!(0.0) {
            total_position_value / self.total_equity
        } else {
            dec!(0.0)
        };

        // Calculate daily loss percentage
        self.risk_metrics.daily_loss_pct = if self.total_equity > dec!(0.0) {
            self.daily_pnl / self.total_equity
        } else {
            dec!(0.0)
        };

        // Update max drawdown (simplified)
        if self.daily_pnl < dec!(0.0) {
            let current_drawdown = self.daily_pnl.abs() / self.total_equity;
            self.risk_metrics.max_drawdown = self.risk_metrics.max_drawdown.max(current_drawdown);
        }

        Ok(())
    }

    pub fn get_portfolio_summary(&self) -> PortfolioSummary {
        let positions_value: Decimal = self.positions.values()
            .map(|p| p.market_value)
            .sum();

        let daily_pnl_pct = if self.total_equity > dec!(0.0) {
            (self.daily_pnl / self.total_equity) * dec!(100.0)
        } else {
            dec!(0.0)
        };

        PortfolioSummary {
            total_value: self.total_equity,
            cash: self.cash_balance,
            positions_value,
            daily_pnl: self.daily_pnl,
            daily_pnl_pct,
            unrealized_pnl: self.unrealized_pnl,
            realized_pnl: self.realized_pnl,
            number_of_positions: self.positions.len(),
            largest_position_pct: self.risk_metrics.concentration_risk * dec!(100.0),
            risk_metrics: self.risk_metrics.clone(),
        }
    }



    pub fn check_risk_limits(&self, max_daily_loss_pct: Decimal, max_position_pct: Decimal) -> Vec<String> {
        let mut violations = Vec::new();

        // Check daily loss limit
        if self.risk_metrics.daily_loss_pct.abs() > max_daily_loss_pct {
            violations.push(format!(
                "Daily loss limit exceeded: {:.2}% (limit: {:.2}%)",
                self.risk_metrics.daily_loss_pct * dec!(100.0),
                max_daily_loss_pct * dec!(100.0)
            ));
        }

        // Check position concentration
        if self.risk_metrics.concentration_risk > max_position_pct {
            violations.push(format!(
                "Position concentration exceeded: {:.2}% (limit: {:.2}%)",
                self.risk_metrics.concentration_risk * dec!(100.0),
                max_position_pct * dec!(100.0)
            ));
        }

        violations
    }

    pub fn can_take_position(&self, symbol: &str, quantity: i32, price: Decimal, max_position_pct: Decimal) -> Result<bool> {
        let trade_value = price * Decimal::from(quantity.abs());
        
        // Check cash availability for buy orders
        if quantity > 0 && trade_value > self.cash_balance {
            return Ok(false);
        }

        // Check position size limits
        let current_position_value = self.positions.get(symbol)
            .map(|p| p.market_value.abs())
            .unwrap_or(dec!(0.0));
        
        let new_position_value = current_position_value + trade_value;
        let max_position_value = self.total_equity * max_position_pct;

        if new_position_value > max_position_value {
            return Ok(false);
        }

        Ok(true)
    }


}

impl Default for PortfolioManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::{TradingSignal, Action};
    use rust_decimal_macros::dec;

    fn create_test_signal(symbol: &str, action: Action, price: Decimal) -> TradingSignal {
        TradingSignal {
            symbol: symbol.to_string(),
            action,
            confidence: 0.8,
            timestamp: Utc::now(),
            price,
            reasoning: "Test signal".to_string(),
            ema_gap: 0.02,
            trend_strength: 0.7,
        }
    }

    #[test]
    fn test_new_portfolio() {
        let portfolio = PortfolioManager::new();
        assert_eq!(portfolio.cash_balance, dec!(100000.0));
        assert_eq!(portfolio.total_equity, dec!(100000.0));
        assert_eq!(portfolio.positions.len(), 0);
    }

    #[test]
    fn test_empty_position() {
        let portfolio = PortfolioManager::new();
        let position = portfolio.get_position("AAPL").unwrap();
        assert_eq!(position.quantity, 0);
        assert_eq!(position.symbol, "AAPL");
    }

    #[test]
    fn test_buy_position() {
        let mut portfolio = PortfolioManager::new();
        let signal = create_test_signal("AAPL", Action::Buy, dec!(150.0));
        
        portfolio.update_position("AAPL", &signal).unwrap();
        
        let position = portfolio.get_position("AAPL").unwrap();
        assert_eq!(position.quantity, 100);
        assert_eq!(position.avg_cost, dec!(150.0));
        assert_eq!(portfolio.cash_balance, dec!(85000.0)); // 100000 - (100 * 150)
    }

    #[test]
    fn test_sell_position() {
        let mut portfolio = PortfolioManager::new();
        
        // First buy
        let buy_signal = create_test_signal("AAPL", Action::Buy, dec!(150.0));
        portfolio.update_position("AAPL", &buy_signal).unwrap();
        
        // Then sell
        let sell_signal = create_test_signal("AAPL", Action::Sell, dec!(160.0));
        portfolio.update_position("AAPL", &sell_signal).unwrap();
        
        let position = portfolio.get_position("AAPL").unwrap();
        assert_eq!(position.quantity, 0); // Position closed
        assert!(portfolio.realized_pnl > dec!(0.0)); // Should have profit
    }

    #[test]
    fn test_position_updates() {
        let mut portfolio = PortfolioManager::new();
        let signal = create_test_signal("AAPL", Action::Buy, dec!(150.0));
        
        portfolio.update_position("AAPL", &signal).unwrap();
        
        // Update market price
        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), dec!(155.0));
        portfolio.update_market_prices(&prices).unwrap();
        
        let position = portfolio.get_position("AAPL").unwrap();
        assert_eq!(position.last_price, dec!(155.0));
        assert!(position.unrealized_pnl > dec!(0.0));
    }

    #[test]
    fn test_portfolio_summary() {
        let mut portfolio = PortfolioManager::new();
        let signal = create_test_signal("AAPL", Action::Buy, dec!(150.0));
        
        portfolio.update_position("AAPL", &signal).unwrap();
        
        let summary = portfolio.get_portfolio_summary();
        assert_eq!(summary.number_of_positions, 1);
        assert_eq!(summary.cash, dec!(85000.0));
        assert_eq!(summary.total_value, dec!(100000.0));
    }

    #[test]
    fn test_insufficient_cash() {
        let mut portfolio = PortfolioManager::with_initial_cash(dec!(1000.0));
        let signal = create_test_signal("AAPL", Action::Buy, dec!(150.0)); // Costs $15,000
        
        let result = portfolio.update_position("AAPL", &signal);
        assert!(result.is_err());
    }

    #[test]
    fn test_risk_limits() {
        let portfolio = PortfolioManager::new();
        let violations = portfolio.check_risk_limits(dec!(0.02), dec!(0.1));
        assert_eq!(violations.len(), 0); // No violations initially
    }

    #[test]
    fn test_can_take_position() {
        let portfolio = PortfolioManager::new();
        let can_buy = portfolio.can_take_position("AAPL", 100, dec!(150.0), dec!(0.2)).unwrap();
        assert!(can_buy); // Should be able to buy with sufficient cash and position limits
        
        let cannot_buy = portfolio.can_take_position("AAPL", 1000, dec!(150.0), dec!(0.1)).unwrap();
        assert!(!cannot_buy); // Should not be able to buy due to position size limits
    }
}