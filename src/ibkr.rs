use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use ibapi::contracts::Contract;
use ibapi::orders::{order_builder, Action as IbAction};
use ibapi::Client;
use log::{debug, error, info, warn};
use rand;
use rust_decimal::Decimal;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

use crate::config::IBKRConfig;

#[derive(Debug)]
pub struct IBKRClient {
    client: Option<Client>,
    config: IBKRConfig,
    connected: Arc<RwLock<bool>>,
    positions: Arc<RwLock<HashMap<String, Position>>>,
    market_data_sender: Arc<RwLock<Option<mpsc::Sender<MarketData>>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub symbol: String,
    pub quantity: i32,
    pub order_type: OrderType,
    pub price: Option<Decimal>,
    pub time_in_force: TimeInForce,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeInForce {
    Day,
    GTC, // Good Till Canceled
    IOC, // Immediate or Cancel
    FOK, // Fill or Kill
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: i32,
    pub avg_cost: Decimal,
    pub market_value: Decimal,
    pub unrealized_pnl: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub bid: Option<Decimal>,
    pub ask: Option<Decimal>,
    pub last: Option<Decimal>,
    pub volume: Option<u64>,
    pub timestamp: DateTime<Utc>,
}

impl IBKRClient {
    pub fn new(config: &IBKRConfig) -> Result<Self> {
        let (sender, _receiver) = mpsc::channel(1000);

        Ok(Self {
            client: None,
            config: config.clone(),
            connected: Arc::new(RwLock::new(false)),
            positions: Arc::new(RwLock::new(HashMap::new())),
            market_data_sender: Arc::new(RwLock::new(Some(sender))),
        })
    }

    pub async fn connect(&mut self) -> Result<()> {
        let connection_url = format!("{}:{}", self.config.host, self.config.port);
        info!("Connecting to IBKR at {}...", connection_url);
        
        // Connect to TWS/Gateway
        match Client::connect(&connection_url, self.config.client_id) {
            Ok(client) => {
                info!("Successfully connected to IBKR at {}", connection_url);
                *self.connected.write().await = true;
                
                // Load current positions
                if let Err(e) = self.load_positions(&client).await {
                    warn!("Failed to load positions: {}", e);
                }
                
                self.client = Some(client);
                Ok(())
            }
            Err(e) => {
                error!("Failed to connect to IBKR: {}", e);
                Err(anyhow!("Connection failed: {}", e))
            }
        }
    }

    pub async fn is_connected(&self) -> bool {
        *self.connected.read().await
    }

    async fn load_positions(&self, client: &Client) -> Result<()> {
        info!("Loading current positions...");
        
        match client.positions() {
            Ok(positions_iter) => {
                let mut positions_map = HashMap::new();
                
                for position in positions_iter {
                    let position_data = Position {
                        symbol: position.contract.symbol.clone(),
                        quantity: position.position as i32,
                        avg_cost: Decimal::from_f64(position.average_cost).unwrap_or_default(),
                        market_value: Decimal::from_f64(position.position * position.average_cost).unwrap_or_default(),
                        unrealized_pnl: Decimal::ZERO, // Would need separate calculation
                    };
                    
                    positions_map.insert(position.contract.symbol.clone(), position_data);
                    info!("Position: {} {} shares at ${:.2}", 
                          position.contract.symbol, position.position, position.average_cost);
                }
                
                *self.positions.write().await = positions_map;
                Ok(())
            }
            Err(e) => {
                error!("Failed to get positions: {}", e);
                Err(anyhow!("Failed to get positions: {}", e))
            }
        }
    }

    pub async fn subscribe_market_data(&mut self, symbol: &str) -> Result<()> {
        if !self.is_connected().await {
            return Err(anyhow!("Not connected to IBKR"));
        }

        info!("Setting up market data subscription for {}", symbol);
        
        // For now, we'll simulate market data since real-time bars in rust-ibapi
        // require careful thread handling that would complicate this example
        let symbol_clone = symbol.to_string();
        let sender = self.market_data_sender.read().await.clone();
        
        if let Some(tx) = sender {
            // Simulate market data updates
            tokio::spawn(async move {
                let mut price = 150.0; // Starting price
                loop {
                    // Simple random walk for demonstration
                    price += (rand::random::<f64>() - 0.5) * 2.0;
                    
                    let market_data = MarketData {
                        symbol: symbol_clone.clone(),
                        bid: Decimal::from_f64(price - 0.01),
                        ask: Decimal::from_f64(price + 0.01),
                        last: Decimal::from_f64(price),
                        volume: Some(1000),
                        timestamp: Utc::now(),
                    };
                    
                    if let Err(_) = tx.send(market_data).await {
                        break; // Channel closed
                    }
                    
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                }
            });
        }
        
        Ok(())
    }

    pub async fn cancel_market_data(&mut self, symbol: &str) -> Result<()> {
        info!("Cancelling market data subscription for {}", symbol);
        Ok(())
    }

    pub async fn place_order(&mut self, order: &Order) -> Result<String> {
        if !self.is_connected().await {
            return Err(anyhow!("Not connected to IBKR"));
        }

        let client = self.client.as_ref()
            .ok_or_else(|| anyhow!("Client not available"))?;

        // Get next order ID
        let order_id = client.next_order_id();
        
        // Create contract
        let contract = Contract::stock(&order.symbol);

        // Convert our order to IB order
        let ib_action = if order.quantity > 0 {
            IbAction::Buy
        } else {
            IbAction::Sell
        };

        let ib_order = match order.order_type {
            OrderType::Market => {
                order_builder::market_order(ib_action, order.quantity.abs() as f64)
            }
            OrderType::Limit => {
                if let Some(price) = order.price {
                    if let Some(price_f64) = price.to_f64() {
                        order_builder::limit_order(ib_action, order.quantity.abs() as f64, price_f64)
                    } else {
                        return Err(anyhow!("Invalid limit price"));
                    }
                } else {
                    return Err(anyhow!("Limit order requires a price"));
                }
            }
            OrderType::Stop => {
                return Err(anyhow!("Stop orders not yet implemented"));
            }
            OrderType::StopLimit => {
                return Err(anyhow!("Stop limit orders not yet implemented"));
            }
        };

        info!("Placing {} order for {} shares of {} (order_id: {})", 
              if order.quantity > 0 { "BUY" } else { "SELL" }, 
              order.quantity.abs(), order.symbol, order_id);

        match client.place_order(order_id, &contract, &ib_order) {
            Ok(notifications) => {
                info!("Order placed successfully: order_id {}", order_id);
                
                // Process order notifications
                for notification in notifications {
                    debug!("Order notification: {:?}", notification);
                }
                
                Ok(order_id.to_string())
            }
            Err(e) => {
                error!("Failed to place order: {}", e);
                Err(anyhow!("Order placement failed: {}", e))
            }
        }
    }









    pub async fn start_market_data_handler<F>(&self, mut callback: F) -> Result<()>
    where
        F: FnMut(MarketData) + Send + 'static,
    {
        let (sender, mut receiver) = mpsc::channel(1000);
        
        // Replace the sender
        *self.market_data_sender.write().await = Some(sender);
        
        // Start the handler task
        tokio::spawn(async move {
            while let Some(market_data) = receiver.recv().await {
                callback(market_data);
            }
        });
        
        Ok(())
    }



    pub async fn update_position(&self, symbol: String, position: Position) {
        let mut positions = self.positions.write().await;
        if position.quantity == 0 {
            positions.remove(&symbol);
        } else {
            positions.insert(symbol, position);
        }
    }

    pub async fn get_account_summary(&self) -> Result<HashMap<String, String>> {
        // For now, return basic account info
        let mut summary = HashMap::new();
        summary.insert("AccountType".to_string(), 
                      if self.config.paper_trading { "Paper" } else { "Live" }.to_string());
        summary.insert("Currency".to_string(), "USD".to_string());
        Ok(summary)
    }
}

impl Drop for IBKRClient {
    fn drop(&mut self) {
        debug!("IBKRClient dropped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::IBKRConfig;

    fn create_test_config() -> IBKRConfig {
        IBKRConfig {
            host: "127.0.0.1".to_string(),
            port: 7497,
            client_id: 1,
            account_id: "DU123456".to_string(),
            paper_trading: true,
        }
    }

    #[test]
    fn test_order_creation() {
        let order = Order {
            symbol: "AAPL".to_string(),
            quantity: 100,
            order_type: OrderType::Market,
            price: None,
            time_in_force: TimeInForce::Day,
        };

        assert_eq!(order.symbol, "AAPL");
        assert_eq!(order.quantity, 100);
    }

    #[tokio::test]
    async fn test_connection_status() {
        // This test will fail without a running TWS/Gateway
        // but shows the API structure
        let config = create_test_config();
        let client = IBKRClient::new(&config).unwrap();
        
        // Should be false initially
        assert!(!client.is_connected().await);
    }

    #[tokio::test]
    async fn test_client_creation() {
        let config = create_test_config();
        let client = IBKRClient::new(&config);
        assert!(client.is_ok());
    }
}