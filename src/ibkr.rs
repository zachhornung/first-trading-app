use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc, NaiveDateTime};
use ibapi::contracts::Contract;
use ibapi::orders::{order_builder, Action as IbAction};
use ibapi::Client;
use ibapi::market_data::realtime::{BarSize, WhatToShow};
use ibapi::market_data::historical::{BarSize as HistBarSize, WhatToShow as HistWhatToShow};
use log::{debug, error, info, warn};
use rust_decimal::Decimal;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use std::sync::Arc;
use time;
use tokio::sync::{mpsc, RwLock};

use crate::config::IBKRConfig;

#[derive(Debug)]
pub struct IBKRClient {
    client: Option<Client>,
    config: IBKRConfig,
    connected: Arc<RwLock<bool>>,
    positions: Arc<RwLock<HashMap<String, Position>>>,
    market_data_sender: Arc<RwLock<Option<mpsc::Sender<MarketData>>>>,
    realtime_bar_req_id: Arc<RwLock<i32>>,
    symbol_to_req_id: Arc<RwLock<HashMap<String, i32>>>,
    tick_req_id: Arc<RwLock<i32>>,
    tick_symbol_to_req_id: Arc<RwLock<HashMap<String, i32>>>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalBar {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: u64,
    pub count: Option<i32>,
    pub wap: Option<Decimal>, // Weighted Average Price
}

#[derive(Debug, Clone)]
pub struct HistoricalDataRequest {
    pub symbol: String,
    pub duration: String,      // "1 Y", "6 M", "1 M", "1 W", "1 D"
    pub bar_size: String,      // "1 day", "1 hour", "30 mins", "15 mins", "5 mins", "1 min"
    pub what_to_show: String,  // "TRADES", "MIDPOINT", "BID", "ASK"
    pub use_rth: bool,         // Regular Trading Hours only
    pub end_date_time: Option<DateTime<Utc>>,
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
            realtime_bar_req_id: Arc::new(RwLock::new(1000)),
            symbol_to_req_id: Arc::new(RwLock::new(HashMap::new())),
            tick_req_id: Arc::new(RwLock::new(2000)),
            tick_symbol_to_req_id: Arc::new(RwLock::new(HashMap::new())),
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

        info!("Setting up comprehensive market data subscription for {}", symbol);
        
        // Subscribe to both real-time bars (for OHLCV data) and tick data (for bid/ask)
        let bars_result = self.subscribe_realtime_bars(symbol).await;
        let tick_result = self.subscribe_tick_data(symbol).await;
        
        match (bars_result, tick_result) {
            (Ok(_), Ok(_)) => {
                info!("Successfully subscribed to all market data for {}", symbol);
                Ok(())
            }
            (Err(e), Ok(_)) => {
                warn!("Failed to subscribe to real-time bars for {}: {}", symbol, e);
                Ok(()) // Still have tick data
            }
            (Ok(_), Err(e)) => {
                warn!("Failed to subscribe to tick data for {}: {}", symbol, e);
                Ok(()) // Still have bars
            }
            (Err(bars_err), Err(tick_err)) => {
                error!("Failed to subscribe to any market data for {}: bars={}, tick={}", 
                       symbol, bars_err, tick_err);
                Err(anyhow!("Failed to subscribe to market data"))
            }
        }
    }

    async fn subscribe_realtime_bars(&mut self, symbol: &str) -> Result<()> {
        let client = self.client.as_ref()
            .ok_or_else(|| anyhow!("Client not available"))?;

        info!("Setting up real-time bars subscription for {}", symbol);
        
        // Create contract for the symbol
        let contract = Contract::stock(symbol);
        
        // Get next request ID
        let mut req_id = self.realtime_bar_req_id.write().await;
        *req_id += 1;
        let current_req_id = *req_id;
        
        // Store the mapping
        self.symbol_to_req_id.write().await.insert(symbol.to_string(), current_req_id);
        
        // Clone necessary data for the callback
        let sender = self.market_data_sender.read().await.clone();
        let symbol_clone = symbol.to_string();
        
        if let Some(tx) = sender {
            // Use the available realtime_bars method
            match client.realtime_bars(
                &contract,
                BarSize::Sec5,
                WhatToShow::Trades,
                false
            ) {
                Ok(bars_iter) => {
                    info!("Successfully subscribed to real-time bars for {}", symbol);
                    
                    // Collect bars to avoid lifetime issues
                    let bars: Vec<_> = bars_iter.collect();
                    
                    // Handle real-time bar updates
                    tokio::spawn(async move {
                        for bar in bars {
                            let market_data = MarketData {
                                symbol: symbol_clone.clone(),
                                bid: None, // Real-time bars don't provide bid/ask
                                ask: None,
                                last: Decimal::from_f64(bar.close),
                                volume: Some(bar.volume as u64),
                                timestamp: Utc::now(),
                            };
                            
                            if let Err(e) = tx.send(market_data).await {
                                error!("Failed to send market data: {}", e);
                                break;
                            }
                        }
                    });
                    
                    Ok(())
                }
                Err(e) => {
                    error!("Failed to request real-time bars for {}: {}", symbol, e);
                    Err(anyhow!("Real-time bars request failed: {}", e))
                }
            }
        } else {
            Err(anyhow!("Market data sender not available"))
        }
    }

    pub async fn cancel_market_data(&mut self, symbol: &str) -> Result<()> {
        if !self.is_connected().await {
            return Err(anyhow!("Not connected to IBKR"));
        }

        info!("Cancelling all market data subscriptions for {}", symbol);
        
        // Cancel both real-time bars and tick data
        let bars_result = self.cancel_realtime_bars(symbol).await;
        let tick_result = self.cancel_tick_data(symbol).await;
        
        // Report results but don't fail if one fails
        match (bars_result, tick_result) {
            (Ok(_), Ok(_)) => info!("Successfully cancelled all subscriptions for {}", symbol),
            (Err(e), Ok(_)) => warn!("Failed to cancel bars for {}: {}", symbol, e),
            (Ok(_), Err(e)) => warn!("Failed to cancel tick data for {}: {}", symbol, e),
            (Err(bars_err), Err(tick_err)) => {
                warn!("Failed to cancel subscriptions for {}: bars={}, tick={}", 
                      symbol, bars_err, tick_err);
            }
        }
        
        Ok(())
    }

    async fn cancel_realtime_bars(&mut self, symbol: &str) -> Result<()> {
        info!("Cancelling real-time bars subscription for {}", symbol);
        
        // For the current ibapi version, we'll just remove from tracking
        // Real cancellation would need specific request ID management
        self.symbol_to_req_id.write().await.remove(symbol);
        info!("Removed real-time bars tracking for {}", symbol);
        Ok(())
    }

    pub async fn subscribe_tick_data(&mut self, symbol: &str) -> Result<()> {
        if !self.is_connected().await {
            return Err(anyhow!("Not connected to IBKR"));
        }

        let client = self.client.as_ref()
            .ok_or_else(|| anyhow!("Client not available"))?;

        info!("Setting up tick-by-tick data subscription for {}", symbol);
        
        // Create contract for the symbol
        let contract = Contract::stock(symbol);
        
        // Get next request ID
        let mut req_id = self.tick_req_id.write().await;
        *req_id += 1;
        let current_req_id = *req_id;
        
        // Store the mapping
        self.tick_symbol_to_req_id.write().await.insert(symbol.to_string(), current_req_id);
        
        // Clone necessary data for the callback
        let sender = self.market_data_sender.read().await.clone();
        let symbol_clone = symbol.to_string();
        
        if let Some(tx) = sender {
            // Use tick_by_tick_bid_ask method
            match client.tick_by_tick_bid_ask(
                &contract,
                100, // number of ticks
                false // ignore size
            ) {
                Ok(tick_iter) => {
                    info!("Successfully subscribed to tick data for {}", symbol);
                    
                    // Collect ticks to avoid lifetime issues
                    let ticks: Vec<_> = tick_iter.collect();
                    
                    // Handle tick-by-tick updates
                    tokio::spawn(async move {
                        for tick in ticks {
                            let market_data = MarketData {
                                symbol: symbol_clone.clone(),
                                bid: Decimal::from_f64(tick.bid_price),
                                ask: Decimal::from_f64(tick.ask_price),
                                last: None,
                                volume: None,
                                timestamp: Utc::now(),
                            };
                            
                            if let Err(e) = tx.send(market_data).await {
                                error!("Failed to send tick data: {}", e);
                                break;
                            }
                        }
                    });
                    
                    Ok(())
                }
                Err(e) => {
                    error!("Failed to request tick data for {}: {}", symbol, e);
                    Err(anyhow!("Tick data request failed: {}", e))
                }
            }
        } else {
            Err(anyhow!("Market data sender not available"))
        }
    }

    pub async fn cancel_tick_data(&mut self, symbol: &str) -> Result<()> {
        info!("Cancelling tick-by-tick data subscription for {}", symbol);
        
        // For the current ibapi version, we'll just remove from tracking
        self.tick_symbol_to_req_id.write().await.remove(symbol);
        info!("Removed tick data tracking for {}", symbol);
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

    pub async fn get_historical_data(&self, request: &HistoricalDataRequest) -> Result<Vec<HistoricalBar>> {
        if !self.is_connected().await {
            return Err(anyhow!("Not connected to IBKR"));
        }

        let client = self.client.as_ref()
            .ok_or_else(|| anyhow!("Client not available"))?;

        info!("Requesting historical data for {} ({} bars of {})", 
              request.symbol, request.duration, request.bar_size);

        // Create contract for the symbol
        let contract = Contract::stock(&request.symbol);

        // Convert bar size string to enum
        let bar_size = match request.bar_size.as_str() {
            "1 day" => HistBarSize::Day,
            "1 hour" => HistBarSize::Hour,
            "30 mins" => HistBarSize::Min30,
            "15 mins" => HistBarSize::Min15,
            "5 mins" => HistBarSize::Min5,
            "1 min" => HistBarSize::Min,
            _ => return Err(anyhow!("Unsupported bar size: {}", request.bar_size)),
        };

        // Convert what to show string to enum
        let what_to_show = match request.what_to_show.as_str() {
            "TRADES" => HistWhatToShow::Trades,
            "MIDPOINT" => HistWhatToShow::MidPoint,
            "BID" => HistWhatToShow::Bid,
            "ASK" => HistWhatToShow::Ask,
            _ => return Err(anyhow!("Unsupported what_to_show: {}", request.what_to_show)),
        };

        // Format end date time
        let end_date_time = request.end_date_time
            .map(|dt| dt.format("%Y%m%d %H:%M:%S").to_string())
            .unwrap_or_else(|| String::new());

        // Use the simplified IBKR API for historical data
        info!("Requesting historical data from IBKR for {}", request.symbol);
        
        // Convert request parameters to proper types
        let bar_size = match request.bar_size.as_str() {
            "1 day" => HistBarSize::Day,
            "1 hour" => HistBarSize::Hour,
            "30 mins" => HistBarSize::Min30,
            "15 mins" => HistBarSize::Min15,
            "5 mins" => HistBarSize::Min5,
            "1 min" => HistBarSize::Min,
            _ => HistBarSize::Day,
        };
        
        let what_to_show = match request.what_to_show.as_str() {
            "TRADES" => HistWhatToShow::Trades,
            "MIDPOINT" => HistWhatToShow::MidPoint,
            "BID" => HistWhatToShow::Bid,
            "ASK" => HistWhatToShow::Ask,
            _ => HistWhatToShow::Trades,
        };
        
        // Use current time as end date if not specified
        let end_time = request.end_date_time.unwrap_or_else(|| Utc::now());
        let end_offset = time::OffsetDateTime::from_unix_timestamp(end_time.timestamp())
            .unwrap_or_else(|_| time::OffsetDateTime::now_utc());
        
        // Parse duration
        let duration_parts: Vec<&str> = request.duration.trim().split_whitespace().collect();
        let duration = if duration_parts.len() == 2 {
            let value: i32 = duration_parts[0].parse().unwrap_or(30);
            match duration_parts[1].to_uppercase().as_str() {
                "D" | "DAY" | "DAYS" => ibapi::market_data::historical::Duration::days(value),
                "W" | "WEEK" | "WEEKS" => ibapi::market_data::historical::Duration::weeks(value),
                "M" | "MONTH" | "MONTHS" => ibapi::market_data::historical::Duration::months(value),
                "Y" | "YEAR" | "YEARS" => ibapi::market_data::historical::Duration::years(value),
                _ => ibapi::market_data::historical::Duration::days(30),
            }
        } else {
            ibapi::market_data::historical::Duration::days(30)
        };
        
        // Use the correct IBKR API method for historical data
        match client.historical_data(&contract, end_offset, duration, bar_size, what_to_show, request.use_rth) {
            Ok(historical_data) => {
                let mut historical_bars = Vec::new();
                
                for bar in historical_data.bars {
                    let timestamp = DateTime::from_timestamp(bar.date.unix_timestamp(), 0)
                        .unwrap_or_else(|| Utc::now());
                    
                    let historical_bar = HistoricalBar {
                        symbol: request.symbol.clone(),
                        timestamp,
                        open: Decimal::from_f64_retain(bar.open).unwrap_or_default(),
                        high: Decimal::from_f64_retain(bar.high).unwrap_or_default(),
                        low: Decimal::from_f64_retain(bar.low).unwrap_or_default(),
                        close: Decimal::from_f64_retain(bar.close).unwrap_or_default(),
                        volume: bar.volume as u64,
                        count: Some(bar.count),
                        wap: Decimal::from_f64_retain(bar.wap),
                    };
                    historical_bars.push(historical_bar);
                }

                info!("Retrieved {} historical bars for {}", historical_bars.len(), request.symbol);
                Ok(historical_bars)
            }
            Err(e) => {
                error!("Failed to get historical data for {}: {}", request.symbol, e);
                Err(anyhow!("Historical data request failed: {}", e))
            }
        }
    }

    // Mock duration parsing - not used with mock data
    fn _parse_duration_string(&self, _duration_str: &str) -> Result<String> {
        Ok("30 D".to_string())
    }

    fn parse_bar_date(&self, time_str: &str) -> Result<DateTime<Utc>> {
        // IBKR returns timestamps in various formats, handle the most common ones
        if let Ok(timestamp) = time_str.parse::<i64>() {
            // Unix timestamp
            if let Some(dt) = DateTime::from_timestamp(timestamp, 0) {
                return Ok(dt);
            }
        }

        // Try parsing as date string (YYYYMMDD format)
        if time_str.len() == 8 {
            if let Ok(naive_date) = NaiveDateTime::parse_from_str(&format!("{} 00:00:00", time_str), "%Y%m%d %H:%M:%S") {
                return Ok(DateTime::from_naive_utc_and_offset(naive_date, Utc));
            }
        }

        // Try parsing as datetime string (YYYYMMDD  HH:MM:SS format)
        if let Ok(naive_dt) = NaiveDateTime::parse_from_str(time_str, "%Y%m%d  %H:%M:%S") {
            return Ok(DateTime::from_naive_utc_and_offset(naive_dt, Utc));
        }

        Err(anyhow!("Could not parse bar time: {}", time_str))
    }

    pub async fn get_multiple_historical_data(&self, requests: &[HistoricalDataRequest]) -> Result<HashMap<String, Vec<HistoricalBar>>> {
        let mut results = HashMap::new();
        
        for request in requests {
            match self.get_historical_data(request).await {
                Ok(bars) => {
                    results.insert(request.symbol.clone(), bars);
                }
                Err(e) => {
                    warn!("Failed to get historical data for {}: {}", request.symbol, e);
                    // Continue with other symbols rather than failing completely
                }
            }
            
            // Add a small delay between requests to avoid rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        
        Ok(results)
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
    use std::env;

    fn create_test_config() -> IBKRConfig {
        // Set required environment variable for tests
        env::set_var("IBKR_ACCOUNT_ID", "DU123456");
        
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
        
        // Clean up
        env::remove_var("IBKR_ACCOUNT_ID");
    }

    #[tokio::test]
    async fn test_client_creation() {
        let config = create_test_config();
        let client = IBKRClient::new(&config);
        assert!(client.is_ok());
        
        // Clean up
        env::remove_var("IBKR_ACCOUNT_ID");
    }

    #[tokio::test]
    async fn test_market_data_subscription() {
        let config = create_test_config();
        let mut client = IBKRClient::new(&config).unwrap();
        
        // Test that subscription fails when not connected
        let result = client.subscribe_market_data("AAPL").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Not connected"));
        
        // Clean up
        env::remove_var("IBKR_ACCOUNT_ID");
    }

    #[tokio::test]
    async fn test_market_data_handler() {
        let config = create_test_config();
        let client = IBKRClient::new(&config).unwrap();
        
        // Test market data handler setup
        let result = client.start_market_data_handler(|_data| {
            // Mock callback
        }).await;
        
        assert!(result.is_ok());
        
        // Clean up
        env::remove_var("IBKR_ACCOUNT_ID");
    }
}