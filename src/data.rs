use anyhow::Result;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use tokio::sync::RwLock;

use crate::ibkr::MarketData;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesData {
    pub symbol: String,
    pub data: VecDeque<OHLCV>,
    pub max_size: usize,
}

#[derive(Debug)]
pub struct MarketDataManager {
    time_series: RwLock<HashMap<String, TimeSeriesData>>,
    latest_quotes: RwLock<HashMap<String, MarketData>>,
    tick_data: RwLock<HashMap<String, VecDeque<TickData>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub price: Decimal,
    pub size: u32,
    pub tick_type: TickType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TickType {
    Trade,
    Bid,
    Ask,
}

impl MarketDataManager {
    pub fn new() -> Self {
        Self {
            time_series: RwLock::new(HashMap::new()),
            latest_quotes: RwLock::new(HashMap::new()),
            tick_data: RwLock::new(HashMap::new()),
        }
    }

    pub async fn add_market_data(&self, market_data: MarketData) -> Result<()> {
        // Update latest quotes
        {
            let mut quotes = self.latest_quotes.write().await;
            quotes.insert(market_data.symbol.clone(), market_data.clone());
        }

        // Convert to tick data if we have a last price
        if let Some(last_price) = market_data.last {
            let tick = TickData {
                symbol: market_data.symbol.clone(),
                timestamp: market_data.timestamp,
                price: last_price,
                size: market_data.volume.unwrap_or(0) as u32,
                tick_type: TickType::Trade,
            };

            self.add_tick_data(tick).await?;
        }

        // Add bid/ask ticks if available
        if let Some(bid_price) = market_data.bid {
            let bid_tick = TickData {
                symbol: market_data.symbol.clone(),
                timestamp: market_data.timestamp,
                price: bid_price,
                size: 0, // Size not typically available for bid/ask
                tick_type: TickType::Bid,
            };
            self.add_tick_data(bid_tick).await?;
        }

        if let Some(ask_price) = market_data.ask {
            let ask_tick = TickData {
                symbol: market_data.symbol.clone(),
                timestamp: market_data.timestamp,
                price: ask_price,
                size: 0,
                tick_type: TickType::Ask,
            };
            self.add_tick_data(ask_tick).await?;
        }

        // Generate OHLCV bar from latest data if we have a price
        if let Some(price) = market_data.last.or(market_data.bid).or(market_data.ask) {
            let ohlcv = OHLCV {
                timestamp: market_data.timestamp,
                open: price,
                high: price,
                low: price,
                close: price,
                volume: market_data.volume.unwrap_or(0),
            };
            
            self.add_ohlcv_data(&market_data.symbol, ohlcv).await?;
        }

        Ok(())
    }

    pub async fn add_tick_data(&self, tick: TickData) -> Result<()> {
        let mut tick_data = self.tick_data.write().await;
        
        let symbol_ticks = tick_data
            .entry(tick.symbol.clone())
            .or_insert_with(|| VecDeque::new());

        symbol_ticks.push_back(tick);

        // Keep only recent ticks (last 10,000)
        while symbol_ticks.len() > 10_000 {
            symbol_ticks.pop_front();
        }

        Ok(())
    }

    pub async fn add_ohlcv_data(&self, symbol: &str, ohlcv: OHLCV) -> Result<()> {
        let mut time_series = self.time_series.write().await;
        
        let series = time_series
            .entry(symbol.to_string())
            .or_insert_with(|| TimeSeriesData {
                symbol: symbol.to_string(),
                data: VecDeque::new(),
                max_size: 1000, // Keep last 1000 bars
            });

        series.data.push_back(ohlcv);

        // Remove old data if we exceed max size
        while series.data.len() > series.max_size {
            series.data.pop_front();
        }

        Ok(())
    }

    pub async fn get_latest_price(&self, symbol: &str) -> Option<Decimal> {
        let quotes = self.latest_quotes.read().await;
        quotes.get(symbol)
            .and_then(|quote| quote.last.or(quote.bid).or(quote.ask))
    }



    pub async fn get_latest_data(&self) -> Result<HashMap<String, PriceData>> {
        let time_series = self.time_series.read().await;
        let mut result = HashMap::new();

        for (symbol, series) in time_series.iter() {
            if let Some(latest_bar) = series.data.back() {
                let price_data = PriceData {
                    symbol: symbol.clone(),
                    timestamp: latest_bar.timestamp,
                    open: latest_bar.open,
                    high: latest_bar.high,
                    low: latest_bar.low,
                    close: latest_bar.close,
                    volume: latest_bar.volume,
                };
                result.insert(symbol.clone(), price_data);
            }
        }

        Ok(result)
    }

    pub async fn get_tick_data(&self, symbol: &str, limit: Option<usize>) -> Vec<TickData> {
        let tick_data = self.tick_data.read().await;
        
        if let Some(ticks) = tick_data.get(symbol) {
            let limit = limit.unwrap_or(ticks.len());
            ticks
                .iter()
                .rev()
                .take(limit)
                .rev()
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    pub async fn get_statistics(&self, symbol: &str) -> Option<MarketStatistics> {
        let time_series = self.time_series.read().await;
        
        if let Some(series) = time_series.get(symbol) {
            if series.data.is_empty() {
                return None;
            }

            let prices: Vec<f64> = series.data
                .iter()
                .map(|bar| bar.close.to_f64().unwrap_or(0.0))
                .collect();

            let sum: f64 = prices.iter().sum();
            let count = prices.len() as f64;
            let mean = sum / count;

            let variance = prices.iter()
                .map(|&price| (price - mean).powi(2))
                .sum::<f64>() / count;
            let std_dev = variance.sqrt();

            let min_price = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_price = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            Some(MarketStatistics {
                symbol: symbol.to_string(),
                count: count as u64,
                mean: Decimal::from_f64_retain(mean).unwrap_or_default(),
                std_dev: Decimal::from_f64_retain(std_dev).unwrap_or_default(),
                min_price: Decimal::from_f64_retain(min_price).unwrap_or_default(),
                max_price: Decimal::from_f64_retain(max_price).unwrap_or_default(),
                last_update: series.data.back().unwrap().timestamp,
            })
        } else {
            None
        }
    }
}



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStatistics {
    pub symbol: String,
    pub count: u64,
    pub mean: Decimal,
    pub std_dev: Decimal,
    pub min_price: Decimal,
    pub max_price: Decimal,
    pub last_update: DateTime<Utc>,
}

impl Default for MarketDataManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_market_data_manager() {
        let manager = MarketDataManager::new();
        
        let market_data = MarketData {
            symbol: "AAPL".to_string(),
            bid: Some(dec!(150.00)),
            ask: Some(dec!(150.05)),
            last: Some(dec!(150.02)),
            volume: Some(1000),
            timestamp: Utc::now(),
        };

        manager.add_market_data(market_data.clone()).await.unwrap();
        
        let latest_price = manager.get_latest_price("AAPL").await;
        assert_eq!(latest_price, Some(dec!(150.02)));
        

    }

    #[tokio::test]
    async fn test_ohlcv_data() {
        let manager = MarketDataManager::new();
        
        let ohlcv = OHLCV {
            timestamp: Utc::now(),
            open: dec!(100.00),
            high: dec!(105.00),
            low: dec!(99.00),
            close: dec!(104.00),
            volume: 5000,
        };

        manager.add_ohlcv_data("TSLA", ohlcv.clone()).await.unwrap();
        

    }

    #[tokio::test]
    async fn test_tick_data() {
        let manager = MarketDataManager::new();
        
        let tick = TickData {
            symbol: "GOOGL".to_string(),
            timestamp: Utc::now(),
            price: dec!(2500.00),
            size: 100,
            tick_type: TickType::Trade,
        };

        manager.add_tick_data(tick.clone()).await.unwrap();
        
        let ticks = manager.get_tick_data("GOOGL", Some(1)).await;
        assert_eq!(ticks.len(), 1);
        assert_eq!(ticks[0].price, dec!(2500.00));
    }

    #[tokio::test]
    async fn test_statistics() {
        let manager = MarketDataManager::new();
        
        // Add some OHLCV data
        for i in 0..10 {
            let ohlcv = OHLCV {
                timestamp: Utc::now(),
                open: Decimal::from(100 + i),
                high: Decimal::from(105 + i),
                low: Decimal::from(95 + i),
                close: Decimal::from(102 + i),
                volume: 1000,
            };
            manager.add_ohlcv_data("TEST", ohlcv).await.unwrap();
        }
        
        let stats = manager.get_statistics("TEST").await;
        assert!(stats.is_some());
        
        let stats = stats.unwrap();
        assert_eq!(stats.symbol, "TEST");
        assert_eq!(stats.count, 10);
        assert!(stats.std_dev > Decimal::ZERO);
    }
}