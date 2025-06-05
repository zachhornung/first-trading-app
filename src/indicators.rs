use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorValue {
    pub value: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BollingerBands {
    pub upper: f64,
    pub middle: f64,
    pub lower: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MACD {
    pub macd_line: f64,
    pub signal_line: f64,
    pub histogram: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stochastic {
    pub k_percent: f64,
    pub d_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorSignal {
    pub indicator_name: String,
    pub signal_type: SignalType,
    pub strength: f64, // 0.0 to 1.0
    pub value: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SignalType {
    Buy,
    Sell,
    Neutral,
    Overbought,
    Oversold,
    Bullish,
    Bearish,
}

// RSI (Relative Strength Index)
#[derive(Debug, Clone)]
pub struct RSI {
    period: usize,
    gains: VecDeque<f64>,
    losses: VecDeque<f64>,
    avg_gain: f64,
    avg_loss: f64,
    last_price: Option<f64>,
    initialized: bool,
}

impl RSI {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            gains: VecDeque::new(),
            losses: VecDeque::new(),
            avg_gain: 0.0,
            avg_loss: 0.0,
            last_price: None,
            initialized: false,
        }
    }

    pub fn update(&mut self, price: f64) -> Option<f64> {
        if let Some(last_price) = self.last_price {
            let change = price - last_price;
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { -change } else { 0.0 };

            self.gains.push_back(gain);
            self.losses.push_back(loss);

            if self.gains.len() > self.period {
                self.gains.pop_front();
                self.losses.pop_front();
            }

            if self.gains.len() == self.period {
                if !self.initialized {
                    self.avg_gain = self.gains.iter().sum::<f64>() / self.period as f64;
                    self.avg_loss = self.losses.iter().sum::<f64>() / self.period as f64;
                    self.initialized = true;
                } else {
                    // Use Wilder's smoothing
                    self.avg_gain = (self.avg_gain * (self.period - 1) as f64 + gain) / self.period as f64;
                    self.avg_loss = (self.avg_loss * (self.period - 1) as f64 + loss) / self.period as f64;
                }

                if self.avg_loss != 0.0 {
                    let rs = self.avg_gain / self.avg_loss;
                    let rsi = 100.0 - (100.0 / (1.0 + rs));
                    self.last_price = Some(price);
                    return Some(rsi);
                }
            }
        }

        self.last_price = Some(price);
        None
    }

    pub fn generate_signal(&self, rsi_value: f64) -> IndicatorSignal {
        let (signal_type, strength) = if rsi_value >= 70.0 {
            (SignalType::Overbought, (rsi_value - 70.0) / 30.0)
        } else if rsi_value <= 30.0 {
            (SignalType::Oversold, (30.0 - rsi_value) / 30.0)
        } else {
            (SignalType::Neutral, 0.0)
        };

        IndicatorSignal {
            indicator_name: "RSI".to_string(),
            signal_type,
            strength: strength.min(1.0).max(0.0),
            value: rsi_value,
            timestamp: chrono::Utc::now(),
        }
    }
}

// MACD (Moving Average Convergence Divergence)
#[derive(Debug, Clone)]
pub struct MACDIndicator {
    _fast_period: usize,
    _slow_period: usize,
    _signal_period: usize,
    fast_ema: EMA,
    slow_ema: EMA,
    signal_ema: EMA,
    macd_values: VecDeque<f64>,
}

impl MACDIndicator {
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            _fast_period: fast_period,
            _slow_period: slow_period,
            _signal_period: signal_period,
            fast_ema: EMA::new(fast_period),
            slow_ema: EMA::new(slow_period),
            signal_ema: EMA::new(signal_period),
            macd_values: VecDeque::new(),
        }
    }

    pub fn update(&mut self, price: f64) -> Option<MACD> {
        if let (Some(fast_ema), Some(slow_ema)) = (
            self.fast_ema.update(price),
            self.slow_ema.update(price),
        ) {
            let macd_line = fast_ema - slow_ema;
            
            if let Some(signal_line) = self.signal_ema.update(macd_line) {
                let histogram = macd_line - signal_line;
                
                self.macd_values.push_back(macd_line);
                if self.macd_values.len() > 100 {
                    self.macd_values.pop_front();
                }

                return Some(MACD {
                    macd_line,
                    signal_line,
                    histogram,
                });
            }
        }
        None
    }

    pub fn generate_signal(&self, macd: &MACD) -> IndicatorSignal {
        let (signal_type, strength) = if macd.macd_line > macd.signal_line && macd.histogram > 0.0 {
            (SignalType::Bullish, macd.histogram.abs().min(1.0))
        } else if macd.macd_line < macd.signal_line && macd.histogram < 0.0 {
            (SignalType::Bearish, macd.histogram.abs().min(1.0))
        } else {
            (SignalType::Neutral, 0.0)
        };

        IndicatorSignal {
            indicator_name: "MACD".to_string(),
            signal_type,
            strength,
            value: macd.macd_line,
            timestamp: chrono::Utc::now(),
        }
    }
}

// Bollinger Bands
#[derive(Debug, Clone)]
pub struct BollingerBandsIndicator {
    period: usize,
    std_dev_multiplier: f64,
    prices: VecDeque<f64>,
    sma: SMA,
}

impl BollingerBandsIndicator {
    pub fn new(period: usize, std_dev_multiplier: f64) -> Self {
        Self {
            period,
            std_dev_multiplier,
            prices: VecDeque::new(),
            sma: SMA::new(period),
        }
    }

    pub fn update(&mut self, price: f64) -> Option<BollingerBands> {
        self.prices.push_back(price);
        if self.prices.len() > self.period {
            self.prices.pop_front();
        }

        if let Some(middle) = self.sma.update(price) {
            if self.prices.len() == self.period {
                let variance = self.prices.iter()
                    .map(|&p| (p - middle).powi(2))
                    .sum::<f64>() / self.period as f64;
                let std_dev = variance.sqrt();
                
                let upper = middle + (std_dev * self.std_dev_multiplier);
                let lower = middle - (std_dev * self.std_dev_multiplier);

                return Some(BollingerBands {
                    upper,
                    middle,
                    lower,
                });
            }
        }
        None
    }

    pub fn generate_signal(&self, bands: &BollingerBands, current_price: f64) -> IndicatorSignal {
        let (signal_type, strength) = if current_price > bands.upper {
            let distance = (current_price - bands.upper) / (bands.upper - bands.middle);
            (SignalType::Overbought, distance.min(1.0))
        } else if current_price < bands.lower {
            let distance = (bands.lower - current_price) / (bands.middle - bands.lower);
            (SignalType::Oversold, distance.min(1.0))
        } else {
            (SignalType::Neutral, 0.0)
        };

        IndicatorSignal {
            indicator_name: "BollingerBands".to_string(),
            signal_type,
            strength,
            value: current_price,
            timestamp: chrono::Utc::now(),
        }
    }
}

// Stochastic Oscillator
#[derive(Debug, Clone)]
pub struct StochasticIndicator {
    k_period: usize,
    _d_period: usize,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    closes: VecDeque<f64>,
    _k_values: VecDeque<f64>,
    d_sma: SMA,
}

impl StochasticIndicator {
    pub fn new(k_period: usize, d_period: usize) -> Self {
        Self {
            k_period,
            _d_period: d_period,
            highs: VecDeque::new(),
            lows: VecDeque::new(),
            closes: VecDeque::new(),
            _k_values: VecDeque::new(),
            d_sma: SMA::new(d_period),
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<Stochastic> {
        self.highs.push_back(high);
        self.lows.push_back(low);
        self.closes.push_back(close);

        if self.highs.len() > self.k_period {
            self.highs.pop_front();
            self.lows.pop_front();
            self.closes.pop_front();
        }

        if self.highs.len() == self.k_period {
            let highest_high = self.highs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lowest_low = self.lows.iter().cloned().fold(f64::INFINITY, f64::min);
            
            let k_percent = if highest_high != lowest_low {
                ((close - lowest_low) / (highest_high - lowest_low)) * 100.0
            } else {
                50.0
            };

            if let Some(d_percent) = self.d_sma.update(k_percent) {
                return Some(Stochastic {
                    k_percent,
                    d_percent,
                });
            }
        }
        None
    }

    pub fn generate_signal(&self, stoch: &Stochastic) -> IndicatorSignal {
        let (signal_type, strength) = if stoch.k_percent >= 80.0 {
            (SignalType::Overbought, (stoch.k_percent - 80.0) / 20.0)
        } else if stoch.k_percent <= 20.0 {
            (SignalType::Oversold, (20.0 - stoch.k_percent) / 20.0)
        } else {
            (SignalType::Neutral, 0.0)
        };

        IndicatorSignal {
            indicator_name: "Stochastic".to_string(),
            signal_type,
            strength: strength.min(1.0).max(0.0),
            value: stoch.k_percent,
            timestamp: chrono::Utc::now(),
        }
    }
}

// Williams %R
#[derive(Debug, Clone)]
pub struct WilliamsR {
    period: usize,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
}

impl WilliamsR {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            highs: VecDeque::new(),
            lows: VecDeque::new(),
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.highs.push_back(high);
        self.lows.push_back(low);

        if self.highs.len() > self.period {
            self.highs.pop_front();
            self.lows.pop_front();
        }

        if self.highs.len() == self.period {
            let highest_high = self.highs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lowest_low = self.lows.iter().cloned().fold(f64::INFINITY, f64::min);
            
            if highest_high != lowest_low {
                let williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100.0;
                return Some(williams_r);
            }
        }
        None
    }

    pub fn generate_signal(&self, williams_r: f64) -> IndicatorSignal {
        let (signal_type, strength) = if williams_r >= -20.0 {
            (SignalType::Overbought, (williams_r + 20.0) / 20.0)
        } else if williams_r <= -80.0 {
            (SignalType::Oversold, (-80.0 - williams_r) / 20.0)
        } else {
            (SignalType::Neutral, 0.0)
        };

        IndicatorSignal {
            indicator_name: "WilliamsR".to_string(),
            signal_type,
            strength: strength.min(1.0).max(0.0),
            value: williams_r,
            timestamp: chrono::Utc::now(),
        }
    }
}

// Commodity Channel Index (CCI)
#[derive(Debug, Clone)]
pub struct CCI {
    period: usize,
    typical_prices: VecDeque<f64>,
    sma: SMA,
}

impl CCI {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            typical_prices: VecDeque::new(),
            sma: SMA::new(period),
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let typical_price = (high + low + close) / 3.0;
        self.typical_prices.push_back(typical_price);

        if self.typical_prices.len() > self.period {
            self.typical_prices.pop_front();
        }

        if let Some(sma_tp) = self.sma.update(typical_price) {
            if self.typical_prices.len() == self.period {
                let mean_deviation = self.typical_prices.iter()
                    .map(|&tp| (tp - sma_tp).abs())
                    .sum::<f64>() / self.period as f64;

                if mean_deviation != 0.0 {
                    let cci = (typical_price - sma_tp) / (0.015 * mean_deviation);
                    return Some(cci);
                }
            }
        }
        None
    }

    pub fn generate_signal(&self, cci: f64) -> IndicatorSignal {
        let (signal_type, strength) = if cci >= 100.0 {
            (SignalType::Overbought, (cci - 100.0) / 100.0)
        } else if cci <= -100.0 {
            (SignalType::Oversold, (-100.0 - cci) / 100.0)
        } else {
            (SignalType::Neutral, 0.0)
        };

        IndicatorSignal {
            indicator_name: "CCI".to_string(),
            signal_type,
            strength: strength.min(1.0).max(0.0),
            value: cci,
            timestamp: chrono::Utc::now(),
        }
    }
}

// Average True Range (ATR)
#[derive(Debug, Clone)]
pub struct ATR {
    _period: usize,
    _true_ranges: VecDeque<f64>,
    atr_ema: EMA,
    previous_close: Option<f64>,
}

impl ATR {
    pub fn new(period: usize) -> Self {
        Self {
            _period: period,
            _true_ranges: VecDeque::new(),
            atr_ema: EMA::new(period),
            previous_close: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let true_range = if let Some(prev_close) = self.previous_close {
            let tr1 = high - low;
            let tr2 = (high - prev_close).abs();
            let tr3 = (low - prev_close).abs();
            tr1.max(tr2).max(tr3)
        } else {
            high - low
        };

        self.previous_close = Some(close);
        self.atr_ema.update(true_range)
    }
}

// Simple Moving Average
#[derive(Debug, Clone)]
pub struct SMA {
    period: usize,
    values: VecDeque<f64>,
}

impl SMA {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            values: VecDeque::new(),
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.values.push_back(value);
        if self.values.len() > self.period {
            self.values.pop_front();
        }

        if self.values.len() == self.period {
            Some(self.values.iter().sum::<f64>() / self.period as f64)
        } else {
            None
        }
    }
}

// Exponential Moving Average
#[derive(Debug, Clone)]
pub struct EMA {
    _period: usize,
    multiplier: f64,
    ema: Option<f64>,
}

impl EMA {
    pub fn new(period: usize) -> Self {
        Self {
            _period: period,
            multiplier: 2.0 / (period as f64 + 1.0),
            ema: None,
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        match self.ema {
            Some(prev_ema) => {
                let new_ema = (value * self.multiplier) + (prev_ema * (1.0 - self.multiplier));
                self.ema = Some(new_ema);
                Some(new_ema)
            }
            None => {
                self.ema = Some(value);
                Some(value)
            }
        }
    }
}

// Volume-based indicators
#[derive(Debug, Clone)]
pub struct VolumeIndicators {
    period: usize,
    volumes: VecDeque<f64>,
    volume_sma: SMA,
}

impl VolumeIndicators {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            volumes: VecDeque::new(),
            volume_sma: SMA::new(period),
        }
    }

    pub fn update(&mut self, volume: f64) -> Option<f64> {
        self.volumes.push_back(volume);
        if self.volumes.len() > self.period {
            self.volumes.pop_front();
        }
        
        self.volume_sma.update(volume)
    }

    pub fn volume_ratio(&self, current_volume: f64) -> Option<f64> {
        if self.volumes.len() == self.period {
            let avg_volume = self.volumes.iter().sum::<f64>() / self.period as f64;
            if avg_volume > 0.0 {
                Some(current_volume / avg_volume)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn generate_signal(&self, volume_ratio: f64) -> IndicatorSignal {
        let (signal_type, strength) = if volume_ratio >= 2.0 {
            (SignalType::Bullish, (volume_ratio - 2.0).min(1.0))
        } else if volume_ratio <= 0.5 {
            (SignalType::Bearish, (0.5 - volume_ratio).min(1.0))
        } else {
            (SignalType::Neutral, 0.0)
        };

        IndicatorSignal {
            indicator_name: "Volume".to_string(),
            signal_type,
            strength,
            value: volume_ratio,
            timestamp: chrono::Utc::now(),
        }
    }
}

// Support and Resistance Levels
#[derive(Debug, Clone)]
pub struct SupportResistance {
    lookback_period: usize,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    resistance_levels: Vec<f64>,
    support_levels: Vec<f64>,
}

impl SupportResistance {
    pub fn new(lookback_period: usize) -> Self {
        Self {
            lookback_period,
            highs: VecDeque::new(),
            lows: VecDeque::new(),
            resistance_levels: Vec::new(),
            support_levels: Vec::new(),
        }
    }

    pub fn update(&mut self, high: f64, low: f64, _close: f64) {
        self.highs.push_back(high);
        self.lows.push_back(low);

        if self.highs.len() > self.lookback_period {
            self.highs.pop_front();
            self.lows.pop_front();
        }

        if self.highs.len() == self.lookback_period {
            self.calculate_levels();
        }
    }

    fn calculate_levels(&mut self) {
        // Find local maxima for resistance
        self.resistance_levels.clear();
        self.support_levels.clear();

        let highs: Vec<f64> = self.highs.iter().cloned().collect();
        let lows: Vec<f64> = self.lows.iter().cloned().collect();

        // Simple peak/trough detection
        for i in 1..highs.len() - 1 {
            if highs[i] > highs[i - 1] && highs[i] > highs[i + 1] {
                self.resistance_levels.push(highs[i]);
            }
            if lows[i] < lows[i - 1] && lows[i] < lows[i + 1] {
                self.support_levels.push(lows[i]);
            }
        }

        // Keep only the most significant levels
        self.resistance_levels.sort_by(|a, b| b.partial_cmp(a).unwrap());
        self.support_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        self.resistance_levels.truncate(3);
        self.support_levels.truncate(3);
    }

    pub fn get_nearest_resistance(&self, price: f64) -> Option<f64> {
        self.resistance_levels.iter()
            .filter(|&&level| level > price)
            .min_by(|a, b| (*a - price).abs().partial_cmp(&(*b - price).abs()).unwrap())
            .cloned()
    }

    pub fn get_nearest_support(&self, price: f64) -> Option<f64> {
        self.support_levels.iter()
            .filter(|&&level| level < price)
            .max_by(|a, b| (*a - price).abs().partial_cmp(&(*b - price).abs()).unwrap())
            .cloned()
    }

    pub fn generate_signal(&self, current_price: f64) -> IndicatorSignal {
        let near_resistance = self.get_nearest_resistance(current_price)
            .map(|level| (current_price - level).abs() / level < 0.01)
            .unwrap_or(false);

        let near_support = self.get_nearest_support(current_price)
            .map(|level| (current_price - level).abs() / level < 0.01)
            .unwrap_or(false);

        let (signal_type, strength) = if near_resistance {
            (SignalType::Bearish, 0.7)
        } else if near_support {
            (SignalType::Bullish, 0.7)
        } else {
            (SignalType::Neutral, 0.0)
        };

        IndicatorSignal {
            indicator_name: "SupportResistance".to_string(),
            signal_type,
            strength,
            value: current_price,
            timestamp: chrono::Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi_calculation() {
        let mut rsi = RSI::new(14);
        
        // Test with known values
        let prices = vec![44.0, 44.25, 44.5, 43.75, 44.0, 44.25, 45.0, 47.0, 46.75, 46.5, 46.25, 47.75, 47.5, 47.25, 48.0];
        
        let mut last_rsi = None;
        for price in prices {
            if let Some(rsi_value) = rsi.update(price) {
                last_rsi = Some(rsi_value);
            }
        }
        
        assert!(last_rsi.is_some());
        assert!(last_rsi.unwrap() > 0.0 && last_rsi.unwrap() <= 100.0);
    }

    #[test]
    fn test_sma_calculation() {
        let mut sma = SMA::new(3);
        
        assert_eq!(sma.update(1.0), None);
        assert_eq!(sma.update(2.0), None);
        assert_eq!(sma.update(3.0), Some(2.0));
        assert_eq!(sma.update(4.0), Some(3.0));
    }

    #[test]
    fn test_ema_calculation() {
        let mut ema = EMA::new(3);
        
        assert_eq!(ema.update(1.0), Some(1.0));
        let second = ema.update(2.0).unwrap();
        assert!(second > 1.0 && second < 2.0);
    }

    #[test]
    fn test_bollinger_bands() {
        let mut bb = BollingerBandsIndicator::new(5, 2.0);
        
        // Add some price data
        for price in vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0] {
            bb.update(price);
        }
        
        if let Some(bands) = bb.update(26.0) {
            assert!(bands.upper > bands.middle);
            assert!(bands.middle > bands.lower);
        }
    }

    #[test]
    fn test_macd_calculation() {
        let mut macd = MACDIndicator::new(12, 26, 9);
        
        // Add enough data points
        for i in 1..=50 {
            macd.update(100.0 + i as f64 * 0.5);
        }
        
        if let Some(macd_data) = macd.update(125.0) {
            // Just verify we get valid data
            assert!(macd_data.macd_line.is_finite());
            assert!(macd_data.signal_line.is_finite());
            assert!(macd_data.histogram.is_finite());
        }
    }
}