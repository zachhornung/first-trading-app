use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
    TrailingStop,
    Bracket,
    OCO, // One Cancels Other
    MOC, // Market on Close
    LOC, // Limit on Close
    TWAP, // Time Weighted Average Price
    VWAP, // Volume Weighted Average Price
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TimeInForce {
    Day,
    GTC, // Good Till Canceled
    IOC, // Immediate or Cancel
    FOK, // Fill or Kill
    GTD(DateTime<Utc>), // Good Till Date
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderStatus {
    Pending,
    PartiallyFilled,
    Filled,
    Canceled,
    Rejected,
    Expired,
    Suspended,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedOrder {
    pub id: String,
    pub symbol: String,
    pub order_type: OrderType,
    pub side: OrderSide,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub stop_price: Option<Decimal>,
    pub time_in_force: TimeInForce,
    pub status: OrderStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub filled_quantity: Decimal,
    pub average_fill_price: Option<Decimal>,
    pub commission: Option<Decimal>,
    pub parent_order_id: Option<String>,
    pub child_orders: Vec<String>,
    pub conditions: Vec<OrderCondition>,
    pub execution_instructions: Vec<ExecutionInstruction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BracketOrder {
    pub parent_order: AdvancedOrder,
    pub take_profit_order: Option<AdvancedOrder>,
    pub stop_loss_order: Option<AdvancedOrder>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCOOrder {
    pub primary_order: AdvancedOrder,
    pub secondary_order: AdvancedOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailingStopOrder {
    pub base_order: AdvancedOrder,
    pub trail_amount: Decimal,
    pub trail_percent: Option<Decimal>,
    pub trail_type: TrailType,
    pub high_water_mark: Option<Decimal>,
    pub low_water_mark: Option<Decimal>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrailType {
    Amount,
    Percent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderCondition {
    pub condition_type: ConditionType,
    pub symbol: Option<String>,
    pub operator: ComparisonOperator,
    pub value: Decimal,
    pub logic_operator: Option<LogicOperator>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConditionType {
    Price,
    Volume,
    Time,
    RSI,
    MACD,
    BollingerBand,
    SMA,
    EMA,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LogicOperator {
    And,
    Or,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionInstruction {
    pub instruction_type: InstructionType,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InstructionType {
    HiddenOrder,
    IcebergOrder,
    PostOnly,
    ReduceOnly,
    CloseOnTrigger,
    PegToMidpoint,
    MinimumQuantity,
    DisplayQuantity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub id: String,
    pub order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: Decimal,
    pub price: Decimal,
    pub commission: Decimal,
    pub timestamp: DateTime<Utc>,
    pub execution_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookLevel {
    pub price: Decimal,
    pub quantity: Decimal,
    pub order_count: u32,
}

pub struct OrderManager {
    orders: HashMap<String, AdvancedOrder>,
    bracket_orders: HashMap<String, BracketOrder>,
    oco_orders: HashMap<String, OCOOrder>,
    trailing_stops: HashMap<String, TrailingStopOrder>,
    fills: Vec<Fill>,
    order_books: HashMap<String, OrderBook>,
}

impl OrderManager {
    pub fn new() -> Self {
        Self {
            orders: HashMap::new(),
            bracket_orders: HashMap::new(),
            oco_orders: HashMap::new(),
            trailing_stops: HashMap::new(),
            fills: Vec::new(),
            order_books: HashMap::new(),
        }
    }

    pub fn create_market_order(
        &mut self,
        symbol: String,
        side: OrderSide,
        quantity: Decimal,
        time_in_force: TimeInForce,
    ) -> AdvancedOrder {
        let order = AdvancedOrder {
            id: Uuid::new_v4().to_string(),
            symbol,
            order_type: OrderType::Market,
            side,
            quantity,
            price: None,
            stop_price: None,
            time_in_force,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            filled_quantity: Decimal::ZERO,
            average_fill_price: None,
            commission: None,
            parent_order_id: None,
            child_orders: Vec::new(),
            conditions: Vec::new(),
            execution_instructions: Vec::new(),
        };

        self.orders.insert(order.id.clone(), order.clone());
        order
    }

    pub fn create_limit_order(
        &mut self,
        symbol: String,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        time_in_force: TimeInForce,
    ) -> AdvancedOrder {
        let order = AdvancedOrder {
            id: Uuid::new_v4().to_string(),
            symbol,
            order_type: OrderType::Limit,
            side,
            quantity,
            price: Some(price),
            stop_price: None,
            time_in_force,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            filled_quantity: Decimal::ZERO,
            average_fill_price: None,
            commission: None,
            parent_order_id: None,
            child_orders: Vec::new(),
            conditions: Vec::new(),
            execution_instructions: Vec::new(),
        };

        self.orders.insert(order.id.clone(), order.clone());
        order
    }

    pub fn create_stop_order(
        &mut self,
        symbol: String,
        side: OrderSide,
        quantity: Decimal,
        stop_price: Decimal,
        time_in_force: TimeInForce,
    ) -> AdvancedOrder {
        let order = AdvancedOrder {
            id: Uuid::new_v4().to_string(),
            symbol,
            order_type: OrderType::Stop,
            side,
            quantity,
            price: None,
            stop_price: Some(stop_price),
            time_in_force,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            filled_quantity: Decimal::ZERO,
            average_fill_price: None,
            commission: None,
            parent_order_id: None,
            child_orders: Vec::new(),
            conditions: Vec::new(),
            execution_instructions: Vec::new(),
        };

        self.orders.insert(order.id.clone(), order.clone());
        order
    }

    pub fn create_stop_limit_order(
        &mut self,
        symbol: String,
        side: OrderSide,
        quantity: Decimal,
        stop_price: Decimal,
        limit_price: Decimal,
        time_in_force: TimeInForce,
    ) -> AdvancedOrder {
        let order = AdvancedOrder {
            id: Uuid::new_v4().to_string(),
            symbol,
            order_type: OrderType::StopLimit,
            side,
            quantity,
            price: Some(limit_price),
            stop_price: Some(stop_price),
            time_in_force,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            filled_quantity: Decimal::ZERO,
            average_fill_price: None,
            commission: None,
            parent_order_id: None,
            child_orders: Vec::new(),
            conditions: Vec::new(),
            execution_instructions: Vec::new(),
        };

        self.orders.insert(order.id.clone(), order.clone());
        order
    }

    pub fn create_bracket_order(
        &mut self,
        symbol: String,
        side: OrderSide,
        quantity: Decimal,
        entry_price: Option<Decimal>,
        take_profit_price: Option<Decimal>,
        stop_loss_price: Option<Decimal>,
    ) -> Result<BracketOrder> {
        // Create parent order
        let parent_order = if let Some(price) = entry_price {
            self.create_limit_order(symbol.clone(), side.clone(), quantity, price, TimeInForce::GTC)
        } else {
            self.create_market_order(symbol.clone(), side.clone(), quantity, TimeInForce::GTC)
        };

        let opposite_side = match side {
            OrderSide::Buy => OrderSide::Sell,
            OrderSide::Sell => OrderSide::Buy,
        };

        // Create take profit order
        let take_profit_order = if let Some(tp_price) = take_profit_price {
            Some(self.create_limit_order(
                symbol.clone(),
                opposite_side.clone(),
                quantity,
                tp_price,
                TimeInForce::GTC,
            ))
        } else {
            None
        };

        // Create stop loss order
        let stop_loss_order = if let Some(sl_price) = stop_loss_price {
            Some(self.create_stop_order(
                symbol.clone(),
                opposite_side,
                quantity,
                sl_price,
                TimeInForce::GTC,
            ))
        } else {
            None
        };

        let bracket = BracketOrder {
            parent_order,
            take_profit_order,
            stop_loss_order,
        };

        self.bracket_orders.insert(bracket.parent_order.id.clone(), bracket.clone());
        Ok(bracket)
    }

    pub fn create_oco_order(
        &mut self,
        symbol: String,
        side: OrderSide,
        quantity: Decimal,
        primary_price: Decimal,
        secondary_price: Decimal,
        primary_type: OrderType,
        secondary_type: OrderType,
    ) -> Result<OCOOrder> {
        let primary_order = match primary_type {
            OrderType::Limit => self.create_limit_order(
                symbol.clone(),
                side.clone(),
                quantity,
                primary_price,
                TimeInForce::GTC,
            ),
            OrderType::Stop => self.create_stop_order(
                symbol.clone(),
                side.clone(),
                quantity,
                primary_price,
                TimeInForce::GTC,
            ),
            _ => return Err(anyhow!("Unsupported order type for OCO primary order")),
        };

        let secondary_order = match secondary_type {
            OrderType::Limit => self.create_limit_order(
                symbol,
                side,
                quantity,
                secondary_price,
                TimeInForce::GTC,
            ),
            OrderType::Stop => self.create_stop_order(
                symbol,
                side,
                quantity,
                secondary_price,
                TimeInForce::GTC,
            ),
            _ => return Err(anyhow!("Unsupported order type for OCO secondary order")),
        };

        let oco = OCOOrder {
            primary_order,
            secondary_order,
        };

        self.oco_orders.insert(oco.primary_order.id.clone(), oco.clone());
        Ok(oco)
    }

    pub fn create_trailing_stop_order(
        &mut self,
        symbol: String,
        side: OrderSide,
        quantity: Decimal,
        trail_amount: Decimal,
        trail_type: TrailType,
    ) -> TrailingStopOrder {
        let base_order = AdvancedOrder {
            id: Uuid::new_v4().to_string(),
            symbol,
            order_type: OrderType::TrailingStop,
            side,
            quantity,
            price: None,
            stop_price: None,
            time_in_force: TimeInForce::GTC,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            filled_quantity: Decimal::ZERO,
            average_fill_price: None,
            commission: None,
            parent_order_id: None,
            child_orders: Vec::new(),
            conditions: Vec::new(),
            execution_instructions: Vec::new(),
        };

        let trailing_stop = TrailingStopOrder {
            base_order: base_order.clone(),
            trail_amount,
            trail_percent: if trail_type == TrailType::Percent {
                Some(trail_amount)
            } else {
                None
            },
            trail_type,
            high_water_mark: None,
            low_water_mark: None,
        };

        self.orders.insert(base_order.id.clone(), base_order);
        self.trailing_stops.insert(trailing_stop.base_order.id.clone(), trailing_stop.clone());
        trailing_stop
    }

    pub fn add_condition_to_order(
        &mut self,
        order_id: &str,
        condition: OrderCondition,
    ) -> Result<()> {
        if let Some(order) = self.orders.get_mut(order_id) {
            order.conditions.push(condition);
            order.updated_at = Utc::now();
            Ok(())
        } else {
            Err(anyhow!("Order not found: {}", order_id))
        }
    }

    pub fn add_execution_instruction(
        &mut self,
        order_id: &str,
        instruction: ExecutionInstruction,
    ) -> Result<()> {
        if let Some(order) = self.orders.get_mut(order_id) {
            order.execution_instructions.push(instruction);
            order.updated_at = Utc::now();
            Ok(())
        } else {
            Err(anyhow!("Order not found: {}", order_id))
        }
    }

    pub fn update_order_status(&mut self, order_id: &str, status: OrderStatus) -> Result<()> {
        if let Some(order) = self.orders.get_mut(order_id) {
            order.status = status;
            order.updated_at = Utc::now();
            Ok(())
        } else {
            Err(anyhow!("Order not found: {}", order_id))
        }
    }

    pub fn fill_order(
        &mut self,
        order_id: &str,
        fill_quantity: Decimal,
        fill_price: Decimal,
        commission: Decimal,
    ) -> Result<Fill> {
        let order = self.orders.get_mut(order_id)
            .ok_or_else(|| anyhow!("Order not found: {}", order_id))?;

        if order.status == OrderStatus::Canceled || order.status == OrderStatus::Filled {
            return Err(anyhow!("Cannot fill order with status: {:?}", order.status));
        }

        let remaining_quantity = order.quantity - order.filled_quantity;
        let actual_fill_quantity = fill_quantity.min(remaining_quantity);

        // Update order fill information
        let previous_filled = order.filled_quantity;
        order.filled_quantity += actual_fill_quantity;

        // Calculate average fill price
        if let Some(avg_price) = order.average_fill_price {
            let total_cost = (avg_price * previous_filled) + (fill_price * actual_fill_quantity);
            order.average_fill_price = Some(total_cost / order.filled_quantity);
        } else {
            order.average_fill_price = Some(fill_price);
        }

        // Update commission
        if let Some(existing_commission) = order.commission {
            order.commission = Some(existing_commission + commission);
        } else {
            order.commission = Some(commission);
        }

        // Update order status
        if order.filled_quantity >= order.quantity {
            order.status = OrderStatus::Filled;
        } else {
            order.status = OrderStatus::PartiallyFilled;
        }

        order.updated_at = Utc::now();

        // Create fill record
        let fill = Fill {
            id: Uuid::new_v4().to_string(),
            order_id: order_id.to_string(),
            symbol: order.symbol.clone(),
            side: order.side.clone(),
            quantity: actual_fill_quantity,
            price: fill_price,
            commission,
            timestamp: Utc::now(),
            execution_id: Uuid::new_v4().to_string(),
        };

        self.fills.push(fill.clone());

        // Handle bracket order logic
        if order.status == OrderStatus::Filled {
            self.handle_bracket_order_completion(order_id)?;
            self.handle_oco_order_completion(order_id)?;
        }

        Ok(fill)
    }

    fn handle_bracket_order_completion(&mut self, parent_order_id: &str) -> Result<()> {
        if let Some(bracket) = self.bracket_orders.get(parent_order_id).cloned() {
            // Activate child orders when parent is filled
            if let Some(tp_order) = bracket.take_profit_order {
                self.update_order_status(&tp_order.id, OrderStatus::Pending)?;
            }
            if let Some(sl_order) = bracket.stop_loss_order {
                self.update_order_status(&sl_order.id, OrderStatus::Pending)?;
            }
        }
        Ok(())
    }

    fn handle_oco_order_completion(&mut self, filled_order_id: &str) -> Result<()> {
        // Find OCO order containing the filled order
        let oco_to_cancel = self.oco_orders.iter()
            .find(|(_, oco)| oco.primary_order.id == filled_order_id || oco.secondary_order.id == filled_order_id)
            .map(|(key, oco)| (key.clone(), oco.clone()));

        if let Some((oco_key, oco)) = oco_to_cancel {
            // Cancel the other order
            let other_order_id = if oco.primary_order.id == filled_order_id {
                &oco.secondary_order.id
            } else {
                &oco.primary_order.id
            };

            self.update_order_status(other_order_id, OrderStatus::Canceled)?;
            self.oco_orders.remove(&oco_key);
        }

        Ok(())
    }

    pub fn update_trailing_stop(&mut self, symbol: &str, current_price: Decimal) -> Result<()> {
        let trailing_stops: Vec<String> = self.trailing_stops.iter()
            .filter(|(_, ts)| ts.base_order.symbol == symbol)
            .map(|(id, _)| id.clone())
            .collect();

        for order_id in trailing_stops {
            if let Some(mut trailing_stop) = self.trailing_stops.get(&order_id).cloned() {
                let should_trigger = match trailing_stop.base_order.side {
                    OrderSide::Buy => {
                        // For buy trailing stops, track the lowest price
                        if trailing_stop.low_water_mark.is_none() || current_price < trailing_stop.low_water_mark.unwrap() {
                            trailing_stop.low_water_mark = Some(current_price);
                        }

                        // Trigger if price rises enough from the low
                        if let Some(low) = trailing_stop.low_water_mark {
                            match trailing_stop.trail_type {
                                TrailType::Amount => current_price >= low + trailing_stop.trail_amount,
                                TrailType::Percent => {
                                    current_price >= low * (Decimal::ONE + trailing_stop.trail_amount / Decimal::from(100))
                                }
                            }
                        } else {
                            false
                        }
                    }
                    OrderSide::Sell => {
                        // For sell trailing stops, track the highest price
                        if trailing_stop.high_water_mark.is_none() || current_price > trailing_stop.high_water_mark.unwrap() {
                            trailing_stop.high_water_mark = Some(current_price);
                        }

                        // Trigger if price falls enough from the high
                        if let Some(high) = trailing_stop.high_water_mark {
                            match trailing_stop.trail_type {
                                TrailType::Amount => current_price <= high - trailing_stop.trail_amount,
                                TrailType::Percent => {
                                    current_price <= high * (Decimal::ONE - trailing_stop.trail_amount / Decimal::from(100))
                                }
                            }
                        } else {
                            false
                        }
                    }
                };

                if should_trigger {
                    // Convert to market order and execute
                    self.update_order_status(&order_id, OrderStatus::Filled)?;
                    // In a real implementation, you would submit a market order here
                }

                // Update the trailing stop record
                self.trailing_stops.insert(order_id, trailing_stop);
            }
        }

        Ok(())
    }

    pub fn cancel_order(&mut self, order_id: &str) -> Result<()> {
        if let Some(order) = self.orders.get_mut(order_id) {
            if order.status == OrderStatus::Pending || order.status == OrderStatus::PartiallyFilled {
                order.status = OrderStatus::Canceled;
                order.updated_at = Utc::now();
                Ok(())
            } else {
                Err(anyhow!("Cannot cancel order with status: {:?}", order.status))
            }
        } else {
            Err(anyhow!("Order not found: {}", order_id))
        }
    }

    pub fn get_order(&self, order_id: &str) -> Option<&AdvancedOrder> {
        self.orders.get(order_id)
    }

    pub fn get_orders_by_symbol(&self, symbol: &str) -> Vec<&AdvancedOrder> {
        self.orders.values()
            .filter(|order| order.symbol == symbol)
            .collect()
    }

    pub fn get_active_orders(&self) -> Vec<&AdvancedOrder> {
        self.orders.values()
            .filter(|order| {
                matches!(order.status, OrderStatus::Pending | OrderStatus::PartiallyFilled)
            })
            .collect()
    }

    pub fn get_fills_by_order(&self, order_id: &str) -> Vec<&Fill> {
        self.fills.iter()
            .filter(|fill| fill.order_id == order_id)
            .collect()
    }

    pub fn get_fills_by_symbol(&self, symbol: &str) -> Vec<&Fill> {
        self.fills.iter()
            .filter(|fill| fill.symbol == symbol)
            .collect()
    }

    pub fn evaluate_conditions(&self, order_id: &str, market_data: &HashMap<String, Decimal>) -> bool {
        if let Some(order) = self.orders.get(order_id) {
            if order.conditions.is_empty() {
                return true;
            }

            // Evaluate all conditions
            let mut results = Vec::new();
            for condition in &order.conditions {
                let result = self.evaluate_single_condition(condition, market_data);
                results.push(result);
            }

            // For simplicity, assume all conditions must be true (AND logic)
            // In a real implementation, you'd handle the logic operators properly
            results.iter().all(|&result| result)
        } else {
            false
        }
    }

    fn evaluate_single_condition(&self, condition: &OrderCondition, market_data: &HashMap<String, Decimal>) -> bool {
        let default_symbol = String::new();
        let symbol = condition.symbol.as_ref().unwrap_or(&default_symbol);
        
        match condition.condition_type {
            ConditionType::Price => {
                if let Some(&current_price) = market_data.get(symbol) {
                    self.compare_values(current_price, condition.value, &condition.operator)
                } else {
                    false
                }
            }
            ConditionType::Volume => {
                // Would need volume data in market_data
                false
            }
            ConditionType::Time => {
                let current_time = Utc::now().timestamp() as f64;
                let condition_time = condition.value.to_f64().unwrap_or(0.0);
                self.compare_values(
                    Decimal::from_f64(current_time).unwrap_or_default(),
                    Decimal::from_f64(condition_time).unwrap_or_default(),
                    &condition.operator
                )
            }
            // For technical indicators, you'd need to calculate them first
            _ => false,
        }
    }

    fn compare_values(&self, left: Decimal, right: Decimal, operator: &ComparisonOperator) -> bool {
        match operator {
            ComparisonOperator::GreaterThan => left > right,
            ComparisonOperator::LessThan => left < right,
            ComparisonOperator::GreaterThanOrEqual => left >= right,
            ComparisonOperator::LessThanOrEqual => left <= right,
            ComparisonOperator::Equal => left == right,
            ComparisonOperator::NotEqual => left != right,
        }
    }

    pub fn update_order_book(&mut self, symbol: String, order_book: OrderBook) {
        self.order_books.insert(symbol, order_book);
    }

    pub fn get_order_book(&self, symbol: &str) -> Option<&OrderBook> {
        self.order_books.get(symbol)
    }
}

impl Default for OrderManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_create_market_order() {
        let mut manager = OrderManager::new();
        let order = manager.create_market_order(
            "AAPL".to_string(),
            OrderSide::Buy,
            dec!(100),
            TimeInForce::Day,
        );

        assert_eq!(order.symbol, "AAPL");
        assert_eq!(order.side, OrderSide::Buy);
        assert_eq!(order.quantity, dec!(100));
        assert_eq!(order.order_type, OrderType::Market);
        assert_eq!(order.status, OrderStatus::Pending);
    }

    #[test]
    fn test_create_bracket_order() {
        let mut manager = OrderManager::new();
        let bracket = manager.create_bracket_order(
            "AAPL".to_string(),
            OrderSide::Buy,
            dec!(100),
            Some(dec!(150)),
            Some(dec!(160)),
            Some(dec!(140)),
        ).unwrap();

        assert_eq!(bracket.parent_order.symbol, "AAPL");
        assert!(bracket.take_profit_order.is_some());
        assert!(bracket.stop_loss_order.is_some());
    }

    #[test]
    fn test_fill_order() {
        let mut manager = OrderManager::new();
        let order = manager.create_limit_order(
            "AAPL".to_string(),
            OrderSide::Buy,
            dec!(100),
            dec!(150),
            TimeInForce::Day,
        );

        let fill = manager.fill_order(&order.id, dec!(50), dec!(149), dec!(1)).unwrap();
        
        assert_eq!(fill.quantity, dec!(50));
        assert_eq!(fill.price, dec!(149));
        
        let updated_order = manager.get_order(&order.id).unwrap();
        assert_eq!(updated_order.status, OrderStatus::PartiallyFilled);
        assert_eq!(updated_order.filled_quantity, dec!(50));
    }

    #[test]
    fn test_trailing_stop_order() {
        let mut manager = OrderManager::new();
        let trailing_stop = manager.create_trailing_stop_order(
            "AAPL".to_string(),
            OrderSide::Sell,
            dec!(100),
            dec!(5), // $5 trail
            TrailType::Amount,
        );

        assert_eq!(trailing_stop.trail_amount, dec!(5));
        assert_eq!(trailing_stop.trail_type, TrailType::Amount);
    }

    #[test]
    fn test_order_conditions() {
        let mut manager = OrderManager::new();
        let order = manager.create_limit_order(
            "AAPL".to_string(),
            OrderSide::Buy,
            dec!(100),
            dec!(150),
            TimeInForce::Day,
        );

        let condition = OrderCondition {
            condition_type: ConditionType::Price,
            symbol: Some("MSFT".to_string()),
            operator: ComparisonOperator::GreaterThan,
            value: dec!(200),
            logic_operator: None,
        };

        manager.add_condition_to_order(&order.id, condition).unwrap();
        
        let mut market_data = HashMap::new();
        market_data.insert("MSFT".to_string(), dec!(205));
        
        assert!(manager.evaluate_conditions(&order.id, &market_data));
    }

    #[test]
    fn test_oco_order() {
        let mut manager = OrderManager::new();
        let oco = manager.create_oco_order(
            "AAPL".to_string(),
            OrderSide::Buy,
            dec!(100),
            dec!(150), // Limit price
            dec!(145), // Stop price
            OrderType::Limit,
            OrderType::Stop,
        ).unwrap();

        assert_eq!(oco.primary_order.order_type, OrderType::Limit);
        assert_eq!(oco.secondary_order.order_type, OrderType::Stop);
    }

    #[test]
    fn test_cancel_order() {
        let mut manager = OrderManager::new();
        let order = manager.create_market_order(
            "AAPL".to_string(),
            OrderSide::Buy,
            dec!(100),
            TimeInForce::Day,
        );

        manager.cancel_order(&order.id).unwrap();
        let updated_order = manager.get_order(&order.id).unwrap();
        assert_eq!(updated_order.status, OrderStatus::Canceled);
    }
}