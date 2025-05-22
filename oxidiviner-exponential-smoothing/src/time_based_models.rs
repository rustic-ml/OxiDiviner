use crate::error::ESError;
use crate::ets::{ETSModel, ErrorType, SeasonalType, TrendType};
use chrono::{DateTime, Utc};
use oxidiviner_core::{OHLCVData, OxiError, Result, TimeSeriesData};
use std::collections::HashMap;

#[cfg(test)]
use chrono::Duration;

/// Represents error, trend, and seasonal components for ETS models
///
/// This enum provides a simplified interface to specify ETS model components
/// compared to the separate component enums in the core ETSModel.
///
/// # Examples
///
/// ```
/// use oxidiviner_exponential_smoothing::{ETSComponent, DailyETSModel};
///
/// // Create a model with additive error, additive trend, and no seasonality
/// let model = DailyETSModel::new(
///     ETSComponent::Additive,  // Error type
///     ETSComponent::Additive,  // Trend type
///     ETSComponent::None,      // No seasonality
///     0.3,                     // alpha (level smoothing)
///     Some(0.1),               // beta (trend smoothing)
///     None,                    // gamma (no seasonality)
///     None,                    // phi (no damping)
///     None,                    // period (no seasonality)
///     None,                    // Use close price
/// ).expect("Failed to create model");
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ETSComponent {
    /// No component
    None,
    /// Additive component
    Additive,
    /// Multiplicative component
    Multiplicative,
    /// Damped trend component
    Damped,
}

impl From<ETSComponent> for ErrorType {
    fn from(component: ETSComponent) -> Self {
        match component {
            ETSComponent::Additive => ErrorType::Additive,
            ETSComponent::Multiplicative => ErrorType::Multiplicative,
            _ => ErrorType::None,
        }
    }
}

impl From<ETSComponent> for TrendType {
    fn from(component: ETSComponent) -> Self {
        match component {
            ETSComponent::Additive => TrendType::Additive,
            ETSComponent::Multiplicative => TrendType::Multiplicative,
            ETSComponent::Damped => TrendType::DampedAdditive,
            ETSComponent::None => TrendType::None,
        }
    }
}

impl From<ETSComponent> for SeasonalType {
    fn from(component: ETSComponent) -> Self {
        match component {
            ETSComponent::Additive => SeasonalType::Additive,
            ETSComponent::Multiplicative => SeasonalType::Multiplicative,
            _ => SeasonalType::None,
        }
    }
}

/// An ETS model specialized for daily OHLCV data
///
/// This model adapts the general ETS framework to work specifically with daily financial data,
/// providing convenient defaults and specialized handling for OHLCV data structures.
///
/// # Features
///
/// - Simplified construction compared to base ETSModel
/// - Works directly with OHLCV data structures
/// - Provides defaults appropriate for daily financial data
/// - Supports all core ETS model variants
///
/// # Common Seasonality Periods for Daily Data
///
/// - Weekly patterns: period = 7
/// - Monthly trading patterns: period ≈ 21 (typical number of trading days in a month)
/// - Quarterly patterns: period ≈ 63 (typical number of trading days in a quarter)
///
/// # Examples
///
/// ```
/// use oxidiviner_core::OHLCVData;
/// use oxidiviner_exponential_smoothing::DailyETSModel;
///
/// // Load your OHLCV data
/// // let data = OHLCVData::from_csv("daily_prices.csv", "%Y-%m-%d", true).unwrap();
///
/// // Create a simple exponential smoothing model (no trend, no seasonality)
/// let mut simple_model = DailyETSModel::simple(
///     0.3,      // alpha (level smoothing)
///     None,     // Use close price by default
/// ).unwrap();
///
/// // Create a Holt's linear trend model (with trend, no seasonality)
/// let mut trend_model = DailyETSModel::holt(
///     0.3,      // alpha (level smoothing)
///     0.1,      // beta (trend smoothing)
///     None,     // Use close price by default
/// ).unwrap();
///
/// // Create a Holt-Winters model with weekly seasonality
/// let mut seasonal_model = DailyETSModel::holt_winters_additive(
///     0.3,      // alpha (level smoothing)
///     0.1,      // beta (trend smoothing)
///     0.1,      // gamma (seasonal smoothing)
///     7,        // period = 7 days (weekly seasonality)
///     None,     // Use close price by default
/// ).unwrap();
///
/// // Fit, forecast and evaluate (assuming you have data)
/// // seasonal_model.fit(&data).unwrap();
/// // let forecasts = seasonal_model.forecast(30).unwrap();  // 30-day forecast
/// // let evaluation = seasonal_model.evaluate(&test_data).unwrap();
/// ```
pub struct DailyETSModel {
    /// The underlying ETS model
    model: ETSModel,
    /// The target column to forecast (default: Close)
    target_column: String,
}

impl DailyETSModel {
    /// Creates a new DailyETSModel with appropriate settings for daily data
    ///
    /// # Arguments
    /// * `error_type` - Type of error component
    /// * `trend_type` - Type of trend component
    /// * `seasonal_type` - Type of seasonal component
    /// * `alpha` - Smoothing parameter for level (0 < α < 1)
    /// * `beta` - Smoothing parameter for trend (0 < β < 1), None if trend_type is None
    /// * `gamma` - Smoothing parameter for seasonality (0 < γ < 1), None if seasonal_type is None
    /// * `phi` - Damping parameter (0 < φ < 1), required if trend is damped
    /// * `period` - Seasonal period, commonly 7 for weekly, 21 for monthly trading days, None if seasonal_type is None
    /// * `target_column` - Target column to forecast (default: "close")
    ///
    /// # Returns
    /// * `Result<Self>` - A new DailyETSModel if parameters are valid
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        error_type: ETSComponent,
        trend_type: ETSComponent,
        seasonal_type: ETSComponent,
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
        period: Option<usize>,
        target_column: Option<String>,
    ) -> std::result::Result<Self, ESError> {
        let error_type_enum: ErrorType = error_type.into();
        let trend_type_enum: TrendType = trend_type.into();
        let seasonal_type_enum: SeasonalType = seasonal_type.into();

        let model = ETSModel::new(
            error_type_enum,
            trend_type_enum,
            seasonal_type_enum,
            alpha,
            beta,
            phi,
            gamma,
            period,
        )?;

        Ok(Self {
            model,
            target_column: target_column.unwrap_or_else(|| "close".to_string()),
        })
    }

    /// Creates a new simple exponential smoothing model for daily data
    pub fn simple(alpha: f64, target_column: Option<String>) -> std::result::Result<Self, ESError> {
        Self::new(
            ETSComponent::Additive,
            ETSComponent::None,
            ETSComponent::None,
            alpha,
            None,
            None,
            None,
            None,
            target_column,
        )
    }

    /// Creates a new Holt's linear trend model for daily data
    pub fn holt(
        alpha: f64,
        beta: f64,
        target_column: Option<String>,
    ) -> std::result::Result<Self, ESError> {
        Self::new(
            ETSComponent::Additive,
            ETSComponent::Additive,
            ETSComponent::None,
            alpha,
            Some(beta),
            None,
            None,
            None,
            target_column,
        )
    }

    /// Creates a new damped trend model for daily data
    pub fn damped_trend(
        alpha: f64,
        beta: f64,
        phi: f64,
        target_column: Option<String>,
    ) -> std::result::Result<Self, ESError> {
        Self::new(
            ETSComponent::Additive,
            ETSComponent::Damped,
            ETSComponent::None,
            alpha,
            Some(beta),
            None,
            Some(phi),
            None,
            target_column,
        )
    }

    /// Creates a new Holt-Winters additive seasonal model for daily data
    pub fn holt_winters_additive(
        alpha: f64,
        beta: f64,
        gamma: f64,
        period: usize,
        target_column: Option<String>,
    ) -> std::result::Result<Self, ESError> {
        Self::new(
            ETSComponent::Additive,
            ETSComponent::Additive,
            ETSComponent::Additive,
            alpha,
            Some(beta),
            Some(gamma),
            None,
            Some(period),
            target_column,
        )
    }

    /// Creates a new Holt-Winters multiplicative seasonal model for daily data
    pub fn holt_winters_multiplicative(
        alpha: f64,
        beta: f64,
        gamma: f64,
        period: usize,
        target_column: Option<String>,
    ) -> std::result::Result<Self, ESError> {
        Self::new(
            ETSComponent::Additive,
            ETSComponent::Additive,
            ETSComponent::Multiplicative,
            alpha,
            Some(beta),
            Some(gamma),
            None,
            Some(period),
            target_column,
        )
    }

    /// Fit the model to OHLCV data
    pub fn fit(&mut self, data: &OHLCVData) -> Result<()> {
        let values = match self.target_column.to_lowercase().as_str() {
            "open" => &data.open,
            "high" => &data.high,
            "low" => &data.low,
            "close" | "" => &data.close,
            "volume" => &data.volume,
            _ => {
                return Err(OxiError::InvalidParameter(format!(
                    "Unknown column: {}",
                    self.target_column
                )))
            }
        };

        let time_series = TimeSeriesData {
            timestamps: data.timestamps.clone(),
            values: values.clone(),
            name: format!("{} {}", data.symbol, self.target_column),
        };

        self.model.fit(&time_series)
    }

    /// Generate forecasts
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        self.model.forecast(horizon)
    }

    /// Evaluate the model
    pub fn evaluate(&self, test_data: &OHLCVData) -> Result<oxidiviner_core::ModelEvaluation> {
        let values = match self.target_column.to_lowercase().as_str() {
            "open" => &test_data.open,
            "high" => &test_data.high,
            "low" => &test_data.low,
            "close" | "" => &test_data.close,
            "volume" => &test_data.volume,
            _ => {
                return Err(OxiError::InvalidParameter(format!(
                    "Unknown column: {}",
                    self.target_column
                )))
            }
        };

        let time_series = TimeSeriesData {
            timestamps: test_data.timestamps.clone(),
            values: values.clone(),
            name: format!("{} {}", test_data.symbol, self.target_column),
        };

        self.model.evaluate(&time_series)
    }

    /// Get fitted values
    pub fn fitted_values(&self) -> Option<&Vec<f64>> {
        self.model.fitted_values()
    }
}

/// An ETS model specialized for minute OHLCV data, with support for data aggregation
///
/// This model adapts the general ETS framework to work specifically with high-frequency
/// financial data, providing specialized handling for minute-level data including optional
/// aggregation to reduce noise.
///
/// # Features
///
/// - Works directly with minute-level OHLCV data
/// - Supports data aggregation (e.g., converting 1-minute data to 5-minute bars)
/// - Provides appropriate defaults for high-frequency data
/// - Higher default smoothing parameters to adapt to faster changes
///
/// # Common Seasonality Periods for Minute Data
///
/// - 15-minute patterns: period = 15
/// - Hourly patterns: period = 60
/// - Half-day session: period ≈ 240 (4-hour trading session)
///
/// # Examples
///
/// ```
/// use oxidiviner_core::OHLCVData;
/// use oxidiviner_exponential_smoothing::MinuteETSModel;
///
/// // Load your minute OHLCV data
/// // let data = OHLCVData::from_csv("minute_prices.csv", "%Y-%m-%dT%H:%M:%S", true).unwrap();
///
/// // Create a simple model with 5-minute aggregation
/// let mut model = MinuteETSModel::simple(
///     0.4,       // alpha (higher than daily to adapt faster)
///     None,      // Use close price by default
///     Some(5),   // 5-minute aggregation
/// ).unwrap();
///
/// // Create a model with trend and hourly seasonality
/// let mut seasonal_model = MinuteETSModel::holt_winters_additive(
///     0.4,       // alpha (level smoothing)
///     0.1,       // beta (trend smoothing)
///     0.1,       // gamma (seasonal smoothing)
///     60,        // period = 60 minutes (hourly seasonality)
///     None,      // Use close price by default
///     Some(5),   // 5-minute aggregation (reduces noise)
/// ).unwrap();
///
/// // Fit, forecast and evaluate (assuming you have data)
/// // seasonal_model.fit(&data).unwrap();
/// // let forecasts = seasonal_model.forecast(60).unwrap();  // 1-hour forecast
/// // let evaluation = seasonal_model.evaluate(&test_data).unwrap();
/// ```
pub struct MinuteETSModel {
    /// The underlying ETS model
    model: ETSModel,
    /// The target column to forecast (default: Close)
    target_column: String,
    /// Aggregation level in minutes (e.g., 5 for 5-minute bars)
    aggregation_minutes: Option<usize>,
}

impl MinuteETSModel {
    /// Creates a new MinuteETSModel with appropriate settings for minute data
    ///
    /// # Arguments
    /// * `error_type` - Type of error component
    /// * `trend_type` - Type of trend component
    /// * `seasonal_type` - Type of seasonal component
    /// * `alpha` - Smoothing parameter for level (0 < α < 1)
    /// * `beta` - Smoothing parameter for trend (0 < β < 1), None if trend_type is None
    /// * `gamma` - Smoothing parameter for seasonality (0 < γ < 1), None if seasonal_type is None
    /// * `phi` - Damping parameter (0 < φ < 1), required if trend is damped
    /// * `period` - Seasonal period, commonly 60 for hourly patterns, 240 for 4-hour session, None if seasonal_type is None
    /// * `target_column` - Target column to forecast (default: "close")
    /// * `aggregation_minutes` - Optional aggregation level in minutes
    ///
    /// # Returns
    /// * `Result<Self>` - A new MinuteETSModel if parameters are valid
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        error_type: ETSComponent,
        trend_type: ETSComponent,
        seasonal_type: ETSComponent,
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
        period: Option<usize>,
        target_column: Option<String>,
        aggregation_minutes: Option<usize>,
    ) -> std::result::Result<Self, ESError> {
        let error_type_enum: ErrorType = error_type.into();
        let trend_type_enum: TrendType = trend_type.into();
        let seasonal_type_enum: SeasonalType = seasonal_type.into();

        // Validate aggregation_minutes if provided
        if let Some(agg) = aggregation_minutes {
            if agg < 1 {
                return Err(ESError::InvalidParameter(format!(
                    "Aggregation minutes must be at least 1, got {}",
                    agg
                )));
            }
        }

        let model = ETSModel::new(
            error_type_enum,
            trend_type_enum,
            seasonal_type_enum,
            alpha,
            beta,
            phi,
            gamma,
            period,
        )?;

        Ok(Self {
            model,
            target_column: target_column.unwrap_or_else(|| "close".to_string()),
            aggregation_minutes,
        })
    }

    /// Creates a new simple exponential smoothing model for minute data
    pub fn simple(
        alpha: f64,
        target_column: Option<String>,
        aggregation_minutes: Option<usize>,
    ) -> std::result::Result<Self, ESError> {
        Self::new(
            ETSComponent::Additive,
            ETSComponent::None,
            ETSComponent::None,
            alpha,
            None,
            None,
            None,
            None,
            target_column,
            aggregation_minutes,
        )
    }

    /// Creates a new Holt's linear trend model for minute data
    pub fn holt(
        alpha: f64,
        beta: f64,
        target_column: Option<String>,
        aggregation_minutes: Option<usize>,
    ) -> std::result::Result<Self, ESError> {
        Self::new(
            ETSComponent::Additive,
            ETSComponent::Additive,
            ETSComponent::None,
            alpha,
            Some(beta),
            None,
            None,
            None,
            target_column,
            aggregation_minutes,
        )
    }

    /// Creates a new Holt-Winters additive seasonal model for minute data
    pub fn holt_winters_additive(
        alpha: f64,
        beta: f64,
        gamma: f64,
        period: usize,
        target_column: Option<String>,
        aggregation_minutes: Option<usize>,
    ) -> std::result::Result<Self, ESError> {
        Self::new(
            ETSComponent::Additive,
            ETSComponent::Additive,
            ETSComponent::Additive,
            alpha,
            Some(beta),
            Some(gamma),
            None,
            Some(period),
            target_column,
            aggregation_minutes,
        )
    }

    /// Get the aggregation minutes value
    pub fn aggregation_minutes(&self) -> usize {
        self.aggregation_minutes.unwrap_or(1)
    }

    /// Fit the model to OHLCV data, with optional aggregation
    pub fn fit(&mut self, data: &OHLCVData) -> Result<()> {
        let values = match self.target_column.to_lowercase().as_str() {
            "open" => &data.open,
            "high" => &data.high,
            "low" => &data.low,
            "close" | "" => &data.close,
            "volume" => &data.volume,
            _ => {
                return Err(OxiError::InvalidParameter(format!(
                    "Unknown column: {}",
                    self.target_column
                )))
            }
        };

        // Check if we need to aggregate the data
        let time_series = if let Some(agg_minutes) = self.aggregation_minutes {
            // Aggregate data
            self.aggregate_minute_data(&data.timestamps, values, agg_minutes)?
        } else {
            // No aggregation needed
            TimeSeriesData {
                timestamps: data.timestamps.clone(),
                values: values.clone(),
                name: format!("{} {}", data.symbol, self.target_column),
            }
        };

        self.model.fit(&time_series)
    }

    /// Generate forecasts
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        self.model.forecast(horizon)
    }

    /// Evaluate the model
    pub fn evaluate(&self, test_data: &OHLCVData) -> Result<oxidiviner_core::ModelEvaluation> {
        let values = match self.target_column.to_lowercase().as_str() {
            "open" => &test_data.open,
            "high" => &test_data.high,
            "low" => &test_data.low,
            "close" | "" => &test_data.close,
            "volume" => &test_data.volume,
            _ => {
                return Err(OxiError::InvalidParameter(format!(
                    "Unknown column: {}",
                    self.target_column
                )))
            }
        };

        // Check if we need to aggregate the data
        let time_series = if let Some(agg_minutes) = self.aggregation_minutes {
            // Aggregate data
            self.aggregate_minute_data(&test_data.timestamps, values, agg_minutes)?
        } else {
            // No aggregation needed
            TimeSeriesData {
                timestamps: test_data.timestamps.clone(),
                values: values.clone(),
                name: format!("{} {}", test_data.symbol, self.target_column),
            }
        };

        self.model.evaluate(&time_series)
    }

    /// Get fitted values
    pub fn fitted_values(&self) -> Option<&Vec<f64>> {
        self.model.fitted_values()
    }

    /// Aggregate minute data to a higher timeframe
    fn aggregate_minute_data(
        &self,
        timestamps: &[DateTime<Utc>],
        values: &[f64],
        agg_minutes: usize,
    ) -> Result<TimeSeriesData> {
        if timestamps.is_empty() || values.is_empty() {
            return Err(OxiError::InvalidParameter(
                "Empty timestamps or values".to_string(),
            ));
        }

        if timestamps.len() != values.len() {
            return Err(OxiError::InvalidParameter(
                "Timestamps and values must have the same length".to_string(),
            ));
        }

        // Calculate the aggregation key for each timestamp
        let mut aggregated_data: HashMap<i64, (DateTime<Utc>, Vec<f64>)> = HashMap::new();

        for (i, timestamp) in timestamps.iter().enumerate() {
            // Truncate to minute and then calculate the aggregation bucket
            // by dividing by agg_minutes and multiplying back
            let epoch_mins = timestamp.timestamp() / 60;
            let agg_key = (epoch_mins / agg_minutes as i64) * agg_minutes as i64;

            // Add value to the appropriate bucket
            aggregated_data
                .entry(agg_key)
                .or_insert_with(|| (*timestamp, Vec::new()))
                .1
                .push(values[i]);
        }

        // Process the aggregated data
        let mut agg_timestamps = Vec::with_capacity(aggregated_data.len());
        let mut agg_values = Vec::with_capacity(aggregated_data.len());

        // Sort by timestamp
        let mut sorted_data: Vec<_> = aggregated_data.into_iter().collect();
        sorted_data.sort_by_key(|&(key, _)| key);

        for (_, (timestamp, values_in_period)) in sorted_data {
            // Calculate typical price (avg of values in the period)
            let avg_value = values_in_period.iter().sum::<f64>() / values_in_period.len() as f64;

            agg_timestamps.push(timestamp);
            agg_values.push(avg_value);
        }

        Ok(TimeSeriesData {
            timestamps: agg_timestamps,
            values: agg_values,
            name: format!("Aggregated {} ({}m)", self.target_column, agg_minutes),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create test daily data
    fn create_test_daily_data() -> OHLCVData {
        let now = Utc::now();
        let mut timestamps = Vec::with_capacity(100);
        let mut open = Vec::with_capacity(100);
        let mut high = Vec::with_capacity(100);
        let mut low = Vec::with_capacity(100);
        let mut close = Vec::with_capacity(100);
        let mut volume = Vec::with_capacity(100);

        // Create 100 days of synthetic data with trend and seasonality
        for i in 0..100 {
            let day = now + Duration::days(i);
            timestamps.push(day);

            // Add trend (0.5 per day) and seasonality (7-day cycle)
            let trend = 0.5 * i as f64;
            let season = 5.0 * (2.0 * std::f64::consts::PI * (i % 7) as f64 / 7.0).sin();
            let noise = (i % 10) as f64 * 0.2 - 1.0; // Deterministic "noise"

            let base_price = 100.0 + trend + season + noise;
            let daily_range = base_price * 0.02; // 2% daily range

            open.push(base_price - daily_range / 2.0);
            close.push(base_price + daily_range / 2.0);
            high.push(base_price + daily_range);
            low.push(base_price - daily_range);
            volume.push(1000.0 + (i % 100) as f64 * 10.0);
        }

        OHLCVData {
            symbol: "TEST_DAILY".to_string(),
            timestamps,
            open,
            high,
            low,
            close,
            volume,
            adjusted_close: None,
        }
    }

    // Helper function to create test minute data
    fn create_test_minute_data() -> OHLCVData {
        let now = Utc::now();
        let mut timestamps = Vec::with_capacity(480); // 8 hours of minute data
        let mut open = Vec::with_capacity(480);
        let mut high = Vec::with_capacity(480);
        let mut low = Vec::with_capacity(480);
        let mut close = Vec::with_capacity(480);
        let mut volume = Vec::with_capacity(480);

        // Create 8 hours of 1-minute data with trend and hourly patterns
        for i in 0..480 {
            let minute = now + Duration::minutes(i);
            timestamps.push(minute);

            // Add small trend + hourly seasonality
            let trend = 0.001 * i as f64;
            let minute_in_hour = i % 60;
            let hour_cycle =
                1.0 * (2.0 * std::f64::consts::PI * minute_in_hour as f64 / 60.0).sin();
            let noise = (i % 5) as f64 * 0.05 - 0.125; // Deterministic "noise"

            let base_price = 100.0 + trend + hour_cycle + noise;
            let minute_range = base_price * 0.001; // Smaller range for minute data

            open.push(base_price - minute_range);
            close.push(base_price + minute_range);
            high.push(base_price + minute_range * 1.5);
            low.push(base_price - minute_range * 1.5);
            volume.push(100.0 + (i % 60) as f64);
        }

        OHLCVData {
            symbol: "TEST_MINUTE".to_string(),
            timestamps,
            open,
            high,
            low,
            close,
            volume,
            adjusted_close: None,
        }
    }

    #[test]
    fn test_daily_model_creation() {
        // Test simple model creation
        let _model = DailyETSModel::simple(0.3, None).unwrap();

        // Test holt model creation
        let _model = DailyETSModel::holt(0.3, 0.1, None).unwrap();

        // Test holt-winters model creation
        let _model = DailyETSModel::holt_winters_additive(0.3, 0.1, 0.1, 7, None).unwrap();
    }

    #[test]
    fn test_daily_model_with_data() {
        let data = create_test_daily_data();

        // Create and fit the model
        let mut model = DailyETSModel::holt_winters_additive(0.3, 0.1, 0.1, 7, None).unwrap();

        // Should fit successfully
        assert!(model.fit(&data).is_ok());

        // Should generate forecasts
        let forecasts = model.forecast(30).unwrap();
        assert_eq!(forecasts.len(), 30);

        // Fitted values should be available
        let fitted = model.fitted_values().unwrap();
        assert_eq!(fitted.len(), data.len());

        // Test with different target columns
        let mut high_model = DailyETSModel::simple(0.3, Some("high".to_string())).unwrap();
        assert!(high_model.fit(&data).is_ok());

        let mut volume_model = DailyETSModel::simple(0.3, Some("volume".to_string())).unwrap();
        assert!(volume_model.fit(&data).is_ok());
    }

    #[test]
    fn test_minute_model_creation() {
        // Test simple model creation
        let _model = MinuteETSModel::simple(0.3, None, None).unwrap();

        // Test with aggregation
        let model = MinuteETSModel::simple(0.3, None, Some(5)).unwrap();
        assert_eq!(model.aggregation_minutes(), 5);

        // Test hourly seasonality with aggregation
        let _model =
            MinuteETSModel::holt_winters_additive(0.3, 0.1, 0.1, 60, None, Some(5)).unwrap();
    }

    #[test]
    fn test_minute_model_with_data() {
        let data = create_test_minute_data();

        // Create and fit a model with 5-minute aggregation
        let mut model = MinuteETSModel::simple(0.3, None, Some(5)).unwrap();
        assert!(model.fit(&data).is_ok());

        // Generate forecasts
        let forecasts = model.forecast(60).unwrap();
        assert_eq!(forecasts.len(), 60);

        // Create and fit a model with 15-minute seasonality
        let mut seasonal_model =
            MinuteETSModel::holt_winters_additive(0.3, 0.1, 0.1, 15, None, Some(5)).unwrap();
        assert!(seasonal_model.fit(&data).is_ok());

        // Test with different target columns
        let mut high_model =
            MinuteETSModel::simple(0.3, Some("high".to_string()), Some(5)).unwrap();
        assert!(high_model.fit(&data).is_ok());
    }

    #[test]
    fn test_minute_data_aggregation() {
        // Create sample minute data
        let now = Utc::now();
        let mut timestamps = Vec::new();
        let mut values = Vec::new();

        // Create 15 minutes of data
        for i in 0..15 {
            timestamps.push(now + Duration::minutes(i));
            values.push(i as f64);
        }

        // Create model with 5-minute aggregation
        let model = MinuteETSModel::simple(0.3, None, Some(5)).unwrap();

        // Aggregate data
        let agg_data = model
            .aggregate_minute_data(&timestamps, &values, 5)
            .unwrap();

        // Debug output
        println!("Original timestamps count: {}", timestamps.len());
        println!("Aggregated timestamps count: {}", agg_data.timestamps.len());

        // Instead of checking exact count, check that aggregation reduces data points
        // There should be fewer aggregated timestamps than original
        assert!(
            agg_data.timestamps.len() < timestamps.len(),
            "Aggregation should reduce the number of data points"
        );

        // We expect approximately 15/5 = 3 points, but allow for boundary conditions
        assert!(
            agg_data.timestamps.len() >= 3 && agg_data.timestamps.len() <= 4,
            "Expected 3-4 aggregated timestamps, got {}",
            agg_data.timestamps.len()
        );
    }

    #[test]
    fn test_invalid_parameters() {
        // Test invalid alpha
        assert!(DailyETSModel::simple(1.5, None).is_err());

        // Test invalid beta
        assert!(DailyETSModel::holt(0.3, 1.5, None).is_err());

        // Test invalid seasonal period
        assert!(DailyETSModel::holt_winters_additive(0.3, 0.1, 0.1, 1, None).is_err());

        // Test invalid aggregation minutes
        assert!(MinuteETSModel::simple(0.3, None, Some(0)).is_err());

        // Test missing required parameters for seasonal model
        assert!(DailyETSModel::new(
            ETSComponent::Additive,
            ETSComponent::Additive,
            ETSComponent::Additive,
            0.3,
            Some(0.1),
            None, // Missing gamma for seasonal model
            None,
            Some(7),
            None
        )
        .is_err());
    }

    #[test]
    fn test_compare_models() {
        let daily_data = create_test_daily_data();

        // Create and fit models with different configurations
        let mut simple_model = DailyETSModel::simple(0.3, None).unwrap();
        simple_model.fit(&daily_data).unwrap();

        let mut trend_model = DailyETSModel::holt(0.3, 0.1, None).unwrap();
        trend_model.fit(&daily_data).unwrap();

        // Use a stronger seasonality to ensure differences
        let mut seasonal_model =
            DailyETSModel::holt_winters_additive(0.3, 0.1, 0.5, 7, None).unwrap();
        seasonal_model.fit(&daily_data).unwrap();

        // Get forecasts - looking further ahead to see more pronounced differences
        let horizon = 20;
        let simple_forecast = simple_model.forecast(horizon).unwrap();
        let _trend_forecast = trend_model.forecast(horizon).unwrap();
        let seasonal_forecast = seasonal_model.forecast(horizon).unwrap();

        // Check that at least some forecasts differ significantly
        let mut differences_found = false;
        for i in 10..horizon {
            // Check later in the forecast horizon where differences are more pronounced
            if (seasonal_forecast[i] - simple_forecast[i]).abs() > 0.1 {
                differences_found = true;
                break;
            }
        }
        assert!(
            differences_found,
            "Models should produce different forecasts"
        );
    }
}
