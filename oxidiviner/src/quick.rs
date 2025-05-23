//! Quick utilities module
//!
//! This module provides one-line functions for common forecasting tasks.
//! These functions use sensible defaults and are designed for rapid prototyping
//! and simple use cases where you don't need to configure complex parameters.

use crate::core::{Result, TimeSeriesData};
use crate::models::autoregressive::ARIMAModel;
use crate::models::exponential_smoothing::SimpleESModel;
use crate::models::moving_average::MAModel;
use chrono::{DateTime, Utc};

/// Quick ARIMA forecast with default parameters (1,1,1)
pub fn arima_forecast(
    timestamps: Vec<DateTime<Utc>>,
    values: Vec<f64>,
    periods: usize,
) -> Result<Vec<f64>> {
    let data = TimeSeriesData::new(timestamps, values, "quick_arima")?;
    let mut model = ARIMAModel::new(1, 1, 1, true)?;
    model.fit(&data)?;
    model.forecast(periods)
}

/// Quick ARIMA forecast with custom parameters
pub fn arima_forecast_custom(
    timestamps: Vec<DateTime<Utc>>,
    values: Vec<f64>,
    periods: usize,
    p: usize,
    d: usize,
    q: usize,
) -> Result<Vec<f64>> {
    let data = TimeSeriesData::new(timestamps, values, "quick_arima_custom")?;
    let mut model = ARIMAModel::new(p, d, q, true)?;
    model.fit(&data)?;
    model.forecast(periods)
}

/// Quick exponential smoothing forecast with default alpha (0.3)
pub fn es_forecast(
    timestamps: Vec<DateTime<Utc>>,
    values: Vec<f64>,
    periods: usize,
) -> Result<Vec<f64>> {
    let data = TimeSeriesData::new(timestamps, values, "quick_es")?;
    let mut model = SimpleESModel::new(0.3)?;
    model.fit(&data)?;
    model.forecast(periods)
}

/// Quick exponential smoothing forecast with custom alpha
pub fn es_forecast_custom(
    timestamps: Vec<DateTime<Utc>>,
    values: Vec<f64>,
    periods: usize,
    alpha: f64,
) -> Result<Vec<f64>> {
    let data = TimeSeriesData::new(timestamps, values, "quick_es_custom")?;
    let mut model = SimpleESModel::new(alpha)?;
    model.fit(&data)?;
    model.forecast(periods)
}

/// Quick moving average forecast with default window (5)
pub fn ma_forecast(
    timestamps: Vec<DateTime<Utc>>,
    values: Vec<f64>,
    periods: usize,
) -> Result<Vec<f64>> {
    let data = TimeSeriesData::new(timestamps, values, "quick_ma")?;
    let mut model = MAModel::new(5)?;
    model.fit(&data)?;
    model.forecast(periods)
}

/// Quick moving average forecast with custom window
pub fn ma_forecast_custom(
    timestamps: Vec<DateTime<Utc>>,
    values: Vec<f64>,
    periods: usize,
    window: usize,
) -> Result<Vec<f64>> {
    let data = TimeSeriesData::new(timestamps, values, "quick_ma_custom")?;
    let mut model = MAModel::new(window)?;
    model.fit(&data)?;
    model.forecast(periods)
}

/// Automatic model selection and forecasting
/// Tries ARIMA -> ES -> MA and returns the first successful one
pub fn auto_forecast(
    timestamps: Vec<DateTime<Utc>>,
    values: Vec<f64>,
    periods: usize,
) -> Result<(Vec<f64>, String)> {
    // Try ARIMA first
    if let Ok(forecast) = arima_forecast(timestamps.clone(), values.clone(), periods) {
        return Ok((forecast, "ARIMA(1,1,1)".to_string()));
    }

    // Try exponential smoothing
    if let Ok(forecast) = es_forecast(timestamps.clone(), values.clone(), periods) {
        return Ok((forecast, "SimpleES(α=0.3)".to_string()));
    }

    // Try moving average
    if let Ok(forecast) = ma_forecast(timestamps, values, periods) {
        return Ok((forecast, "MA(5)".to_string()));
    }

    Err(crate::core::OxiError::ModelError(
        "All quick models failed".to_string(),
    ))
}

/// Compare multiple models and return all successful forecasts
pub fn compare_models(
    timestamps: Vec<DateTime<Utc>>,
    values: Vec<f64>,
    periods: usize,
) -> Result<Vec<(String, Vec<f64>)>> {
    let mut results = Vec::new();

    // Try ARIMA
    if let Ok(forecast) = arima_forecast(timestamps.clone(), values.clone(), periods) {
        results.push(("ARIMA(1,1,1)".to_string(), forecast));
    }

    // Try exponential smoothing
    if let Ok(forecast) = es_forecast(timestamps.clone(), values.clone(), periods) {
        results.push(("SimpleES(α=0.3)".to_string(), forecast));
    }

    // Try moving average
    if let Ok(forecast) = ma_forecast(timestamps.clone(), values.clone(), periods) {
        results.push(("MA(5)".to_string(), forecast));
    }

    if results.is_empty() {
        return Err(crate::core::OxiError::ModelError(
            "All models failed in comparison".to_string(),
        ));
    }

    Ok(results)
}

/// Quick forecast from just values (uses index-based timestamps)
pub fn values_only_forecast(
    values: Vec<f64>,
    periods: usize,
) -> Result<(Vec<f64>, String)> {
    use chrono::{Duration, Utc};
    
    let start_time = Utc::now();
    let timestamps: Vec<DateTime<Utc>> = (0..values.len())
        .map(|i| start_time + Duration::days(i as i64))
        .collect();

    auto_forecast(timestamps, values, periods)
}

/// Quick forecast for financial data (assumes daily prices)
pub fn daily_price_forecast(
    prices: Vec<f64>,
    periods: usize,
) -> Result<(Vec<f64>, String)> {
    use chrono::{Duration, Utc};
    
    let start_date = Utc::now() - Duration::days(prices.len() as i64);
    let timestamps: Vec<DateTime<Utc>> = (0..prices.len())
        .map(|i| start_date + Duration::days(i as i64))
        .collect();

    auto_forecast(timestamps, prices, periods)
}

/// Simple forecast that takes a closure for data generation
pub fn forecast_with_data<F>(
    data_fn: F,
    periods: usize,
) -> Result<(Vec<f64>, String)>
where
    F: FnOnce() -> (Vec<DateTime<Utc>>, Vec<f64>),
{
    let (timestamps, values) = data_fn();
    auto_forecast(timestamps, values, periods)
}

/// ARIMA forecast from TimeSeriesData with default parameters
pub fn arima(data: TimeSeriesData, periods: usize) -> Result<Vec<f64>> {
    let mut model = ARIMAModel::new(1, 1, 1, true)?;
    model.fit(&data)?;
    model.forecast(periods)
}

/// ARIMA forecast from TimeSeriesData with custom parameters
pub fn arima_with_config(
    data: TimeSeriesData,
    periods: usize,
    config: Option<(usize, usize, usize)>,
) -> Result<Vec<f64>> {
    let (p, d, q) = config.unwrap_or((1, 1, 1));
    let mut model = ARIMAModel::new(p, d, q, true)?;
    model.fit(&data)?;
    model.forecast(periods)
}

/// Moving average forecast from TimeSeriesData
pub fn moving_average(
    data: TimeSeriesData,
    periods: usize,
    window: Option<usize>,
) -> Result<Vec<f64>> {
    let window_size = window.unwrap_or(5);
    let mut model = MAModel::new(window_size)?;
    model.fit(&data)?;
    model.forecast(periods)
}

/// Auto-select best model and forecast
pub fn auto_select(data: TimeSeriesData, periods: usize) -> Result<(Vec<f64>, String)> {
    // Try ARIMA first
    if let Ok(forecast) = arima(data.clone(), periods) {
        return Ok((forecast, "ARIMA(1,1,1)".to_string()));
    }

    // Try exponential smoothing
    let mut es_model = SimpleESModel::new(0.3)?;
    if let Ok(()) = es_model.fit(&data) {
        if let Ok(forecast) = es_model.forecast(periods) {
            return Ok((forecast, "SimpleES(α=0.3)".to_string()));
        }
    }

    // Try moving average
    if let Ok(forecast) = moving_average(data, periods, Some(5)) {
        return Ok((forecast, "MA(5)".to_string()));
    }

    Err(crate::core::OxiError::ModelError(
        "All auto-selection models failed".to_string(),
    ))
}

/// Forecast with model configuration
pub fn forecast_with_config(
    data: TimeSeriesData,
    periods: usize,
    config: crate::core::ModelConfig,
) -> Result<Vec<f64>> {
    match config.model_type.as_str() {
        "ARIMA" => {
            let p = *config.parameters.get("p").unwrap_or(&1.0) as usize;
            let d = *config.parameters.get("d").unwrap_or(&1.0) as usize;
            let q = *config.parameters.get("q").unwrap_or(&1.0) as usize;
            arima_with_config(data, periods, Some((p, d, q)))
        }
        "MA" => {
            let window = *config.parameters.get("window").unwrap_or(&5.0) as usize;
            moving_average(data, periods, Some(window))
        }
        "ES" => {
            let alpha = config.parameters.get("alpha").unwrap_or(&0.3);
            let mut model = SimpleESModel::new(*alpha)?;
            model.fit(&data)?;
            model.forecast(periods)
        }
        _ => Err(crate::core::OxiError::InvalidParameter(format!(
            "Unsupported model type: {}",
            config.model_type
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    fn generate_test_data() -> (Vec<DateTime<Utc>>, Vec<f64>) {
        let start = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..20)
            .map(|i| start + Duration::days(i))
            .collect();
        let values: Vec<f64> = (0..20)
            .map(|i| 100.0 + (i as f64) * 2.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();
        (timestamps, values)
    }

    #[test]
    fn test_quick_arima() {
        let (timestamps, values) = generate_test_data();
        let result = arima_forecast(timestamps, values, 5);
        assert!(result.is_ok());
        let forecast = result.unwrap();
        assert_eq!(forecast.len(), 5);
    }

    #[test]
    fn test_quick_es() {
        let (timestamps, values) = generate_test_data();
        let result = es_forecast(timestamps, values, 5);
        assert!(result.is_ok());
        let forecast = result.unwrap();
        assert_eq!(forecast.len(), 5);
    }

    #[test]
    fn test_quick_ma() {
        let (timestamps, values) = generate_test_data();
        let result = ma_forecast(timestamps, values, 5);
        assert!(result.is_ok());
        let forecast = result.unwrap();
        assert_eq!(forecast.len(), 5);
    }

    #[test]
    fn test_auto_forecast() {
        let (timestamps, values) = generate_test_data();
        let result = auto_forecast(timestamps, values, 5);
        assert!(result.is_ok());
        let (forecast, model_name) = result.unwrap();
        assert_eq!(forecast.len(), 5);
        assert!(!model_name.is_empty());
    }

    #[test]
    fn test_compare_models() {
        let (timestamps, values) = generate_test_data();
        let result = compare_models(timestamps, values, 5);
        assert!(result.is_ok());
        let comparisons = result.unwrap();
        assert!(!comparisons.is_empty());
        for (name, forecast) in comparisons {
            assert!(!name.is_empty());
            assert_eq!(forecast.len(), 5);
        }
    }

    #[test]
    fn test_values_only_forecast() {
        let values: Vec<f64> = (0..20)
            .map(|i| 100.0 + (i as f64) * 2.0)
            .collect();
        let result = values_only_forecast(values, 5);
        assert!(result.is_ok());
        let (forecast, model_name) = result.unwrap();
        assert_eq!(forecast.len(), 5);
        assert!(!model_name.is_empty());
    }
} 