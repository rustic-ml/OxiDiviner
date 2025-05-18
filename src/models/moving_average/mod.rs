use crate::models::{Forecaster, ModelEvaluation, ModelOutput};
use crate::data::TimeSeriesData;
use crate::error::{OxiError, Result as OxiResult};
use crate::indicators::sma;

/// Simple Moving Average model for forecasting
/// Uses the average of the last `window_size` observations as the forecast
pub struct MAModel {
    /// Model name
    name: String,
    /// Window size for the moving average
    window_size: usize,
    /// Last values from the time series (used for forecasting)
    last_values: Option<Vec<f64>>,
}

impl MAModel {
    /// Create a new Moving Average model
    ///
    /// # Arguments
    /// * `window_size` - The number of most recent observations to use for the average
    ///
    /// # Returns
    /// * `Result<Self, String>` - A new MA model if parameters are valid
    pub fn new(window_size: usize) -> Result<Self, String> {
        if window_size == 0 {
            return Err("Window size must be greater than 0".to_string());
        }
        
        let name = format!("MA({})", window_size);
        
        Ok(MAModel {
            name,
            window_size,
            last_values: None,
        })
    }
    
    /// Fit the model to the provided time series data.
    /// This is a convenience method that calls the trait method directly.
    pub fn fit(&mut self, data: &TimeSeriesData) -> OxiResult<()> {
        <Self as Forecaster>::fit(self, data)
    }
    
    /// Forecast future values.
    /// This is a convenience method that calls the trait method directly.
    pub fn forecast(&self, horizon: usize) -> OxiResult<Vec<f64>> {
        <Self as Forecaster>::forecast(self, horizon)
    }
    
    /// Evaluate the model on test data.
    /// This is a convenience method that calls the trait method directly.
    pub fn evaluate(&self, test_data: &TimeSeriesData) -> OxiResult<ModelEvaluation> {
        <Self as Forecaster>::evaluate(self, test_data)
    }
    
    /// Generate forecasts and evaluation in a standardized format.
    /// This is a convenience method that calls the trait method directly.
    pub fn predict(&self, horizon: usize, test_data: Option<&TimeSeriesData>) -> OxiResult<ModelOutput> {
        <Self as Forecaster>::predict(self, horizon, test_data)
    }
}

impl Forecaster for MAModel {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn fit(&mut self, data: &TimeSeriesData) -> OxiResult<()> {
        if data.is_empty() {
            return Err(OxiError::Model("Cannot fit model on empty data".to_string()));
        }
        
        if data.len() < self.window_size {
            return Err(OxiError::Model(format!(
                "Time series length ({}) must be at least equal to window size ({})",
                data.len(), self.window_size
            )));
        }
        
        // Store the last window_size values for forecasting
        self.last_values = Some(data.values[data.len() - self.window_size..].to_vec());
        
        Ok(())
    }
    
    fn forecast(&self, horizon: usize) -> OxiResult<Vec<f64>> {
        if let Some(last_values) = &self.last_values {
            // Calculate the moving average of the last window_size values
            let avg = last_values.iter().sum::<f64>() / last_values.len() as f64;
            
            // For simple MA model, all forecasts are the same value
            Ok(vec![avg; horizon])
        } else {
            Err(OxiError::Model("Model has not been fitted yet".to_string()))
        }
    }
    
    fn evaluate(&self, test_data: &TimeSeriesData) -> OxiResult<ModelEvaluation> {
        if self.last_values.is_none() {
            return Err(OxiError::Model("Model has not been fitted yet".to_string()));
        }
        
        let forecast = self.forecast(test_data.values.len())?;
        
        // Calculate error metrics
        let mae = crate::utils::mae(&test_data.values, &forecast);
        let mse = crate::utils::mse(&test_data.values, &forecast);
        let rmse = crate::utils::rmse(&test_data.values, &forecast);
        let mape = crate::utils::mape(&test_data.values, &forecast);
        let smape = crate::utils::smape(&test_data.values, &forecast);
        
        Ok(ModelEvaluation {
            model_name: self.name.clone(),
            mae,
            mse,
            rmse,
            mape,
            smape,
        })
    }
    
    // Using the default predict implementation from the trait
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc, TimeZone};

    #[test]
    fn test_ma_model() {
        // Create a simple time series
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();
        
        // Linear trend data: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        let values: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        
        let time_series = TimeSeriesData::new(timestamps, values, "test_series").unwrap();
        
        // Create and fit the MA model with window size 3
        let mut model = MAModel::new(3).unwrap();
        model.fit(&time_series).unwrap();
        
        // Forecast 5 periods ahead
        let forecast_horizon = 5;
        let forecasts = model.forecast(forecast_horizon).unwrap();
        
        // Check that the number of forecasts matches the horizon
        assert_eq!(forecasts.len(), forecast_horizon);
        
        // Expected forecast value: average of last 3 values (8, 9, 10) = 9
        let expected_forecast = 9.0;
        for forecast in forecasts.iter() {
            assert_eq!(*forecast, expected_forecast);
        }
        
        // Test the standardized ModelOutput from predict()
        let output = model.predict(forecast_horizon, None).unwrap();
        
        // Check basic properties of the ModelOutput
        assert_eq!(output.model_name, model.name());
        assert_eq!(output.forecasts.len(), forecast_horizon);
        assert_eq!(output.forecasts, forecasts);
        
        // Test with evaluation
        let output_with_eval = model.predict(forecast_horizon, Some(&time_series)).unwrap();
        
        // Should have evaluation metrics
        assert!(output_with_eval.evaluation.is_some());
        let eval = output_with_eval.evaluation.unwrap();
        assert_eq!(eval.model_name, model.name());
    }
} 