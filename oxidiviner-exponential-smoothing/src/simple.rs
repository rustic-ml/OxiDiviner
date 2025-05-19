use oxidiviner_core::{Forecaster, ModelEvaluation, ModelOutput, OxiError, Result, TimeSeriesData};
use oxidiviner_math::metrics::{mae, mse, rmse, mape, smape};
use crate::error::ESError;

/// Simple Exponential Smoothing (SES) model for forecasting.
///
/// This model only has a level component and is suitable for forecasting time series 
/// without clear trend or seasonality. It's equivalent to ETS(A,N,N) in the ETS framework.
///
/// # Model Equation:
/// - Level: l_t = α * y_t + (1 - α) * l_{t-1}
/// - Forecast: ŷ_{t+h|t} = l_t
///
/// where:
/// - l_t is the level at time t
/// - y_t is the observed value at time t
/// - α is the smoothing parameter (0 < α < 1)
/// - ŷ_{t+h|t} is the h-step ahead forecast from time t
pub struct SimpleESModel {
    /// Model name
    name: String,
    /// Smoothing parameter for level (0 < α < 1)
    alpha: f64,
    /// Current level value (after fitting)
    level: Option<f64>,
    /// Fitted values over the training period
    fitted_values: Option<Vec<f64>>,
}

impl SimpleESModel {
    /// Creates a new Simple Exponential Smoothing model.
    ///
    /// # Arguments
    /// * `alpha` - Smoothing parameter for level (0 < α < 1)
    ///
    /// # Returns
    /// * `Result<Self>` - A new SES model if parameters are valid
    pub fn new(alpha: f64) -> std::result::Result<Self, ESError> {
        // Validate parameters
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(ESError::InvalidAlpha(alpha));
        }
        
        let name = format!("SES(α={:.3})", alpha);
        
        Ok(SimpleESModel {
            name,
            alpha,
            level: None,
            fitted_values: None,
        })
    }
    
    /// Fit the model to the provided time series data.
    /// This is a convenience method that calls the trait method directly.
    pub fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        <Self as Forecaster>::fit(self, data)
    }
    
    /// Forecast future values.
    /// This is a convenience method that calls the trait method directly.
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        <Self as Forecaster>::forecast(self, horizon)
    }
    
    /// Evaluate the model on test data.
    /// This is a convenience method that calls the trait method directly.
    pub fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        <Self as Forecaster>::evaluate(self, test_data)
    }
    
    /// Generate forecasts and evaluation in a standardized format.
    /// This is a convenience method that calls the trait method directly.
    pub fn predict(&self, horizon: usize, test_data: Option<&TimeSeriesData>) -> Result<ModelOutput> {
        <Self as Forecaster>::predict(self, horizon, test_data)
    }
    
    /// Get the fitted values if available.
    pub fn fitted_values(&self) -> Option<&Vec<f64>> {
        self.fitted_values.as_ref()
    }
}

impl Forecaster for SimpleESModel {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        if data.is_empty() {
            return Err(OxiError::from(ESError::EmptyData));
        }
        
        let n = data.values.len();
        
        // Initialize the level with the first observation
        let mut level = data.values[0];
        
        // Prepare to store fitted values
        let mut fitted_values = Vec::with_capacity(n);
        fitted_values.push(level); // First fitted value is the initial level
        
        // Apply the SES model recursively
        for i in 1..n {
            // Calculate forecast for this step (which is just the previous level)
            let forecast = level;
            
            // Update the level based on the observed value
            level = self.alpha * data.values[i] + (1.0 - self.alpha) * level;
            
            // Store the forecast
            fitted_values.push(forecast);
        }
        
        // Store the final level and fitted values
        self.level = Some(level);
        self.fitted_values = Some(fitted_values);
        
        Ok(())
    }
    
    fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        if let Some(level) = self.level {
            // For SES, all future forecasts are just the last level
            Ok(vec![level; horizon])
        } else {
            Err(OxiError::from(ESError::NotFitted))
        }
    }
    
    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        if self.level.is_none() {
            return Err(OxiError::from(ESError::NotFitted));
        }
        
        let forecast = self.forecast(test_data.values.len())?;
        
        // Calculate error metrics
        let mae = mae(&test_data.values, &forecast);
        let mse = mse(&test_data.values, &forecast);
        let rmse = rmse(&test_data.values, &forecast);
        let mape = mape(&test_data.values, &forecast);
        let smape = smape(&test_data.values, &forecast);
        
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
    fn test_ses_forecast() {
        // Create a simple time series
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();
        
        // Linear trend data: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        let values: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        
        let time_series = TimeSeriesData::new(timestamps, values, "test_series").unwrap();
        
        // Create and fit the SES model with alpha = 0.3
        let mut model = SimpleESModel::new(0.3).unwrap();
        model.fit(&time_series).unwrap();
        
        // Forecast 5 periods ahead
        let forecast_horizon = 5;
        let forecasts = model.forecast(forecast_horizon).unwrap();
        
        // Check that the number of forecasts matches the horizon
        assert_eq!(forecasts.len(), forecast_horizon);
        
        // All forecasts should be the same value (the last level)
        let first_forecast = forecasts[0];
        for forecast in forecasts.iter() {
            assert_eq!(*forecast, first_forecast);
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