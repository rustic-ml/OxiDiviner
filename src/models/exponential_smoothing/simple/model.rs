use crate::models::data::OHLCVData;
use crate::models::{ModelEvaluation, Forecaster, ModelOutput};
use crate::data::TimeSeriesData;
use crate::error::{OxiError, Result as OxiResult};

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
pub struct SESModel {
    /// Model name
    name: String,
    /// Smoothing parameter for level (0 < α < 1)
    alpha: f64,
    /// Current level value (after fitting)
    level: Option<f64>,
    /// Fitted values over the training period
    fitted_values: Option<Vec<f64>>,
    /// Which column from OHLCV data to use for prediction (defaults to close)
    target_column: TargetColumn,
}

/// Defines which price column to use for the SES model
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TargetColumn {
    /// Use the Open price
    Open,
    /// Use the High price 
    High,
    /// Use the Low price
    Low,
    /// Use the Close price (default)
    Close,
    /// Use the Volume
    Volume,
}

impl SESModel {
    /// Creates a new Simple Exponential Smoothing model.
    ///
    /// # Arguments
    /// * `alpha` - Smoothing parameter for level (0 < α < 1)
    /// * `target_column` - Which column of OHLCV data to use (defaults to Close)
    ///
    /// # Returns
    /// * `Result<Self, String>` - A new SES model if parameters are valid, or an error message if invalid
    ///
    /// # Examples
    /// ```
    /// use oxidiviner::models::exponential_smoothing::simple::SESModel;
    ///
    /// // Create a Simple Exponential Smoothing model with alpha = 0.3
    /// let model = SESModel::new(0.3, None).unwrap();
    /// ```
    pub fn new(
        alpha: f64,
        target_column: Option<TargetColumn>,
    ) -> Result<Self, String> {
        // Validate parameters
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err("Alpha must be between 0 and 1 exclusive".to_string());
        }
        
        // Default to close price if target column not specified
        let target_column = target_column.unwrap_or(TargetColumn::Close);
        
        // Create model name
        let column_name = match target_column {
            TargetColumn::Open => "Open",
            TargetColumn::High => "High",
            TargetColumn::Low => "Low",
            TargetColumn::Close => "Close",
            TargetColumn::Volume => "Volume",
        };
        
        let name = format!("SES({} | α={:.3})", column_name, alpha);
        
        Ok(SESModel {
            name,
            alpha,
            level: None,
            fitted_values: None,
            target_column,
        })
    }
    
    /// Extract the target data from OHLCV based on the selected column
    fn extract_target_data(&self, data: &OHLCVData) -> Vec<f64> {
        match self.target_column {
            TargetColumn::Open => data.open.clone(),
            TargetColumn::High => data.high.clone(),
            TargetColumn::Low => data.low.clone(),
            TargetColumn::Close => data.close.clone(),
            TargetColumn::Volume => data.volume.clone(),
        }
    }
    
    /// Fit the model to the provided time series data.
    ///
    /// # Arguments
    /// * `data` - The time series data to fit the model to
    ///
    /// # Returns
    /// * `Result<(), String>` - Ok if fitting is successful, or an error message if not
    pub fn fit_ohlcv(&mut self, data: &OHLCVData) -> Result<(), String> {
        if data.is_empty() {
            return Err("Cannot fit model on empty data".to_string());
        }
        
        let values = self.extract_target_data(data);
        let n = values.len();
        
        // Initialize the level with the first observation
        let mut level = values[0];
        
        // Prepare to store fitted values
        let mut fitted_values = Vec::with_capacity(n);
        fitted_values.push(level); // First fitted value is the initial level
        
        // Apply the SES model recursively
        for i in 1..n {
            // Calculate forecast for this step (which is just the previous level)
            let forecast = level;
            
            // Update the level based on the observed value
            level = self.alpha * values[i] + (1.0 - self.alpha) * level;
            
            // Store the forecast
            fitted_values.push(forecast);
        }
        
        // Store the final level and fitted values
        self.level = Some(level);
        self.fitted_values = Some(fitted_values);
        
        Ok(())
    }
    
    /// Generate forecasts for future periods.
    ///
    /// # Arguments
    /// * `horizon` - The number of periods to forecast ahead
    ///
    /// # Returns
    /// * `Result<Vec<f64>, String>` - Vector of forecasted values, or an error message if forecasting fails
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>, String> {
        if let Some(level) = self.level {
            // For SES, all future forecasts are just the last level
            Ok(vec![level; horizon])
        } else {
            Err("Model has not been fitted yet".to_string())
        }
    }
    
    /// Evaluate the model on test data.
    ///
    /// # Arguments
    /// * `test_data` - The test OHLCV data to evaluate against
    ///
    /// # Returns
    /// * `Result<ModelEvaluation, String>` - Evaluation metrics, or an error message if evaluation fails
    pub fn evaluate_ohlcv(&self, test_data: &OHLCVData) -> Result<ModelEvaluation, String> {
        if self.level.is_none() {
            return Err("Model has not been fitted yet".to_string());
        }
        
        let actual = self.extract_target_data(test_data);
        let forecast = self.forecast(actual.len())?;
        
        // Calculate error metrics
        let mae = mean_absolute_error(&actual, &forecast);
        let mse = mean_squared_error(&actual, &forecast);
        let rmse = mse.sqrt();
        let mape = mean_absolute_percentage_error(&actual, &forecast);
        let smape = symmetric_mean_absolute_percentage_error(&actual, &forecast);
        
        Ok(ModelEvaluation {
            model_name: self.name.clone(),
            mae,
            mse,
            rmse,
            mape,
            smape,
        })
    }
    
    /// Get the model name.
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the fitted values if available.
    pub fn fitted_values(&self) -> Option<&Vec<f64>> {
        self.fitted_values.as_ref()
    }
    
    /// Wrapper for the Forecaster trait's fit method.
    pub fn fit_impl(&mut self, data: &TimeSeriesData) -> OxiResult<()> {
        <Self as Forecaster>::fit(self, data)
    }
    
    /// Wrapper for the Forecaster trait's forecast method.
    pub fn forecast_impl(&self, horizon: usize) -> OxiResult<Vec<f64>> {
        <Self as Forecaster>::forecast(self, horizon)
    }
    
    /// Wrapper for the Forecaster trait's evaluate method.
    pub fn evaluate_impl(&self, test_data: &TimeSeriesData) -> OxiResult<ModelEvaluation> {
        <Self as Forecaster>::evaluate(self, test_data)
    }
    
    /// Wrapper for the Forecaster trait's predict method.
    pub fn predict_impl(&self, horizon: usize, test_data: Option<&TimeSeriesData>) -> OxiResult<ModelOutput> {
        <Self as Forecaster>::predict(self, horizon, test_data)
    }
}

// Error metrics functions
fn mean_absolute_error(actual: &[f64], forecast: &[f64]) -> f64 {
    let n = actual.len().min(forecast.len());
    if n == 0 {
        return 0.0;
    }
    
    let mut sum = 0.0;
    for i in 0..n {
        sum += (actual[i] - forecast[i]).abs();
    }
    
    sum / n as f64
}

fn mean_squared_error(actual: &[f64], forecast: &[f64]) -> f64 {
    let n = actual.len().min(forecast.len());
    if n == 0 {
        return 0.0;
    }
    
    let mut sum = 0.0;
    for i in 0..n {
        let error = actual[i] - forecast[i];
        sum += error * error;
    }
    
    sum / n as f64
}

fn mean_absolute_percentage_error(actual: &[f64], forecast: &[f64]) -> f64 {
    let n = actual.len().min(forecast.len());
    if n == 0 {
        return 0.0;
    }
    
    let mut sum = 0.0;
    let mut count = 0;
    
    for i in 0..n {
        if actual[i] != 0.0 {
            sum += ((actual[i] - forecast[i]).abs() / actual[i].abs()) * 100.0;
            count += 1;
        }
    }
    
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

fn symmetric_mean_absolute_percentage_error(actual: &[f64], forecast: &[f64]) -> f64 {
    let n = actual.len().min(forecast.len());
    if n == 0 {
        return 0.0;
    }
    
    let mut sum = 0.0;
    let mut count = 0;
    
    for i in 0..n {
        let abs_actual = actual[i].abs();
        let abs_forecast = forecast[i].abs();
        if abs_actual + abs_forecast > 0.0 {
            sum += 200.0 * (abs_actual - abs_forecast).abs() / (abs_actual + abs_forecast);
            count += 1;
        }
    }
    
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

impl Forecaster for SESModel {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn fit(&mut self, data: &TimeSeriesData) -> OxiResult<()> {
        if data.is_empty() {
            return Err(OxiError::Model("Cannot fit model on empty data".to_string()));
        }
        
        let values = &data.values;
        let n = values.len();
        
        // Initialize the level with the first observation
        let mut level = values[0];
        
        // Prepare to store fitted values
        let mut fitted_values = Vec::with_capacity(n);
        fitted_values.push(level); // First fitted value is the initial level
        
        // Apply the SES model recursively
        for i in 1..n {
            // Calculate forecast for this step (which is just the previous level)
            let forecast = level;
            
            // Update the level based on the observed value
            level = self.alpha * values[i] + (1.0 - self.alpha) * level;
            
            // Store the forecast
            fitted_values.push(forecast);
        }
        
        // Store the final level and fitted values
        self.level = Some(level);
        self.fitted_values = Some(fitted_values);
        
        Ok(())
    }
    
    fn forecast(&self, horizon: usize) -> OxiResult<Vec<f64>> {
        if let Some(level) = self.level {
            // For SES, all future forecasts are just the last level
            Ok(vec![level; horizon])
        } else {
            Err(OxiError::Model("Model has not been fitted yet".to_string()))
        }
    }
    
    fn evaluate(&self, test_data: &TimeSeriesData) -> OxiResult<ModelEvaluation> {
        if self.level.is_none() {
            return Err(OxiError::Model("Model has not been fitted yet".to_string()));
        }
        
        let actual = &test_data.values;
        let forecast = self.forecast(actual.len())?;
        
        // Calculate error metrics using utility functions
        let mae = crate::utils::mae(actual, &forecast);
        let mse = crate::utils::mse(actual, &forecast);
        let rmse = crate::utils::rmse(actual, &forecast);
        let mape = crate::utils::mape(actual, &forecast);
        let smape = crate::utils::smape(actual, &forecast);
        
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
    use crate::data::TimeSeriesData;
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
        
        // Create and fit the SES model
        let mut model = SESModel::new(0.3, None).unwrap();
        model.fit(&time_series).unwrap();
        
        // Forecast 5 periods ahead
        let forecast_horizon = 5;
        let forecasts = model.forecast(forecast_horizon).unwrap();
        
        // Check that the number of forecasts matches the horizon
        assert_eq!(forecasts.len(), forecast_horizon);
        
        // Check that all forecasts are the same value (last level)
        let first_forecast = forecasts[0];
        for forecast in forecasts.iter() {
            assert_eq!(*forecast, first_forecast);
        }
        
        // Now test the standardized ModelOutput from predict()
        let output = model.predict(forecast_horizon, None).unwrap();
        
        // Check basic properties of the ModelOutput
        assert_eq!(output.model_name, model.name());
        assert_eq!(output.forecasts.len(), forecast_horizon);
        assert_eq!(output.forecasts, forecasts);
        
        // Evaluation should be None since we didn't provide test data
        assert!(output.evaluation.is_none());
        
        // Now test with evaluation
        let output_with_eval = model.predict(forecast_horizon, Some(&time_series)).unwrap();
        
        // Should now have evaluation metrics
        assert!(output_with_eval.evaluation.is_some());
        let eval = output_with_eval.evaluation.unwrap();
        assert_eq!(eval.model_name, model.name());
        
        // Metrics should be non-negative
        assert!(eval.mae >= 0.0);
        assert!(eval.mse >= 0.0);
        assert!(eval.rmse >= 0.0);
    }
} 