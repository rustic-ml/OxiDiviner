use crate::error::MAError;
use oxidiviner_core::{Forecaster, ModelEvaluation, ModelOutput, OxiError, Result, TimeSeriesData};
use oxidiviner_math::metrics::{mae, mape, mse, rmse, smape};
use oxidiviner_math::transforms::moving_average;

/// Moving Average model for time series forecasting.
///
/// This is a simple model that predicts future values
/// as the average of the last `window_size` observations.
pub struct MAModel {
    /// Model name
    name: String,
    /// Window size (number of past observations to average)
    window_size: usize,
    /// Last `window_size` values from the training data
    last_values: Option<Vec<f64>>,
    /// Fitted values over the training period
    fitted_values: Option<Vec<f64>>,
}

impl MAModel {
    /// Creates a new Moving Average model.
    ///
    /// # Arguments
    /// * `window_size` - Number of past observations to average
    ///
    /// # Returns
    /// * `Result<Self>` - A new model instance if parameters are valid
    pub fn new(window_size: usize) -> std::result::Result<Self, MAError> {
        // Validate parameters
        if window_size == 0 {
            return Err(MAError::InvalidWindowSize(window_size));
        }

        let name = format!("MA({})", window_size);

        Ok(MAModel {
            name,
            window_size,
            last_values: None,
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
    pub fn predict(
        &self,
        horizon: usize,
        test_data: Option<&TimeSeriesData>,
    ) -> Result<ModelOutput> {
        <Self as Forecaster>::predict(self, horizon, test_data)
    }

    /// Get the fitted values if available.
    pub fn fitted_values(&self) -> Option<&Vec<f64>> {
        self.fitted_values.as_ref()
    }
}

impl Forecaster for MAModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        if data.is_empty() {
            return Err(OxiError::from(MAError::EmptyData));
        }

        let n = data.values.len();

        if n < self.window_size {
            return Err(OxiError::from(MAError::TimeSeriesTooShort {
                actual: n,
                expected: self.window_size,
            }));
        }

        // Calculate fitted values
        let ma_values = moving_average(&data.values, self.window_size);
        let mut fitted_values = vec![f64::NAN; self.window_size - 1];
        fitted_values.extend(ma_values);

        // Store the last window_size values for forecasting
        self.last_values = Some(data.values[n - self.window_size..].to_vec());
        self.fitted_values = Some(fitted_values);

        Ok(())
    }

    fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        if let Some(last_values) = &self.last_values {
            if horizon == 0 {
                return Err(OxiError::from(MAError::InvalidHorizon(horizon)));
            }

            // For MA model, all forecasts are the same value: the average of the last window_size values
            let forecast_value = last_values.iter().sum::<f64>() / last_values.len() as f64;

            // Return the same value for all horizons
            Ok(vec![forecast_value; horizon])
        } else {
            Err(OxiError::from(MAError::NotFitted))
        }
    }

    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        if self.last_values.is_none() {
            return Err(OxiError::from(MAError::NotFitted));
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

    // Use the default predict implementation from the trait
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, TimeZone, Utc};

    #[test]
    fn test_ma_forecast() {
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

        // All forecasts should be the average of the last 3 values: (8+9+10)/3 = 9
        let expected_forecast = 9.0;
        for forecast in forecasts.iter() {
            assert!((forecast - expected_forecast).abs() < 1e-6);
        }

        // Test the standardized ModelOutput from predict()
        let output = model.predict(forecast_horizon, None).unwrap();

        // Check basic properties of the ModelOutput
        assert_eq!(output.model_name, model.name());
        assert_eq!(output.forecasts.len(), forecast_horizon);

        // Test with evaluation
        let output_with_eval = model.predict(forecast_horizon, Some(&time_series)).unwrap();

        // Should have evaluation metrics
        assert!(output_with_eval.evaluation.is_some());
        let eval = output_with_eval.evaluation.unwrap();
        assert_eq!(eval.model_name, model.name());
    }

    #[test]
    fn test_ma_fitted_values() {
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let time_series = TimeSeriesData::new(timestamps, values, "test_series").unwrap();

        // MA model with window size 3
        let mut model = MAModel::new(3).unwrap();
        model.fit(&time_series).unwrap();

        // First 2 fitted values should be NaN (window_size - 1)
        let fitted_values = model.fitted_values().unwrap();
        assert!(fitted_values[0].is_nan());
        assert!(fitted_values[1].is_nan());

        // Check subsequent values
        // MA(3) at index 2 = (1+2+3)/3 = 2
        assert!((fitted_values[2] - 2.0).abs() < 1e-6);

        // MA(3) at index 3 = (2+3+4)/3 = 3
        assert!((fitted_values[3] - 3.0).abs() < 1e-6);

        // MA(3) at index 9 = (8+9+10)/3 = 9
        assert!((fitted_values[9] - 9.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_ma_invalid_parameters() {
        // Test with invalid window size
        let result = MAModel::new(0);
        assert!(result.is_err());
        
        if let Err(MAError::InvalidWindowSize(size)) = result {
            assert_eq!(size, 0);
        } else {
            panic!("Expected InvalidWindowSize error");
        }
    }
    
    #[test]
    fn test_ma_fit_errors() {
        // Create a model
        let mut model = MAModel::new(3).unwrap();
        
        // Test with insufficient data
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..2)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();
        
        let values = vec![1.0, 2.0]; // Only 2 values, but window size is 3
        let short_data = TimeSeriesData::new(timestamps, values, "short").unwrap();
        
        let result = model.fit(&short_data);
        assert!(result.is_err());
        
        if let Err(err) = result {
            // Check the error message contains the right information
            let err_msg = format!("{:?}", err);
            assert!(err_msg.contains("series length"));
            assert!(err_msg.contains("2"));  // actual length
            assert!(err_msg.contains("3"));  // expected length
        }
    }
    
    #[test]
    fn test_ma_forecast_errors() {
        // Test forecasting with unfitted model
        let model = MAModel::new(3).unwrap();
        let result = model.forecast(5);
        assert!(result.is_err());
        
        if let Err(err) = result {
            match err {
                OxiError::DataError(_) |
                OxiError::ModelError(_) => {
                    // This is fine - we just want to check that forecasting fails
                }
                _ => panic!("Unexpected error type: {:?}", err),
            }
        }
        
        // Test invalid forecast horizon
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();
        
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let data = TimeSeriesData::new(timestamps, values, "test").unwrap();
        
        let mut fitted_model = MAModel::new(3).unwrap();
        fitted_model.fit(&data).unwrap();
        
        let result = fitted_model.forecast(0);
        assert!(result.is_err());
        
        if let Err(err) = result {
            // We don't check the specific error type since it's wrapped in OxiError
            // Just make sure it failed due to the horizon being 0
            assert!(format!("{:?}", err).contains("0"));
        }
    }
    
    #[test]
    fn test_ma_evaluate_errors() {
        // Test evaluating with unfitted model
        let model = MAModel::new(3).unwrap();
        
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..5)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();
        
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let test_data = TimeSeriesData::new(timestamps, values, "test").unwrap();
        
        let result = model.evaluate(&test_data);
        assert!(result.is_err());
        
        if let Err(err) = result {
            // Check that we got an error, but don't verify specific error type
            // since it's wrapped in OxiError
            assert!(format!("{:?}", err).contains("fitted"));
        }
    }
    
    #[test]
    fn test_ma_constant_data() {
        // Test with constant data
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();
        
        let values = vec![5.0; 10]; // All values are 5.0
        let data = TimeSeriesData::new(timestamps, values, "constant").unwrap();
        
        let mut model = MAModel::new(3).unwrap();
        model.fit(&data).unwrap();
        
        // Forecast should be constant
        let forecasts = model.forecast(5).unwrap();
        for forecast in forecasts {
            assert!((forecast - 5.0).abs() < 1e-6);
        }
    }
}
