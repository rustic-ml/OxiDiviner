use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod data;
mod error;

// Re-export the main components
pub use data::{OHLCVData, TimeSeriesData};
pub use error::{OxiError, Result};

// Define the Forecaster trait for time series forecasting models
pub trait Forecaster {
    /// Get the name of the model
    fn name(&self) -> &str;
    
    /// Fit the model to training data
    fn fit(&mut self, data: &TimeSeriesData) -> Result<()>;
    
    /// Generate forecasts for the specified horizon
    fn forecast(&self, horizon: usize) -> Result<Vec<f64>>;
    
    /// Evaluate the model on test data
    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation>;
    
    /// Generate forecasts and evaluation in a standardized output format
    fn predict(&self, horizon: usize, test_data: Option<&TimeSeriesData>) -> Result<ModelOutput> {
        // Generate forecasts
        let forecasts = self.forecast(horizon)?;
        
        // If test data is provided, evaluate the model
        let evaluation = if let Some(test_data) = test_data {
            Some(self.evaluate(test_data)?)
        } else {
            None
        };
        
        Ok(ModelOutput {
            model_name: self.name().to_string(),
            forecasts,
            evaluation,
        })
    }
}

/// Model evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEvaluation {
    /// Name of the model
    pub model_name: String,
    /// Mean Absolute Error
    pub mae: f64,
    /// Mean Squared Error
    pub mse: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Percentage Error
    pub mape: f64,
    /// Symmetric Mean Absolute Percentage Error
    pub smape: f64,
}

/// Standardized output from a forecasting model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutput {
    /// Name of the model
    pub model_name: String,
    /// Forecasted values
    pub forecasts: Vec<f64>,
    /// Optional evaluation metrics (if test data was provided)
    pub evaluation: Option<ModelEvaluation>,
}