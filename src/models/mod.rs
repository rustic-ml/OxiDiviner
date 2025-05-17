pub mod data;
pub mod exponential_smoothing;

// Re-export common data types
pub use data::OHLCVData;
pub use exponential_smoothing::*;

use crate::data::TimeSeriesData;
use crate::error::Result;

/// Trait for forecasting models
pub trait Forecaster {
    /// Get the name of the model
    fn name(&self) -> &str;
    
    /// Fit the model to training data
    fn fit(&mut self, data: &TimeSeriesData) -> Result<()>;
    
    /// Forecast future values
    fn forecast(&self, horizon: usize) -> Result<Vec<f64>>;
    
    /// Evaluate the model on test data
    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation>;
}

/// Represents model evaluation metrics
#[derive(Debug, Clone)]
pub struct ModelEvaluation {
    pub model_name: String,
    pub mae: f64,     // Mean Absolute Error
    pub mse: f64,     // Mean Squared Error
    pub rmse: f64,    // Root Mean Squared Error
    pub mape: f64,    // Mean Absolute Percentage Error
    pub smape: f64,   // Symmetric Mean Absolute Percentage Error
} 