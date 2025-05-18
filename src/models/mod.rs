pub mod data;
pub mod exponential_smoothing;
pub mod moving_average;

// Re-export common data types
pub use data::OHLCVData;
pub use data::ModelOutput;
pub use exponential_smoothing::*;
pub use moving_average::*;

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
    
    /// Standard entry point for all models
    /// Returns a standardized ModelOutput containing forecasts and other useful information
    fn predict(&self, horizon: usize, test_data: Option<&TimeSeriesData>) -> Result<ModelOutput> {
        // Get forecast values
        let forecasts = self.forecast(horizon)?;
        
        // Create base output with forecasts
        let mut output = ModelOutput::new(self.name().to_string(), forecasts);
        
        // If test data is provided, add evaluation metrics
        if let Some(test_data) = test_data {
            let evaluation = self.evaluate(test_data)?;
            output = output.with_evaluation(evaluation);
        }
        
        Ok(output)
    }
}

/// Represents model evaluation metrics
#[derive(Debug, Clone)]
pub struct ModelEvaluation {
    /// Name of the model being evaluated
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