/*!
# OxiDiviner Core

The foundation of the OxiDiviner time series forecasting ecosystem, providing data structures,
interfaces, and utility traits used across all forecasting models.

[![Crates.io](https://img.shields.io/crates/v/oxidiviner-core.svg)](https://crates.io/crates/oxidiviner-core)
[![Documentation](https://docs.rs/oxidiviner-core/badge.svg)](https://docs.rs/oxidiviner-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Core Components

### Data Structures

* [`TimeSeriesData`] - A flexible container for time series data with timestamps
* [`OHLCVData`] - A specialized container for financial time series (Open-High-Low-Close-Volume)

### Interfaces

* [`Forecaster`] - The central trait implemented by all forecasting models
* [`ModelEvaluation`] - Common metrics for evaluating forecast accuracy
* [`ModelOutput`] - Standard output format for forecasts and evaluations

## Usage Example

```rust
use oxidiviner_core::{TimeSeriesData, Forecaster, Result};
use chrono::{Utc, TimeZone};

// Create a custom forecasting model
struct SimpleAverageForecast {
    name: String,
    values: Vec<f64>,
}

impl SimpleAverageForecast {
    fn new() -> Self {
        Self {
            name: "Simple Average".to_string(),
            values: Vec::new(),
        }
    }
}

impl Forecaster for SimpleAverageForecast {
    fn name(&self) -> &str {
        &self.name
    }

    fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        self.values = data.values().to_vec();
        Ok(())
    }

    fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        if self.values.is_empty() {
            return Err(oxidiviner_core::OxiError::NotFitted);
        }

        // Calculate the average of all values
        let avg = self.values.iter().sum::<f64>() / self.values.len() as f64;

        // Return the average for each point in the forecast horizon
        Ok(vec![avg; horizon])
    }

    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<oxidiviner_core::ModelEvaluation> {
        // Implementation would calculate various error metrics
        // between forecasts and actual test data
        unimplemented!()
    }
}

fn example() -> Result<()> {
    // Create sample time series data
    let dates = vec![
        Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
        Utc.with_ymd_and_hms(2023, 1, 2, 0, 0, 0).unwrap(),
        Utc.with_ymd_and_hms(2023, 1, 3, 0, 0, 0).unwrap(),
    ];
    let values = vec![1.0, 2.0, 3.0];

    let data = TimeSeriesData::new(dates, values)?;

    // Create our model
    let mut model = SimpleAverageForecast::new();

    // Fit and forecast
    model.fit(&data)?;
    let forecast = model.forecast(2)?;

    println!("Forecast: {:?}", forecast);
    Ok(())
}
*/

use serde::{Deserialize, Serialize};

mod data;
mod error;

// Re-export the main components
pub use data::{OHLCVData, TimeSeriesData};
pub use error::{OxiError, Result};

/// Central trait that all forecasting models must implement
///
/// The Forecaster trait provides a common interface for time series models.
/// It includes methods for fitting models to data, generating forecasts,
/// and evaluating model performance.
pub trait Forecaster {
    /// Get the name of the model
    fn name(&self) -> &str;

    /// Fit the model to training data
    ///
    /// # Arguments
    ///
    /// * `data` - The time series data to fit the model to
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Success or an error if fitting fails
    fn fit(&mut self, data: &TimeSeriesData) -> Result<()>;

    /// Generate forecasts for the specified horizon
    ///
    /// # Arguments
    ///
    /// * `horizon` - The number of future time steps to forecast
    ///
    /// # Returns
    ///
    /// * `Result<Vec<f64>>` - The forecasted values or an error
    fn forecast(&self, horizon: usize) -> Result<Vec<f64>>;

    /// Evaluate the model on test data
    ///
    /// # Arguments
    ///
    /// * `test_data` - The time series data to evaluate against
    ///
    /// # Returns
    ///
    /// * `Result<ModelEvaluation>` - Evaluation metrics or an error
    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation>;

    /// Generate forecasts and evaluation in a standardized output format
    ///
    /// # Arguments
    ///
    /// * `horizon` - The number of future time steps to forecast
    /// * `test_data` - Optional test data for evaluation
    ///
    /// # Returns
    ///
    /// * `Result<ModelOutput>` - Standardized output or an error
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
