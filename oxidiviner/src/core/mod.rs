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
use crate::core::{TimeSeriesData, Forecaster, Result};
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
        self.values = data.values.clone();
        Ok(())
    }

    fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        if self.values.is_empty() {
            return Err(crate::core::OxiError::ModelError("Model not fitted".to_string()));
        }

        // Calculate the average of all values
        let avg = self.values.iter().sum::<f64>() / self.values.len() as f64;

        // Return the average for each point in the forecast horizon
        Ok(vec![avg; horizon])
    }

    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<crate::core::ModelEvaluation> {
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

    let data = TimeSeriesData::new(dates, values, "sample_data")?;

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

pub mod data;
pub mod diagnostics;
pub mod error;
pub mod persistence;
pub mod streaming;
pub mod validation;
// pub mod optimization; // Temporarily disabled due to enum visibility issues

// Re-export the main components
pub use data::{OHLCVData, TimeSeriesData};
pub use diagnostics::{
    DiagnosticReport, ForecastDiagnostics, ModelDiagnostics, ResidualAnalysis, SpecificationTests,
    TestResult,
};
pub use error::{OxiError, Result};
pub use persistence::{ModelPersistence, ModelState, Persistable, PersistedModel};
pub use streaming::{batch_process, RunningStats, StreamingBuffer, StreamingProcessor};
pub use validation::{
    AccuracyReport, BacktestConfig, BacktestResult, ModelValidator, ValidationUtils,
};

/// The central trait that all forecasting models must implement
///
/// The Forecaster trait provides a common interface for time series models.
/// All forecasting models in OxiDiviner implement this trait, ensuring
/// consistency and enabling polymorphic usage.
///
/// # Basic Usage Pattern
///
/// 1. Create a model instance
/// 2. Fit the model to training data using `fit()`
/// 3. Generate forecasts using `forecast()` or `predict()`
/// 4. Evaluate the model using `evaluate()`
///
/// # Example
///
/// ```rust,ignore
/// use oxidiviner::core::{Forecaster, TimeSeriesData};
/// use oxidiviner::models::ARModel;
///
/// let mut model = ARModel::new(2)?; // AR(2) model
/// model.fit(&training_data)?;
/// let forecasts = model.forecast(5)?; // Forecast 5 periods
/// let evaluation = model.evaluate(&test_data)?;
/// ```
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
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
    /// Akaike Information Criterion
    pub aic: Option<f64>,
    /// Bayesian Information Criterion
    pub bic: Option<f64>,
}

/// Standardized output from a forecasting model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutput {
    /// Name of the model
    pub model_name: String,
    /// Forecasted values
    pub forecasts: Vec<f64>,
    /// Model evaluation metrics (if test data was provided)
    pub evaluation: Option<ModelEvaluation>,
}

/// Unified trait for quick forecasting with consistent interface across all models
pub trait QuickForecaster: std::fmt::Debug {
    /// Fit the model to time series data
    fn quick_fit(&mut self, data: &TimeSeriesData) -> Result<()>;

    /// Generate forecasts for the specified number of periods
    fn quick_forecast(&self, periods: usize) -> Result<Vec<f64>>;

    /// Get the name of the model type
    fn model_name(&self) -> &'static str;

    /// Get fitted values if available
    fn fitted_values(&self) -> Option<Vec<f64>> {
        None
    }

    /// Evaluate the model on test data
    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation>;
}

/// Enhanced forecasting trait with confidence intervals
pub trait ConfidenceForecaster: QuickForecaster {
    /// Generate forecasts with confidence intervals
    fn forecast_with_confidence(&self, periods: usize, confidence: f64) -> Result<ForecastResult>;
}

/// Trait for cloning QuickForecaster trait objects
pub trait CloneableQuickForecaster: QuickForecaster {
    fn clone_box(&self) -> Box<dyn CloneableQuickForecaster>;
}

impl<T> CloneableQuickForecaster for T
where
    T: 'static + QuickForecaster + Clone,
{
    fn clone_box(&self) -> Box<dyn CloneableQuickForecaster> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn CloneableQuickForecaster> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Result of a forecast with optional confidence intervals
#[derive(Debug, Clone)]
pub struct ForecastResult {
    /// Point forecasts
    pub point_forecast: Vec<f64>,
    /// Lower confidence bound
    pub lower_bound: Option<Vec<f64>>,
    /// Upper confidence bound
    pub upper_bound: Option<Vec<f64>>,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: Option<f64>,
    /// Model name used for forecasting
    pub model_name: String,
}

impl ForecastResult {
    /// Create a new forecast result with just point forecasts
    pub fn new(point_forecast: Vec<f64>, model_name: String) -> Self {
        Self {
            point_forecast,
            lower_bound: None,
            upper_bound: None,
            confidence_level: None,
            model_name,
        }
    }

    /// Create a new forecast result with confidence intervals
    pub fn with_confidence(
        point_forecast: Vec<f64>,
        lower_bound: Vec<f64>,
        upper_bound: Vec<f64>,
        confidence_level: f64,
        model_name: String,
    ) -> Self {
        Self {
            point_forecast,
            lower_bound: Some(lower_bound),
            upper_bound: Some(upper_bound),
            confidence_level: Some(confidence_level),
            model_name,
        }
    }

    /// Check if confidence intervals are available
    pub fn has_confidence_intervals(&self) -> bool {
        self.lower_bound.is_some() && self.upper_bound.is_some()
    }
}

/// Builder pattern for creating forecasting models with fluent interface
///
/// This provides a convenient way to configure and create models with
/// a clean, readable syntax.
///
/// # Example
/// ```rust
/// # use crate::core::{ModelBuilder, Result};
/// # fn example() -> Result<()> {
/// let config = ModelBuilder::arima()
///     .with_ar(2)
///     .with_differencing(1)
///     .with_ma(1)
///     .build_config();
/// # Ok(())
/// # }
/// ```
pub struct ModelBuilder {
    model_type: String,
    parameters: std::collections::HashMap<String, f64>,
}

impl ModelBuilder {
    /// Start building an ARIMA model
    pub fn arima() -> Self {
        Self {
            model_type: "ARIMA".to_string(),
            parameters: std::collections::HashMap::new(),
        }
    }

    /// Start building an AR model
    pub fn ar() -> Self {
        Self {
            model_type: "AR".to_string(),
            parameters: std::collections::HashMap::new(),
        }
    }

    /// Start building a Moving Average model
    pub fn moving_average() -> Self {
        Self {
            model_type: "MA".to_string(),
            parameters: std::collections::HashMap::new(),
        }
    }

    /// Start building an Exponential Smoothing model
    pub fn exponential_smoothing() -> Self {
        Self {
            model_type: "ES".to_string(),
            parameters: std::collections::HashMap::new(),
        }
    }

    /// Start building a GARCH model
    pub fn garch() -> Self {
        Self {
            model_type: "GARCH".to_string(),
            parameters: std::collections::HashMap::new(),
        }
    }

    /// Set autoregressive order (for ARIMA/AR models)
    pub fn with_ar(mut self, p: usize) -> Self {
        self.parameters.insert("p".to_string(), p as f64);
        self
    }

    /// Set differencing order (for ARIMA models)
    pub fn with_differencing(mut self, d: usize) -> Self {
        self.parameters.insert("d".to_string(), d as f64);
        self
    }

    /// Set moving average order (for ARIMA models)
    pub fn with_ma(mut self, q: usize) -> Self {
        self.parameters.insert("q".to_string(), q as f64);
        self
    }

    /// Set window size (for MA models)
    pub fn with_window(mut self, window: usize) -> Self {
        self.parameters.insert("window".to_string(), window as f64);
        self
    }

    /// Set alpha parameter (for ES models)
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.parameters.insert("alpha".to_string(), alpha);
        self
    }

    /// Set beta parameter (for ES models with trend)
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.parameters.insert("beta".to_string(), beta);
        self
    }

    /// Set gamma parameter (for ES models with seasonality)
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.parameters.insert("gamma".to_string(), gamma);
        self
    }

    /// Set GARCH order (for GARCH models)
    pub fn with_garch_order(mut self, p: usize) -> Self {
        self.parameters.insert("garch_p".to_string(), p as f64);
        self
    }

    /// Set ARCH order (for GARCH models)
    pub fn with_arch_order(mut self, q: usize) -> Self {
        self.parameters.insert("arch_q".to_string(), q as f64);
        self
    }

    /// Add a custom parameter
    pub fn with_parameter(mut self, name: &str, value: f64) -> Self {
        self.parameters.insert(name.to_string(), value);
        self
    }

    /// Get the model type
    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    /// Get the parameters
    pub fn parameters(&self) -> &std::collections::HashMap<String, f64> {
        &self.parameters
    }

    /// Build the model (placeholder - actual implementation would depend on available models)
    ///
    /// Note: The actual build() method would need to be implemented in the main crate
    /// where all model types are available.
    pub fn build_config(self) -> ModelConfig {
        ModelConfig {
            model_type: self.model_type,
            parameters: self.parameters,
        }
    }
}

/// Configuration for a forecasting model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Type of model (e.g., "ARIMA", "AR", "MA", "ES", "GARCH")
    pub model_type: String,
    /// Model parameters
    pub parameters: std::collections::HashMap<String, f64>,
}

/// Criteria for automatic model selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionCriteria {
    /// Akaike Information Criterion
    AIC,
    /// Bayesian Information Criterion  
    BIC,
    /// Cross-validation with specified number of folds
    CrossValidation { folds: usize },
    /// Hold-out validation with specified test size
    HoldOut { test_ratio: f64 },
    /// Multiple criteria combined
    Combined(Vec<SelectionCriteria>),
}

/// Smart model selection for automatic forecasting
///
/// This utility tries multiple models and selects the best one based
/// on specified criteria.
pub struct AutoSelector {
    criteria: SelectionCriteria,
    candidate_models: Vec<ModelConfig>,
}

impl AutoSelector {
    /// Create a new auto selector with AIC criterion
    pub fn with_aic() -> Self {
        Self {
            criteria: SelectionCriteria::AIC,
            candidate_models: Self::default_candidates(),
        }
    }

    /// Create a new auto selector with BIC criterion
    pub fn with_bic() -> Self {
        Self {
            criteria: SelectionCriteria::BIC,
            candidate_models: Self::default_candidates(),
        }
    }

    /// Create a new auto selector with cross-validation
    pub fn with_cross_validation(folds: usize) -> Self {
        Self {
            criteria: SelectionCriteria::CrossValidation { folds },
            candidate_models: Self::default_candidates(),
        }
    }

    /// Create a new auto selector with hold-out validation
    pub fn with_hold_out(test_ratio: f64) -> Self {
        Self {
            criteria: SelectionCriteria::HoldOut { test_ratio },
            candidate_models: Self::default_candidates(),
        }
    }

    /// Add a custom model to the candidates
    pub fn add_candidate(mut self, config: ModelConfig) -> Self {
        self.candidate_models.push(config);
        self
    }

    /// Set custom candidate models (replaces defaults)
    pub fn with_candidates(mut self, candidates: Vec<ModelConfig>) -> Self {
        self.candidate_models = candidates;
        self
    }

    /// Get the selection criteria
    pub fn criteria(&self) -> &SelectionCriteria {
        &self.criteria
    }

    /// Get the candidate models
    pub fn candidates(&self) -> &[ModelConfig] {
        &self.candidate_models
    }

    /// Default set of candidate models to try
    fn default_candidates() -> Vec<ModelConfig> {
        vec![
            // ARIMA models
            ModelBuilder::arima()
                .with_ar(1)
                .with_differencing(1)
                .with_ma(1)
                .build_config(),
            ModelBuilder::arima()
                .with_ar(2)
                .with_differencing(1)
                .with_ma(1)
                .build_config(),
            ModelBuilder::arima()
                .with_ar(1)
                .with_differencing(1)
                .with_ma(2)
                .build_config(),
            // AR models
            ModelBuilder::ar().with_ar(1).build_config(),
            ModelBuilder::ar().with_ar(2).build_config(),
            ModelBuilder::ar().with_ar(3).build_config(),
            // MA models
            ModelBuilder::moving_average().with_window(3).build_config(),
            ModelBuilder::moving_average().with_window(5).build_config(),
            ModelBuilder::moving_average().with_window(7).build_config(),
            // ES models
            ModelBuilder::exponential_smoothing()
                .with_alpha(0.3)
                .build_config(),
            ModelBuilder::exponential_smoothing()
                .with_alpha(0.5)
                .build_config(),
            ModelBuilder::exponential_smoothing()
                .with_alpha(0.7)
                .build_config(),
        ]
    }
}

/// Result of model selection process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionResult {
    /// Best model configuration
    pub best_model: ModelConfig,
    /// Score of the best model (lower is better for information criteria)
    pub best_score: f64,
    /// All model results sorted by score
    pub all_results: Vec<(ModelConfig, f64)>,
    /// Criteria used for selection
    pub criteria: SelectionCriteria,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forecast_result_creation() {
        let forecasts = vec![1.0, 2.0, 3.0];
        let result = ForecastResult::new(forecasts.clone(), "TestModel".to_string());

        assert_eq!(result.point_forecast, forecasts);
        assert!(result.lower_bound.is_none());
        assert!(result.upper_bound.is_none());
        assert!(!result.has_confidence_intervals());
        assert_eq!(result.model_name, "TestModel");
    }

    #[test]
    fn test_forecast_result_with_confidence() {
        let forecasts = vec![1.0, 2.0, 3.0];
        let lower = vec![0.5, 1.5, 2.5];
        let upper = vec![1.5, 2.5, 3.5];

        let result = ForecastResult::with_confidence(
            forecasts.clone(),
            lower.clone(),
            upper.clone(),
            0.95,
            "TestModel".to_string(),
        );

        assert_eq!(result.point_forecast, forecasts);
        assert_eq!(result.lower_bound, Some(lower));
        assert_eq!(result.upper_bound, Some(upper));
        assert!(result.has_confidence_intervals());
        assert_eq!(result.model_name, "TestModel");
    }

    #[test]
    fn test_model_evaluation_creation() {
        let eval = ModelEvaluation {
            model_name: "TestModel".to_string(),
            mae: 1.0,
            mse: 2.0,
            rmse: 1.414,
            mape: 10.0,
            smape: 15.0,
            r_squared: 0.95,
            aic: Some(-100.0),
            bic: Some(-95.0),
        };

        assert_eq!(eval.model_name, "TestModel");
        assert_eq!(eval.mae, 1.0);
        assert_eq!(eval.r_squared, 0.95);
        assert_eq!(eval.aic, Some(-100.0));
    }
}
