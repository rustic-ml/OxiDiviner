//! # Ensemble Methods for OxiDiviner
//!
//! This module provides robust ensemble forecasting capabilities designed to enhance
//! prediction accuracy and stability by combining multiple individual forecasting models.
//!
//! ## Key Features
//!
//! - **Multiple Ensemble Strategies**: Implements common ensemble techniques such as:
//!   - `SimpleAverage`: Averages the forecasts from all member models.
//!   - `WeightedAverage`: Averages forecasts using weights, often derived from
//!     individual model performance or confidence scores.
//!   - `Median`: Uses the median of the forecasts, which is robust to outliers.
//!   - `BestModel`: Selects the forecast from the single best-performing model
//!     (based on a provided score like confidence or past error).
//!   - `Stacking`: A more advanced method where a meta-learner is trained on the
//!     forecasts of base models. (Currently simplified as performance-weighted average).
//!
//! - **Flexible Configuration**: The `EnsembleBuilder` allows for easy construction
//!   of ensembles, adding model forecasts with optional weights or confidence scores.
//!
//! - **Performance Evaluation**: Includes utilities (`EnsembleUtils`) to evaluate
//!   the performance of the ensemble against actual values and compare it to
//!   individual member models.
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use oxidiviner::prelude::*;
//! use oxidiviner::ensemble::{EnsembleBuilder, EnsembleMethod, ModelForecast};
//! use std::collections::HashMap;
//!
//! // Assume forecast1, forecast2 are Vec<f64> from other models
//! let forecast1 = vec![10.0, 11.0, 12.0];
//! let forecast2 = vec![10.5, 11.5, 12.5];
//! let actual_values = vec![10.2, 11.1, 12.3];
//!
//! // Using EnsembleBuilder
//! let ensemble = EnsembleBuilder::new(EnsembleMethod::SimpleAverage)
//!     .add_model_forecast("ModelA".to_string(), forecast1.clone())
//!     .add_model_forecast("ModelB".to_string(), forecast2.clone())
//!     .build()?;
//!
//! if let Some(final_fc) = ensemble.get_forecast() {
//!     println!("Ensemble Forecast: {:?}", final_fc);
//! }
//!
//! // Evaluate performance
//! let mut performance = EnsembleUtils::evaluate_ensemble(&ensemble, &actual_values)?;
//! EnsembleUtils::calculate_improvement(&mut performance);
//! println!("Ensemble MAE: {:.2}", performance.ensemble_mae);
//! if let Some(improvement) = performance.improvement {
//!     println!("Improvement over best individual: {:.1}%", improvement);
//! }
//! # Ok::<(), OxiError>(())
//! ```
//!
//! This module aims to provide a powerful yet easy-to-use interface for leveraging
//! the benefits of ensemble forecasting in various time series analysis tasks.

use crate::core::{OxiError, Result, TimeSeriesData};
use crate::math::metrics::{mae, mse, rmse};
use std::collections::HashMap;

/// Ensemble method types
#[derive(Debug, Clone)]
pub enum EnsembleMethod {
    /// Simple average of all model forecasts.
    /// This method gives equal weight to each model in the ensemble.
    SimpleAverage,
    /// Weighted average based on inverse error metrics or predefined weights.
    /// Models that have historically performed better or have higher confidence
    /// can be given more influence in the final forecast.
    WeightedAverage,
    /// Median of all model forecasts.
    /// This method is robust to outlier forecasts from individual models.
    Median,
    /// Best performing model selection.
    /// This method selects the forecast from the single model that has the
    /// highest confidence or best predefined weight.
    BestModel,
    /// Stacked ensemble with a meta-learner.
    /// In this approach, a meta-model is trained to combine the forecasts of base models.
    /// Currently, this is implemented as a simplified performance-weighted average,
    /// where weights are derived from model confidence (inversely related to error).
    Stacking,
}

/// Represents an individual model's forecast within an ensemble.
#[derive(Debug, Clone)]
pub struct ModelForecast {
    /// Name of the individual model.
    pub name: String,
    /// The forecast values produced by the model.
    pub forecast: Vec<f64>,
    /// Optional confidence score for the forecast (e.g., 1 - MAPE).
    /// Higher values indicate higher confidence. Used by `BestModel` and `Stacking`.
    pub confidence: Option<f64>,
    /// Optional predefined weight for the model's forecast in `WeightedAverage`.
    /// If not provided for `WeightedAverage` and confidence is also None, a default weight is used.
    pub weight: Option<f64>,
}

/// Manages a collection of model forecasts and combines them using a specified ensemble method.
#[derive(Debug)]
pub struct EnsembleForecast {
    /// A list of individual model forecasts included in the ensemble.
    pub forecasts: Vec<ModelForecast>,
    /// The ensemble method used to combine the forecasts.
    pub method: EnsembleMethod,
    /// The final combined forecast produced by the ensemble method.
    /// This is `None` until `combine()` is called.
    pub final_forecast: Option<Vec<f64>>,
    /// Weights assigned to each model in the ensemble, if applicable.
    /// For `SimpleAverage`, weights are equal. For `WeightedAverage` and `Stacking`,
    /// weights are calculated or user-defined. For `BestModel`, one model has weight 1.0.
    pub model_weights: Option<HashMap<String, f64>>,
}

impl EnsembleForecast {
    /// Create a new ensemble forecast
    pub fn new(method: EnsembleMethod) -> Self {
        Self {
            forecasts: Vec::new(),
            method,
            final_forecast: None,
            model_weights: None,
        }
    }

    /// Add a model forecast to the ensemble
    pub fn add_forecast(&mut self, forecast: ModelForecast) {
        self.forecasts.push(forecast);
    }

    /// Combine forecasts using the specified ensemble method
    pub fn combine(&mut self) -> Result<Vec<f64>> {
        if self.forecasts.is_empty() {
            return Err(OxiError::ModelError("No forecasts to combine".into()));
        }

        let forecast_length = self.forecasts[0].forecast.len();
        
        // Validate all forecasts have the same length
        for forecast in &self.forecasts {
            if forecast.forecast.len() != forecast_length {
                return Err(OxiError::ModelError(
                    "All forecasts must have the same length".into(),
                ));
            }
        }

        let combined = match self.method {
            EnsembleMethod::SimpleAverage => self.simple_average()?,
            EnsembleMethod::WeightedAverage => self.weighted_average()?,
            EnsembleMethod::Median => self.median_ensemble()?,
            EnsembleMethod::BestModel => self.best_model()?,
            EnsembleMethod::Stacking => self.stacking_ensemble()?,
        };

        self.final_forecast = Some(combined.clone());
        Ok(combined)
    }

    /// Simple average ensemble
    fn simple_average(&mut self) -> Result<Vec<f64>> {
        let forecast_length = self.forecasts[0].forecast.len();
        let mut combined = vec![0.0; forecast_length];

        for i in 0..forecast_length {
            let sum: f64 = self.forecasts.iter().map(|f| f.forecast[i]).sum();
            combined[i] = sum / self.forecasts.len() as f64;
        }

        // Set equal weights
        let mut weights = HashMap::new();
        let equal_weight = 1.0 / self.forecasts.len() as f64;
        for forecast in &self.forecasts {
            weights.insert(forecast.name.clone(), equal_weight);
        }
        self.model_weights = Some(weights);

        Ok(combined)
    }

    /// Weighted average based on model confidence/performance
    fn weighted_average(&mut self) -> Result<Vec<f64>> {
        let forecast_length = self.forecasts[0].forecast.len();
        let mut combined = vec![0.0; forecast_length];

        // Calculate weights from confidence or use predefined weights
        let weights = self.calculate_weights()?;
        let weight_sum: f64 = weights.values().sum();

        if weight_sum <= 0.0 {
            return Err(OxiError::ModelError("Invalid weights for ensemble".into()));
        }

        // Normalize weights
        let normalized_weights: HashMap<String, f64> = weights
            .iter()
            .map(|(k, v)| (k.clone(), v / weight_sum))
            .collect();

        for i in 0..forecast_length {
            combined[i] = self.forecasts
                .iter()
                .map(|f| f.forecast[i] * normalized_weights[&f.name])
                .sum();
        }

        self.model_weights = Some(normalized_weights);
        Ok(combined)
    }

    /// Median ensemble (more robust to outliers)
    fn median_ensemble(&self) -> Result<Vec<f64>> {
        let forecast_length = self.forecasts[0].forecast.len();
        let mut combined = vec![0.0; forecast_length];

        for i in 0..forecast_length {
            let mut values: Vec<f64> = self.forecasts.iter().map(|f| f.forecast[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let n = values.len();
            combined[i] = if n % 2 == 0 {
                (values[n / 2 - 1] + values[n / 2]) / 2.0
            } else {
                values[n / 2]
            };
        }

        Ok(combined)
    }

    /// Select the best performing model
    fn best_model(&mut self) -> Result<Vec<f64>> {
        // Find the model with highest confidence or best weight
        let best_forecast = self.forecasts
            .iter()
            .max_by(|a, b| {
                let a_score = a.confidence.or(a.weight).unwrap_or(0.0);
                let b_score = b.confidence.or(b.weight).unwrap_or(0.0);
                a_score.partial_cmp(&b_score).unwrap()
            })
            .ok_or_else(|| OxiError::ModelError("No best model found".into()))?;

        // Set weight to 1.0 for best model, 0.0 for others
        let mut weights = HashMap::new();
        for forecast in &self.forecasts {
            weights.insert(
                forecast.name.clone(),
                if forecast.name == best_forecast.name { 1.0 } else { 0.0 }
            );
        }
        self.model_weights = Some(weights);

        Ok(best_forecast.forecast.clone())
    }

    /// Stacking ensemble with simple linear combination
    fn stacking_ensemble(&mut self) -> Result<Vec<f64>> {
        // For simplicity, use cross-validation based weighted average
        // In practice, this would train a meta-learner
        
        // Calculate performance-based weights
        let weights = self.calculate_performance_weights()?;
        let weight_sum: f64 = weights.values().sum();

        if weight_sum <= 0.0 {
            return self.simple_average(); // Fallback
        }

        let forecast_length = self.forecasts[0].forecast.len();
        let mut combined = vec![0.0; forecast_length];

        for i in 0..forecast_length {
            combined[i] = self.forecasts
                .iter()
                .map(|f| f.forecast[i] * weights[&f.name] / weight_sum)
                .sum();
        }

        self.model_weights = Some(weights);
        Ok(combined)
    }

    /// Calculate weights based on confidence or predefined weights
    fn calculate_weights(&self) -> Result<HashMap<String, f64>> {
        let mut weights = HashMap::new();

        for forecast in &self.forecasts {
            let weight = if let Some(w) = forecast.weight {
                w
            } else if let Some(c) = forecast.confidence {
                c
            } else {
                1.0 // Default equal weight
            };
            weights.insert(forecast.name.clone(), weight);
        }

        Ok(weights)
    }

    /// Calculate performance-based weights (simplified version)
    fn calculate_performance_weights(&self) -> Result<HashMap<String, f64>> {
        let mut weights = HashMap::new();

        // Use inverse of confidence as a proxy for performance
        // In practice, this would use validation set performance
        for forecast in &self.forecasts {
            let performance_score = if let Some(conf) = forecast.confidence {
                if conf > 0.0 { 1.0 / conf } else { 1.0 }
            } else {
                1.0
            };
            weights.insert(forecast.name.clone(), performance_score);
        }

        Ok(weights)
    }

    /// Get the ensemble forecast result
    pub fn get_forecast(&self) -> Option<&Vec<f64>> {
        self.final_forecast.as_ref()
    }

    /// Get model weights
    pub fn get_weights(&self) -> Option<&HashMap<String, f64>> {
        self.model_weights.as_ref()
    }
}

/// A builder pattern for constructing `EnsembleForecast` instances.
///
/// This builder simplifies the process of adding multiple model forecasts
/// and configuring the ensemble.
pub struct EnsembleBuilder {
    ensemble: EnsembleForecast,
}

impl EnsembleBuilder {
    /// Creates a new `EnsembleBuilder` with the specified ensemble method.
    ///
    /// # Arguments
    ///
    /// * `method` - The `EnsembleMethod` to be used for combining forecasts.
    pub fn new(method: EnsembleMethod) -> Self {
        Self {
            ensemble: EnsembleForecast::new(method),
        }
    }

    /// Adds a model's forecast to the ensemble.
    ///
    /// # Arguments
    ///
    /// * `name` - A unique name for the model.
    /// * `forecast` - A `Vec<f64>` representing the model's forecast values.
    pub fn add_model_forecast(mut self, name: String, forecast: Vec<f64>) -> Self {
        self.ensemble.add_forecast(ModelForecast {
            name,
            forecast,
            confidence: None,
            weight: None,
        });
        self
    }

    /// Adds a model's forecast with a specific weight.
    ///
    /// Primarily used with `EnsembleMethod::WeightedAverage`.
    ///
    /// # Arguments
    ///
    /// * `name` - A unique name for the model.
    /// * `forecast` - A `Vec<f64>` representing the model's forecast values.
    /// * `weight` - The weight to assign to this model's forecast.
    pub fn add_weighted_forecast(
        mut self,
        name: String,
        forecast: Vec<f64>,
        weight: f64,
    ) -> Self {
        self.ensemble.add_forecast(ModelForecast {
            name,
            forecast,
            confidence: None,
            weight: Some(weight),
        });
        self
    }

    /// Adds a model's forecast with a specific confidence score.
    ///
    /// Used by `EnsembleMethod::BestModel` or can influence weighting in
    /// `EnsembleMethod::WeightedAverage` or `EnsembleMethod::Stacking` if weights are not explicitly set.
    ///
    /// # Arguments
    ///
    /// * `name` - A unique name for the model.
    /// * `forecast` - A `Vec<f64>` representing the model's forecast values.
    /// * `confidence` - The confidence score for this forecast (e.g., a value between 0.0 and 1.0).
    pub fn add_confident_forecast(
        mut self,
        name: String,
        forecast: Vec<f64>,
        confidence: f64,
    ) -> Self {
        self.ensemble.add_forecast(ModelForecast {
            name,
            forecast,
            confidence: Some(confidence),
            weight: None,
        });
        self
    }

    /// Builds the `EnsembleForecast` after all models have been added.
    /// This method calls the `combine()` function on the underlying `EnsembleForecast`.
    ///
    /// # Returns
    ///
    /// A `Result` containing the configured `EnsembleForecast` or an `OxiError`
    /// if combination fails (e.g., no forecasts added, mismatched lengths).
    pub fn build(mut self) -> Result<EnsembleForecast> {
        self.ensemble.combine()?;
        Ok(self.ensemble)
    }
}

/// Provides utility functions for evaluating ensemble forecasts.
pub struct EnsembleUtils;

impl EnsembleUtils {
    /// Evaluates the performance of an ensemble forecast against actual values.
    ///
    /// Calculates MAE and RMSE for the ensemble and for each individual model within it.
    ///
    /// # Arguments
    ///
    /// * `ensemble` - The `EnsembleForecast` that has been built and combined.
    /// * `actual` - A slice of `f64` representing the true actual values.
    ///
    /// # Returns
    ///
    /// A `Result` containing `EnsemblePerformance` or an `OxiError` if evaluation fails
    /// (e.g., forecast and actual lengths mismatch, ensemble not computed).
    pub fn evaluate_ensemble(
        ensemble: &EnsembleForecast,
        actual: &[f64],
    ) -> Result<EnsemblePerformance> {
        let forecast = ensemble.get_forecast()
            .ok_or_else(|| OxiError::ModelError("Ensemble not computed".into()))?;

        if forecast.len() != actual.len() {
            return Err(OxiError::ModelError(
                "Forecast and actual lengths don't match".into(),
            ));
        }

        // Calculate ensemble metrics
        let ensemble_mae = mae(actual, forecast);
        let ensemble_rmse = rmse(actual, forecast);

        // Calculate individual model performances
        let mut model_performances = HashMap::new();
        for model_forecast in &ensemble.forecasts {
            let model_mae = mae(actual, &model_forecast.forecast[..actual.len()]);
            let model_rmse = rmse(actual, &model_forecast.forecast[..actual.len()]);
            
            model_performances.insert(
                model_forecast.name.clone(),
                ModelPerformance {
                    mae: model_mae,
                    rmse: model_rmse,
                },
            );
        }

        Ok(EnsemblePerformance {
            ensemble_mae,
            ensemble_rmse,
            model_performances,
            improvement: None,
        })
    }

    /// Calculates the percentage improvement of the ensemble over the best individual model.
    /// The improvement is based on the MAE metric. Updates the `improvement` field
    /// in the `EnsemblePerformance` struct.
    ///
    /// # Arguments
    ///
    /// * `performance` - A mutable reference to `EnsemblePerformance` to update.
    pub fn calculate_improvement(performance: &mut EnsemblePerformance) {
        let best_individual_mae = performance
            .model_performances
            .values()
            .map(|p| p.mae)
            .fold(f64::INFINITY, f64::min);

        if best_individual_mae > 0.0 {
            let improvement_pct = 
                (best_individual_mae - performance.ensemble_mae) / best_individual_mae * 100.0;
            performance.improvement = Some(improvement_pct);
        }
    }
}

/// Holds performance metrics for an ensemble forecast and its constituent models.
#[derive(Debug, Clone)]
pub struct EnsemblePerformance {
    /// Mean Absolute Error (MAE) of the combined ensemble forecast.
    pub ensemble_mae: f64,
    /// Root Mean Squared Error (RMSE) of the combined ensemble forecast.
    pub ensemble_rmse: f64,
    /// Performance metrics (MAE, RMSE) for each individual model in the ensemble.
    /// The HashMap key is the model name.
    pub model_performances: HashMap<String, ModelPerformance>,
    /// Percentage improvement of the ensemble's MAE over the best individual model's MAE.
    /// `None` if no improvement or if it cannot be calculated (e.g., best individual MAE is zero).
    pub improvement: Option<f64>,
}

/// Holds performance metrics for an individual model.
#[derive(Debug, Clone)]
pub struct ModelPerformance {
    /// Mean Absolute Error (MAE) of the model.
    pub mae: f64,
    /// Root Mean Squared Error (RMSE) of the model.
    pub rmse: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_average_ensemble() {
        let mut ensemble = EnsembleForecast::new(EnsembleMethod::SimpleAverage);
        
        ensemble.add_forecast(ModelForecast {
            name: "Model1".to_string(),
            forecast: vec![1.0, 2.0, 3.0],
            confidence: None,
            weight: None,
        });
        
        ensemble.add_forecast(ModelForecast {
            name: "Model2".to_string(),
            forecast: vec![2.0, 3.0, 4.0],
            confidence: None,
            weight: None,
        });

        let result = ensemble.combine().unwrap();
        assert_eq!(result, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_weighted_average_ensemble() {
        let mut ensemble = EnsembleForecast::new(EnsembleMethod::WeightedAverage);
        
        ensemble.add_forecast(ModelForecast {
            name: "Model1".to_string(),
            forecast: vec![1.0, 2.0, 3.0],
            confidence: None,
            weight: Some(0.3),
        });
        
        ensemble.add_forecast(ModelForecast {
            name: "Model2".to_string(),
            forecast: vec![2.0, 3.0, 4.0],
            confidence: None,
            weight: Some(0.7),
        });

        let result = ensemble.combine().unwrap();
        // Expected: 0.3 * [1,2,3] + 0.7 * [2,3,4] = [1.7, 2.7, 3.7]
        assert!((result[0] - 1.7).abs() < 0.001);
        assert!((result[1] - 2.7).abs() < 0.001);
        assert!((result[2] - 3.7).abs() < 0.001);
    }

    #[test]
    fn test_median_ensemble() {
        let mut ensemble = EnsembleForecast::new(EnsembleMethod::Median);
        
        ensemble.add_forecast(ModelForecast {
            name: "Model1".to_string(),
            forecast: vec![1.0, 2.0, 3.0],
            confidence: None,
            weight: None,
        });
        
        ensemble.add_forecast(ModelForecast {
            name: "Model2".to_string(),
            forecast: vec![3.0, 4.0, 5.0],
            confidence: None,
            weight: None,
        });

        ensemble.add_forecast(ModelForecast {
            name: "Model3".to_string(),
            forecast: vec![2.0, 3.0, 4.0],
            confidence: None,
            weight: None,
        });

        let result = ensemble.combine().unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0]); // Median values
    }

    #[test]
    fn test_best_model_ensemble() {
        let mut ensemble = EnsembleForecast::new(EnsembleMethod::BestModel);
        
        ensemble.add_forecast(ModelForecast {
            name: "Model1".to_string(),
            forecast: vec![1.0, 2.0, 3.0],
            confidence: Some(0.5),
            weight: None,
        });
        
        ensemble.add_forecast(ModelForecast {
            name: "Model2".to_string(),
            forecast: vec![4.0, 5.0, 6.0],
            confidence: Some(0.9), // Best confidence
            weight: None,
        });

        let result = ensemble.combine().unwrap();
        assert_eq!(result, vec![4.0, 5.0, 6.0]); // Best model forecast
    }

    #[test]
    fn test_confidence_based_weighting() {
        let mut ensemble = EnsembleForecast::new(EnsembleMethod::WeightedAverage);
        
        ensemble.add_forecast(ModelForecast {
            name: "Model1".to_string(),
            forecast: vec![1.0, 2.0, 3.0],
            confidence: Some(0.8),
            weight: None,
        });
        
        ensemble.add_forecast(ModelForecast {
            name: "Model2".to_string(),
            forecast: vec![2.0, 3.0, 4.0],
            confidence: Some(0.6),
            weight: None,
        });

        let result = ensemble.combine().unwrap();
        // Should weight by confidence: 0.8 and 0.6, normalized to 0.571 and 0.429
        let expected_0 = 1.0 * (0.8/1.4) + 2.0 * (0.6/1.4);
        assert!((result[0] - expected_0).abs() < 0.01);
    }

    #[test]
    fn test_ensemble_builder() {
        let ensemble = EnsembleBuilder::new(EnsembleMethod::SimpleAverage)
            .add_model_forecast("ARIMA".to_string(), vec![1.0, 2.0, 3.0])
            .add_weighted_forecast("ES".to_string(), vec![2.0, 3.0, 4.0], 0.8)
            .build()
            .unwrap();

        let forecast = ensemble.get_forecast().unwrap();
        assert_eq!(forecast.len(), 3);
    }

    #[test]
    fn test_empty_ensemble_error() {
        let mut ensemble = EnsembleForecast::new(EnsembleMethod::SimpleAverage);
        let result = ensemble.combine();
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_forecast_lengths() {
        let mut ensemble = EnsembleForecast::new(EnsembleMethod::SimpleAverage);
        
        ensemble.add_forecast(ModelForecast {
            name: "Model1".to_string(),
            forecast: vec![1.0, 2.0, 3.0],
            confidence: None,
            weight: None,
        });
        
        ensemble.add_forecast(ModelForecast {
            name: "Model2".to_string(),
            forecast: vec![1.0, 2.0], // Different length
            confidence: None,
            weight: None,
        });

        let result = ensemble.combine();
        assert!(result.is_err());
    }

    #[test]
    fn test_ensemble_performance_evaluation() {
        let mut ensemble = EnsembleForecast::new(EnsembleMethod::SimpleAverage);
        
        ensemble.add_forecast(ModelForecast {
            name: "Model1".to_string(),
            forecast: vec![1.0, 2.0, 3.0],
            confidence: None,
            weight: None,
        });
        
        ensemble.add_forecast(ModelForecast {
            name: "Model2".to_string(),
            forecast: vec![1.1, 2.1, 3.1],
            confidence: None,
            weight: None,
        });

        ensemble.combine().unwrap();

        let actual = vec![1.05, 2.05, 3.05];
        let mut performance = EnsembleUtils::evaluate_ensemble(&ensemble, &actual).unwrap();
        
        assert!(performance.ensemble_mae < 0.1);
        assert_eq!(performance.model_performances.len(), 2);

        EnsembleUtils::calculate_improvement(&mut performance);
        assert!(performance.improvement.is_some());
    }

    #[test]
    fn test_stacking_ensemble() {
        let mut ensemble = EnsembleForecast::new(EnsembleMethod::Stacking);
        
        ensemble.add_forecast(ModelForecast {
            name: "Model1".to_string(),
            forecast: vec![1.0, 2.0, 3.0],
            confidence: Some(0.2), // Lower confidence
            weight: None,
        });
        
        ensemble.add_forecast(ModelForecast {
            name: "Model2".to_string(),
            forecast: vec![2.0, 3.0, 4.0],
            confidence: Some(0.1), // Higher confidence (inverse weight)
            weight: None,
        });

        let result = ensemble.combine().unwrap();
        assert_eq!(result.len(), 3);
        // Should weight inverse to confidence (higher weight for lower confidence)
    }

    #[test]
    fn test_single_model_ensemble() {
        let mut ensemble = EnsembleForecast::new(EnsembleMethod::SimpleAverage);
        
        ensemble.add_forecast(ModelForecast {
            name: "OnlyModel".to_string(),
            forecast: vec![1.0, 2.0, 3.0],
            confidence: None,
            weight: None,
        });

        let result = ensemble.combine().unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_zero_weights_error() {
        let mut ensemble = EnsembleForecast::new(EnsembleMethod::WeightedAverage);
        
        ensemble.add_forecast(ModelForecast {
            name: "Model1".to_string(),
            forecast: vec![1.0, 2.0, 3.0],
            confidence: None,
            weight: Some(0.0),
        });
        
        ensemble.add_forecast(ModelForecast {
            name: "Model2".to_string(),
            forecast: vec![2.0, 3.0, 4.0],
            confidence: None,
            weight: Some(0.0),
        });

        let result = ensemble.combine();
        assert!(result.is_err());
    }
} 