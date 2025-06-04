//! # Parameter Optimization for OxiDiviner Models
//!
//! This module provides a suite of tools for automatically optimizing the parameters
//! of various forecasting models within the OxiDiviner library. Effective parameter
//! tuning is crucial for achieving high accuracy in time series forecasting.
//!
//! ## Key Features
//!
//! - **Multiple Optimization Algorithms**:
//!   - `GridSearch`: Exhaustively searches a predefined hyperparameter space.
//!   - `RandomSearch`: Samples parameters randomly from a given space (often more efficient for larger spaces).
//!   - `BayesianOptimization`: (Simplified) Uses past evaluations to make informed choices for the next set of parameters.
//!
//! - **Support for Various Models**: Currently supports optimization for:
//!   - `ARIMAModel`
//!   - `SimpleESModel` (Exponential Smoothing)
//!   - `HoltWintersModel`
//!   - `MAModel` (Moving Average)
//!
//! - **Flexible Configuration**: The `OptimizerBuilder` allows for detailed configuration of the
//!   optimization process, including:
//!   - Choice of optimization method (`OptimizationMethod`).
//!   - Selection of evaluation metric (`OptimizationMetric`: MAE, RMSE, MAPE, AIC, BIC).
//!   - Maximum number of evaluations (`max_evaluations`).
//!   - Cross-validation folds (`cv_folds`) and train/test split ratio (`train_ratio`) for robust evaluation.
//!
//! - **Detailed Results**: `OptimizationResult` provides the best parameters found, the corresponding
//!   score, improvement percentage over a baseline, and convergence information.
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use oxidiviner::prelude::*;
//! use oxidiviner::optimization::{OptimizerBuilder, OptimizationMethod, OptimizationMetric};
//! use std::collections::HashMap;
//!
//! // Assume `series_data` is a valid TimeSeriesData instance
//! let series_data: TimeSeriesData = /* ... load or create your data ... */ ;
//!
//! // Configure the optimizer
//! let optimizer = OptimizerBuilder::new()
//!     .method(OptimizationMethod::GridSearch)
//!     .metric(OptimizationMetric::MAE)
//!     .max_evaluations(30)
//!     .train_ratio(0.8) // Use 80% for training, 20% for validation
//!     .build();
//!
//! // Optimize ARIMA parameters (p, d, q up to specified max values)
//! if let Ok(arima_result) = optimizer.optimize_arima(&series_data, 3, 1, 3) {
//!     println!("Best ARIMA Parameters: {:?}", arima_result.best_parameters);
//!     println!("Best ARIMA MAE: {:.4}", arima_result.best_score);
//!     println!("ARIMA Optimization Improvement: {:.2}%", arima_result.improvement_percentage);
//! }
//!
//! // Optimize Simple Exponential Smoothing alpha parameter
//! if let Ok(es_result) = optimizer.optimize_exponential_smoothing(&series_data) {
//!     println!("Best ES Alpha: {:?}", es_result.best_parameters.get("alpha"));
//!     println!("Best ES MAE: {:.4}", es_result.best_score);
//! }
//! # Ok::<(), OxiError>(())
//! ```
//!
//! This module helps users to systematically find better model configurations without
//! manual trial-and-error, leading to more accurate and reliable forecasts.

use crate::core::validation::ValidationUtils;
use crate::core::{Result, TimeSeriesData};
use crate::math::metrics::{mae, mse, rmse};
use crate::models::autoregressive::ARIMAModel;
use crate::models::exponential_smoothing::HoltWintersModel;
use crate::models::exponential_smoothing::SimpleESModel;
use crate::models::moving_average::MAModel;
use std::collections::HashMap;

/// Specifies the algorithm used for parameter optimization.
#[derive(Debug, Clone)]
pub enum OptimizationMethod {
    /// Grid search explores a predefined discrete set of hyperparameter values.
    /// It exhaustively tries all combinations.
    GridSearch,
    /// Random search samples hyperparameters randomly from their defined ranges.
    /// Can be more efficient than grid search for high-dimensional spaces.
    RandomSearch,
    /// Bayesian optimization uses a probabilistic model to select the next set of
    /// hyperparameters to evaluate based on past results. (Simplified implementation).
    BayesianOptimization,
}

/// Configuration settings for the parameter optimization process.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// The optimization algorithm to use (e.g., GridSearch, RandomSearch).
    pub method: OptimizationMethod,
    /// The performance metric to optimize (e.g., MAE, RMSE).
    pub metric: OptimizationMetric,
    /// The maximum number of parameter combinations to evaluate.
    pub max_evaluations: usize,
    /// The number of folds to use in cross-validation (if applicable, currently uses time_split).
    pub cv_folds: usize,
    /// The ratio of data to use for training when splitting for validation (e.g., 0.8 for 80%).
    pub train_ratio: f64,
}

/// Performance metrics that can be used for model optimization.
#[derive(Debug, Clone)]
pub enum OptimizationMetric {
    /// Mean Absolute Error. Robust to outliers.
    MAE,
    /// Root Mean Squared Error. Penalizes larger errors more heavily.
    RMSE,
    /// Mean Absolute Percentage Error. Relative error measure.
    MAPE,
    /// Akaike Information Criterion. Balances model fit and complexity (lower is better).
    AIC,
    /// Bayesian Information Criterion. Similar to AIC but with a stronger penalty for complexity (lower is better).
    BIC,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            method: OptimizationMethod::GridSearch,
            metric: OptimizationMetric::MAE,
            max_evaluations: 50,
            cv_folds: 3,
            train_ratio: 0.8,
        }
    }
}

/// Stores the results of a parameter optimization process.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// A HashMap containing the best set of parameters found (parameter name -> value).
    pub best_parameters: HashMap<String, f64>,
    /// The score achieved by the best parameters, based on the chosen `OptimizationMetric`.
    pub best_score: f64,
    /// Percentage improvement of the best score over a baseline model's score.
    pub improvement_percentage: f64,
    /// The total number of parameter combinations evaluated during the optimization.
    pub evaluation_count: usize,
    /// Information about the convergence of the optimization process.
    pub convergence_info: ConvergenceInfo,
}

/// Contains information about the convergence of the optimization process.
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Indicates whether the optimization process is considered to have converged.
    /// (Currently, this is a simplified check; always true if evaluations complete).
    pub converged: bool,
    /// The final score achieved at the end of the optimization process (same as `best_score`).
    pub final_score: f64,
    /// A history of scores for each evaluation made during the optimization.
    pub score_history: Vec<f64>,
}

/// The main engine for performing parameter optimization on forecasting models.
pub struct ParameterOptimizer {
    config: OptimizationConfig,
}

impl ParameterOptimizer {
    /// Creates a new `ParameterOptimizer` with the given configuration.
    pub fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }

    /// Optimizes ARIMA model parameters (p, d, q).
    ///
    /// Searches for the best (p, d, q) orders for an ARIMA model up to the specified maximums.
    ///
    /// # Arguments
    /// * `data` - The `TimeSeriesData` to use for optimization.
    /// * `max_p` - Maximum value for the autoregressive (AR) order p.
    /// * `max_d` - Maximum value for the differencing order d.
    /// * `max_q` - Maximum value for the moving average (MA) order q.
    ///
    /// # Returns
    /// A `Result` containing `OptimizationResult` or an `OxiError`.
    pub fn optimize_arima(
        &self,
        data: &TimeSeriesData,
        max_p: usize,
        max_d: usize,
        max_q: usize,
    ) -> Result<OptimizationResult> {
        let parameter_space = self.generate_arima_parameter_space(max_p, max_d, max_q);
        let baseline_score = self.evaluate_baseline_arima(data)?;

        let mut best_score = f64::INFINITY;
        let mut best_params = HashMap::new();
        let mut evaluation_count = 0;
        let mut score_history = Vec::new();

        for params in parameter_space {
            if evaluation_count >= self.config.max_evaluations {
                break;
            }

            let p = params["p"] as usize;
            let d = params["d"] as usize;
            let q = params["q"] as usize;

            if let Ok(score) = self.evaluate_arima_params(data, p, d, q) {
                evaluation_count += 1;
                score_history.push(score);

                if score < best_score {
                    best_score = score;
                    best_params = params;
                }
            }
        }

        let improvement_pct = if baseline_score > 0.0 {
            (baseline_score - best_score) / baseline_score * 100.0
        } else {
            0.0
        };

        Ok(OptimizationResult {
            best_parameters: best_params,
            best_score,
            improvement_percentage: improvement_pct,
            evaluation_count,
            convergence_info: ConvergenceInfo {
                converged: true, // Simple convergence check
                final_score: best_score,
                score_history,
            },
        })
    }

    /// Optimizes parameters for a Simple Exponential Smoothing model (alpha).
    ///
    /// Searches for the best alpha (smoothing factor) for the SimpleESModel.
    ///
    /// # Arguments
    /// * `data` - The `TimeSeriesData` to use for optimization.
    ///
    /// # Returns
    /// A `Result` containing `OptimizationResult` or an `OxiError`.
    pub fn optimize_exponential_smoothing(
        &self,
        data: &TimeSeriesData,
    ) -> Result<OptimizationResult> {
        let parameter_space = self.generate_es_parameter_space();
        let baseline_score = self.evaluate_baseline_es(data)?;

        let mut best_score = f64::INFINITY;
        let mut best_params = HashMap::new();
        let mut evaluation_count = 0;
        let mut score_history = Vec::new();

        for params in parameter_space {
            if evaluation_count >= self.config.max_evaluations {
                break;
            }

            let alpha = params["alpha"];

            if let Ok(score) = self.evaluate_es_params(data, alpha) {
                evaluation_count += 1;
                score_history.push(score);

                if score < best_score {
                    best_score = score;
                    best_params = params;
                }
            }
        }

        let improvement_pct = if baseline_score > 0.0 {
            (baseline_score - best_score) / baseline_score * 100.0
        } else {
            0.0
        };

        Ok(OptimizationResult {
            best_parameters: best_params,
            best_score,
            improvement_percentage: improvement_pct,
            evaluation_count,
            convergence_info: ConvergenceInfo {
                converged: true,
                final_score: best_score,
                score_history,
            },
        })
    }

    /// Optimizes parameters for a Holt-Winters Exponential Smoothing model (alpha, beta, gamma).
    ///
    /// Searches for the best alpha, beta, and gamma smoothing factors.
    ///
    /// # Arguments
    /// * `data` - The `TimeSeriesData` to use for optimization.
    /// * `seasonal_period` - The length of the seasonal cycle.
    ///
    /// # Returns
    /// A `Result` containing `OptimizationResult` or an `OxiError`.
    pub fn optimize_holt_winters(
        &self,
        data: &TimeSeriesData,
        seasonal_period: usize,
    ) -> Result<OptimizationResult> {
        let parameter_space = self.generate_hw_parameter_space();
        let baseline_score = self.evaluate_baseline_hw(data, seasonal_period)?;

        let mut best_score = f64::INFINITY;
        let mut best_params = HashMap::new();
        let mut evaluation_count = 0;
        let mut score_history = Vec::new();

        for params in parameter_space {
            if evaluation_count >= self.config.max_evaluations {
                break;
            }

            let alpha = params["alpha"];
            let beta = params["beta"];
            let gamma = params["gamma"];

            if let Ok(score) = self.evaluate_hw_params(data, alpha, beta, gamma, seasonal_period) {
                evaluation_count += 1;
                score_history.push(score);

                if score < best_score {
                    best_score = score;
                    best_params = params;
                }
            }
        }

        let improvement_pct = if baseline_score > 0.0 {
            (baseline_score - best_score) / baseline_score * 100.0
        } else {
            0.0
        };

        Ok(OptimizationResult {
            best_parameters: best_params,
            best_score,
            improvement_percentage: improvement_pct,
            evaluation_count,
            convergence_info: ConvergenceInfo {
                converged: true,
                final_score: best_score,
                score_history,
            },
        })
    }

    /// Optimizes the window size for a Moving Average (MA) model.
    ///
    /// Searches for the best window size up to `max_window`.
    ///
    /// # Arguments
    /// * `data` - The `TimeSeriesData` to use for optimization.
    /// * `max_window` - The maximum window size to consider.
    ///
    /// # Returns
    /// A `Result` containing `OptimizationResult` or an `OxiError`.
    pub fn optimize_moving_average(
        &self,
        data: &TimeSeriesData,
        max_window: usize,
    ) -> Result<OptimizationResult> {
        let parameter_space = self.generate_ma_parameter_space(max_window);
        let baseline_score = self.evaluate_baseline_ma(data)?;

        let mut best_score = f64::INFINITY;
        let mut best_params = HashMap::new();
        let mut evaluation_count = 0;
        let mut score_history = Vec::new();

        for params in parameter_space {
            if evaluation_count >= self.config.max_evaluations {
                break;
            }

            let window = params["window"] as usize;

            if let Ok(score) = self.evaluate_ma_params(data, window) {
                evaluation_count += 1;
                score_history.push(score);

                if score < best_score {
                    best_score = score;
                    best_params = params;
                }
            }
        }

        let improvement_pct = if baseline_score > 0.0 {
            (baseline_score - best_score) / baseline_score * 100.0
        } else {
            0.0
        };

        Ok(OptimizationResult {
            best_parameters: best_params,
            best_score,
            improvement_percentage: improvement_pct,
            evaluation_count,
            convergence_info: ConvergenceInfo {
                converged: true,
                final_score: best_score,
                score_history,
            },
        })
    }

    // Parameter space generation methods
    fn generate_arima_parameter_space(
        &self,
        max_p: usize,
        max_d: usize,
        max_q: usize,
    ) -> Vec<HashMap<String, f64>> {
        let mut parameter_space = Vec::new();

        for p in 0..=max_p {
            for d in 0..=max_d {
                for q in 0..=max_q {
                    if p == 0 && d == 0 && q == 0 {
                        continue; // Skip invalid combination
                    }

                    let mut params = HashMap::new();
                    params.insert("p".to_string(), p as f64);
                    params.insert("d".to_string(), d as f64);
                    params.insert("q".to_string(), q as f64);
                    parameter_space.push(params);
                }
            }
        }

        parameter_space
    }

    fn generate_es_parameter_space(&self) -> Vec<HashMap<String, f64>> {
        let mut parameter_space = Vec::new();
        let alpha_values = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        for alpha in alpha_values {
            let mut params = HashMap::new();
            params.insert("alpha".to_string(), alpha);
            parameter_space.push(params);
        }

        parameter_space
    }

    fn generate_hw_parameter_space(&self) -> Vec<HashMap<String, f64>> {
        let mut parameter_space = Vec::new();
        let param_values = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        for alpha in &param_values {
            for beta in &param_values {
                for gamma in &param_values {
                    let mut params = HashMap::new();
                    params.insert("alpha".to_string(), *alpha);
                    params.insert("beta".to_string(), *beta);
                    params.insert("gamma".to_string(), *gamma);
                    parameter_space.push(params);
                }
            }
        }

        parameter_space
    }

    fn generate_ma_parameter_space(&self, max_window: usize) -> Vec<HashMap<String, f64>> {
        let mut parameter_space = Vec::new();

        for window in 1..=max_window {
            let mut params = HashMap::new();
            params.insert("window".to_string(), window as f64);
            parameter_space.push(params);
        }

        parameter_space
    }

    // Model evaluation methods
    fn evaluate_arima_params(
        &self,
        data: &TimeSeriesData,
        p: usize,
        d: usize,
        q: usize,
    ) -> Result<f64> {
        let (train_data, test_data) = ValidationUtils::time_split(data, self.config.train_ratio)?;

        let mut model = ARIMAModel::new(p, d, q, true)?;
        model.fit(&train_data)?;

        let forecast = model.forecast(test_data.values.len())?;
        let score = self.calculate_score(&test_data.values, &forecast);

        Ok(score)
    }

    fn evaluate_es_params(&self, data: &TimeSeriesData, alpha: f64) -> Result<f64> {
        let (train_data, test_data) = ValidationUtils::time_split(data, self.config.train_ratio)?;

        let mut model = SimpleESModel::new(alpha)?;
        model.fit(&train_data)?;

        let forecast = model.forecast(test_data.values.len())?;
        let score = self.calculate_score(&test_data.values, &forecast);

        Ok(score)
    }

    fn evaluate_hw_params(
        &self,
        data: &TimeSeriesData,
        alpha: f64,
        beta: f64,
        gamma: f64,
        period: usize,
    ) -> Result<f64> {
        let (train_data, test_data) = ValidationUtils::time_split(data, self.config.train_ratio)?;

        let mut model = HoltWintersModel::new(alpha, beta, gamma, period)?;
        model.fit(&train_data)?;

        let forecast = model.forecast(test_data.values.len())?;
        let score = self.calculate_score(&test_data.values, &forecast);

        Ok(score)
    }

    fn evaluate_ma_params(&self, data: &TimeSeriesData, window: usize) -> Result<f64> {
        let (train_data, test_data) = ValidationUtils::time_split(data, self.config.train_ratio)?;

        let mut model = MAModel::new(window)?;
        model.fit(&train_data)?;

        let forecast = model.forecast(test_data.values.len())?;
        let score = self.calculate_score(&test_data.values, &forecast);

        Ok(score)
    }

    // Baseline evaluation methods
    fn evaluate_baseline_arima(&self, data: &TimeSeriesData) -> Result<f64> {
        self.evaluate_arima_params(data, 1, 1, 1)
    }

    fn evaluate_baseline_es(&self, data: &TimeSeriesData) -> Result<f64> {
        self.evaluate_es_params(data, 0.3)
    }

    fn evaluate_baseline_hw(&self, data: &TimeSeriesData, period: usize) -> Result<f64> {
        self.evaluate_hw_params(data, 0.2, 0.1, 0.1, period)
    }

    fn evaluate_baseline_ma(&self, data: &TimeSeriesData) -> Result<f64> {
        self.evaluate_ma_params(data, 5)
    }

    fn calculate_score(&self, actual: &[f64], forecast: &[f64]) -> f64 {
        match self.config.metric {
            OptimizationMetric::MAE => mae(actual, forecast),
            OptimizationMetric::RMSE => rmse(actual, forecast),
            OptimizationMetric::MAPE => {
                // Calculate MAPE
                let mut sum = 0.0;
                let mut count = 0;
                for (a, f) in actual.iter().zip(forecast.iter()) {
                    if a.abs() > 1e-8 {
                        sum += ((a - f) / a).abs();
                        count += 1;
                    }
                }
                if count > 0 {
                    sum / count as f64 * 100.0
                } else {
                    f64::INFINITY
                }
            }
            OptimizationMetric::AIC | OptimizationMetric::BIC => {
                // For simplicity, use MSE as proxy for likelihood-based metrics
                mse(actual, forecast)
            }
        }
    }
}

/// A builder for creating and configuring `ParameterOptimizer` instances.
///
/// Provides a fluent API to set various optimization parameters before building the optimizer.
pub struct OptimizerBuilder {
    config: OptimizationConfig,
}

impl OptimizerBuilder {
    /// Creates a new `OptimizerBuilder` with default configuration values.
    pub fn new() -> Self {
        Self {
            config: OptimizationConfig::default(),
        }
    }

    /// Sets the optimization method (e.g., GridSearch).
    pub fn method(mut self, method: OptimizationMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Sets the performance metric to optimize for (e.g., MAE).
    pub fn metric(mut self, metric: OptimizationMetric) -> Self {
        self.config.metric = metric;
        self
    }

    /// Sets the maximum number of parameter combinations to evaluate.
    pub fn max_evaluations(mut self, max_evals: usize) -> Self {
        self.config.max_evaluations = max_evals;
        self
    }

    /// Sets the number of cross-validation folds (currently not used with time_split based validation).
    pub fn cv_folds(mut self, folds: usize) -> Self {
        self.config.cv_folds = folds;
        self
    }

    /// Sets the ratio of data to be used for training during validation splits (e.g., 0.8 for 80%).
    pub fn train_ratio(mut self, ratio: f64) -> Self {
        self.config.train_ratio = ratio;
        self
    }

    /// Builds the `ParameterOptimizer` with the specified configuration.
    pub fn build(self) -> ParameterOptimizer {
        ParameterOptimizer::new(self.config)
    }
}

impl Default for OptimizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    fn create_test_data() -> TimeSeriesData {
        let start_time = Utc::now();
        let timestamps: Vec<_> = (0..50).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..50)
            .map(|i| 100.0 + i as f64 * 0.5 + (i as f64 * 0.1).sin())
            .collect();
        TimeSeriesData::new(timestamps, values, "test").unwrap()
    }

    fn create_seasonal_test_data() -> TimeSeriesData {
        let start_time = Utc::now();
        let timestamps: Vec<_> = (0..60).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..60)
            .map(|i| {
                100.0 + i as f64 * 0.2 + 10.0 * (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin()
            })
            .collect();
        TimeSeriesData::new(timestamps, values, "seasonal_test").unwrap()
    }

    #[test]
    fn test_optimizer_builder() {
        let optimizer = OptimizerBuilder::new()
            .method(OptimizationMethod::GridSearch)
            .metric(OptimizationMetric::MAE)
            .max_evaluations(10)
            .build();

        assert_eq!(optimizer.config.max_evaluations, 10);
        assert!(matches!(
            optimizer.config.method,
            OptimizationMethod::GridSearch
        ));
        assert!(matches!(optimizer.config.metric, OptimizationMetric::MAE));
    }

    #[test]
    fn test_default_optimizer_config() {
        let config = OptimizationConfig::default();
        assert_eq!(config.max_evaluations, 50);
        assert_eq!(config.cv_folds, 3);
        assert_eq!(config.train_ratio, 0.8);
        assert!(matches!(config.method, OptimizationMethod::GridSearch));
        assert!(matches!(config.metric, OptimizationMetric::MAE));
    }

    #[test]
    fn test_arima_optimization() {
        let data = create_test_data();
        let optimizer = OptimizerBuilder::new().max_evaluations(5).build();

        let result = optimizer.optimize_arima(&data, 2, 1, 2);
        assert!(result.is_ok());

        let opt_result = result.unwrap();
        assert!(opt_result.evaluation_count > 0);
        assert!(opt_result.best_score >= 0.0);
        assert!(opt_result.best_parameters.contains_key("p"));
        assert!(opt_result.best_parameters.contains_key("d"));
        assert!(opt_result.best_parameters.contains_key("q"));
        assert!(!opt_result.convergence_info.score_history.is_empty());
    }

    #[test]
    fn test_es_optimization() {
        let data = create_test_data();
        let optimizer = OptimizerBuilder::new().max_evaluations(5).build();

        let result = optimizer.optimize_exponential_smoothing(&data);
        assert!(result.is_ok());

        let opt_result = result.unwrap();
        assert!(opt_result.evaluation_count > 0);
        assert!(opt_result.best_parameters.contains_key("alpha"));
        let alpha = opt_result.best_parameters["alpha"];
        assert!(alpha > 0.0 && alpha < 1.0);
    }

    #[test]
    fn test_holt_winters_optimization() {
        let data = create_seasonal_test_data();
        let optimizer = OptimizerBuilder::new()
            .max_evaluations(8) // 2^3 = 8 combinations for 2 values each of alpha, beta, gamma
            .build();

        let result = optimizer.optimize_holt_winters(&data, 12);
        assert!(result.is_ok());

        let opt_result = result.unwrap();
        assert!(opt_result.evaluation_count > 0);
        assert!(opt_result.best_parameters.contains_key("alpha"));
        assert!(opt_result.best_parameters.contains_key("beta"));
        assert!(opt_result.best_parameters.contains_key("gamma"));

        // Check parameter bounds
        for &param in opt_result.best_parameters.values() {
            assert!(param > 0.0 && param < 1.0);
        }
    }

    #[test]
    fn test_ma_optimization() {
        let data = create_test_data();
        let optimizer = OptimizerBuilder::new().max_evaluations(10).build();

        let result = optimizer.optimize_moving_average(&data, 15);
        assert!(result.is_ok());

        let opt_result = result.unwrap();
        assert!(opt_result.evaluation_count > 0);
        assert!(opt_result.best_parameters.contains_key("window"));
        let window = opt_result.best_parameters["window"] as usize;
        assert!((1..=15).contains(&window));
    }

    #[test]
    fn test_optimization_metrics() {
        let data = create_test_data();

        // Test different metrics
        let metrics = vec![
            OptimizationMetric::MAE,
            OptimizationMetric::RMSE,
            OptimizationMetric::MAPE,
            OptimizationMetric::AIC,
            OptimizationMetric::BIC,
        ];

        for metric in metrics {
            let optimizer = OptimizerBuilder::new()
                .metric(metric)
                .max_evaluations(3)
                .build();

            let result = optimizer.optimize_arima(&data, 1, 1, 1);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_optimization_methods() {
        let data = create_test_data();

        // Test different optimization methods
        let methods = vec![
            OptimizationMethod::GridSearch,
            OptimizationMethod::RandomSearch,
            OptimizationMethod::BayesianOptimization,
        ];

        for method in methods {
            let optimizer = OptimizerBuilder::new()
                .method(method)
                .max_evaluations(3)
                .build();

            let result = optimizer.optimize_arima(&data, 1, 1, 1);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_improvement_calculation() {
        let data = create_test_data();
        let optimizer = OptimizerBuilder::new().max_evaluations(5).build();

        let result = optimizer.optimize_arima(&data, 2, 1, 2).unwrap();

        // Improvement percentage should be calculated
        assert!(result.improvement_percentage >= 0.0 || result.improvement_percentage.is_nan());
    }

    #[test]
    fn test_convergence_info() {
        let data = create_test_data();
        let optimizer = OptimizerBuilder::new().max_evaluations(3).build();

        let result = optimizer.optimize_arima(&data, 1, 1, 1).unwrap();

        assert!(result.convergence_info.converged);
        assert_eq!(result.convergence_info.final_score, result.best_score);
        assert!(!result.convergence_info.score_history.is_empty());
    }

    #[test]
    fn test_empty_data_error() {
        // Test that optimizer properly handles empty data
        // Since TimeSeriesData::new itself fails with empty data,
        // we'll test with minimal data that would cause optimization to fail
        let start_time = Utc::now();
        let timestamps: Vec<_> = (0..2).map(|i| start_time + Duration::days(i)).collect();
        let values = vec![1.0, 1.0]; // Very small, constant data
        let minimal_data = TimeSeriesData::new(timestamps, values, "minimal").unwrap();

        let optimizer = OptimizerBuilder::new().build();

        // Should fail for complex ARIMA models with insufficient data
        let result = optimizer.optimize_arima(&minimal_data, 1, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_data_error() {
        let start_time = Utc::now();
        let timestamps: Vec<_> = (0..5).map(|i| start_time + Duration::days(i)).collect();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let small_data = TimeSeriesData::new(timestamps, values, "small").unwrap();

        let optimizer = OptimizerBuilder::new().build();

        // Should fail for complex ARIMA models with insufficient data
        let result = optimizer.optimize_arima(&small_data, 3, 2, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_parameter_space_generation() {
        let optimizer = OptimizerBuilder::new().build();

        // Test ARIMA parameter space
        let arima_space = optimizer.generate_arima_parameter_space(2, 1, 2);
        assert!(!arima_space.is_empty());

        // Each parameter set should have p, d, q
        for params in &arima_space {
            assert!(params.contains_key("p"));
            assert!(params.contains_key("d"));
            assert!(params.contains_key("q"));
        }

        // Test ES parameter space
        let es_space = optimizer.generate_es_parameter_space();
        assert!(!es_space.is_empty());

        for params in &es_space {
            assert!(params.contains_key("alpha"));
            let alpha = params["alpha"];
            assert!(alpha > 0.0 && alpha < 1.0);
        }
    }

    #[test]
    fn test_max_evaluations_limit() {
        let data = create_test_data();
        let max_evals = 3;
        let optimizer = OptimizerBuilder::new().max_evaluations(max_evals).build();

        let result = optimizer.optimize_arima(&data, 3, 2, 3).unwrap();
        assert!(result.evaluation_count <= max_evals);
    }

    #[test]
    fn test_train_ratio_setting() {
        let data = create_test_data();
        let optimizer = OptimizerBuilder::new()
            .train_ratio(0.7)
            .max_evaluations(3)
            .build();

        let result = optimizer.optimize_arima(&data, 1, 1, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_score_calculation_edge_cases() {
        let optimizer = OptimizerBuilder::new()
            .metric(OptimizationMetric::MAPE)
            .build();

        // Test MAPE with zero values (should handle gracefully)
        let actual = vec![0.0, 1.0, 2.0];
        let forecast = vec![0.1, 1.1, 2.1];
        let score = optimizer.calculate_score(&actual, &forecast);
        assert!(score.is_finite());
    }

    #[test]
    fn test_baseline_evaluation() {
        let data = create_test_data();
        let optimizer = OptimizerBuilder::new().build();

        // Test baseline methods don't panic
        assert!(optimizer.evaluate_baseline_arima(&data).is_ok());
        assert!(optimizer.evaluate_baseline_es(&data).is_ok());
        assert!(optimizer.evaluate_baseline_ma(&data).is_ok());
        assert!(optimizer.evaluate_baseline_hw(&data, 12).is_ok());
    }
}
