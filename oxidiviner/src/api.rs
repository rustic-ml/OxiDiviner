//! High-level API module
//!
//! This module provides a simplified, high-level interface for forecasting
//! that abstracts away the complexity of individual models and provides
//! a unified way to work with different forecasting algorithms.

use crate::core::{ModelEvaluation, OxiError, QuickForecaster, Result, TimeSeriesData};
use crate::models::autoregressive::ARIMAModel;
use crate::models::exponential_smoothing::{HoltLinearModel, HoltWintersModel, SimpleESModel};
use crate::models::moving_average::MAModel;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// High-level forecaster interface
pub struct Forecaster {
    config: ForecastConfig,
}

/// Configuration for forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastConfig {
    /// The forecasting model to use
    pub model_type: ModelType,
    /// Model-specific parameters
    pub parameters: ModelParameters,
    /// Whether to automatically select the best model
    pub auto_select: bool,
}

/// Available forecasting models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    /// ARIMA model
    ARIMA,
    /// Simple Exponential Smoothing
    SimpleES,
    /// Holt's Linear Trend model
    HoltLinear,
    /// Holt-Winters Seasonal model
    HoltWinters,
    /// Moving Average
    MovingAverage,
    /// Automatic model selection
    Auto,
}

/// Model-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    // ARIMA parameters
    pub arima_p: Option<usize>,
    pub arima_d: Option<usize>,
    pub arima_q: Option<usize>,

    // Exponential smoothing parameters
    pub alpha: Option<f64>,
    pub beta: Option<f64>,
    pub gamma: Option<f64>,

    // Moving average parameters
    pub ma_window: Option<usize>,

    // Seasonal parameters
    pub seasonal_period: Option<usize>,
}

/// Output from forecasting
#[derive(Debug, Clone)]
pub struct ForecastOutput {
    /// The forecast values
    pub forecast: Vec<f64>,
    /// The model that was used
    pub model_used: String,
    /// Confidence intervals (if available)
    pub confidence_intervals: Option<Vec<(f64, f64)>>,
    /// Model performance metrics
    pub metrics: Option<ModelMetrics>,
}

/// Model performance metrics
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    /// Mean Absolute Error
    pub mae: Option<f64>,
    /// Mean Squared Error
    pub mse: Option<f64>,
    /// Root Mean Squared Error
    pub rmse: Option<f64>,
    /// Mean Absolute Percentage Error
    pub mape: Option<f64>,
}

impl Default for ForecastConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::Auto,
            parameters: ModelParameters::default(),
            auto_select: true,
        }
    }
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            arima_p: Some(1),
            arima_d: Some(1),
            arima_q: Some(1),
            alpha: Some(0.3),
            beta: Some(0.1),
            gamma: Some(0.1),
            ma_window: Some(5),
            seasonal_period: Some(12),
        }
    }
}

impl Forecaster {
    /// Create a new forecaster with default configuration
    pub fn new() -> Self {
        Self {
            config: ForecastConfig::default(),
        }
    }

    /// Create a new forecaster with custom configuration
    pub fn with_config(config: ForecastConfig) -> Self {
        Self { config }
    }

    /// Set the model type
    pub fn model(mut self, model_type: ModelType) -> Self {
        self.config.model_type = model_type;
        self
    }

    /// Set ARIMA parameters
    pub fn arima_params(mut self, p: usize, d: usize, q: usize) -> Self {
        self.config.parameters.arima_p = Some(p);
        self.config.parameters.arima_d = Some(d);
        self.config.parameters.arima_q = Some(q);
        self
    }

    /// Set exponential smoothing alpha parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.parameters.alpha = Some(alpha);
        self
    }

    /// Set moving average window size
    pub fn ma_window(mut self, window: usize) -> Self {
        self.config.parameters.ma_window = Some(window);
        self
    }

    /// Enable automatic model selection
    pub fn auto_select(mut self) -> Self {
        self.config.auto_select = true;
        self.config.model_type = ModelType::Auto;
        self
    }

    /// Perform forecasting on the given data
    pub fn forecast(&self, data: &TimeSeriesData, periods: usize) -> Result<ForecastOutput> {
        match self.config.model_type {
            ModelType::ARIMA => self.forecast_arima(data, periods),
            ModelType::SimpleES => self.forecast_simple_es(data, periods),
            ModelType::HoltLinear => self.forecast_holt_linear(data, periods),
            ModelType::HoltWinters => self.forecast_holt_winters(data, periods),
            ModelType::MovingAverage => self.forecast_ma(data, periods),
            ModelType::Auto => self.forecast_auto(data, periods),
        }
    }

    /// Create forecaster from timestamps and values
    pub fn from_data(
        timestamps: Vec<DateTime<Utc>>,
        values: Vec<f64>,
        name: &str,
    ) -> Result<(Self, TimeSeriesData)> {
        let data = TimeSeriesData::new(timestamps, values, name)?;
        Ok((Self::new(), data))
    }

    // Private implementation methods
    fn forecast_arima(&self, data: &TimeSeriesData, periods: usize) -> Result<ForecastOutput> {
        let p = self.config.parameters.arima_p.unwrap_or(1);
        let d = self.config.parameters.arima_d.unwrap_or(1);
        let q = self.config.parameters.arima_q.unwrap_or(1);

        let mut model = ARIMAModel::new(p, d, q, true)?;
        model.fit(data)?;
        let forecast = model.forecast(periods)?;

        Ok(ForecastOutput {
            forecast,
            model_used: format!("ARIMA({},{},{})", p, d, q),
            confidence_intervals: None,
            metrics: None,
        })
    }

    fn forecast_simple_es(&self, data: &TimeSeriesData, periods: usize) -> Result<ForecastOutput> {
        let alpha = self.config.parameters.alpha.unwrap_or(0.3);

        let mut model = SimpleESModel::new(alpha)?;
        model.fit(data)?;
        let forecast = model.forecast(periods)?;

        Ok(ForecastOutput {
            forecast,
            model_used: format!("SimpleES(α={})", alpha),
            confidence_intervals: None,
            metrics: None,
        })
    }

    fn forecast_holt_linear(
        &self,
        data: &TimeSeriesData,
        periods: usize,
    ) -> Result<ForecastOutput> {
        let alpha = self.config.parameters.alpha.unwrap_or(0.3);
        let beta = self.config.parameters.beta.unwrap_or(0.1);

        let mut model = HoltLinearModel::new(alpha, beta)?;
        model.fit(data)?;
        let forecast = model.forecast(periods)?;

        Ok(ForecastOutput {
            forecast,
            model_used: format!("HoltLinear(α={}, β={})", alpha, beta),
            confidence_intervals: None,
            metrics: None,
        })
    }

    fn forecast_holt_winters(
        &self,
        data: &TimeSeriesData,
        periods: usize,
    ) -> Result<ForecastOutput> {
        let alpha = self.config.parameters.alpha.unwrap_or(0.3);
        let beta = self.config.parameters.beta.unwrap_or(0.1);
        let gamma = self.config.parameters.gamma.unwrap_or(0.1);
        let period = self.config.parameters.seasonal_period.unwrap_or(12);

        let mut model = HoltWintersModel::new(alpha, beta, gamma, period)?;
        model.fit(data)?;
        let forecast = model.forecast(periods)?;

        Ok(ForecastOutput {
            forecast,
            model_used: format!(
                "HoltWinters(α={}, β={}, γ={}, s={})",
                alpha, beta, gamma, period
            ),
            confidence_intervals: None,
            metrics: None,
        })
    }

    fn forecast_ma(&self, data: &TimeSeriesData, periods: usize) -> Result<ForecastOutput> {
        let window = self.config.parameters.ma_window.unwrap_or(5);

        let mut model =
            MAModel::new(window).map_err(|e| OxiError::ModelError(format!("{:?}", e)))?;
        model.fit(data)?;
        let forecast = model.forecast(periods)?;

        Ok(ForecastOutput {
            forecast,
            model_used: format!("MA({})", window),
            confidence_intervals: None,
            metrics: None,
        })
    }

    fn forecast_auto(&self, data: &TimeSeriesData, periods: usize) -> Result<ForecastOutput> {
        // Simple auto-selection: try a few models and pick the one with lowest MAE
        let models = vec![
            ("ARIMA(1,1,1)", ModelType::ARIMA),
            ("SimpleES", ModelType::SimpleES),
            ("MA(5)", ModelType::MovingAverage),
        ];

        let mut best_model = ModelType::ARIMA;
        let mut best_error = f64::INFINITY;
        let split_point = (data.len() as f64 * 0.8) as usize;

        if split_point > 10 {
            let train_data = TimeSeriesData::new(
                data.timestamps[..split_point].to_vec(),
                data.values[..split_point].to_vec(),
                "train",
            )?;
            let test_data = TimeSeriesData::new(
                data.timestamps[split_point..].to_vec(),
                data.values[split_point..].to_vec(),
                "test",
            )?;

            for (_name, model_type) in &models {
                let mut config = self.config.clone();
                config.model_type = model_type.clone();
                let forecaster = Forecaster::with_config(config);

                if let Ok(forecast_output) = forecaster.forecast(&train_data, test_data.len()) {
                    let mae = test_data
                        .values
                        .iter()
                        .zip(forecast_output.forecast.iter())
                        .map(|(actual, predicted)| (actual - predicted).abs())
                        .sum::<f64>()
                        / test_data.len() as f64;

                    if mae < best_error {
                        best_error = mae;
                        best_model = model_type.clone();
                    }
                }
            }
        }

        // Use the best model to forecast on full data
        let mut config = self.config.clone();
        config.model_type = best_model;
        let forecaster = Forecaster::with_config(config);
        forecaster.forecast(data, periods)
    }
}

impl Default for Forecaster {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder pattern for creating forecasters
pub struct ForecastBuilder {
    config: ForecastConfig,
}

impl ForecastBuilder {
    /// Create a new forecast builder
    pub fn new() -> Self {
        Self {
            config: ForecastConfig::default(),
        }
    }

    /// Set the model type
    pub fn model(mut self, model_type: ModelType) -> Self {
        self.config.model_type = model_type;
        self
    }

    /// Configure for ARIMA forecasting
    pub fn arima(mut self, p: usize, d: usize, q: usize) -> Self {
        self.config.model_type = ModelType::ARIMA;
        self.config.parameters.arima_p = Some(p);
        self.config.parameters.arima_d = Some(d);
        self.config.parameters.arima_q = Some(q);
        self
    }

    /// Configure for Simple Exponential Smoothing
    pub fn simple_es(mut self, alpha: f64) -> Self {
        self.config.model_type = ModelType::SimpleES;
        self.config.parameters.alpha = Some(alpha);
        self
    }

    /// Configure for Moving Average
    pub fn moving_average(mut self, window: usize) -> Self {
        self.config.model_type = ModelType::MovingAverage;
        self.config.parameters.ma_window = Some(window);
        self
    }

    /// Enable automatic model selection
    pub fn auto(mut self) -> Self {
        self.config.model_type = ModelType::Auto;
        self.config.auto_select = true;
        self
    }

    /// Build the forecaster
    pub fn build(self) -> Forecaster {
        Forecaster::with_config(self.config)
    }
}

impl Default for ForecastBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating individual model wrappers with fluent interface
#[derive(Clone)]
pub struct ModelBuilder {
    model_type: String,
    params: HashMap<String, f64>,
    int_params: HashMap<String, usize>,
}

impl ModelBuilder {
    /// Start building an ARIMA model
    pub fn arima() -> Self {
        Self {
            model_type: "ARIMA".to_string(),
            params: HashMap::new(),
            int_params: HashMap::new(),
        }
    }

    /// Start building an Exponential Smoothing model
    pub fn exponential_smoothing() -> Self {
        Self {
            model_type: "ES".to_string(),
            params: HashMap::new(),
            int_params: HashMap::new(),
        }
    }

    /// Start building a Moving Average model
    pub fn moving_average() -> Self {
        Self {
            model_type: "MA".to_string(),
            params: HashMap::new(),
            int_params: HashMap::new(),
        }
    }

    /// Start building a GARCH model
    pub fn garch() -> Self {
        Self {
            model_type: "GARCH".to_string(),
            params: HashMap::new(),
            int_params: HashMap::new(),
        }
    }

    /// Set autoregressive order
    pub fn with_ar(mut self, p: usize) -> Self {
        self.int_params.insert("p".to_string(), p);
        self
    }

    /// Set differencing order
    pub fn with_differencing(mut self, d: usize) -> Self {
        self.int_params.insert("d".to_string(), d);
        self
    }

    /// Set moving average order
    pub fn with_ma(mut self, q: usize) -> Self {
        self.int_params.insert("q".to_string(), q);
        self
    }

    /// Set alpha parameter
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.params.insert("alpha".to_string(), alpha);
        self
    }

    /// Set beta parameter
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.params.insert("beta".to_string(), beta);
        self
    }

    /// Set gamma parameter
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.params.insert("gamma".to_string(), gamma);
        self
    }

    /// Set damping parameter
    pub fn with_phi(mut self, phi: f64) -> Self {
        self.params.insert("phi".to_string(), phi);
        self
    }

    /// Set seasonal period
    pub fn with_period(mut self, period: usize) -> Self {
        self.int_params.insert("period".to_string(), period);
        self
    }

    /// Set window size
    pub fn with_window(mut self, window: usize) -> Self {
        self.int_params.insert("window".to_string(), window);
        self
    }

    /// Set GARCH p parameter
    pub fn with_garch_p(mut self, p: usize) -> Self {
        self.int_params.insert("garch_p".to_string(), p);
        self
    }

    /// Set GARCH q parameter
    pub fn with_garch_q(mut self, q: usize) -> Self {
        self.int_params.insert("garch_q".to_string(), q);
        self
    }

    /// Build the model wrapper
    pub fn build(self) -> Result<Box<dyn QuickForecaster>> {
        match self.model_type.as_str() {
            "ARIMA" => {
                let p = self.int_params.get("p").copied().unwrap_or(1);
                let d = self.int_params.get("d").copied().unwrap_or(1);
                let q = self.int_params.get("q").copied().unwrap_or(1);
                Ok(Box::new(ARIMAWrapper::new(p, d, q)?))
            }
            "ES" => {
                let alpha = self.params.get("alpha").copied().unwrap_or(0.3);
                Ok(Box::new(ESWrapper::new(alpha)?))
            }
            "MA" => {
                let window = self.int_params.get("window").copied().unwrap_or(5);
                Ok(Box::new(MAWrapper::new(window)?))
            }
            "GARCH" => {
                let p = self.int_params.get("garch_p").copied().unwrap_or(1);
                let q = self.int_params.get("garch_q").copied().unwrap_or(1);
                Ok(Box::new(GARCHWrapper::new(p, q)?))
            }
            _ => Err(OxiError::ModelError(format!(
                "Unknown model type: {}",
                self.model_type
            ))),
        }
    }

    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    pub fn get_param(&self, key: &str) -> Option<f64> {
        self.params.get(key).copied()
    }

    pub fn get_int_param(&self, key: &str) -> Option<usize> {
        self.int_params.get(key).copied()
    }
}

/// Configuration for ARIMA models built with the builder pattern
#[derive(Debug, Clone)]
pub struct ARIMABuilderConfig {
    pub p: usize,
    pub d: usize,
    pub q: usize,
}

/// Configuration for Exponential Smoothing models built with the builder pattern
#[derive(Debug, Clone)]
pub struct ESBuilderConfig {
    pub alpha: f64,
    pub beta: Option<f64>,
    pub gamma: Option<f64>,
    pub phi: Option<f64>,
    pub period: Option<usize>,
}

/// Configuration for Moving Average models built with the builder pattern
#[derive(Debug, Clone)]
pub struct MABuilderConfig {
    pub window: usize,
}

/// Configuration for GARCH models built with the builder pattern
#[derive(Debug, Clone)]
pub struct GARCHBuilderConfig {
    pub p: usize,
    pub q: usize,
}

/// Wrapper for ARIMA models to implement QuickForecaster
pub struct ARIMAWrapper {
    model: crate::models::autoregressive::ARIMAModel,
}

impl std::fmt::Debug for ARIMAWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ARIMAWrapper")
            .field("model_type", &"ARIMA")
            .finish()
    }
}

impl ARIMAWrapper {
    pub fn new(p: usize, d: usize, q: usize) -> Result<Self> {
        let model = ARIMAModel::new(p, d, q, true)?;
        Ok(Self { model })
    }
}

impl QuickForecaster for ARIMAWrapper {
    fn quick_fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        self.model.fit(data)
    }

    fn quick_forecast(&self, periods: usize) -> Result<Vec<f64>> {
        self.model.forecast(periods)
    }

    fn model_name(&self) -> &'static str {
        "ARIMA"
    }

    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        self.model.evaluate(test_data)
    }

    fn fitted_values(&self) -> Option<Vec<f64>> {
        None // TODO: implement if needed
    }
}

/// Wrapper for Exponential Smoothing models to implement QuickForecaster
pub struct ESWrapper {
    model: crate::models::exponential_smoothing::SimpleESModel,
}

impl std::fmt::Debug for ESWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ESWrapper")
            .field("model_type", &"Exponential Smoothing")
            .finish()
    }
}

impl ESWrapper {
    pub fn new(alpha: f64) -> Result<Self> {
        let model = SimpleESModel::new(alpha)?;
        Ok(Self { model })
    }
}

impl QuickForecaster for ESWrapper {
    fn quick_fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        self.model.fit(data)
    }

    fn quick_forecast(&self, periods: usize) -> Result<Vec<f64>> {
        self.model.forecast(periods)
    }

    fn model_name(&self) -> &'static str {
        "Exponential Smoothing"
    }

    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        self.model.evaluate(test_data)
    }

    fn fitted_values(&self) -> Option<Vec<f64>> {
        None // TODO: implement if needed
    }
}

/// Wrapper for Moving Average models to implement QuickForecaster
pub struct MAWrapper {
    model: crate::models::moving_average::MAModel,
}

impl std::fmt::Debug for MAWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MAWrapper")
            .field("model_type", &"Moving Average")
            .finish()
    }
}

impl MAWrapper {
    pub fn new(window: usize) -> Result<Self> {
        let model = MAModel::new(window).map_err(|e| OxiError::ModelError(format!("{:?}", e)))?;
        Ok(Self { model })
    }
}

impl QuickForecaster for MAWrapper {
    fn quick_fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        self.model.fit(data)
    }

    fn quick_forecast(&self, periods: usize) -> Result<Vec<f64>> {
        self.model.forecast(periods)
    }

    fn model_name(&self) -> &'static str {
        "Moving Average"
    }

    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        self.model.evaluate(test_data)
    }

    fn fitted_values(&self) -> Option<Vec<f64>> {
        self.model.fitted_values().cloned()
    }
}

/// Wrapper for GARCH models to implement QuickForecaster
pub struct GARCHWrapper {
    model: crate::models::garch::GARCHModel,
}

impl std::fmt::Debug for GARCHWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GARCHWrapper")
            .field("model_type", &"GARCH")
            .finish()
    }
}

impl GARCHWrapper {
    pub fn new(p: usize, q: usize) -> Result<Self> {
        let model = crate::models::garch::GARCHModel::new(p, q, None)?;
        Ok(Self { model })
    }
}

impl QuickForecaster for GARCHWrapper {
    fn quick_fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        // Convert TimeSeriesData to the format expected by GARCH
        self.model
            .fit(&data.values, Some(&data.timestamps))
            .map_err(|e| OxiError::ModelError(format!("GARCH fit error: {:?}", e)))
    }

    fn quick_forecast(&self, periods: usize) -> Result<Vec<f64>> {
        // GARCH forecasts variance, not the actual values
        // For simplicity, we'll return variance forecasts
        self.model
            .forecast_variance(periods)
            .map_err(|e| OxiError::ModelError(format!("GARCH forecast error: {:?}", e)))
    }

    fn model_name(&self) -> &'static str {
        "GARCH"
    }

    fn evaluate(&self, _test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        // Simple placeholder implementation
        Ok(ModelEvaluation {
            model_name: "GARCH".to_string(),
            mae: 0.0,
            mse: 0.0,
            rmse: 0.0,
            mape: 0.0,
            smape: 0.0,
            r_squared: 0.0,
            aic: self.model.info_criteria.map(|(aic, _)| aic),
            bic: self.model.info_criteria.map(|(_, bic)| bic),
        })
    }

    fn fitted_values(&self) -> Option<Vec<f64>> {
        self.model.fitted_variance.clone()
    }
}

/// Automatic model selector for finding the best model
pub struct AutoSelector {
    criteria: SelectionCriteria,
    models_to_try: Vec<String>,
    max_models: usize,
}

/// Criteria for selecting the best model
#[derive(Debug, Clone)]
pub enum SelectionCriteria {
    /// Akaike Information Criterion (penalizes model complexity)
    AIC,
    /// Bayesian Information Criterion (stronger penalty for complexity)
    BIC,
    /// Cross-validation with specified number of folds
    CrossValidation { folds: usize },
    /// Out-of-sample validation with specified test ratio
    OutOfSample { test_ratio: f64 },
    /// Mean Absolute Error on validation set
    MAE,
    /// Mean Squared Error on validation set
    MSE,
}

/// Result of model selection
#[derive(Debug)]
pub struct SelectionResult {
    /// The best model found
    pub best_model: Box<dyn QuickForecaster>,
    /// Score achieved by the best model
    pub best_score: f64,
    /// Name of the best model
    pub model_name: String,
    /// Scores for all models tried
    pub all_scores: Vec<(String, f64)>,
    /// Details about the selection process
    pub selection_info: String,
}

impl AutoSelector {
    /// Create selector using AIC criterion
    pub fn with_aic() -> Self {
        Self {
            criteria: SelectionCriteria::AIC,
            models_to_try: vec!["ARIMA".to_string(), "ES".to_string(), "MA".to_string()],
            max_models: 10,
        }
    }

    /// Create selector using BIC criterion
    pub fn with_bic() -> Self {
        Self {
            criteria: SelectionCriteria::BIC,
            models_to_try: vec!["ARIMA".to_string(), "ES".to_string(), "MA".to_string()],
            max_models: 10,
        }
    }

    /// Create selector using cross-validation
    pub fn with_cross_validation(folds: usize) -> Self {
        Self {
            criteria: SelectionCriteria::CrossValidation { folds },
            models_to_try: vec!["ARIMA".to_string(), "ES".to_string(), "MA".to_string()],
            max_models: 10,
        }
    }

    /// Create selector using out-of-sample validation
    pub fn with_out_of_sample(test_ratio: f64) -> Self {
        Self {
            criteria: SelectionCriteria::OutOfSample { test_ratio },
            models_to_try: vec!["ARIMA".to_string(), "ES".to_string(), "MA".to_string()],
            max_models: 10,
        }
    }

    /// Set maximum number of models to try
    pub fn max_models(mut self, max: usize) -> Self {
        self.max_models = max;
        self
    }

    /// Add a model type to try
    pub fn add_model(mut self, model_type: &str) -> Self {
        if !self.models_to_try.contains(&model_type.to_string()) {
            self.models_to_try.push(model_type.to_string());
        }
        self
    }

    /// Select the best model for the given data
    pub fn select_best(
        &self,
        data: &TimeSeriesData,
    ) -> Result<(Box<dyn QuickForecaster>, f64, String)> {
        let configs = self.generate_model_configs()?;

        let mut best_score = f64::INFINITY;
        let mut best_model: Option<Box<dyn QuickForecaster>> = None;
        let mut best_name = String::new();
        let mut all_scores = Vec::new();

        for config in configs.iter().take(self.max_models) {
            match self.evaluate_model_config(config, data) {
                Ok((score, model, name)) => {
                    all_scores.push((name.clone(), score));
                    if score < best_score {
                        best_score = score;
                        best_model = Some(model);
                        best_name = name;
                    }
                }
                Err(_) => {
                    // Skip models that fail to fit
                    continue;
                }
            }
        }

        match best_model {
            Some(model) => Ok((model, best_score, best_name)),
            None => Err(OxiError::ModelError(
                "No models could be fitted".to_string(),
            )),
        }
    }

    fn generate_model_configs(&self) -> Result<Vec<ModelBuilder>> {
        let mut configs = Vec::new();

        if self.models_to_try.contains(&"ARIMA".to_string()) {
            // Try different ARIMA configurations
            for p in 1..=3 {
                for d in 0..=2 {
                    for q in 1..=3 {
                        configs.push(
                            ModelBuilder::arima()
                                .with_ar(p)
                                .with_differencing(d)
                                .with_ma(q),
                        );
                    }
                }
            }
        }

        if self.models_to_try.contains(&"ES".to_string()) {
            // Try different alpha values
            for alpha in [0.1, 0.3, 0.5, 0.7, 0.9] {
                configs.push(ModelBuilder::exponential_smoothing().with_alpha(alpha));
            }
        }

        if self.models_to_try.contains(&"MA".to_string()) {
            // Try different window sizes
            for window in [3, 5, 7, 10, 15] {
                configs.push(ModelBuilder::moving_average().with_window(window));
            }
        }

        Ok(configs)
    }

    fn evaluate_model_config(
        &self,
        config: &ModelBuilder,
        data: &TimeSeriesData,
    ) -> Result<(f64, Box<dyn QuickForecaster>, String)> {
        let mut model = config.clone().build()?;
        let model_name = format!("{:?}", config.model_type());

        match &self.criteria {
            SelectionCriteria::AIC => {
                model.quick_fit(data)?;
                let evaluation = model.evaluate(data)?;
                let score = evaluation.aic.unwrap_or(f64::INFINITY);
                Ok((score, model, model_name))
            }
            SelectionCriteria::BIC => {
                model.quick_fit(data)?;
                let evaluation = model.evaluate(data)?;
                let score = evaluation.bic.unwrap_or(f64::INFINITY);
                Ok((score, model, model_name))
            }
            SelectionCriteria::OutOfSample { test_ratio } => {
                let split_point = ((1.0 - test_ratio) * data.len() as f64) as usize;
                if split_point < 10 {
                    return Err(OxiError::ModelError(
                        "Not enough data for out-of-sample validation".to_string(),
                    ));
                }

                let train_data = TimeSeriesData::new(
                    data.timestamps[..split_point].to_vec(),
                    data.values[..split_point].to_vec(),
                    "train",
                )?;
                let test_data = TimeSeriesData::new(
                    data.timestamps[split_point..].to_vec(),
                    data.values[split_point..].to_vec(),
                    "test",
                )?;

                model.quick_fit(&train_data)?;
                let forecast = model.quick_forecast(test_data.len())?;

                let mae = test_data
                    .values
                    .iter()
                    .zip(forecast.iter())
                    .map(|(actual, predicted)| (actual - predicted).abs())
                    .sum::<f64>()
                    / test_data.len() as f64;

                Ok((mae, model, model_name))
            }
            SelectionCriteria::CrossValidation { folds } => {
                let score = self.cross_validate(config, data, *folds)?;
                model.quick_fit(data)?; // Fit on full data for final model
                Ok((score, model, model_name))
            }
            SelectionCriteria::MAE => {
                model.quick_fit(data)?;
                let evaluation = model.evaluate(data)?;
                Ok((evaluation.mae, model, model_name))
            }
            SelectionCriteria::MSE => {
                model.quick_fit(data)?;
                let evaluation = model.evaluate(data)?;
                Ok((evaluation.mse, model, model_name))
            }
        }
    }

    fn cross_validate(
        &self,
        config: &ModelBuilder,
        data: &TimeSeriesData,
        folds: usize,
    ) -> Result<f64> {
        if data.len() < folds * 10 {
            return Err(OxiError::ModelError(
                "Not enough data for cross-validation".to_string(),
            ));
        }

        let fold_size = data.len() / folds;
        let mut total_error = 0.0;
        let mut valid_folds = 0;

        for i in 0..folds {
            let test_start = i * fold_size;
            let test_end = if i == folds - 1 {
                data.len()
            } else {
                (i + 1) * fold_size
            };

            // Create train and test sets
            let mut train_timestamps = Vec::new();
            let mut train_values = Vec::new();

            // Add data before test fold
            train_timestamps.extend_from_slice(&data.timestamps[..test_start]);
            train_values.extend_from_slice(&data.values[..test_start]);

            // Add data after test fold
            if test_end < data.len() {
                train_timestamps.extend_from_slice(&data.timestamps[test_end..]);
                train_values.extend_from_slice(&data.values[test_end..]);
            }

            if train_values.len() < 10 {
                continue; // Skip if not enough training data
            }

            let train_data = TimeSeriesData::new(train_timestamps, train_values, "cv_train")?;
            let test_values = &data.values[test_start..test_end];

            let mut model = config.clone().build()?;
            if model.quick_fit(&train_data).is_ok() {
                if let Ok(forecast) = model.quick_forecast(test_values.len()) {
                    let fold_error = test_values
                        .iter()
                        .zip(forecast.iter())
                        .map(|(actual, predicted)| (actual - predicted).abs())
                        .sum::<f64>()
                        / test_values.len() as f64;

                    total_error += fold_error;
                    valid_folds += 1;
                }
            }
        }

        if valid_folds > 0 {
            Ok(total_error / valid_folds as f64)
        } else {
            Err(OxiError::ModelError(
                "No valid folds in cross-validation".to_string(),
            ))
        }
    }
}
