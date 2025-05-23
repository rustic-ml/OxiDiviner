//! High-level API module
//!
//! This module provides a simplified, high-level interface for forecasting
//! that abstracts away the complexity of individual models and provides
//! a unified way to work with different forecasting algorithms.

use crate::core::{OxiError, Result, TimeSeriesData};
use crate::models::autoregressive::ARIMAModel;
use crate::models::exponential_smoothing::{SimpleESModel, HoltLinearModel, HoltWintersModel};
use crate::models::moving_average::MAModel;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

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
    pub fn forecast(
        &self,
        data: &TimeSeriesData,
        periods: usize,
    ) -> Result<ForecastOutput> {
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

    fn forecast_holt_linear(&self, data: &TimeSeriesData, periods: usize) -> Result<ForecastOutput> {
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

    fn forecast_holt_winters(&self, data: &TimeSeriesData, periods: usize) -> Result<ForecastOutput> {
        let alpha = self.config.parameters.alpha.unwrap_or(0.3);
        let beta = self.config.parameters.beta.unwrap_or(0.1);
        let gamma = self.config.parameters.gamma.unwrap_or(0.1);
        let seasonal_period = self.config.parameters.seasonal_period.unwrap_or(12);

        let mut model = HoltWintersModel::new(alpha, beta, gamma, seasonal_period)?;
        model.fit(data)?;
        let forecast = model.forecast(periods)?;

        Ok(ForecastOutput {
            forecast,
            model_used: format!("HoltWinters(α={}, β={}, γ={})", alpha, beta, gamma),
            confidence_intervals: None,
            metrics: None,
        })
    }

    fn forecast_ma(&self, data: &TimeSeriesData, periods: usize) -> Result<ForecastOutput> {
        let window = self.config.parameters.ma_window.unwrap_or(5);

        let mut model = MAModel::new(window)?;
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
        // Try different models and return the first successful one
        // Use individual function calls instead of closures to avoid type issues
        
        // Try ARIMA first
        if let Ok(mut output) = self.forecast_arima(data, periods) {
            output.model_used = format!("Auto({})", output.model_used);
            return Ok(output);
        }

        // Try Simple ES next
        if let Ok(mut output) = self.forecast_simple_es(data, periods) {
            output.model_used = format!("Auto({})", output.model_used);
            return Ok(output);
        }

        // Try Moving Average last
        if let Ok(mut output) = self.forecast_ma(data, periods) {
            output.model_used = format!("Auto({})", output.model_used);
            return Ok(output);
        }

        Err(OxiError::ModelError("All models failed in auto mode".to_string())) // Use existing error type
    }
}

impl Default for Forecaster {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder pattern for easy configuration
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

    /// Set ARIMA parameters
    pub fn arima(mut self, p: usize, d: usize, q: usize) -> Self {
        self.config.model_type = ModelType::ARIMA;
        self.config.parameters.arima_p = Some(p);
        self.config.parameters.arima_d = Some(d);
        self.config.parameters.arima_q = Some(q);
        self
    }

    /// Set simple exponential smoothing
    pub fn simple_es(mut self, alpha: f64) -> Self {
        self.config.model_type = ModelType::SimpleES;
        self.config.parameters.alpha = Some(alpha);
        self
    }

    /// Set moving average
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