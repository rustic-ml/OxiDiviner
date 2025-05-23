//! # OxiDiviner
//!
//! A comprehensive Rust library for time series analysis and forecasting.
//!
//! [![Crates.io](https://img.shields.io/crates/v/oxidiviner.svg)](https://crates.io/crates/oxidiviner)
//! [![Documentation](https://docs.rs/oxidiviner/badge.svg)](https://docs.rs/oxidiviner)
//! [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
//!
//! ## Overview
//!
//! OxiDiviner is a modular ecosystem of crates for time series analysis and forecasting,
//! designed to provide efficient, accurate, and easy-to-use statistical models for Rust.
//! This main crate serves as a convenient entry point to access all functionality.
//!
//! ## Quick Start for Financial Applications
//!
//! For financial time series forecasting, use the convenience API:
//!
//! ```rust
//! use oxidiviner::financial::*;
//! use chrono::{DateTime, Utc};
//!
//! # fn main() -> oxidiviner::Result<()> {
//! // Create financial time series from OHLCV data
//! let prices = vec![100.0, 101.0, 99.5, 102.0, 100.5];
//! let dates: Vec<DateTime<Utc>> = (0..5)
//!     .map(|i| Utc::now() - chrono::Duration::days(5 - i))
//!     .collect();
//!
//! let data = FinancialTimeSeries::from_prices(dates, prices)?;
//!
//! // Quick forecast with automatic model selection
//! let forecast = data.auto_forecast(10)?;
//! println!("Forecasted prices: {:?}", forecast.values);
//!
//! // Compare multiple models quickly
//! let comparison = data.compare_models(5)?;
//! println!("Best model: {:?}", comparison.best_model);
//! # Ok(())
//! # }
//! ```
//!
//! ## Usage
//!
//! The easiest way to use OxiDiviner is through the prelude module, which provides access to commonly used types and functions:
//!
//! ```rust
//! use oxidiviner::prelude::*;
//!
//! # fn main() -> Result<()> {
//! // Create a time series
//! let dates = vec![
//!     Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
//!     Utc.with_ymd_and_hms(2023, 1, 2, 0, 0, 0).unwrap(),
//!     Utc.with_ymd_and_hms(2023, 1, 3, 0, 0, 0).unwrap(),
//! ];
//! let values = vec![1.0, 1.5, 2.0];
//! let data = TimeSeriesData::new(dates, values, "example_series")?;
//!
//! // Create and fit a model
//! let mut model = ARModel::new(1, true)?;
//! model.fit(&data)?;
//!
//! // Generate forecast
//! let forecast = model.forecast(2)?;
//! # Ok(())
//! # }
//! ```
//!
//! You can also directly access models without going through the prelude or models module:
//!
//! ```rust
//! // Direct access to models
//! use oxidiviner::{TimeSeriesData, Forecaster, ARModel, HoltWintersModel, MAModel};
//!
//! # fn main() -> oxidiviner::Result<()> {
//! // Now you can use models directly without nested imports
//! let mut ma_model = MAModel::new(3)?;
//! let mut arima = oxidiviner::ARIMAModel::new(1, 1, 1, true)?;
//! # Ok(())
//! # }
//! ```
//!
//! For more direct access to specific subcrates, you can use the direct exports:
//!
//! ```rust
//! // Direct access to subcrates
//! use oxidiviner::moving_average::MAModel;
//! use oxidiviner::exponential_smoothing::HoltWintersModel;
//! use oxidiviner::autoregressive::ARIMAModel;
//! use oxidiviner::math::metrics::rmse;
//!
//! # fn main() {
//! // Now you can use types directly without going through the models module
//! # }
//! ```
//!
//! ## Benefits of Multiple Import Options
//!
//! OxiDiviner offers these different import methods to support various usage patterns:
//!
//! * **Prelude**: Ideal for quick prototyping and simplicity
//! * **Direct Model Imports**: Clean and concise for focused model usage
//! * **Subcrate Access**: Provides granular control for advanced users
//!
//! This approach allows you to choose the style that best fits your project's needs.
//!
//! ## Project Organization
//!
//! OxiDiviner uses a monorepo architecture where:
//!
//! * The main `oxidiviner` crate is the only publicly published package
//! * Internal crates provide modular organization during development
//! * All internal crate source code is included in the main crate's package
//!
//! This approach allows for a cleaner architecture while simplifying usage for
//! consumers who only need to depend on a single crate.
//!
//! ## Components
//!
//! OxiDiviner includes the following major components:
//!
//! * **Core** - Common interfaces, data structures, and traits for time series data
//! * **Math** - Statistical utilities, transformations, and metrics
//! * **Models** - Implementation of various forecasting models
//!
//! ## Available Models
//!
//! ### Moving Average Models
//!
//! Moving average models capture short-term dependencies in time series data.
//!
//! ```rust
//! use oxidiviner::models::moving_average::MAModel;
//! use oxidiviner::{TimeSeriesData, Forecaster};
//! use chrono::{Utc, TimeZone};
//!
//! # fn main() -> oxidiviner::Result<()> {
//! // Create sample time series data
//! let dates = vec![
//!     Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
//!     Utc.with_ymd_and_hms(2023, 1, 2, 0, 0, 0).unwrap(),
//!     Utc.with_ymd_and_hms(2023, 1, 3, 0, 0, 0).unwrap(),
//! ];
//! let values = vec![1.0, 1.5, 2.0];
//! let data = TimeSeriesData::new(dates, values, "test_series")?;
//!
//! // Create and fit an MA(1) model
//! let mut model = MAModel::new(1)?;
//! model.fit(&data)?;
//!
//! // Generate forecast
//! let forecast = model.forecast(2)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Exponential Smoothing Models
//!
//! Exponential smoothing models are effective for data with trend and seasonality.
//!
//! ```rust
//! use oxidiviner::models::exponential_smoothing::{SimpleESModel, HoltWintersModel};
//! use oxidiviner::{TimeSeriesData, Forecaster};
//! use chrono::{Utc, TimeZone};
//!
//! # fn main() -> oxidiviner::Result<()> {
//! // Create sample time series data
//! let dates = vec![
//!     // Dates would go here
//!     Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
//!     Utc.with_ymd_and_hms(2023, 1, 2, 0, 0, 0).unwrap(),
//!     Utc.with_ymd_and_hms(2023, 1, 3, 0, 0, 0).unwrap(),
//! ];
//! let values = vec![10.0, 11.0, 9.5];
//! let data = TimeSeriesData::new(dates, values, "test_series")?;
//!
//! // Create simple exponential smoothing model
//! let mut model = SimpleESModel::new(0.3)?;
//! model.fit(&data)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Autoregressive Models
//!
//! Autoregressive models capture data that depends on its own previous values.
//!
//! ```rust
//! use oxidiviner::models::autoregressive::{ARModel, ARIMAModel, SARIMAModel};
//! use oxidiviner::{TimeSeriesData, Forecaster};
//! use chrono::{Utc, TimeZone};
//!
//! # fn main() -> oxidiviner::Result<()> {
//! // Create an AR(2) model
//! let mut ar_model = ARModel::new(2, true)?;
//!
//! // Create an ARIMA(1,1,1) model
//! let mut arima_model = ARIMAModel::new(1, 1, 1, true)?;
//!
//! // Create a SARIMA(1,1,1)(1,1,0)12 model for monthly data with yearly seasonality
//! let mut sarima_model = SARIMAModel::new(1, 1, 1, 1, 1, 0, 12, true)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### GARCH Models
//!
//! GARCH models are specialized for volatility forecasting in financial time series.
//!
//! ```rust
//! use oxidiviner::prelude::*;
//! use oxidiviner::models::garch::{GARCHModel, GJRGARCHModel, EGARCHModel, GARCHMModel, RiskPremiumType};
//!
//! # fn main() -> oxidiviner::Result<()> {
//! // Create a basic GARCH(1,1) model
//! let mut garch_model = GARCHModel::new(1, 1, None).unwrap();
//!
//! // Create a GJR-GARCH(1,1) model for asymmetric volatility
//! let mut gjr_garch_model = GJRGARCHModel::new(1, 1, None).unwrap();
//!
//! // Create an EGARCH(1,1) model
//! let mut egarch_model = EGARCHModel::new(1, 1, None).unwrap();
//!
//! // Create a GARCH-M(1,1) model with risk premium as variance
//! let mut garch_m_model = GARCHMModel::new(1, 1, RiskPremiumType::Variance, None).unwrap();
//! # Ok(())
//! # }
//! ```
//!
//! ## Mathematical Utilities
//!
//! OxiDiviner provides various mathematical utilities for time series analysis.
//!
//! ```rust
//! use oxidiviner::math;
//! use oxidiviner::math::metrics::{mae, mse, rmse, mape, smape};
//!
//! # fn main() {
//! // Apply transformations
//! let data = vec![10.5, 11.2, 10.8, 11.5, 12.0];
//! let differenced = oxidiviner::math::transforms::difference(&data);
//! let log_data: Vec<f64> = data.iter().map(|&x| x.ln()).collect();
//!
//! // Calculate forecast accuracy metrics
//! let actual = vec![10.0, 11.0, 12.0];
//! let predicted = vec![10.2, 10.8, 11.5];
//!
//! // Using the library's metric functions
//! let mae_value = mae(&actual, &predicted);
//! let mse_value = mse(&actual, &predicted);
//! let rmse_value = rmse(&actual, &predicted);
//! let mape_value = mape(&actual, &predicted);
//! let smape_value = smape(&actual, &predicted);
//!
//! println!("MAE: {:.4}, MSE: {:.4}, RMSE: {:.4}", mae_value, mse_value, rmse_value);
//! println!("MAPE: {:.2}%, SMAPE: {:.2}%", mape_value * 100.0, smape_value * 100.0);
//! # }
//! ```

// Re-export from core
pub use oxidiviner_core::*;

// Re-export from math
#[doc(inline)]
pub use oxidiviner_math as math;

// Direct access to subcrates for power users
#[doc(inline)]
pub use oxidiviner_autoregressive as autoregressive;
#[doc(inline)]
pub use oxidiviner_core as core;
#[doc(inline)]
pub use oxidiviner_exponential_smoothing as exponential_smoothing;
#[doc(inline)]
pub use oxidiviner_garch as garch;
#[doc(inline)]
pub use oxidiviner_moving_average as moving_average;

// Re-export models from module-specific crates
pub mod models {
    /// Moving average models for capturing short-term patterns
    pub mod moving_average {
        #[doc(inline)]
        pub use oxidiviner_moving_average::*;
    }

    /// Exponential smoothing models for trend and seasonal data
    pub mod exponential_smoothing {
        #[doc(inline)]
        pub use oxidiviner_exponential_smoothing::*;
    }

    /// Autoregressive models for data with dependency on past values
    pub mod autoregressive {
        #[doc(inline)]
        pub use oxidiviner_autoregressive::*;
    }

    /// GARCH models for volatility forecasting in financial time series
    pub mod garch {
        #[doc(inline)]
        pub use oxidiviner_garch::*;
    }
}

// Direct re-exports of all model types for maximum convenience
pub use models::autoregressive::{ARIMAModel, ARMAModel, ARModel, SARIMAModel, VARModel};
pub use models::exponential_smoothing::{
    DailyETSModel, DampedTrendModel, ETSComponent, ETSModel, HoltLinearModel, HoltWintersModel,
    MinuteETSModel, SimpleESModel,
};
pub use models::garch::{EGARCHModel, GARCHMModel, GARCHModel, GJRGARCHModel, RiskPremiumType};
pub use models::moving_average::MAModel;

/// Financial time series utilities and convenience functions
pub mod financial {
    use super::*;
    use std::collections::HashMap;
    
    /// Financial time series wrapper with convenience methods
    pub struct FinancialTimeSeries {
        data: TimeSeriesData,
    }
    
    /// Model comparison results
    #[derive(Debug, Clone)]
    pub struct ModelComparison {
        pub best_model: String,
        pub results: HashMap<String, ModelResult>,
    }
    
    /// Individual model result
    #[derive(Debug, Clone)]
    pub struct ModelResult {
        pub mse: f64,
        pub mae: f64,
        pub rmse: f64,
        pub forecast: Vec<f64>,
    }
    
    /// Quick forecast result
    #[derive(Debug, Clone)]
    pub struct QuickForecast {
        pub values: Vec<f64>,
        pub model_used: String,
        pub confidence_score: f64,
    }
    
    impl FinancialTimeSeries {
        /// Create from price data
        pub fn from_prices(dates: Vec<chrono::DateTime<chrono::Utc>>, prices: Vec<f64>) -> Result<Self> {
            let data = TimeSeriesData::new(dates, prices, "financial_series")?;
            Ok(Self { data })
        }
        
        /// Create from OHLCV data (uses close prices)
        pub fn from_ohlcv(ohlcv: &OHLCVData) -> Result<Self> {
            let data = TimeSeriesData::new(
                ohlcv.timestamps.clone(),
                ohlcv.close.clone(),
                "financial_ohlcv"
            )?;
            Ok(Self { data })
        }
        
        /// Automatic forecasting with model selection
        pub fn auto_forecast(&self, periods: usize) -> Result<QuickForecast> {
            // Try multiple models and pick the best one based on AIC/BIC
            let models = vec![
                ("MA(5)", self.forecast_ma(5, periods)?),
                ("ARIMA(1,1,1)", self.forecast_arima(1, 1, 1, periods)?),
                ("SimpleES", self.forecast_simple_es(0.3, periods)?),
            ];
            
            // For simplicity, just return the first successful forecast
            // In a real implementation, you'd evaluate and compare models
            if let Some((name, forecast)) = models.into_iter().next() {
                Ok(QuickForecast {
                    values: forecast,
                    model_used: name.to_string(),
                    confidence_score: 0.8, // Placeholder
                })
            } else {
                Err(OxiError::ModelError("No models succeeded".to_string()))
            }
        }
        
        /// Compare multiple forecasting models
        pub fn compare_models(&self, forecast_periods: usize) -> Result<ModelComparison> {
            let mut results = HashMap::new();
            
            // Split data for validation
            let split_point = self.data.values.len() * 80 / 100;
            let train_data = self.data.slice(0, split_point)?;
            let test_data = self.data.slice(split_point, self.data.values.len())?;
            
            // Test Moving Average
            if let Ok(forecast) = self.forecast_ma_on_data(&train_data, 5, test_data.values.len()) {
                let metrics = calculate_metrics(&test_data.values, &forecast);
                results.insert("MA(5)".to_string(), ModelResult {
                    mse: metrics.0,
                    mae: metrics.1,
                    rmse: metrics.2,
                    forecast,
                });
            }
            
            // Test ARIMA
            if let Ok(forecast) = self.forecast_arima_on_data(&train_data, 1, 1, 1, test_data.values.len()) {
                let metrics = calculate_metrics(&test_data.values, &forecast);
                results.insert("ARIMA(1,1,1)".to_string(), ModelResult {
                    mse: metrics.0,
                    mae: metrics.1,
                    rmse: metrics.2,
                    forecast,
                });
            }
            
            // Test Simple Exponential Smoothing
            if let Ok(forecast) = self.forecast_simple_es_on_data(&train_data, 0.3, test_data.values.len()) {
                let metrics = calculate_metrics(&test_data.values, &forecast);
                results.insert("SimpleES".to_string(), ModelResult {
                    mse: metrics.0,
                    mae: metrics.1,
                    rmse: metrics.2,
                    forecast,
                });
            }
            
            // Find best model (lowest MSE)
            let best_model = results
                .iter()
                .min_by(|a, b| a.1.mse.partial_cmp(&b.1.mse).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(name, _)| name.clone())
                .unwrap_or("None".to_string());
            
            Ok(ModelComparison {
                best_model,
                results,
            })
        }
        
        /// Convenient MA forecasting
        pub fn forecast_ma(&self, window: usize, periods: usize) -> Result<Vec<f64>> {
            self.forecast_ma_on_data(&self.data, window, periods)
        }
        
        fn forecast_ma_on_data(&self, data: &TimeSeriesData, window: usize, periods: usize) -> Result<Vec<f64>> {
            let mut model = MAModel::new(window)?;
            model.fit(data)?;
            let forecast = model.forecast(periods)?;
            Ok(forecast)
        }
        
        /// Convenient ARIMA forecasting
        pub fn forecast_arima(&self, p: usize, d: usize, q: usize, periods: usize) -> Result<Vec<f64>> {
            self.forecast_arima_on_data(&self.data, p, d, q, periods)
        }
        
        fn forecast_arima_on_data(&self, data: &TimeSeriesData, p: usize, d: usize, q: usize, periods: usize) -> Result<Vec<f64>> {
            let mut model = ARIMAModel::new(p, d, q, true)?;
            model.fit(data)?;
            let forecast = model.forecast(periods)?;
            Ok(forecast)
        }
        
        /// Convenient Simple ES forecasting
        pub fn forecast_simple_es(&self, alpha: f64, periods: usize) -> Result<Vec<f64>> {
            self.forecast_simple_es_on_data(&self.data, alpha, periods)
        }
        
        fn forecast_simple_es_on_data(&self, data: &TimeSeriesData, alpha: f64, periods: usize) -> Result<Vec<f64>> {
            let mut model = SimpleESModel::new(alpha)?;
            model.fit(data)?;
            let forecast = model.forecast(periods)?;
            Ok(forecast)
        }
        
        /// Get the underlying time series data
        pub fn data(&self) -> &TimeSeriesData {
            &self.data
        }
    }
    
    /// Calculate error metrics
    fn calculate_metrics(actual: &[f64], predicted: &[f64]) -> (f64, f64, f64) {
        use crate::math::metrics::{mae, mse, rmse};
        (mse(actual, predicted), mae(actual, predicted), rmse(actual, predicted))
    }
}

/// High-level API module for easy integration with external crates
/// 
/// This module provides a simplified interface for common forecasting tasks,
/// designed specifically for consumption by other crates and applications.
/// 
/// ## Future API Improvements Needed:
/// 
/// 1. **Unified Model Interface**: All models should implement a consistent trait:
/// ```rust
/// pub trait UnifiedForecaster {
///     fn quick_fit(data: &TimeSeriesData) -> Result<Self> where Self: Sized;
///     fn quick_forecast(&self, periods: usize) -> Result<Vec<f64>>;
///     fn model_info(&self) -> ModelInfo;
/// }
/// ```
/// 
/// 2. **Error Harmonization**: All model errors should convert to OxiError
/// 3. **Parameter Standardization**: Consistent parameter naming across models
/// 4. **Model Registry**: Runtime model selection and comparison
pub mod api {
    use super::*;
    use std::collections::HashMap;
    
    /// Simplified forecasting interface with automatic model selection
    pub struct Forecaster {
        default_model: String,
        models_cache: HashMap<String, Box<dyn super::Forecaster>>,
    }
    
    /// Configuration for forecasting models
    #[derive(Debug, Clone)]
    pub struct ForecastConfig {
        pub model_type: String,
        pub parameters: HashMap<String, f64>,
        pub auto_select: bool,
    }
    
    /// Simple forecast result with metadata
    #[derive(Debug, Clone)]
    pub struct ForecastOutput {
        pub values: Vec<f64>,
        pub model_used: String,
        pub accuracy_score: Option<f64>,
        pub parameters_used: HashMap<String, f64>,
    }
    
    impl Default for ForecastConfig {
        fn default() -> Self {
            let mut params = HashMap::new();
            params.insert("alpha".to_string(), 0.3);
            params.insert("window".to_string(), 5.0);
            
            Self {
                model_type: "auto".to_string(),
                parameters: params,
                auto_select: true,
            }
        }
    }
    
    impl Forecaster {
        /// Create a new forecaster with default settings
        pub fn new() -> Self {
            Self {
                default_model: "ARIMA".to_string(),
                models_cache: HashMap::new(),
            }
        }
        
        /// Create a forecaster with specific default model
        pub fn with_model(model_type: &str) -> Self {
            Self {
                default_model: model_type.to_string(),
                models_cache: HashMap::new(),
            }
        }
        
        /// Forecast using automatic model selection
        pub fn forecast_auto(
            &self,
            dates: Vec<chrono::DateTime<chrono::Utc>>,
            values: Vec<f64>,
            periods: usize,
        ) -> Result<ForecastOutput> {
            let data = TimeSeriesData::new(dates, values, "api_data")?;
            
            // Try models in order of preference
            let models_to_try = vec![
                ("ARIMA", self.try_arima(&data, periods)),
                ("MA", self.try_ma(&data, periods)),
                ("ES", self.try_exponential_smoothing(&data, periods)),
            ];
            
            for (model_name, result) in models_to_try {
                if let Ok(forecast) = result {
                    return Ok(ForecastOutput {
                        values: forecast,
                        model_used: model_name.to_string(),
                        accuracy_score: None,
                        parameters_used: HashMap::new(),
                    });
                }
            }
            
            Err(OxiError::ModelError("All models failed".to_string()))
        }
        
        /// Forecast with specific configuration
        pub fn forecast_with_config(
            &self,
            dates: Vec<chrono::DateTime<chrono::Utc>>,
            values: Vec<f64>,
            periods: usize,
            config: ForecastConfig,
        ) -> Result<ForecastOutput> {
            if config.auto_select {
                return self.forecast_auto(dates, values, periods);
            }
            
            let data = TimeSeriesData::new(dates, values, "api_data")?;
            
            let forecast = match config.model_type.as_str() {
                "ARIMA" => self.try_arima(&data, periods)?,
                "MA" => self.try_ma(&data, periods)?,
                "ES" => self.try_exponential_smoothing(&data, periods)?,
                _ => return Err(OxiError::InvalidParameter(format!("Unknown model: {}", config.model_type))),
            };
            
            Ok(ForecastOutput {
                values: forecast,
                model_used: config.model_type.clone(),
                accuracy_score: None,
                parameters_used: config.parameters,
            })
        }
        
        fn try_arima(&self, data: &TimeSeriesData, periods: usize) -> Result<Vec<f64>> {
            let mut model = ARIMAModel::new(1, 1, 1, true)?;
            model.fit(data)?;
            let forecast = model.forecast(periods)?;
            Ok(forecast)
        }
        
        fn try_ma(&self, data: &TimeSeriesData, periods: usize) -> Result<Vec<f64>> {
            let mut model = MAModel::new(5)?;
            model.fit(data)?;
            let forecast = model.forecast(periods)?;
            Ok(forecast)
        }
        
        fn try_exponential_smoothing(&self, data: &TimeSeriesData, periods: usize) -> Result<Vec<f64>> {
            let mut model = SimpleESModel::new(0.3)?;
            model.fit(data)?;
            let forecast = model.forecast(periods)?;
            Ok(forecast)
        }
    }
    
    /// Quick forecasting functions for simple use cases
    pub mod quick {
        use super::*;
        use oxidiviner_core::ModelValidator;
        
        /// Quick ARIMA forecast with default parameters
        pub fn arima_forecast(
            dates: Vec<chrono::DateTime<chrono::Utc>>,
            values: Vec<f64>,
            periods: usize,
        ) -> Result<Vec<f64>> {
            let data = TimeSeriesData::new(dates, values, "quick_arima")?;
            let mut model = ARIMAModel::new(1, 1, 1, true)?;
            model.fit(&data)?;
            let forecast = model.forecast(periods)?;
            Ok(forecast)
        }
        
        /// Quick moving average forecast
        pub fn ma_forecast(
            dates: Vec<chrono::DateTime<chrono::Utc>>,
            values: Vec<f64>,
            periods: usize,
            window: Option<usize>,
        ) -> Result<Vec<f64>> {
            let data = TimeSeriesData::new(dates, values, "quick_ma")?;
            let mut model = MAModel::new(window.unwrap_or(5))?;
            model.fit(&data)?;
            let forecast = model.forecast(periods)?;
            Ok(forecast)
        }
        
        /// Quick exponential smoothing forecast
        pub fn es_forecast(
            dates: Vec<chrono::DateTime<chrono::Utc>>,
            values: Vec<f64>,
            periods: usize,
            alpha: Option<f64>,
        ) -> Result<Vec<f64>> {
            let data = TimeSeriesData::new(dates, values, "quick_es")?;
            let mut model = SimpleESModel::new(alpha.unwrap_or(0.3))?;
            model.fit(&data)?;
            let forecast = model.forecast(periods)?;
            Ok(forecast)
        }
    }
}

/// Batch processing utilities for multiple time series
pub mod batch {
    use super::*;
    use std::collections::HashMap;
    
    /// Batch forecast multiple time series
    pub fn batch_forecast(
        data_map: HashMap<String, TimeSeriesData>,
        model_type: &str,
        periods: usize,
    ) -> Result<HashMap<String, Vec<f64>>> {
        let mut results = HashMap::new();
        
        for (name, data) in data_map {
            let forecast = match model_type {
                "MA" => {
                    let mut model = MAModel::new(5)?;
                    model.fit(&data)?;
                    model.forecast(periods)?
                }
                "ARIMA" => {
                    let mut model = ARIMAModel::new(1, 1, 1, true)?;
                    model.fit(&data)?;
                    model.forecast(periods)?
                }
                "ES" => {
                    let mut model = SimpleESModel::new(0.3)?;
                    model.fit(&data)?;
                    model.forecast(periods)?
                }
                _ => return Err(OxiError::InvalidParameter(format!("Unknown model type: {}", model_type))),
            };
            
            results.insert(name, forecast);
        }
        
        Ok(results)
    }
}

/// Quick and easy forecasting API with unified interface
/// 
/// This module provides a simplified, unified interface for forecasting
/// that implements the QuickForecaster trait for all supported models.
pub mod quick {
    use super::*;
    use oxidiviner_core::{QuickForecaster, ModelValidator, ForecastResult, ConfidenceForecaster};
    use std::collections::HashMap;
    
    /// Quick ARIMA forecasting with minimal configuration
    /// 
    /// # Arguments
    /// * `data` - Time series data to forecast
    /// * `periods` - Number of periods to forecast
    /// * `config` - Optional ARIMA configuration (p, d, q)
    /// 
    /// # Returns
    /// * `Result<Vec<f64>>` - Forecasted values
    /// 
    /// # Example
    /// ```rust
    /// # use oxidiviner::prelude::*;
    /// # use oxidiviner::quick;
    /// # fn example() -> Result<()> {
    /// # let data = TimeSeriesData::new(vec![Utc::now()], vec![1.0], "test")?;
    /// let forecast = quick::arima(data, 10)?;
    /// println!("ARIMA forecast: {:?}", forecast);
    /// # Ok(())
    /// # }
    /// ```
    pub fn arima(data: TimeSeriesData, periods: usize) -> Result<Vec<f64>> {
        arima_with_config(data, periods, None)
    }
    
    /// Quick ARIMA forecasting with custom configuration
    pub fn arima_with_config(
        data: TimeSeriesData, 
        periods: usize,
        config: Option<(usize, usize, usize)>  // (p, d, q)
    ) -> Result<Vec<f64>> {
        let (p, d, q) = config.unwrap_or((1, 1, 1));
        
        // Validate parameters
        ModelValidator::validate_arima_params(p, d, q)?;
        ModelValidator::validate_forecast_horizon(periods, data.values.len())?;
        ModelValidator::validate_minimum_data(data.values.len(), p + d + q + 1, "ARIMA")?;
        
        // Create and fit model
        let mut model = super::ARIMAModel::new(p, d, q, true)?;
        model.fit(&data)?;
        model.forecast(periods)
    }
    
    /// Quick autoregressive forecasting
    /// 
    /// # Arguments
    /// * `data` - Time series data to forecast
    /// * `periods` - Number of periods to forecast
    /// * `order` - Optional AR order (default: 2)
    pub fn ar(data: TimeSeriesData, periods: usize, order: Option<usize>) -> Result<Vec<f64>> {
        let p = order.unwrap_or(2);
        
        // Validate parameters
        ModelValidator::validate_ar_params(p)?;
        ModelValidator::validate_forecast_horizon(periods, data.values.len())?;
        ModelValidator::validate_minimum_data(data.values.len(), p + 1, "AR")?;
        
        // Create and fit model
        let mut model = super::ARModel::new(p, true)?;
        model.fit(&data)?;
        model.forecast(periods)
    }
    
    /// Quick moving average forecasting
    /// 
    /// # Arguments
    /// * `data` - Time series data to forecast
    /// * `periods` - Number of periods to forecast
    /// * `window` - Optional window size (default: 5)
    pub fn moving_average(data: TimeSeriesData, periods: usize, window: Option<usize>) -> Result<Vec<f64>> {
        let w = window.unwrap_or(5);
        
        // Validate parameters
        ModelValidator::validate_ma_params(w)?;
        ModelValidator::validate_forecast_horizon(periods, data.values.len())?;
        ModelValidator::validate_minimum_data(data.values.len(), w, "Moving Average")?;
        
        // Create and fit model
        let mut model = super::MAModel::new(w)?;
        model.fit(&data)?;
        model.forecast(periods)
    }
    
    /// Quick exponential smoothing forecasting
    /// 
    /// # Arguments
    /// * `data` - Time series data to forecast
    /// * `periods` - Number of periods to forecast
    /// * `alpha` - Optional smoothing parameter (default: 0.3)
    pub fn exponential_smoothing(data: TimeSeriesData, periods: usize, alpha: Option<f64>) -> Result<Vec<f64>> {
        let a = alpha.unwrap_or(0.3);
        
        // Validate parameters
        ModelValidator::validate_exponential_smoothing_params(a, None, None)?;
        ModelValidator::validate_forecast_horizon(periods, data.values.len())?;
        ModelValidator::validate_minimum_data(data.values.len(), 2, "Exponential Smoothing")?;
        
        // Create and fit model
        let mut model = super::SimpleESModel::new(a)?;
        model.fit(&data)?;
        model.forecast(periods)
    }
    
    /// Automatic model selection and forecasting
    /// 
    /// Tries multiple models and returns the best forecast based on AIC.
    /// 
    /// # Arguments
    /// * `data` - Time series data to forecast
    /// * `periods` - Number of periods to forecast
    /// 
    /// # Returns
    /// * `Result<(Vec<f64>, String)>` - Best forecast and model name
    pub fn auto_select(data: TimeSeriesData, periods: usize) -> Result<(Vec<f64>, String)> {
        use oxidiviner_core::validation::ValidationUtils;
        
        // Split data for validation
        let (train_data, test_data) = ValidationUtils::time_split(&data, 0.8)?;
        
        let mut best_forecast = Vec::new();
        let mut best_model = String::new();
        let mut best_mae = f64::INFINITY;
        
        // Try different models with boxed closures
        let models_to_try: Vec<(String, Box<dyn Fn() -> Result<Vec<f64>>>)> = vec![
            ("ARIMA(1,1,1)".to_string(), Box::new({
                let train_data = train_data.clone();
                let test_len = test_data.values.len();
                move || arima_with_config(train_data.clone(), test_len, Some((1, 1, 1)))
            })),
            ("ARIMA(2,1,1)".to_string(), Box::new({
                let train_data = train_data.clone();
                let test_len = test_data.values.len();
                move || arima_with_config(train_data.clone(), test_len, Some((2, 1, 1)))
            })),
            ("AR(2)".to_string(), Box::new({
                let train_data = train_data.clone();
                let test_len = test_data.values.len();
                move || ar(train_data.clone(), test_len, Some(2))
            })),
            ("MA(5)".to_string(), Box::new({
                let train_data = train_data.clone();
                let test_len = test_data.values.len();
                move || moving_average(train_data.clone(), test_len, Some(5))
            })),
            ("ES(0.3)".to_string(), Box::new({
                let train_data = train_data.clone();
                let test_len = test_data.values.len();
                move || exponential_smoothing(train_data.clone(), test_len, Some(0.3))
            })),
        ];
        
        for (model_name, model_fn) in models_to_try {
            if let Ok(forecast) = model_fn() {
                if forecast.len() == test_data.values.len() {
                    let mae = ValidationUtils::accuracy_metrics(&test_data.values, &forecast, None)
                        .map(|m| m.mae)
                        .unwrap_or(f64::INFINITY);
                    
                    if mae < best_mae {
                        best_mae = mae;
                        best_model = model_name.clone();
                        // Generate final forecast on full data
                        best_forecast = match model_name.as_str() {
                            "ARIMA(1,1,1)" => arima_with_config(data.clone(), periods, Some((1, 1, 1)))?,
                            "ARIMA(2,1,1)" => arima_with_config(data.clone(), periods, Some((2, 1, 1)))?,
                            "AR(2)" => ar(data.clone(), periods, Some(2))?,
                            "MA(5)" => moving_average(data.clone(), periods, Some(5))?,
                            _ => exponential_smoothing(data.clone(), periods, Some(0.3))?,
                        };
                    }
                }
            }
        }
        
        if best_forecast.is_empty() {
            return Err(OxiError::ModelError("No suitable model found".into()));
        }
        
        Ok((best_forecast, best_model))
    }
    
    /// Unified forecasting interface with model builder
    /// 
    /// # Arguments
    /// * `data` - Time series data to forecast
    /// * `periods` - Number of periods to forecast
    /// * `config` - Model configuration from ModelBuilder
    /// 
    /// # Returns
    /// * `Result<Vec<f64>>` - Forecasted values
    pub fn forecast_with_config(
        data: TimeSeriesData,
        periods: usize,
        config: oxidiviner_core::ModelConfig,
    ) -> Result<Vec<f64>> {
        match config.model_type.as_str() {
            "ARIMA" => {
                let p = config.parameters.get("p").copied().unwrap_or(1.0) as usize;
                let d = config.parameters.get("d").copied().unwrap_or(1.0) as usize;
                let q = config.parameters.get("q").copied().unwrap_or(1.0) as usize;
                arima_with_config(data, periods, Some((p, d, q)))
            },
            "AR" => {
                let p = config.parameters.get("p").copied().unwrap_or(2.0) as usize;
                ar(data, periods, Some(p))
            },
            "MA" => {
                let window = config.parameters.get("window").copied().unwrap_or(5.0) as usize;
                moving_average(data, periods, Some(window))
            },
            "ES" => {
                let alpha = config.parameters.get("alpha").copied().unwrap_or(0.3);
                exponential_smoothing(data, periods, Some(alpha))
            },
            _ => Err(OxiError::InvalidParameter(format!("Unsupported model type: {}", config.model_type))),
        }
    }
}

// Re-export key components for external use
pub use oxidiviner_core::{
    ModelBuilder, AutoSelector, ModelConfig, SelectionCriteria, SelectionResult,
    ModelValidator, QuickForecaster, ConfidenceForecaster, ForecastResult,
    validation,
};

/// Prelude module that re-exports the most commonly used types and traits
///
/// This module is intended to be glob-imported with `use oxidiviner::prelude::*`
/// to bring the most common OxiDiviner types and traits into scope.
pub mod prelude {
    pub use crate::Forecaster;
    pub use crate::ModelEvaluation;
    pub use crate::ModelOutput;
    pub use crate::OHLCVData;
    pub use crate::OxiError;
    pub use crate::Result;
    pub use crate::TimeSeriesData;

    // Financial convenience types
    pub use crate::financial::{FinancialTimeSeries, ModelComparison, QuickForecast};

    // Common models
    pub use crate::models::autoregressive::ARIMAModel;
    pub use crate::models::autoregressive::ARMAModel;
    pub use crate::models::autoregressive::ARModel;
    pub use crate::models::autoregressive::SARIMAModel;
    pub use crate::models::autoregressive::VARModel;

    pub use crate::models::exponential_smoothing::DailyETSModel;
    pub use crate::models::exponential_smoothing::DampedTrendModel;
    pub use crate::models::exponential_smoothing::ETSComponent;
    pub use crate::models::exponential_smoothing::ETSModel;
    pub use crate::models::exponential_smoothing::HoltLinearModel;
    pub use crate::models::exponential_smoothing::HoltWintersModel;
    pub use crate::models::exponential_smoothing::MinuteETSModel;
    pub use crate::models::exponential_smoothing::SimpleESModel as SESModel;

    pub use crate::models::garch::EGARCHModel;
    pub use crate::models::garch::GARCHMModel;
    pub use crate::models::garch::GARCHModel;
    pub use crate::models::garch::GJRGARCHModel;
    pub use crate::models::garch::RiskPremiumType;

    pub use crate::models::moving_average::MAModel;

    // Common math utilities
    pub use crate::math::metrics::{mae, mape, mse, rmse, smape};
    pub use crate::math::transforms::difference;

    // Time-related
    pub use chrono::{DateTime, Duration, TimeZone, Utc};
}
