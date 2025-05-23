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
//! OxiDiviner is a comprehensive library for time series analysis and forecasting,
//! designed to provide efficient, accurate, and easy-to-use statistical models for Rust.
//! This library includes all functionality in a single crate for ease of use.
//!
//! ## Enhanced API Modules
//!
//! OxiDiviner provides several enhanced API modules for different use cases:
//!
//! - [`financial`] - Specialized functionality for financial time series analysis
//! - [`api`] - High-level unified interface for all forecasting models
//! - [`quick`] - One-line utility functions for rapid prototyping
//! - [`batch`] - Batch processing for multiple time series simultaneously
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use oxidiviner::prelude::*;
//! use oxidiviner::quick;
//! use chrono::{Duration, Utc};
//!
//! // Generate sample data
//! let start = Utc::now();
//! let timestamps: Vec<_> = (0..30).map(|i| start + Duration::days(i)).collect();
//! let values: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 + (i as f64 * 0.1).sin() * 5.0).collect();
//!
//! // Quick forecasting
//! let (forecast, model_used) = quick::auto_forecast(timestamps, values, 5)?;
//! println!("Used {} model, forecast: {:?}", model_used, forecast);
//! ```

// Internal modules - organized for development but packaged as single crate
pub mod core;
pub mod math;
pub mod models;

// Enhanced API modules
pub mod api;
pub mod batch;
pub mod financial;
pub mod quick;

// Re-export from core
pub use crate::core::*;

// Re-export enhanced API components
pub use api::{Forecaster, ForecastBuilder, ForecastConfig, ForecastOutput, ModelType, ModelParameters};
pub use batch::{BatchTimeSeries, BatchConfig, BatchForecastResult, BatchModelType};
pub use financial::{FinancialTimeSeries, ModelComparison, ModelResult};

// Direct re-exports of all model types for maximum convenience
pub use models::autoregressive::{ARIMAModel, ARMAModel, ARModel, SARIMAModel, VARModel};
pub use models::exponential_smoothing::{
    DailyETSModel, DampedTrendModel, ETSComponent, ETSModel, HoltLinearModel, HoltWintersModel,
    MinuteETSModel, SimpleESModel,
};
pub use models::garch::{EGARCHModel, GARCHMModel, GARCHModel, GJRGARCHModel, RiskPremiumType};
pub use models::moving_average::MAModel;

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

    // Enhanced API exports
    pub use crate::api::{ForecastBuilder, ForecastConfig, ForecastOutput};
    pub use crate::batch::BatchTimeSeries;
    pub use crate::financial::FinancialTimeSeries;

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
