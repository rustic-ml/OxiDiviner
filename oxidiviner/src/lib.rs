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

/// # Ensemble Forecasting Methods
///
/// Provides tools for combining multiple forecasting models to improve
/// prediction accuracy and robustness. Includes various ensemble strategies
/// like averaging, weighted averaging, median, and stacking.
///
/// For detailed usage and a list of available methods, see the [`ensemble`]
/// module documentation.
///
/// [`ensemble`]: ensemble
pub mod ensemble;

/// # Parameter Optimization Engine
///
/// Offers automated parameter tuning for various forecasting models.
/// Helps in finding optimal model configurations by searching through
/// parameter spaces using methods like Grid Search and evaluating
/// performance with metrics such as MAE, RMSE, etc.
///
/// For detailed usage and configuration options, see the [`optimization`]
/// module documentation.
///
/// [`optimization`]: optimization
pub mod optimization;

// Re-export from core
pub use crate::core::*;

// Re-export enhanced API components
pub use api::{
    ForecastBuilder, ForecastConfig, ForecastOutput, Forecaster, ModelParameters, ModelType,
};
pub use batch::{BatchConfig, BatchForecastResult, BatchModelType, BatchTimeSeries};
pub use financial::{FinancialTimeSeries, ModelComparison, ModelResult};

// Direct re-exports of all model types for maximum convenience
pub use models::autoregressive::{ARIMAModel, ARMAModel, ARModel, SARIMAModel, VARModel};
pub use models::exponential_smoothing::{
    DailyETSModel, DampedTrendModel, ETSComponent, ETSModel, HoltLinearModel, HoltWintersModel,
    MinuteETSModel, SimpleESModel,
};
pub use models::garch::{EGARCHModel, GARCHMModel, GARCHModel, GJRGARCHModel, RiskPremiumType};
pub use models::moving_average::MAModel;

// Advanced forecasting models
pub use models::cointegration::VECMModel;
pub use models::decomposition::STLModel;
pub use models::nonlinear::TARModel;
pub use models::regime_switching::MarkovSwitchingModel;
pub use models::state_space::KalmanFilter;

/// Prelude module for convenient imports
///
/// This module re-exports the most commonly used types and traits,
/// making it easy to get started with OxiDiviner:
///
/// ```
/// use oxidiviner::prelude::*;
/// ```
pub mod prelude {
    // Core types
    pub use crate::core::{
        ConfidenceForecaster, ForecastResult, ModelEvaluation, ModelValidator, OHLCVData, OxiError,
        QuickForecaster, Result, TimeSeriesData,
    };

    // API builders and selectors
    pub use crate::api::{
        ARIMABuilderConfig, ARIMAWrapper, AutoSelector, ESBuilderConfig, ESWrapper,
        GARCHBuilderConfig, GARCHWrapper, MABuilderConfig, MAWrapper, ModelBuilder,
        SelectionCriteria,
    };

    // Quick functions
    pub use crate::quick::{arima, arima_with_config, auto_select, moving_average};

    // Individual models for advanced usage
    pub use crate::models::{
        autoregressive::{ARIMAModel, ARMAModel, ARModel, SARIMAModel, VARModel},
        cointegration::VECMModel,
        decomposition::STLModel,
        exponential_smoothing::{
            DailyETSModel, ETSModel, HoltLinearModel, HoltWintersModel, MinuteETSModel,
            SimpleESModel,
        },
        garch::{EGARCHModel, GARCHMModel, GARCHModel, GJRGARCHModel},
        moving_average::MAModel,
        nonlinear::TARModel,
        regime_switching::MarkovSwitchingModel,
        state_space::KalmanFilter,
    };

    // Re-export key components from ensemble and optimization modules
    pub use crate::ensemble::{
        EnsembleBuilder, EnsembleForecast, EnsembleMethod, EnsemblePerformance,
        EnsembleUtils, ModelForecast, ModelPerformance as EnsembleModelPerformance
    };
    pub use crate::optimization::{
        OptimizerBuilder, ParameterOptimizer, OptimizationConfig, OptimizationMethod,
        OptimizationMetric, OptimizationResult, ConvergenceInfo
    };
}

/// Convenience functions for quick forecasting
///
/// This module provides the most convenient functions for common use cases:
///
/// ```
/// use oxidiviner::convenience::*;
///
/// // One-line forecasting
/// let forecast = auto_forecast(timestamps, values, 10)?;
/// ```
pub mod convenience {
    pub use crate::quick::*;
}

/// Builder API namespace for fluent model construction
///
/// This module provides the builder pattern interface:
///
/// ```
/// use oxidiviner::builder::*;
///
/// let model = ModelBuilder::arima()
///     .with_ar(2)
///     .with_differencing(1)
///     .with_ma(1)
///     .build()?;
/// ```
pub mod builder {
    pub use crate::api::{AutoSelector, ModelBuilder, SelectionCriteria};
}

/// Advanced API for specialized use cases
///
/// This module exposes the full model interfaces for users who need
/// fine-grained control over model parameters and behavior.
pub mod advanced {
    pub use crate::core::validation::BacktestConfig;
    pub use crate::core::{Forecaster, ModelConfig};
    pub use crate::math::*;
    pub use crate::models::*;
}
