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

    // Common models
    pub use crate::models::autoregressive::ARIMAModel;
    pub use crate::models::autoregressive::ARModel;
    pub use crate::models::exponential_smoothing::HoltWintersModel;
    pub use crate::models::exponential_smoothing::SimpleESModel as SESModel;
    pub use crate::models::garch::GARCHModel;
    pub use crate::models::moving_average::MAModel;
}
