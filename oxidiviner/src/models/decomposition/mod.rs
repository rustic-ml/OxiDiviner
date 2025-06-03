//! Time series decomposition models
//!
//! This module provides models for decomposing time series into components
//! such as trend, seasonal, and remainder, useful for forecasting and analysis.

pub mod stl;

pub use stl::STLModel;
