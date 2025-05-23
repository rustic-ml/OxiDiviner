//! # Models Module
//!
//! This module contains all the forecasting models organized by type.
//! Each submodule provides specific implementations for different forecasting approaches.

/// Moving average models for capturing short-term patterns
pub mod moving_average;

/// Exponential smoothing models for trend and seasonal data
pub mod exponential_smoothing;

/// Autoregressive models for data with dependency on past values
pub mod autoregressive;

/// GARCH models for volatility forecasting in financial time series
pub mod garch;

// Re-export all models for convenience
pub use moving_average::*;
pub use exponential_smoothing::*;
pub use autoregressive::*;
pub use garch::*; 