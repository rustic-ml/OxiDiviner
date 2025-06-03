//! Non-linear forecasting models
//!
//! This module provides models that capture non-linear dynamics in time series,
//! including threshold effects and regime-dependent behavior.

pub mod tar;

pub use tar::TARModel;
