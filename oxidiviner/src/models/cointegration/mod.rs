//! Cointegration-based forecasting models
//!
//! This module provides models for time series that share long-run equilibrium
//! relationships, particularly useful for multi-asset and pairs trading.

pub mod vecm;

pub use vecm::VECMModel;
