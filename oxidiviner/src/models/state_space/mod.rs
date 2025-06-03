//! State-space models for dynamic forecasting
//!
//! This module provides state-space models that estimate hidden states and forecast
//! future observations. State-space models are particularly useful for financial
//! time series where the underlying process has unobservable components.
//!
//! ## Available Models
//!
//! - **Kalman Filter**: Optimal linear filter for Gaussian state-space models
//! - **Unobserved Components**: State-space models for trend, seasonal, and cyclical components
//!
//! ## Key Features
//!
//! - Real-time state estimation and forecasting
//! - Confidence intervals for forecasts
//! - Model diagnostics and validation
//! - Support for various state-space specifications

pub mod kalman_filter;

pub use kalman_filter::KalmanFilter;
