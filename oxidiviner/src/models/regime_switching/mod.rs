//! Regime-switching models for capturing different market states
//!
//! This module provides Markov regime-switching models that can identify and
//! forecast transitions between different market regimes (bull/bear markets,
//! high/low volatility periods, etc.).
//!
//! ## Models Available
//!
//! - **MarkovSwitchingModel**: Univariate regime detection for single time series
//! - **MultivariateMarkovSwitchingModel**: Multivariate regime detection for multiple correlated series
//! - **HigherOrderMarkovModel**: Higher-order dependencies in regime transitions
//! - **DurationDependentMarkovModel**: Duration-dependent regime persistence
//! - **RegimeSwitchingARModel**: Regime-switching with autoregressive dynamics
//!
//! ## Enhanced Capabilities
//!
//! The multivariate extension provides:
//! - Cross-asset regime detection
//! - Portfolio-wide regime analysis
//! - Correlation regime switching
//! - Risk factor regime modeling
//!
//! ## Higher-Order Extensions
//!
//! The higher-order models provide:
//! - Second and third-order Markov dependencies
//! - Duration-dependent transition probabilities
//! - Regime-dependent autoregressive dynamics
//! - Complex temporal pattern recognition

pub mod higher_order_regime_switching;
pub mod markov_switching;
pub mod multivariate_markov_switching;

pub use higher_order_regime_switching::{
    DurationDependentMarkovModel, HigherOrderMarkovModel, RegimeSwitchingARModel,
};
pub use markov_switching::MarkovSwitchingModel;
pub use multivariate_markov_switching::{
    CrossCorrelationStats, MultivariateMarkovSwitchingModel, PortfolioRegimeAnalysis,
};
