//! Regime-switching models for capturing different market states
//!
//! This module provides Markov regime-switching models that can identify and
//! forecast transitions between different market regimes (bull/bear markets,
//! high/low volatility periods, etc.).

pub mod markov_switching;

pub use markov_switching::MarkovSwitchingModel;
