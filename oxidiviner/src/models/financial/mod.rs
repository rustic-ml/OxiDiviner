//! Financial Models Module
//!
//! This module contains specialized financial time series models for quantitative finance
//! applications including options pricing, risk management, and portfolio optimization.
//!
//! ## Available Models
//!
//! ### Jump-Diffusion Models
//! - **Merton Jump-Diffusion**: Industry standard for crash modeling with Gaussian jumps
//! - **Kou Jump-Diffusion**: Asymmetric jump distributions with double-exponential tails
//!
//! ### Stochastic Volatility Models  
//! - **Heston Model**: Gold standard for volatility modeling with mean reversion
//! - **SABR Model**: Industry standard for FX and rates volatility surface modeling
//!
//! ### Applications
//! - Options pricing and derivatives valuation
//! - Value-at-Risk (VaR) calculations with realistic distributions
//! - Portfolio risk management and stress testing
//! - Algorithmic trading strategy development
//! - Regulatory capital calculations (Basel III compliance)

pub mod heston_stochastic_volatility;
pub mod kou_jump_diffusion;
pub mod merton_jump_diffusion;
pub mod sabr_volatility;

// Re-export main model types for easy access
pub use merton_jump_diffusion::{
    EstimatedParams as MertonEstimatedParams, JumpEvent, MertonJumpDiffusionModel,
    ModelDiagnostics as MertonModelDiagnostics,
};

pub use kou_jump_diffusion::{
    AsymmetricJumpEvent, EstimatedParams as KouEstimatedParams, KouJumpDiffusionModel,
    ModelDiagnostics as KouModelDiagnostics,
};

pub use heston_stochastic_volatility::{
    EstimatedParams as HestonEstimatedParams, HestonPath, HestonStochasticVolatilityModel,
    ModelDiagnostics as HestonModelDiagnostics, VolatilitySurfacePoint,
};

pub use sabr_volatility::{
    EstimatedParams as SABREstimatedParams, ModelDiagnostics as SABRModelDiagnostics,
    SABRCalibration, SABRPath, SABRVolatilityModel, SABRVolatilitySurfacePoint,
};
