mod error;
mod garch;
mod egarch;
mod gjr_garch;
mod garch_m;

// Re-export the public models
pub use garch::GARCHModel;
pub use egarch::EGARCHModel;
pub use gjr_garch::GJRGARCHModel;
pub use garch_m::{GARCHMModel, RiskPremiumType};

// Re-export the error types
pub use error::{GARCHError, Result}; 