//! # OxiDiviner
//! 
//! A comprehensive Rust library for time series analysis and forecasting.
//! 
//! This crate provides a unified interface to all OxiDiviner components:
//! - Core functionality and common interfaces
//! - Mathematical utilities
//! - Moving average models
//! - Exponential smoothing models
//! - Autoregressive models
//! - GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models

// Re-export from core
pub use oxidiviner_core::*;

// Re-export from math
pub use oxidiviner_math as math;

// Re-export models from module-specific crates
pub mod models {
    // Moving average models
    pub mod moving_average {
        pub use oxidiviner_moving_average::*;
    }

    // Exponential smoothing models
    pub mod exponential_smoothing {
        pub use oxidiviner_exponential_smoothing::*;
    }

    // Autoregressive models
    pub mod autoregressive {
        pub use oxidiviner_autoregressive::*;
    }

    // GARCH models
    pub mod garch {
        pub use oxidiviner_garch::*;
    }
} 