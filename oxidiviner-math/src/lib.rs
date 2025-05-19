#[cfg(feature = "ndarray_support")]
use ndarray::Array1;
use statrs::statistics::Data;
use statrs::statistics::Distribution;

// Internal modules
mod error;
pub mod metrics;
pub mod statistics;
pub mod transforms;

// Public exports
pub use error::MathError;
pub use metrics::*;
pub use statistics::*;
pub use transforms::*;
