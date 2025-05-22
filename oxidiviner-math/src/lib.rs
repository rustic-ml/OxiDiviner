#[cfg(feature = "ndarray_support")]
use ndarray::Array1;

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
