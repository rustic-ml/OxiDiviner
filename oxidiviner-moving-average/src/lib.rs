
mod error;
mod model;

// Re-export the public models
pub use model::MAModel;

// Re-export the error types
pub use error::{MAError, Result};
