use std::io;
use thiserror::Error;

/// Custom error type for OxiDiviner operations
#[derive(Error, Debug)]
pub enum OxiError {
    /// Error related to IO operations
    #[error("IO Error: {0}")]
    IoError(#[from] io::Error),

    /// Error related to data validation or processing
    #[error("Data Error: {0}")]
    DataError(String),

    /// Error related to model operations
    #[error("Model Error: {0}")]
    ModelError(String),

    /// Error related to invalid parameters
    #[error("Invalid Parameter: {0}")]
    InvalidParameter(String),

    /// Error when a feature is not enabled
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),
}

/// Custom Result type for OxiDiviner operations
pub type Result<T> = std::result::Result<T, OxiError>;
