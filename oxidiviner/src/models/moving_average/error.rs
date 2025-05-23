use crate::core::OxiError;
use thiserror::Error;

/// Error type for moving average models
#[derive(Error, Debug)]
pub enum MAError {
    #[error("Empty data provided")]
    EmptyData,

    #[error("Invalid window size: {0}")]
    InvalidWindowSize(usize),

    #[error("Time series length ({actual}) must be at least equal to window size ({expected})")]
    TimeSeriesTooShort { actual: usize, expected: usize },

    #[error("Model has not been fitted yet")]
    NotFitted,

    #[error("Invalid horizon: {0}")]
    InvalidHorizon(usize),
}

// Convert from MAError to OxiError
impl From<MAError> for OxiError {
    fn from(e: MAError) -> Self {
        match e {
            MAError::EmptyData => OxiError::DataError("Empty data provided".into()),
            MAError::InvalidWindowSize(v) => {
                OxiError::InvalidParameter(format!("Invalid window size: {}", v))
            }
            MAError::TimeSeriesTooShort { actual, expected } => OxiError::DataError(format!(
                "Time series length ({}) must be at least equal to window size ({})",
                actual, expected
            )),
            MAError::NotFitted => OxiError::ModelError("Model has not been fitted yet".into()),
            MAError::InvalidHorizon(v) => {
                OxiError::InvalidParameter(format!("Invalid horizon: {}", v))
            }
        }
    }
}

// Define a Result type for internal module use
pub type Result<T> = std::result::Result<T, MAError>;
