use oxidiviner_core::OxiError;
use thiserror::Error;

/// Error type for exponential smoothing models
#[derive(Error, Debug)]
pub enum ESError {
    #[error("Empty data provided")]
    EmptyData,

    #[error("Invalid alpha value: {0}")]
    InvalidAlpha(f64),

    #[error("Invalid beta value: {0}")]
    InvalidBeta(f64),

    #[error("Invalid gamma value: {0}")]
    InvalidGamma(f64),

    #[error("Invalid period: {0}")]
    InvalidPeriod(usize),

    #[error("Model has not been fitted yet")]
    NotFitted,

    #[error("Invalid horizon: {0}")]
    InvalidHorizon(usize),

    #[error("Insufficient data: {actual} points provided, {expected} required")]
    InsufficientData { actual: usize, expected: usize },

    #[error("Invalid damping factor: {0}")]
    InvalidDampingFactor(f64),

    #[error("Missing required parameter: {0}")]
    MissingParameter(String),

    #[error("Unsupported model type: {0}")]
    UnsupportedModelType(String),
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

// Convert from ESError to OxiError
impl From<ESError> for OxiError {
    fn from(e: ESError) -> Self {
        match e {
            ESError::EmptyData => OxiError::DataError("Empty data provided".into()),
            ESError::InvalidAlpha(v) => {
                OxiError::InvalidParameter(format!("Invalid alpha value: {}", v))
            }
            ESError::InvalidBeta(v) => {
                OxiError::InvalidParameter(format!("Invalid beta value: {}", v))
            }
            ESError::InvalidGamma(v) => {
                OxiError::InvalidParameter(format!("Invalid gamma value: {}", v))
            }
            ESError::InvalidPeriod(v) => {
                OxiError::InvalidParameter(format!("Invalid period: {}", v))
            }
            ESError::NotFitted => OxiError::ModelError("Model has not been fitted yet".into()),
            ESError::InvalidHorizon(v) => {
                OxiError::InvalidParameter(format!("Invalid horizon: {}", v))
            }
            ESError::InsufficientData { actual, expected } => OxiError::DataError(format!(
                "Insufficient data: {} points provided, {} required",
                actual, expected
            )),
            ESError::InvalidDampingFactor(v) => {
                OxiError::InvalidParameter(format!("Invalid damping factor: {}", v))
            }
            ESError::MissingParameter(param) => {
                OxiError::InvalidParameter(format!("Missing required parameter: {}", param))
            }
            ESError::UnsupportedModelType(model) => {
                OxiError::ModelError(format!("Unsupported model type: {}", model))
            }
            ESError::InvalidParameter(msg) => {
                OxiError::InvalidParameter(msg)
            }
        }
    }
}

// Define a Result type for internal module use
pub type Result<T> = std::result::Result<T, ESError>;
