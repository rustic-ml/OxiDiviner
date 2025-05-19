use thiserror::Error;
use oxidiviner_core::OxiError;

/// Error type for autoregressive models
#[derive(Error, Debug)]
pub enum ARError {
    #[error("Empty data provided")]
    EmptyData,
    
    #[error("Invalid lag order p: {0}")]
    InvalidLagOrder(usize),
    
    #[error("Invalid horizon: {0}")]
    InvalidHorizon(usize),
    
    #[error("Insufficient data: {actual} points provided, {expected} required")]
    InsufficientData {
        actual: usize,
        expected: usize,
    },
    
    #[error("Model has not been fitted yet")]
    NotFitted,
    
    #[error("Failed to solve linear system for AR coefficients: {0}")]
    LinearSolveError(String),
    
    #[error("Invalid coefficient detected: NaN or Infinity")]
    InvalidCoefficient,
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Missing variable in multivariate model: {0}")]
    MissingVariable(String),
    
    #[error("Inconsistent timestamps in multivariate data")]
    InconsistentTimestamps,
    
    #[error("Invalid seasonal period: {0}")]
    InvalidSeasonalPeriod(usize),
}

// Convert from ARError to OxiError
impl From<ARError> for OxiError {
    fn from(e: ARError) -> Self {
        match e {
            ARError::EmptyData => OxiError::DataError("Empty data provided".into()),
            ARError::InvalidLagOrder(p) => OxiError::InvalidParameter(format!("Invalid lag order p: {}", p)),
            ARError::InvalidHorizon(v) => OxiError::InvalidParameter(format!("Invalid horizon: {}", v)),
            ARError::InsufficientData { actual, expected } => 
                OxiError::DataError(format!("Insufficient data: {} points provided, {} required", actual, expected)),
            ARError::NotFitted => OxiError::ModelError("Model has not been fitted yet".into()),
            ARError::LinearSolveError(msg) => OxiError::ModelError(format!("Failed to solve AR coefficients: {}", msg)),
            ARError::InvalidCoefficient => OxiError::ModelError("Invalid coefficient detected: NaN or Infinity".into()),
            ARError::InvalidParameter(msg) => OxiError::InvalidParameter(msg),
            ARError::MissingVariable(var) => OxiError::DataError(format!("Missing variable in multivariate model: {}", var)),
            ARError::InconsistentTimestamps => OxiError::DataError("Inconsistent timestamps in multivariate data".into()),
            ARError::InvalidSeasonalPeriod(p) => OxiError::InvalidParameter(format!("Invalid seasonal period: {}", p)),
        }
    }
}

// Define a Result type for internal module use
pub type Result<T> = std::result::Result<T, ARError>; 