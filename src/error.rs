use thiserror::Error;
use std::error::Error as StdError;
use std::result::Result as StdResult;

/// Custom result type for OxiDiviner operations
pub type Result<T> = StdResult<T, OxiError>;

/// Error types for OxiDiviner
#[derive(Error, Debug)]
pub enum OxiError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Data error: {0}")]
    Data(String),
    
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),
    
    #[error("Forecast error: {0}")]
    Forecast(String),
    
    #[error("Model error: {0}")]
    Model(String),
    
    #[error("Other error: {0}")]
    Other(String),
}

impl OxiError {
    pub fn data_error<S: Into<String>>(msg: S) -> Self {
        OxiError::Data(msg.into())
    }
    
    pub fn invalid_params<S: Into<String>>(msg: S) -> Self {
        OxiError::InvalidParams(msg.into())
    }
    
    pub fn forecast_error<S: Into<String>>(msg: S) -> Self {
        OxiError::Forecast(msg.into())
    }
    
    pub fn model_error<S: Into<String>>(msg: S) -> Self {
        OxiError::Model(msg.into())
    }
}

impl From<String> for OxiError {
    fn from(s: String) -> Self {
        OxiError::Other(s)
    }
}

impl From<&str> for OxiError {
    fn from(s: &str) -> Self {
        OxiError::Other(s.to_string())
    }
}

impl From<Box<dyn StdError>> for OxiError {
    fn from(e: Box<dyn StdError>) -> Self {
        OxiError::Other(e.to_string())
    }
}

impl From<Box<dyn StdError + Send + Sync>> for OxiError {
    fn from(e: Box<dyn StdError + Send + Sync>) -> Self {
        OxiError::Other(e.to_string())
    }
} 