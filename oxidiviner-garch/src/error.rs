use thiserror::Error;

/// Result type for GARCH operations
pub type Result<T> = std::result::Result<T, GARCHError>;

/// Error types for GARCH models
#[derive(Error, Debug)]
pub enum GARCHError {
    /// Error when input data is invalid
    #[error("Invalid input data: {0}")]
    InvalidData(String),

    /// Error when model parameters are invalid
    #[error("Invalid model parameters: {0}")]
    InvalidParameters(String),

    /// Error during estimation process
    #[error("Estimation error: {0}")]
    EstimationError(String),

    /// Error during forecasting process
    #[error("Forecasting error: {0}")]
    ForecastError(String),

    /// Non-convergence error during optimization
    #[error("Optimization did not converge")]
    NonConvergence,

    /// Numerical errors
    #[error("Numerical error: {0}")]
    NumericalError(String),
}
