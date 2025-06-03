use std::io;
use thiserror::Error;

/// Unified error type for all OxiDiviner operations
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

    /// Error related to validation failures
    #[error("Validation Error: {0}")]
    ValidationError(String),

    /// Error when a feature is not enabled
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),

    // GARCH-specific errors
    /// Error when GARCH input data is invalid
    #[error("GARCH invalid input data: {0}")]
    GarchInvalidData(String),

    /// Error when GARCH model parameters are invalid
    #[error("GARCH invalid model parameters: {0}")]
    GarchInvalidParameters(String),

    /// Error during GARCH estimation process
    #[error("GARCH estimation error: {0}")]
    GarchEstimationError(String),

    /// Error during GARCH forecasting process
    #[error("GARCH forecasting error: {0}")]
    GarchForecastError(String),

    /// GARCH non-convergence error during optimization
    #[error("GARCH optimization did not converge")]
    GarchNonConvergence,

    /// GARCH numerical errors
    #[error("GARCH numerical error: {0}")]
    GarchNumericalError(String),

    // Exponential Smoothing-specific errors
    /// Empty data provided to ES model
    #[error("ES empty data provided")]
    ESEmptyData,

    /// Invalid alpha value in ES model
    #[error("ES invalid alpha value: {0}")]
    ESInvalidAlpha(f64),

    /// Invalid beta value in ES model
    #[error("ES invalid beta value: {0}")]
    ESInvalidBeta(f64),

    /// Invalid gamma value in ES model
    #[error("ES invalid gamma value: {0}")]
    ESInvalidGamma(f64),

    /// Invalid period in ES model
    #[error("ES invalid period: {0}")]
    ESInvalidPeriod(usize),

    /// ES model has not been fitted yet
    #[error("ES model has not been fitted yet")]
    ESNotFitted,

    /// Invalid horizon in ES model
    #[error("ES invalid horizon: {0}")]
    ESInvalidHorizon(usize),

    /// Insufficient data in ES model
    #[error("ES insufficient data: {actual} points provided, {expected} required")]
    ESInsufficientData { actual: usize, expected: usize },

    /// Invalid damping factor in ES model
    #[error("ES invalid damping factor: {0}")]
    ESInvalidDampingFactor(f64),

    /// Missing required parameter in ES model
    #[error("ES missing required parameter: {0}")]
    ESMissingParameter(String),

    /// Unsupported model type in ES
    #[error("ES unsupported model type: {0}")]
    ESUnsupportedModelType(String),

    // Autoregressive-specific errors
    /// Empty data provided to AR model
    #[error("AR empty data provided")]
    AREmptyData,

    /// Invalid lag order in AR model
    #[error("AR invalid lag order p: {0}")]
    ARInvalidLagOrder(usize),

    /// Invalid horizon in AR model
    #[error("AR invalid horizon: {0}")]
    ARInvalidHorizon(usize),

    /// Insufficient data in AR model
    #[error("AR insufficient data: {actual} points provided, {expected} required")]
    ARInsufficientData { actual: usize, expected: usize },

    /// AR model has not been fitted yet
    #[error("AR model has not been fitted yet")]
    ARNotFitted,

    /// Failed to solve linear system for AR coefficients
    #[error("AR failed to solve linear system for AR coefficients: {0}")]
    ARLinearSolveError(String),

    /// Invalid coefficient detected in AR model
    #[error("AR invalid coefficient detected: NaN or Infinity")]
    ARInvalidCoefficient,

    /// Missing variable in multivariate AR model
    #[error("AR missing variable in multivariate model: {0}")]
    ARMissingVariable(String),

    /// Inconsistent timestamps in multivariate AR data
    #[error("AR inconsistent timestamps in multivariate data")]
    ARInconsistentTimestamps,

    /// Invalid seasonal period in AR model
    #[error("AR invalid seasonal period: {0}")]
    ARInvalidSeasonalPeriod(usize),

    // Moving Average-specific errors
    /// Empty data provided to MA model
    #[error("MA empty data provided")]
    MAEmptyData,

    /// Invalid window size in MA model
    #[error("MA invalid window size: {0}")]
    MAInvalidWindowSize(usize),

    /// Time series too short for MA model
    #[error("MA time series length ({actual}) must be at least equal to window size ({expected})")]
    MATimeSeriesTooShort { actual: usize, expected: usize },

    /// MA model has not been fitted yet
    #[error("MA model has not been fitted yet")]
    MANotFitted,

    /// Invalid horizon in MA model
    #[error("MA invalid horizon: {0}")]
    MAInvalidHorizon(usize),

    // Math operation errors
    /// Empty data provided to math operation
    #[error("Math empty data provided")]
    MathEmptyData,

    /// Invalid window size in math operation
    #[error("Math invalid window size: {0}")]
    MathInvalidWindowSize(usize),

    /// Invalid alpha value in math operation
    #[error("Math invalid alpha value: {0}")]
    MathInvalidAlpha(f64),

    /// Data length mismatch in math operation
    #[error("Math data length mismatch: expected {expected}, got {actual}")]
    MathDataLengthMismatch { expected: usize, actual: usize },

    /// Division by zero in math operation
    #[error("Math division by zero")]
    MathDivisionByZero,

    /// Invalid lag value in math operation
    #[error("Math invalid lag value: {0}")]
    MathInvalidLag(usize),

    // Optimization-specific errors
    /// Error related to optimization operations
    #[error("Optimization Error: {0}")]
    OptimizationError(String),
}

/// Unified Result type for all OxiDiviner operations
pub type Result<T> = std::result::Result<T, OxiError>;
