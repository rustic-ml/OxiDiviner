use thiserror::Error;

/// Error type for math operations
#[derive(Error, Debug)]
pub enum MathError {
    #[error("Empty data provided")]
    EmptyData,
    
    #[error("Invalid window size: {0}")]
    InvalidWindowSize(usize),
    
    #[error("Invalid alpha value: {0}")]
    InvalidAlpha(f64),
    
    #[error("Data length mismatch: expected {expected}, got {actual}")]
    DataLengthMismatch {
        expected: usize,
        actual: usize,
    },
    
    #[error("Division by zero")]
    DivisionByZero,
    
    #[error("Invalid lag value: {0}")]
    InvalidLag(usize),
} 