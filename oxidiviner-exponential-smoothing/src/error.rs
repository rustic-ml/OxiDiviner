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
            ESError::InvalidParameter(msg) => OxiError::InvalidParameter(msg),
        }
    }
}

// Define a Result type for internal module use
pub type Result<T> = std::result::Result<T, ESError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_conversions() {
        // Test each error type and its conversion to OxiError

        // Empty data
        let es_err = ESError::EmptyData;
        let oxi_err = OxiError::from(es_err);
        if let OxiError::DataError(msg) = oxi_err {
            assert!(msg.contains("Empty data"));
        } else {
            panic!("Expected DataError");
        }

        // Invalid alpha
        let es_err = ESError::InvalidAlpha(1.5);
        let oxi_err = OxiError::from(es_err);
        if let OxiError::InvalidParameter(msg) = oxi_err {
            assert!(msg.contains("alpha"));
            assert!(msg.contains("1.5"));
        } else {
            panic!("Expected InvalidParameter");
        }

        // Invalid beta
        let es_err = ESError::InvalidBeta(-0.1);
        let oxi_err = OxiError::from(es_err);
        if let OxiError::InvalidParameter(msg) = oxi_err {
            assert!(msg.contains("beta"));
            assert!(msg.contains("-0.1"));
        } else {
            panic!("Expected InvalidParameter");
        }

        // Invalid gamma
        let es_err = ESError::InvalidGamma(2.0);
        let oxi_err = OxiError::from(es_err);
        if let OxiError::InvalidParameter(msg) = oxi_err {
            assert!(msg.contains("gamma"));
            assert!(msg.contains("2"));
        } else {
            panic!("Expected InvalidParameter");
        }

        // Invalid period
        let es_err = ESError::InvalidPeriod(0);
        let oxi_err = OxiError::from(es_err);
        if let OxiError::InvalidParameter(msg) = oxi_err {
            assert!(msg.contains("period"));
            assert!(msg.contains("0"));
        } else {
            panic!("Expected InvalidParameter");
        }

        // Not fitted
        let es_err = ESError::NotFitted;
        let oxi_err = OxiError::from(es_err);
        if let OxiError::ModelError(msg) = oxi_err {
            assert!(msg.contains("not been fitted"));
        } else {
            panic!("Expected ModelError");
        }

        // Invalid horizon
        let es_err = ESError::InvalidHorizon(0);
        let oxi_err = OxiError::from(es_err);
        if let OxiError::InvalidParameter(msg) = oxi_err {
            assert!(msg.contains("horizon"));
            assert!(msg.contains("0"));
        } else {
            panic!("Expected InvalidParameter");
        }

        // Insufficient data
        let es_err = ESError::InsufficientData {
            actual: 2,
            expected: 5,
        };
        let oxi_err = OxiError::from(es_err);
        if let OxiError::DataError(msg) = oxi_err {
            assert!(msg.contains("Insufficient data"));
            assert!(msg.contains("2"));
            assert!(msg.contains("5"));
        } else {
            panic!("Expected DataError");
        }

        // Invalid damping factor
        let es_err = ESError::InvalidDampingFactor(1.5);
        let oxi_err = OxiError::from(es_err);
        if let OxiError::InvalidParameter(msg) = oxi_err {
            assert!(msg.contains("damping factor"));
            assert!(msg.contains("1.5"));
        } else {
            panic!("Expected InvalidParameter");
        }

        // Missing parameter
        let es_err = ESError::MissingParameter("alpha".to_string());
        let oxi_err = OxiError::from(es_err);
        if let OxiError::InvalidParameter(msg) = oxi_err {
            assert!(msg.contains("Missing"));
            assert!(msg.contains("alpha"));
        } else {
            panic!("Expected InvalidParameter");
        }

        // Unsupported model type
        let es_err = ESError::UnsupportedModelType("XYZ".to_string());
        let oxi_err = OxiError::from(es_err);
        if let OxiError::ModelError(msg) = oxi_err {
            assert!(msg.contains("Unsupported"));
            assert!(msg.contains("XYZ"));
        } else {
            panic!("Expected ModelError");
        }

        // Invalid parameter
        let es_err = ESError::InvalidParameter("test parameter".to_string());
        let oxi_err = OxiError::from(es_err);
        if let OxiError::InvalidParameter(msg) = oxi_err {
            assert!(msg.contains("test parameter"));
        } else {
            panic!("Expected InvalidParameter");
        }
    }

    #[test]
    fn test_error_display() {
        // Test the display implementation for a few error types
        let err = ESError::EmptyData;
        assert_eq!(format!("{}", err), "Empty data provided");

        let err = ESError::InvalidAlpha(0.5);
        assert_eq!(format!("{}", err), "Invalid alpha value: 0.5");

        let err = ESError::InsufficientData {
            actual: 3,
            expected: 10,
        };
        assert_eq!(
            format!("{}", err),
            "Insufficient data: 3 points provided, 10 required"
        );
    }
}
