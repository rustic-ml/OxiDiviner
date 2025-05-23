use crate::core::OxiError;
use thiserror::Error;

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
    InsufficientData { actual: usize, expected: usize },

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
            ARError::InvalidLagOrder(p) => {
                OxiError::InvalidParameter(format!("Invalid lag order p: {}", p))
            }
            ARError::InvalidHorizon(v) => {
                OxiError::InvalidParameter(format!("Invalid horizon: {}", v))
            }
            ARError::InsufficientData { actual, expected } => OxiError::DataError(format!(
                "Insufficient data: {} points provided, {} required",
                actual, expected
            )),
            ARError::NotFitted => OxiError::ModelError("Model has not been fitted yet".into()),
            ARError::LinearSolveError(msg) => {
                OxiError::ModelError(format!("Failed to solve AR coefficients: {}", msg))
            }
            ARError::InvalidCoefficient => {
                OxiError::ModelError("Invalid coefficient detected: NaN or Infinity".into())
            }
            ARError::InvalidParameter(msg) => OxiError::InvalidParameter(msg),
            ARError::MissingVariable(var) => {
                OxiError::DataError(format!("Missing variable in multivariate model: {}", var))
            }
            ARError::InconsistentTimestamps => {
                OxiError::DataError("Inconsistent timestamps in multivariate data".into())
            }
            ARError::InvalidSeasonalPeriod(p) => {
                OxiError::InvalidParameter(format!("Invalid seasonal period: {}", p))
            }
        }
    }
}

// Define a Result type for internal module use
pub type Result<T> = std::result::Result<T, ARError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ar_error_conversions() {
        // Test conversion of each error type to OxiError

        // EmptyData
        let ar_err = ARError::EmptyData;
        let oxi_err = OxiError::from(ar_err);
        if let OxiError::DataError(msg) = oxi_err {
            assert!(msg.contains("Empty data"));
        } else {
            panic!("Expected DataError");
        }

        // InvalidLagOrder
        let ar_err = ARError::InvalidLagOrder(0);
        let oxi_err = OxiError::from(ar_err);
        if let OxiError::InvalidParameter(msg) = oxi_err {
            assert!(msg.contains("lag order"));
            assert!(msg.contains("0"));
        } else {
            panic!("Expected InvalidParameter");
        }

        // InvalidHorizon
        let ar_err = ARError::InvalidHorizon(0);
        let oxi_err = OxiError::from(ar_err);
        if let OxiError::InvalidParameter(msg) = oxi_err {
            assert!(msg.contains("horizon"));
            assert!(msg.contains("0"));
        } else {
            panic!("Expected InvalidParameter");
        }

        // InsufficientData
        let ar_err = ARError::InsufficientData {
            actual: 2,
            expected: 5,
        };
        let oxi_err = OxiError::from(ar_err);
        if let OxiError::DataError(msg) = oxi_err {
            assert!(msg.contains("Insufficient data"));
            assert!(msg.contains("2"));
            assert!(msg.contains("5"));
        } else {
            panic!("Expected DataError");
        }

        // NotFitted
        let ar_err = ARError::NotFitted;
        let oxi_err = OxiError::from(ar_err);
        if let OxiError::ModelError(msg) = oxi_err {
            assert!(msg.contains("not been fitted"));
        } else {
            panic!("Expected ModelError");
        }

        // LinearSolveError
        let ar_err = ARError::LinearSolveError("singular matrix".to_string());
        let oxi_err = OxiError::from(ar_err);
        if let OxiError::ModelError(msg) = oxi_err {
            assert!(msg.contains("Failed to solve"));
            assert!(msg.contains("singular matrix"));
        } else {
            panic!("Expected ModelError");
        }

        // InvalidCoefficient
        let ar_err = ARError::InvalidCoefficient;
        let oxi_err = OxiError::from(ar_err);
        if let OxiError::ModelError(msg) = oxi_err {
            assert!(msg.contains("coefficient"));
            assert!(msg.contains("NaN"));
        } else {
            panic!("Expected ModelError");
        }

        // InvalidParameter
        let ar_err = ARError::InvalidParameter("bad parameter".to_string());
        let oxi_err = OxiError::from(ar_err);
        if let OxiError::InvalidParameter(msg) = oxi_err {
            assert_eq!(msg, "bad parameter");
        } else {
            panic!("Expected InvalidParameter");
        }

        // MissingVariable
        let ar_err = ARError::MissingVariable("price".to_string());
        let oxi_err = OxiError::from(ar_err);
        if let OxiError::DataError(msg) = oxi_err {
            assert!(msg.contains("Missing variable"));
            assert!(msg.contains("price"));
        } else {
            panic!("Expected DataError");
        }

        // InconsistentTimestamps
        let ar_err = ARError::InconsistentTimestamps;
        let oxi_err = OxiError::from(ar_err);
        if let OxiError::DataError(msg) = oxi_err {
            assert!(msg.contains("timestamps"));
        } else {
            panic!("Expected DataError");
        }

        // InvalidSeasonalPeriod
        let ar_err = ARError::InvalidSeasonalPeriod(0);
        let oxi_err = OxiError::from(ar_err);
        if let OxiError::InvalidParameter(msg) = oxi_err {
            assert!(msg.contains("seasonal period"));
            assert!(msg.contains("0"));
        } else {
            panic!("Expected InvalidParameter");
        }
    }

    #[test]
    fn test_ar_error_display() {
        // Test the display implementation for a few error types
        let err = ARError::EmptyData;
        assert_eq!(format!("{}", err), "Empty data provided");

        let err = ARError::InvalidLagOrder(0);
        assert_eq!(format!("{}", err), "Invalid lag order p: 0");

        let err = ARError::InsufficientData {
            actual: 3,
            expected: 10,
        };
        assert_eq!(
            format!("{}", err),
            "Insufficient data: 3 points provided, 10 required"
        );

        let err = ARError::MissingVariable("volume".to_string());
        assert_eq!(
            format!("{}", err),
            "Missing variable in multivariate model: volume"
        );
    }
}
