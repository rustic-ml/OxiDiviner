mod ar;
mod arima;
mod arma;
mod error;
mod sarima;
mod var;

// Re-export the public models
pub use ar::ARModel;
pub use arima::ARIMAModel;
pub use arma::ARMAModel;
pub use sarima::SARIMAModel;
pub use var::VARModel;

// Re-export the error types
pub use error::{ARError, Result};
