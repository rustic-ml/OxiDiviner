

mod error;
mod ar;
mod arma;
mod arima;
mod sarima;
mod var;

// Re-export the public models
pub use ar::ARModel;
pub use arma::ARMAModel;
pub use arima::ARIMAModel;
pub use sarima::SARIMAModel;
pub use var::VARModel;

// Re-export the error types
pub use error::{ARError, Result}; 