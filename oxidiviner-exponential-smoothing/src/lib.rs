use oxidiviner_core::{Forecaster, ModelEvaluation, ModelOutput, TimeSeriesData};
use oxidiviner_math::transforms::exponential_moving_average;

mod damped_trend;
mod error;
mod ets;
mod holt;
mod holt_winters;
mod simple;

// Re-export the public models
pub use damped_trend::DampedTrendModel;
pub use ets::ETSModel;
pub use holt::HoltLinearModel;
pub use holt_winters::HoltWintersModel;
pub use simple::SimpleESModel;

// Re-export the error types
pub use error::{ESError, Result};

// We'll create stub modules for now and implement them later
// When implementing them, uncomment these lines
// mod holt;
// mod holt_winters;
// mod ets;

// pub use holt::HoltLinearModel;
// pub use holt_winters::HoltWintersModel;
// pub use ets::ETSModel;
