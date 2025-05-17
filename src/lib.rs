// OxiDiviner: Time Series Analysis and Forecasting Library

pub mod data;
pub mod error;
pub mod models;
pub mod utils;

// Re-exports
pub use data::TimeSeriesData;
pub use error::*;
pub use models::data::OHLCVData as ModelsOHLCVData;
pub use models::*;
pub use utils::*;

pub mod prelude {
    pub use crate::data::TimeSeriesData;
    pub use crate::error::{OxiError, Result};
    pub use crate::models::data::OHLCVData as ModelsOHLCVData;
    pub use crate::models::exponential_smoothing::ets::{ETSComponent, DailyETSModel, MinuteETSModel};
    pub use crate::models::exponential_smoothing::simple::SESModel;
    pub use crate::models::exponential_smoothing::holt::HoltModel;
    pub use crate::models::exponential_smoothing::holt_winters::{HoltWintersModel, SeasonalType};
}

// Re-export common types
pub use models::exponential_smoothing::ets::{ETSComponent, DailyETSModel, MinuteETSModel};
pub use models::exponential_smoothing::simple::SESModel;
pub use models::exponential_smoothing::holt::HoltModel;
pub use models::exponential_smoothing::holt_winters::{HoltWintersModel, SeasonalType}; 