/*!
# OxiDiviner GARCH Models

A comprehensive implementation of Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models
for time series analysis, volatility forecasting, and risk modeling in financial data.

## Available Models

This crate provides several variants of GARCH models:

* [`GARCHModel`] - Standard GARCH(p,q) model
* [`EGARCHModel`] - Exponential GARCH model for asymmetric volatility
* [`GJRGARCHModel`] - Glosten-Jagannathan-Runkle GARCH for leverage effects
* [`GARCHMModel`] - GARCH-in-Mean model with risk premium in the mean equation

## Usage Example

```rust
use oxidiviner_garch::GARCHModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a GARCH(1,1) model
    let mut model = GARCHModel::new(1, 1, None)?;
    
    // Time series data
    let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02, 0.01];
    
    // Fit the model
    model.fit(&returns, None)?;
    
    // Display model parameters
    println!("{}", model);
    
    // Forecast volatility for the next 5 periods
    let forecast = model.forecast_variance(5)?;
    println!("Volatility forecast: {:?}", forecast);
    
    Ok(())
}
*/

mod error;
mod garch;
mod egarch;
mod gjr_garch;
mod garch_m;

// Re-export the public models
pub use garch::GARCHModel;
pub use egarch::EGARCHModel;
pub use gjr_garch::GJRGARCHModel;
pub use garch_m::{GARCHMModel, RiskPremiumType};

// Re-export the error types
pub use error::{GARCHError, Result}; 