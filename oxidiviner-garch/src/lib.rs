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

mod egarch;
mod error;
mod garch;
mod garch_m;
mod gjr_garch;

// Re-export the public models
pub use egarch::EGARCHModel;
pub use garch::GARCHModel;
pub use garch_m::{GARCHMModel, RiskPremiumType};
pub use gjr_garch::GJRGARCHModel;

// Re-export the error types
pub use error::{GARCHError, Result};

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    #[test]
    fn test_garch_model_creation() {
        // Test creating a GARCH(1,1) model with default parameters
        let result = GARCHModel::new(1, 1, None);
        assert!(result.is_ok());

        let model = result.unwrap();
        assert_eq!(model.alpha.len(), 1);
        assert_eq!(model.beta.len(), 1);
        assert_eq!(model.order(), (1, 1));

        // Test with invalid parameters (p=0, q=0)
        let result = GARCHModel::new(0, 0, None);
        assert!(result.is_err());

        // Test with explicit parameters
        let params = vec![0.0, 0.01, 0.05, 0.85]; // mean, omega, alpha1, beta1
        let result = GARCHModel::new(1, 1, Some(params));
        assert!(result.is_ok());

        let model = result.unwrap();
        assert_eq!(model.omega, 0.01);
        assert_eq!(model.alpha[0], 0.05);
        assert_eq!(model.beta[0], 0.85);
    }

    #[test]
    fn test_garch_model_fit_and_forecast() {
        // Create synthetic returns data with volatility clustering
        let n = 200;
        let mut returns: Vec<f64> = Vec::with_capacity(n);
        let mut volatility: f64 = 0.01;

        for i in 0..n {
            // Simulate volatility clustering
            if i > 0 {
                // Simple GARCH-like process
                volatility = 0.01 + 0.05 * returns[i - 1].powi(2) + 0.90 * volatility;
            }

            // Generate return with current volatility
            let shock = (i as f64 * 0.1).sin() * f64::sqrt(volatility);
            returns.push(shock);
        }

        // Create timestamps
        let now = Utc::now();
        let timestamps: Vec<_> = (0..n)
            .map(|i| {
                Utc.timestamp_opt(now.timestamp() + (i as i64) * 86400, 0)
                    .unwrap()
            })
            .collect();

        // Create and fit GARCH(1,1) model
        let mut model = GARCHModel::new(1, 1, None).unwrap();
        let result = model.fit(&returns, Some(&timestamps));
        assert!(result.is_ok());

        // Check that fitted parameters are available
        assert!(model.fitted_variance.is_some());
        assert!(model.residuals.is_some());

        // Test forecasting
        let forecast_horizon = 10;
        let forecast = model.forecast_variance(forecast_horizon);
        assert!(forecast.is_ok());
        assert_eq!(forecast.unwrap().len(), forecast_horizon);
    }

    #[test]
    fn test_garch_m_model() {
        // Create a GARCH-M(1,1) model
        let risk_type = RiskPremiumType::Variance;
        let result = GARCHMModel::new(1, 1, risk_type, None);
        assert!(result.is_ok());

        // Test with invalid parameters
        let result = GARCHMModel::new(0, 0, risk_type, None);
        assert!(result.is_err());

        // Test with explicit parameters
        let params = vec![0.0, 0.1, 0.01, 0.05, 0.85]; // mean, lambda, omega, alpha1, beta1
        let result = GARCHMModel::new(1, 1, risk_type, Some(params));
        assert!(result.is_ok());

        let model = result.unwrap();
        assert_eq!(model.lambda, 0.1);
        assert_eq!(model.omega, 0.01);
        assert_eq!(model.alpha[0], 0.05);
        assert_eq!(model.beta[0], 0.85);
    }

    #[test]
    fn test_gjr_garch_model() {
        // Create a GJR-GARCH(1,1) model
        let result = GJRGARCHModel::new(1, 1, None);
        assert!(result.is_ok());

        // Test with explicit parameters (mean, omega, alpha1, gamma1, beta1)
        let params = vec![0.0, 0.01, 0.03, 0.05, 0.85];
        let result = GJRGARCHModel::new(1, 1, Some(params));
        assert!(result.is_ok());

        let model = result.unwrap();
        assert_eq!(model.omega, 0.01);
        assert_eq!(model.alpha[0], 0.03);
        assert_eq!(model.gamma[0], 0.05);
        assert_eq!(model.beta[0], 0.85);

        // Test news impact curve
        let (shock_values, variance_values) = model.news_impact_curve(10, 2.0);
        assert_eq!(shock_values.len(), 10);
        assert_eq!(variance_values.len(), 10);

        // Verify asymmetry: negative shock should have higher impact than positive
        let _mid_point = shock_values.len() / 2;
        let positive_idx = shock_values.len() - 1;
        let negative_idx = 0;

        // Find a positive and negative shock of similar magnitude
        assert!(shock_values[positive_idx] > 0.0);
        assert!(shock_values[negative_idx] < 0.0);

        // Due to positive gamma, negative shocks should have higher impact
        assert!(variance_values[negative_idx] > variance_values[positive_idx]);
    }

    #[test]
    fn test_egarch_model() {
        // Create an EGARCH(1,1) model
        let result = EGARCHModel::new(1, 1, None);
        assert!(result.is_ok());

        // Test with explicit parameters (omega, alpha1, gamma1, beta1)
        let params = vec![0.0, -0.1, 0.1, 0.05, 0.9];
        let result = EGARCHModel::new(1, 1, Some(params));
        assert!(result.is_ok());

        let model = result.unwrap();
        assert_eq!(model.omega, -0.1);
        assert_eq!(model.alpha[0], 0.1);
        assert_eq!(model.gamma[0], 0.05);
        assert_eq!(model.beta[0], 0.9);

        // Test with invalid parameters
        let result = EGARCHModel::new(0, 0, None);
        assert!(result.is_err());

        // Test model fit with synthetic data
        let mut model = EGARCHModel::new(1, 1, None).unwrap();
        let n = 100;
        let mut returns = Vec::with_capacity(n);

        // Generate some returns with volatility clustering
        let mut volatility: f64 = 0.01;
        for i in 0..n {
            if i > 0 {
                // Add asymmetric effect where negative returns increase volatility more
                let prev_return: f64 = returns[i - 1];
                let asym_effect: f64 = if prev_return < 0.0 { 1.5 } else { 0.5 };
                volatility = 0.01 + 0.1 * prev_return.powi(2) * asym_effect + 0.85 * volatility;
            }
            returns.push((i as f64 * 0.01).sin() * volatility.sqrt());
        }

        // Fit the model
        let fit_result = model.fit(&returns, None);
        assert!(fit_result.is_ok());

        // Test forecasting
        let forecast_result = model.forecast_variance(5);
        assert!(forecast_result.is_ok());
        assert_eq!(forecast_result.unwrap().len(), 5);
    }
}
