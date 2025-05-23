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
    use chrono::{TimeZone, Utc, DateTime};

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

    #[test]
    fn test_egarch_with_timestamps() {
        let mut model = EGARCHModel::new(1, 1, None).unwrap();
        
        let n = 50;
        let returns: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..n)
            .map(|i| {
                Utc.timestamp_opt(now.timestamp() + (i as i64) * 86400, 0)
                    .unwrap()
            })
            .collect();

        let result = model.fit(&returns, Some(&timestamps));
        assert!(result.is_ok());
        
        // Check that timestamps were stored
        assert!(model.timestamps.is_some());
        assert_eq!(model.timestamps.as_ref().unwrap().len(), n);
    }

    #[test]
    fn test_egarch_news_impact_curve() {
        let model = EGARCHModel::new(1, 1, Some(vec![0.0, -0.1, 0.1, 0.05, 0.9])).unwrap();
        
        let (shock_values, variance_values) = model.news_impact_curve(10, 2.0);
        assert_eq!(shock_values.len(), 10);
        assert_eq!(variance_values.len(), 10);
        
        // Check that we have both positive and negative shocks
        assert!(shock_values.iter().any(|&x| x > 0.0));
        assert!(shock_values.iter().any(|&x| x < 0.0));
        
        // All variance values should be positive
        assert!(variance_values.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_egarch_validation() {
        // Test invalid parameters that would make model non-stationary
        let params = vec![0.0, -0.1, 0.1, 0.05, 1.1]; // beta > 1
        let result = EGARCHModel::new(1, 1, Some(params));
        assert!(result.is_err());
    }

    #[test]
    fn test_gjr_garch_advanced() {
        // Test with complex parameter sets
        let params = vec![0.0, 0.01, 0.03, 0.05, 0.85];
        let mut model = GJRGARCHModel::new(1, 1, Some(params)).unwrap();
        
        // Generate realistic financial returns data
        let n = 200;
        let mut returns: Vec<f64> = Vec::with_capacity(n);
        let mut volatility: f64 = 0.01;
        
        for i in 0..n {
            if i > 0 {
                let prev_return: f64 = returns[i - 1];
                let arch_term = 0.03 * prev_return.powi(2);
                let asymmetric_term = if prev_return < 0.0 { 0.05 * prev_return.powi(2) } else { 0.0 };
                let garch_term = 0.85 * volatility;
                volatility = 0.01 + arch_term + asymmetric_term + garch_term;
            }
            
            // Generate return with current volatility
            let shock = (i as f64 * 0.7).sin() * volatility.sqrt();
            returns.push(shock);
        }
        
        let result = model.fit(&returns, None);
        assert!(result.is_ok());
        
        // Test that fitted values are available
        assert!(model.fitted_variance.is_some());
        assert!(model.residuals.is_some());
        assert!(model.log_likelihood.is_some());
        assert!(model.info_criteria.is_some());
        
        // Test forecasting with longer horizon
        let forecast = model.forecast_variance(20);
        assert!(forecast.is_ok());
        assert_eq!(forecast.unwrap().len(), 20);
    }

    #[test]
    fn test_gjr_garch_parameter_validation() {
        // Test omega <= 0
        let params = vec![0.0, -0.01, 0.03, 0.05, 0.85];
        let result = GJRGARCHModel::new(1, 1, Some(params));
        assert!(result.is_err());
        
        // Test negative alpha
        let params = vec![0.0, 0.01, -0.03, 0.05, 0.85];
        let result = GJRGARCHModel::new(1, 1, Some(params));
        assert!(result.is_err());
        
        // Test alpha + gamma/2 < 0
        let params = vec![0.0, 0.01, 0.01, -0.05, 0.85];
        let result = GJRGARCHModel::new(1, 1, Some(params));
        assert!(result.is_err());
        
        // Test non-stationarity (sum >= 1)
        let params = vec![0.0, 0.01, 0.5, 0.5, 0.5];
        let result = GJRGARCHModel::new(1, 1, Some(params));
        assert!(result.is_err());
    }

    #[test]
    fn test_garch_m_advanced() {
        // Test different risk premium types
        let variance_type = RiskPremiumType::Variance;
        let std_dev_type = RiskPremiumType::StdDev;
        
        let params = vec![0.0, 0.1, 0.01, 0.05, 0.85];
        
        let model_var = GARCHMModel::new(1, 1, variance_type, Some(params.clone());
        assert!(model_var.is_ok());
        
        let model_vol = GARCHMModel::new(1, 1, std_dev_type, Some(params));
        assert!(model_vol.is_ok());
        
        // Test that lambda parameter was set correctly
        assert_eq!(model_var.unwrap().lambda, 0.1);
        assert_eq!(model_vol.unwrap().lambda, 0.1);
    }

    #[test]
    fn test_garch_m_fitting() {
        let mut model = GARCHMModel::new(1, 1, RiskPremiumType::Variance, None).unwrap();
        
        // Create returns with risk premium effects
        let n = 100;
        let mut returns: Vec<f64> = Vec::with_capacity(n);
        let mut variance = 0.01;
        
        for i in 0..n {
            if i > 0 {
                variance = 0.01 + 0.05 * returns[i - 1].powi(2) + 0.85 * variance;
            }
            
            // Add risk premium to mean return
            let risk_premium = 0.1 * variance; // lambda * variance
            let shock = (i as f64 * 0.3).sin() * variance.sqrt();
            returns.push(risk_premium + shock);
        }
        
        let result = model.fit(&returns, None);
        assert!(result.is_ok());
        
        // Test forecasting (GARCH-M returns both mean and variance forecasts)
        let forecast = model.forecast(10);
        assert!(forecast.is_ok());
        let (mean_forecast, variance_forecast) = forecast.unwrap();
        assert_eq!(mean_forecast.len(), 10);
        assert_eq!(variance_forecast.len(), 10);
    }

    #[test]
    fn test_basic_garch_advanced() {
        // Test higher order GARCH model
        let mut model = GARCHModel::new(2, 2, None).unwrap();
        assert_eq!(model.alpha.len(), 2);
        assert_eq!(model.beta.len(), 2);
        assert_eq!(model.order(), (2, 2));
        
        // Test with specific parameters for GARCH(2,2)
        let params = vec![0.0, 0.01, 0.05, 0.03, 0.4, 0.5]; // mean, omega, alpha1, alpha2, beta1, beta2
        let model_with_params = GARCHModel::new(2, 2, Some(params));
        assert!(model_with_params.is_ok());
        
        let model = model_with_params.unwrap();
        assert_eq!(model.alpha[0], 0.05);
        assert_eq!(model.alpha[1], 0.03);
        assert_eq!(model.beta[0], 0.4);
        assert_eq!(model.beta[1], 0.5);
    }

    #[test]
    fn test_garch_error_conditions() {
        // Test insufficient data
        let mut model = GARCHModel::new(1, 1, None).unwrap();
        let insufficient_data = vec![1.0]; // Only one data point
        
        let result = model.fit(&insufficient_data, None);
        assert!(result.is_err());
        
        // Test empty data
        let empty_data: Vec<f64> = vec![];
        let result = model.fit(&empty_data, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_display_traits() {
        let garch_model = GARCHModel::new(1, 1, None).unwrap();
        let garch_str = format!("{}", garch_model);
        assert!(garch_str.contains("GARCH"));
        
        let gjr_model = GJRGARCHModel::new(1, 1, None).unwrap();
        let gjr_str = format!("{}", gjr_model);
        assert!(gjr_str.contains("GJR-GARCH"));
        
        let egarch_model = EGARCHModel::new(1, 1, None).unwrap();
        let egarch_str = format!("{}", egarch_model);
        assert!(egarch_str.contains("EGARCH"));
        
        let garch_m_model = GARCHMModel::new(1, 1, RiskPremiumType::Variance, None).unwrap();
        let garch_m_str = format!("{}", garch_m_model);
        assert!(garch_m_str.contains("GARCH-M"));
    }

    #[test]
    fn test_garch_forecast_edge_cases() {
        let mut model = GARCHModel::new(1, 1, None).unwrap();
        
        // Fit with simple data
        let data = vec![0.01, -0.02, 0.015, -0.01, 0.005];
        model.fit(&data, None).unwrap();
        
        // Test forecasting with zero horizon
        let forecast = model.forecast_variance(0);
        assert!(forecast.is_ok());
        assert_eq!(forecast.unwrap().len(), 0);
        
        // Test forecasting with large horizon
        let forecast = model.forecast_variance(100);
        assert!(forecast.is_ok());
        assert_eq!(forecast.unwrap().len(), 100);
    }

    #[test]
    fn test_egarch_asymmetric_effects() {
        let params = vec![0.0, -0.1, 0.1, -0.05, 0.9]; // negative gamma for leverage effect
        let model = EGARCHModel::new(1, 1, Some(params)).unwrap();
        
        let (shocks, variances) = model.news_impact_curve(21, 3.0);
        
        // Find indices for positive and negative shocks of similar magnitude
        let mid_idx = shocks.len() / 2;
        let positive_shock_idx = shocks.len() - 1;
        let negative_shock_idx = 0;
        
        // With negative gamma, negative shocks should have higher impact
        if shocks[negative_shock_idx] < 0.0 && shocks[positive_shock_idx] > 0.0 {
            // Due to leverage effect (negative gamma), negative shocks should increase variance more
            assert!(variances[negative_shock_idx] > variances[mid_idx]);
        }
    }

    #[test]
    fn test_gjr_garch_asymmetric_effects() {
        let params = vec![0.0, 0.01, 0.03, 0.05, 0.85]; // positive gamma
        let model = GJRGARCHModel::new(1, 1, Some(params)).unwrap();
        
        let (shocks, variances) = model.news_impact_curve(21, 2.0);
        
        // With positive gamma, negative shocks should have higher impact than positive
        let negative_idx = shocks.iter().position(|&x| x < -1.0).unwrap_or(0);
        let positive_idx = shocks.iter().rposition(|&x| x > 1.0).unwrap_or(shocks.len() - 1);
        
        if shocks[negative_idx] < 0.0 && shocks[positive_idx] > 0.0 {
            // GJR effect: negative shock has alpha + gamma effect
            assert!(variances[negative_idx] > variances[positive_idx]);
        }
    }

    #[test]
    fn test_model_information_criteria() {
        let mut model = GARCHModel::new(1, 1, None).unwrap();
        
        let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        model.fit(&data, None).unwrap();
        
        // After fitting, information criteria should be available
        assert!(model.log_likelihood.is_some());
        assert!(model.info_criteria.is_some());
        
        let (aic, bic) = model.info_criteria.unwrap();
        
        // AIC and BIC should be finite and reasonable
        assert!(aic.is_finite());
        assert!(bic.is_finite());
        assert!(bic > aic); // BIC should be larger than AIC due to additional penalty
    }

    #[test]
    fn test_garch_convergence() {
        let mut model = GARCHModel::new(1, 1, None).unwrap();
        
        // Create a challenging dataset with high volatility clustering
        let n = 200;
        let mut returns: Vec<f64> = Vec::with_capacity(n);
        let mut variance: f64 = 0.01;
        
        for i in 0..n {
            if i > 0 {
                // Strong volatility clustering
                variance = 0.001 + 0.1 * returns[i - 1].powi(2) + 0.85 * variance;
            }
            
            // Add occasional large shocks
            let shock_multiplier = if i % 20 == 0 { 3.0 } else { 1.0 };
            returns.push((i as f64 * 0.05).sin() * variance.sqrt() * shock_multiplier);
        }
        
        let result = model.fit(&returns, None);
        assert!(result.is_ok());
        
        // Model should converge and produce reasonable fitted variance
        assert!(model.fitted_variance.is_some());
        let fitted_var = model.fitted_variance.as_ref().unwrap();
        assert_eq!(fitted_var.len(), returns.len());
        
        // All fitted variances should be positive
        assert!(fitted_var.iter().all(|&v| v > 0.0));
    }

    // Additional comprehensive tests for better coverage
    #[test]
    fn test_garch_model_comprehensive() {
        // Test different GARCH orders
        for p in 1..=3 {
            for q in 1..=3 {
                let model = GARCHModel::new(p, q, None);
                assert!(model.is_ok());
                
                let model = model.unwrap();
                assert_eq!(model.alpha.len(), p);
                assert_eq!(model.beta.len(), q);
                assert_eq!(model.order(), (p, q));
                
                // Test with explicit parameters
                let param_count = 2 + p + q; // mean, omega, alphas, betas
                let params = vec![0.0; param_count];
                let param_model = GARCHModel::new(p, q, Some(params));
                assert!(param_model.is_ok());
            }
        }
        
        // Test edge cases
        let result = GARCHModel::new(0, 1, None);
        assert!(result.is_err());
        
        let result = GARCHModel::new(1, 0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_egarch_model_comprehensive() {
        // Test different EGARCH orders
        for p in 1..=2 {
            for q in 1..=2 {
                let model = EGARCHModel::new(p, q, None);
                assert!(model.is_ok());
                
                let model = model.unwrap();
                assert_eq!(model.alpha.len(), p);
                assert_eq!(model.gamma.len(), p);
                assert_eq!(model.beta.len(), q);
                
                // Test parameter validation
                assert!(model.omega <= 0.0 || model.omega > 0.0); // Just check it's a valid float
            }
        }
        
        // Test asymmetric effects with different gamma values
        let positive_gamma = vec![0.0, -0.1, 0.1, 0.05, 0.9];
        let model_pos = EGARCHModel::new(1, 1, Some(positive_gamma));
        assert!(model_pos.is_ok());
        
        let negative_gamma = vec![0.0, -0.1, 0.1, -0.05, 0.9];
        let model_neg = EGARCHModel::new(1, 1, Some(negative_gamma));
        assert!(model_neg.is_ok());
    }

    #[test]
    fn test_gjr_garch_model_comprehensive() {
        // Test different GJR-GARCH orders
        for p in 1..=2 {
            for q in 1..=2 {
                let model = GJRGARCHModel::new(p, q, None);
                assert!(model.is_ok());
                
                let model = model.unwrap();
                assert_eq!(model.alpha.len(), p);
                assert_eq!(model.gamma.len(), p);
                assert_eq!(model.beta.len(), q);
                
                // Test parameter bounds
                assert!(model.omega > 0.0);
                for &alpha in &model.alpha {
                    assert!(alpha >= 0.0);
                }
                for &gamma in &model.gamma {
                    assert!(gamma >= 0.0);
                }
                for &beta in &model.beta {
                    assert!(beta >= 0.0 && beta <= 1.0);
                }
            }
        }
        
        // Test news impact curve properties
        let model = GJRGARCHModel::new(1, 1, None).unwrap();
        let (shocks, impacts) = model.news_impact_curve(10, 2.0);
        
        assert_eq!(shocks.len(), 10);
        assert_eq!(impacts.len(), 10);
        assert!(impacts.iter().all(|&v| v >= 0.0)); // All impacts should be non-negative
    }

    #[test]
    fn test_garch_m_model_comprehensive() {
        // Test both risk premium types
        let risk_types = vec![RiskPremiumType::Variance, RiskPremiumType::StdDev];
        
        for risk_type in risk_types {
            // Test different orders
            for p in 1..=2 {
                for q in 1..=2 {
                    let model = GARCHMModel::new(p, q, risk_type, None);
                    assert!(model.is_ok());
                    
                    let model = model.unwrap();
                    assert_eq!(model.alpha.len(), p);
                    assert_eq!(model.beta.len(), q);
                    
                    // Test parameter initialization
                    assert!(model.lambda.is_finite());
                    assert!(model.omega > 0.0);
                }
            }
        }
        
        // Test parameter validation
        let invalid_params = vec![0.0, 0.1, -0.01, 0.05, 0.85]; // negative omega
        let result = GARCHMModel::new(1, 1, RiskPremiumType::Variance, Some(invalid_params));
        assert!(result.is_err());
    }

    #[test]
    fn test_garch_fitting_comprehensive() {
        let data_sets = vec![
            // Trending data
            (0..100).map(|i| (i as f64) * 0.001 + (i as f64 * 0.1).sin() * 0.01).collect::<Vec<f64>>(),
            // High volatility data
            (0..100).map(|i| (i as f64 * 0.05).sin() * 0.05).collect::<Vec<f64>>(),
            // Low volatility data
            (0..100).map(|i| (i as f64 * 0.2).cos() * 0.001).collect::<Vec<f64>>(),
        ];
        
        for (i, data) in data_sets.iter().enumerate() {
            let mut garch_model = GARCHModel::new(1, 1, None).unwrap();
            let result = garch_model.fit(data, None);
            assert!(result.is_ok(), "GARCH fitting failed for dataset {}", i);
            
            let mut egarch_model = EGARCHModel::new(1, 1, None).unwrap();
            let result = egarch_model.fit(data, None);
            assert!(result.is_ok(), "EGARCH fitting failed for dataset {}", i);
            
            let mut gjr_model = GJRGARCHModel::new(1, 1, None).unwrap();
            let result = gjr_model.fit(data, None);
            assert!(result.is_ok(), "GJR-GARCH fitting failed for dataset {}", i);
            
            let mut garch_m_model = GARCHMModel::new(1, 1, RiskPremiumType::Variance, None).unwrap();
            let result = garch_m_model.fit(data, None);
            assert!(result.is_ok(), "GARCH-M fitting failed for dataset {}", i);
        }
    }

    #[test]
    fn test_garch_forecasting_comprehensive() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        
        // Test different forecast horizons
        let horizons = vec![1, 5, 10, 20];
        
        let mut garch_model = GARCHModel::new(1, 1, None).unwrap();
        garch_model.fit(&data, None).unwrap();
        
        for horizon in horizons {
            let forecast = garch_model.forecast_variance(horizon);
            assert!(forecast.is_ok());
            
            let forecast_values = forecast.unwrap();
            assert_eq!(forecast_values.len(), horizon);
            assert!(forecast_values.iter().all(|&v| v > 0.0)); // All forecasts should be positive
        }
        
        // Test EGARCH forecasting
        let mut egarch_model = EGARCHModel::new(1, 1, None).unwrap();
        egarch_model.fit(&data, None).unwrap();
        
        let forecast = egarch_model.forecast_variance(10);
        assert!(forecast.is_ok());
        assert_eq!(forecast.unwrap().len(), 10);
        
        // Test GJR-GARCH forecasting
        let mut gjr_model = GJRGARCHModel::new(1, 1, None).unwrap();
        gjr_model.fit(&data, None).unwrap();
        
        let forecast = gjr_model.forecast_variance(10);
        assert!(forecast.is_ok());
        assert_eq!(forecast.unwrap().len(), 10);
        
        // Test GARCH-M forecasting
        let mut garch_m_model = GARCHMModel::new(1, 1, RiskPremiumType::Variance, None).unwrap();
        garch_m_model.fit(&data, None).unwrap();
        
        let forecast = garch_m_model.forecast(10);
        assert!(forecast.is_ok());
        
        let (mean_forecast, var_forecast) = forecast.unwrap();
        assert_eq!(mean_forecast.len(), 10);
        assert_eq!(var_forecast.len(), 10);
    }

    #[test]
    fn test_model_parameter_access() {
        // Test parameter access for all models
        let garch_model = GARCHModel::new(2, 2, None).unwrap();
        assert_eq!(garch_model.order(), (2, 2));
        assert_eq!(garch_model.alpha.len(), 2);
        assert_eq!(garch_model.beta.len(), 2);
        
        let egarch_model = EGARCHModel::new(1, 1, None).unwrap();
        assert_eq!(egarch_model.alpha.len(), 1);
        assert_eq!(egarch_model.gamma.len(), 1);
        assert_eq!(egarch_model.beta.len(), 1);
        
        let gjr_model = GJRGARCHModel::new(1, 1, None).unwrap();
        assert_eq!(gjr_model.alpha.len(), 1);
        assert_eq!(gjr_model.gamma.len(), 1);
        assert_eq!(gjr_model.beta.len(), 1);
        
        let garch_m_model = GARCHMModel::new(1, 1, RiskPremiumType::Variance, None).unwrap();
        assert_eq!(garch_m_model.alpha.len(), 1);
        assert_eq!(garch_m_model.beta.len(), 1);
        assert!(garch_m_model.lambda.is_finite());
    }

    #[test]
    fn test_news_impact_curves() {
        // Test news impact curves for asymmetric models
        let gjr_model = GJRGARCHModel::new(1, 1, Some(vec![0.0, 0.01, 0.03, 0.05, 0.85])).unwrap();
        let (shocks, impacts) = gjr_model.news_impact_curve(21, 3.0);
        
        assert_eq!(shocks.len(), 21);
        assert_eq!(impacts.len(), 21);
        
        // Check that we have both positive and negative shocks
        assert!(shocks.iter().any(|&x| x > 0.0));
        assert!(shocks.iter().any(|&x| x < 0.0));
        
        // Check asymmetric effect: negative shocks should have higher impact
        let mid_idx = shocks.len() / 2;
        let negative_idx = 0; // First element should be most negative
        let positive_idx = shocks.len() - 1; // Last element should be most positive
        
        if shocks[negative_idx] < 0.0 && shocks[positive_idx] > 0.0 {
            // GJR effect: negative shock has additional gamma effect
            assert!(impacts[negative_idx] >= impacts[positive_idx]);
        }
        
        // Test EGARCH news impact curve
        let egarch_model = EGARCHModel::new(1, 1, Some(vec![0.0, -0.1, 0.1, -0.05, 0.9])).unwrap();
        let (e_shocks, e_impacts) = egarch_model.news_impact_curve(21, 3.0);
        
        assert_eq!(e_shocks.len(), 21);
        assert_eq!(e_impacts.len(), 21);
        assert!(e_impacts.iter().all(|&v| v > 0.0)); // All variance impacts should be positive
    }

    #[test]
    fn test_error_handling_comprehensive() {
        // Test various error conditions
        
        // 1. Invalid model orders
        assert!(GARCHModel::new(0, 1, None).is_err());
        assert!(GARCHModel::new(1, 0, None).is_err());
        assert!(EGARCHModel::new(0, 1, None).is_err());
        assert!(GJRGARCHModel::new(0, 1, None).is_err());
        assert!(GARCHMModel::new(0, 1, RiskPremiumType::Variance, None).is_err());
        
        // 2. Insufficient data
        let mut garch_model = GARCHModel::new(1, 1, None).unwrap();
        let short_data = vec![1.0, 2.0]; // Too little data
        assert!(garch_model.fit(&short_data, None).is_err());
        
        // 3. Empty data
        let empty_data: Vec<f64> = vec![];
        assert!(garch_model.fit(&empty_data, None).is_err());
        
        // 4. Forecasting without fitting
        let unfitted_model = GARCHModel::new(1, 1, None).unwrap();
        assert!(unfitted_model.forecast_variance(5).is_err());
    }

    #[test]
    fn test_model_convergence_properties() {
        // Test that models converge to reasonable parameters
        let n = 200;
        let mut returns: Vec<f64> = Vec::with_capacity(n);
        let mut true_variance: f64 = 0.01;
        
        // Generate data with known GARCH properties
        for i in 0..n {
            if i > 0 {
                true_variance = 0.001 + 0.05 * returns[i-1].powi(2) + 0.9 * true_variance;
            }
            returns.push((i as f64 * 0.01).sin() * true_variance.sqrt());
        }
        
        let mut model = GARCHModel::new(1, 1, None).unwrap();
        let result = model.fit(&returns, None);
        assert!(result.is_ok());
        
        // Check that fitted parameters are reasonable
        assert!(model.omega > 0.0);
        assert!(model.alpha[0] >= 0.0);
        assert!(model.beta[0] >= 0.0);
        assert!(model.alpha[0] + model.beta[0] < 1.0); // Stationarity condition
        
        // Check that residuals and fitted values are available
        assert!(model.residuals.is_some());
        assert!(model.fitted_variance.is_some());
        
        let fitted_var = model.fitted_variance.as_ref().unwrap();
        assert_eq!(fitted_var.len(), returns.len());
        assert!(fitted_var.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_model_display_functionality() {
        // Test that all models implement Display trait properly
        let garch_model = GARCHModel::new(1, 1, None).unwrap();
        let garch_display = format!("{}", garch_model);
        assert!(garch_display.contains("GARCH"));
        assert!(garch_display.contains("(1,1)"));
        
        let egarch_model = EGARCHModel::new(1, 1, None).unwrap();
        let egarch_display = format!("{}", egarch_model);
        assert!(egarch_display.contains("EGARCH"));
        
        let gjr_model = GJRGARCHModel::new(1, 1, None).unwrap();
        let gjr_display = format!("{}", gjr_model);
        assert!(gjr_display.contains("GJR-GARCH"));
        
        let garch_m_model = GARCHMModel::new(1, 1, RiskPremiumType::Variance, None).unwrap();
        let garch_m_display = format!("{}", garch_m_model);
        assert!(garch_m_display.contains("GARCH-M"));
        assert!(garch_m_display.contains("Variance"));
    }

    #[test]
    fn test_timestamp_handling() {
        // Test models with and without timestamps
        let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..50)
            .map(|i| {
                Utc.timestamp_opt(now.timestamp() + (i as i64) * 86400, 0)
                    .unwrap()
            })
            .collect();
        
        // Test GARCH with timestamps
        let mut garch_model = GARCHModel::new(1, 1, None).unwrap();
        let result = garch_model.fit(&data, Some(&timestamps));
        assert!(result.is_ok());
        
        // Test EGARCH with timestamps
        let mut egarch_model = EGARCHModel::new(1, 1, None).unwrap();
        let result = egarch_model.fit(&data, Some(&timestamps));
        assert!(result.is_ok());
        assert!(egarch_model.timestamps.is_some());
        assert_eq!(egarch_model.timestamps.as_ref().unwrap().len(), 50);
        
        // Test without timestamps
        let mut no_ts_model = GARCHModel::new(1, 1, None).unwrap();
        let result = no_ts_model.fit(&data, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_different_risk_premium_types() {
        let data: Vec<f64> = (0..50).map(|i| 0.001 + (i as f64 * 0.1).sin() * 0.01).collect();
        
        // Test Variance risk premium
        let mut var_model = GARCHMModel::new(1, 1, RiskPremiumType::Variance, None).unwrap();
        let result = var_model.fit(&data, None);
        assert!(result.is_ok());
        
        let forecast = var_model.forecast(5);
        assert!(forecast.is_ok());
        let (mean_forecast, _) = forecast.unwrap();
        assert_eq!(mean_forecast.len(), 5);
        
        // Test StdDev risk premium
        let mut vol_model = GARCHMModel::new(1, 1, RiskPremiumType::StdDev, None).unwrap();
        let result = vol_model.fit(&data, None);
        assert!(result.is_ok());
        
        let forecast = vol_model.forecast(5);
        assert!(forecast.is_ok());
        let (mean_forecast, _) = forecast.unwrap();
        assert_eq!(mean_forecast.len(), 5);
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_garch_error_types() {
        use crate::models::GARCHError;
        
        // Test InvalidParameters error
        let invalid_params_error = GARCHError::InvalidParameters("test message".to_string());
        assert!(format!("{}", invalid_params_error).contains("test message"));
        
        // Test InvalidData error
        let invalid_data_error = GARCHError::InvalidData("data message".to_string());
        assert!(format!("{}", invalid_data_error).contains("data message"));
    }

    #[test]
    fn test_error_propagation() {
        // Test that errors propagate correctly through the Result type
        let result: Result<()> = Err(OxiError::from(GARCHError::InvalidParameters("test".to_string());
        assert!(result.is_err());
        
        match result {
            Err(OxiError::from(GARCHError::InvalidParameters(msg)) => assert_eq!(msg, "test"),
            _ => panic!("Expected InvalidParameters error"),
        }
    }
}
