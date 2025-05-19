use oxidiviner_garch::{EGARCHModel, GARCHMModel, GJRGARCHModel, RiskPremiumType, Result};
use rand::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

// Create synthetic EGARCH(1,1) data with asymmetric effects
fn generate_egarch_series(n: usize, mean: f64, omega: f64, alpha: f64, gamma: f64, beta: f64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    
    let mut y = Vec::with_capacity(n);
    let mut log_variance = omega / (1.0 - beta); // Start at unconditional log-variance
    let mut variance = log_variance.exp();
    
    for _ in 0..n {
        let z = normal.sample(&mut rng);
        let epsilon = z * variance.sqrt();
        let y_t = mean + epsilon;
        
        y.push(y_t);
        
        // Update log-variance for next period
        log_variance = omega + alpha * (z.abs() - (2.0/std::f64::consts::PI).sqrt()) + gamma * z + beta * log_variance;
        variance = log_variance.exp();
    }
    
    y
}

// Create synthetic GJR-GARCH(1,1) data with asymmetric effects
fn generate_gjr_garch_series(n: usize, mean: f64, omega: f64, alpha: f64, gamma: f64, beta: f64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    
    let mut y = Vec::with_capacity(n);
    // Long-run average variance considering asymmetry effect (gamma/2 assuming 50% negative shocks)
    let mut sigma_squared = omega / (1.0 - alpha - gamma/2.0 - beta);
    
    for _ in 0..n {
        let z = normal.sample(&mut rng);
        let epsilon = z * sigma_squared.sqrt();
        let y_t = mean + epsilon;
        
        y.push(y_t);
        
        // Update variance for next period with asymmetric effect
        let indicator = if z < 0.0 { 1.0 } else { 0.0 };
        sigma_squared = omega + alpha * epsilon.powi(2) + gamma * epsilon.powi(2) * indicator + beta * sigma_squared;
    }
    
    y
}

// Create synthetic GARCH-M(1,1) data with risk premium
fn generate_garch_m_series(n: usize, mean: f64, lambda: f64, omega: f64, alpha: f64, beta: f64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    
    let mut y = Vec::with_capacity(n);
    let mut sigma_squared = omega / (1.0 - alpha - beta); // Start at unconditional variance
    
    for _ in 0..n {
        // Mean depends on volatility (risk premium)
        let mean_t = mean + lambda * sigma_squared.sqrt();
        
        let z = normal.sample(&mut rng);
        let epsilon = z * sigma_squared.sqrt();
        let y_t = mean_t + epsilon;
        
        y.push(y_t);
        
        // Update variance for next period
        sigma_squared = omega + alpha * epsilon.powi(2) + beta * sigma_squared;
    }
    
    y
}

#[test]
fn test_egarch_model() -> Result<()> {
    // Generate synthetic EGARCH(1,1) data
    let data = generate_egarch_series(500, 0.0, -0.1, 0.15, -0.08, 0.85);
    
    // Create and fit EGARCH model
    let mut model = EGARCHModel::new(1, 1, None)?;
    model.fit(&data, None)?;
    
    // Check that all required fields are populated
    assert!(model.fitted_variance.is_some());
    assert!(model.residuals.is_some());
    assert!(model.std_residuals.is_some());
    assert!(model.log_likelihood.is_some());
    assert!(model.info_criteria.is_some());
    
    // Test news impact curve
    let (shocks, variances) = model.news_impact_curve(100, 3.0);
    assert_eq!(shocks.len(), 100);
    assert_eq!(variances.len(), 100);
    
    // Verify the asymmetry effect - negative shocks should produce higher volatility
    let neg_idx = 25; // Index for a negative shock
    let pos_idx = 75; // Index for a positive shock of the same absolute value
    
    // Should have roughly the same absolute value but opposite signs
    assert!((shocks[neg_idx].abs() - shocks[pos_idx].abs()).abs() < 0.1);
    assert!(shocks[neg_idx] < 0.0);
    assert!(shocks[pos_idx] > 0.0);
    
    // With negative gamma, negative shocks should typically produce higher volatility
    if model.gamma[0] < 0.0 {
        assert!(variances[neg_idx] > variances[pos_idx]);
    }
    
    // Test forecasting
    let forecast = model.forecast_variance(10)?;
    assert_eq!(forecast.len(), 10);
    
    // All forecasted values should be positive
    for &v in &forecast {
        assert!(v > 0.0);
    }
    
    Ok(())
}

#[test]
fn test_gjr_garch_model() -> Result<()> {
    // Generate synthetic GJR-GARCH(1,1) data
    let data = generate_gjr_garch_series(500, 0.0, 0.01, 0.05, 0.10, 0.80);
    
    // Create and fit GJR-GARCH model
    let mut model = GJRGARCHModel::new(1, 1, None)?;
    model.fit(&data, None)?;
    
    // Check that all required fields are populated
    assert!(model.fitted_variance.is_some());
    assert!(model.residuals.is_some());
    assert!(model.log_likelihood.is_some());
    assert!(model.info_criteria.is_some());
    
    // Test news impact curve
    let (shocks, variances) = model.news_impact_curve(100, 3.0);
    assert_eq!(shocks.len(), 100);
    assert_eq!(variances.len(), 100);
    
    // Verify the asymmetry effect - negative shocks should produce higher volatility
    let neg_idx = 25; // Index for a negative shock
    let pos_idx = 75; // Index for a positive shock of the same absolute value
    
    // Should have roughly the same absolute value but opposite signs
    assert!((shocks[neg_idx].abs() - shocks[pos_idx].abs()).abs() < 0.1);
    assert!(shocks[neg_idx] < 0.0);
    assert!(shocks[pos_idx] > 0.0);
    
    // With positive gamma, negative shocks should typically produce higher volatility
    if model.gamma[0] > 0.0 {
        assert!(variances[neg_idx] > variances[pos_idx]);
    }
    
    // Test forecasting
    let forecast = model.forecast_variance(10)?;
    assert_eq!(forecast.len(), 10);
    
    // All forecasted values should be positive
    for &v in &forecast {
        assert!(v > 0.0);
    }
    
    Ok(())
}

#[test]
fn test_garch_m_model() -> Result<()> {
    // Generate synthetic GARCH-M(1,1) data
    let data = generate_garch_m_series(500, 0.0, 0.15, 0.01, 0.05, 0.90);
    
    // Create and fit GARCH-M model
    let mut model = GARCHMModel::new(1, 1, RiskPremiumType::StdDev, None)?;
    model.fit(&data, None)?;
    
    // Check that all required fields are populated
    assert!(model.fitted_variance.is_some());
    assert!(model.fitted_mean.is_some());
    assert!(model.residuals.is_some());
    assert!(model.log_likelihood.is_some());
    assert!(model.info_criteria.is_some());
    
    // Verify that lambda (risk premium) is non-zero
    assert!(model.lambda != 0.0);
    
    // Test forecasting
    let (forecast_mean, forecast_var) = model.forecast(10)?;
    assert_eq!(forecast_mean.len(), 10);
    assert_eq!(forecast_var.len(), 10);
    
    // All forecasted variance values should be positive
    for &v in &forecast_var {
        assert!(v > 0.0);
    }
    
    // For positive lambda, higher variance should lead to higher forecast mean
    if model.lambda > 0.0 {
        for i in 0..forecast_var.len()-1 {
            if forecast_var[i] < forecast_var[i+1] {
                assert!(forecast_mean[i] < forecast_mean[i+1]);
            }
        }
    }
    
    Ok(())
}

#[test]
fn test_different_risk_premium_types() -> Result<()> {
    // Test different risk premium types for GARCH-M
    let data = generate_garch_m_series(200, 0.0, 0.15, 0.01, 0.05, 0.90);
    
    // Create models with different risk premium types
    let mut model_variance = GARCHMModel::new(1, 1, RiskPremiumType::Variance, None)?;
    let mut model_stddev = GARCHMModel::new(1, 1, RiskPremiumType::StdDev, None)?;
    let mut model_logvar = GARCHMModel::new(1, 1, RiskPremiumType::LogVariance, None)?;
    
    // Fit all models
    model_variance.fit(&data, None)?;
    model_stddev.fit(&data, None)?;
    model_logvar.fit(&data, None)?;
    
    // Verify all models fit successfully
    assert!(model_variance.fitted_mean.is_some());
    assert!(model_stddev.fitted_mean.is_some());
    assert!(model_logvar.fitted_mean.is_some());
    
    // Compare log-likelihood to see which specification fits better
    // (this is for information only, not a strict test, as the best form
    // depends on the specific dataset)
    if let (Some(ll_var), Some(ll_std), Some(ll_log)) = (
        model_variance.log_likelihood,
        model_stddev.log_likelihood,
        model_logvar.log_likelihood
    ) {
        println!("Variance LL: {}, StdDev LL: {}, LogVar LL: {}", ll_var, ll_std, ll_log);
    }
    
    // Forecast with all models
    let (mean_var, _) = model_variance.forecast(10)?;
    let (mean_std, _) = model_stddev.forecast(10)?;
    let (mean_log, _) = model_logvar.forecast(10)?;
    
    // Different risk premium types should produce different forecasts
    assert!(mean_var != mean_std);
    assert!(mean_var != mean_log);
    assert!(mean_std != mean_log);
    
    Ok(())
} 