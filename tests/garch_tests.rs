use oxidiviner_garch::{GARCHModel, Result};
use rand::distributions::{Distribution, Normal};
use rand::rngs::StdRng;
use rand::SeedableRng;

// Create a synthetic GARCH(1,1) series
fn generate_garch_series(n: usize, mean: f64, omega: f64, alpha: f64, beta: f64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
    let normal = Normal::new(0.0, 1.0).unwrap();
    
    let mut y = Vec::with_capacity(n);
    let mut sigma_squared = omega / (1.0 - alpha - beta); // Start at unconditional variance
    
    for _ in 0..n {
        let z = normal.sample(&mut rng);
        let epsilon = z * sigma_squared.sqrt();
        let y_t = mean + epsilon;
        
        y.push(y_t);
        
        // Update variance for next period
        sigma_squared = omega + alpha * epsilon.powi(2) + beta * sigma_squared;
    }
    
    y
}

#[test]
fn test_garch_model_creation() -> Result<()> {
    // Test with valid parameters
    let model = GARCHModel::new(1, 1, Some(vec![0.0, 0.01, 0.3, 0.6]))?;
    assert_eq!(model.order(), (1, 1));
    
    // Test default parameters
    let model = GARCHModel::new(2, 1, None)?;
    assert_eq!(model.order(), (2, 1));
    
    Ok(())
}

#[test]
fn test_garch_parameter_validation() {
    // Test with invalid omega
    let invalid_model = GARCHModel::new(1, 1, Some(vec![0.0, -0.01, 0.3, 0.6]));
    assert!(invalid_model.is_err());
    
    // Test with invalid alpha
    let invalid_model = GARCHModel::new(1, 1, Some(vec![0.0, 0.01, -0.3, 0.6]));
    assert!(invalid_model.is_err());
    
    // Test with invalid beta
    let invalid_model = GARCHModel::new(1, 1, Some(vec![0.0, 0.01, 0.3, -0.6]));
    assert!(invalid_model.is_err());
    
    // Test with non-stationary parameters
    let invalid_model = GARCHModel::new(1, 1, Some(vec![0.0, 0.01, 0.5, 0.6]));
    assert!(invalid_model.is_err());
}

#[test]
fn test_garch_fit_and_forecast() -> Result<()> {
    // Generate synthetic GARCH(1,1) data
    let data = generate_garch_series(500, 0.0, 0.01, 0.2, 0.7);
    
    // Create and fit GARCH(1,1) model
    let mut model = GARCHModel::new(1, 1, None)?;
    model.fit(&data, None)?;
    
    // Check that all required fields are populated
    assert!(model.fitted_variance.is_some());
    assert!(model.residuals.is_some());
    assert!(model.log_likelihood.is_some());
    assert!(model.info_criteria.is_some());
    
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
fn test_garch_display() -> Result<()> {
    let model = GARCHModel::new(1, 1, Some(vec![0.0, 0.01, 0.3, 0.6]))?;
    let display_string = format!("{}", model);
    
    // Check that the display string contains key information
    assert!(display_string.contains("GARCH(1, 1) Model"));
    assert!(display_string.contains("Mean: 0."));
    assert!(display_string.contains("Omega: 0.01"));
    assert!(display_string.contains("Alpha[1]: 0.3"));
    assert!(display_string.contains("Beta[1]: 0.6"));
    
    Ok(())
} 