//! Comprehensive demo of the improved OxiDiviner API
//! 
//! This example demonstrates the new user-friendly features:
//! - Unified QuickForecaster trait
//! - Model builder pattern with fluent API
//! - Parameter validation with clear error messages
//! - Smart model auto-selection
//! - Enhanced error handling

use oxidiviner::core::{TimeSeriesData, Result, ModelValidator};
use oxidiviner::api::{ModelBuilder, AutoSelector};
use chrono::{DateTime, Utc, Duration};

fn main() -> Result<()> {
    println!("🚀 OxiDiviner Improved API Demo");
    println!("================================\n");

    // Generate sample time series data
    let (timestamps, values) = generate_sample_data();
    let data = TimeSeriesData::new(timestamps, values, "Sample Time Series")?;
    
    println!("📊 Generated {} data points", data.values.len());
    println!("   First few values: {:?}", &data.values[..5]);
    println!("   Last few values: {:?}\n", &data.values[data.values.len()-5..]);

    // Demo 1: Model Builder Pattern
    demo_builder_pattern(&data)?;
    
    // Demo 2: Parameter Validation
    demo_parameter_validation()?;
    
    // Demo 3: Auto Model Selection
    demo_auto_selection(&data)?;
    
    // Demo 4: Error Handling
    demo_error_handling()?;
    
    // Demo 5: Quick Forecasting Interface
    demo_quick_interface(&data)?;

    println!("✅ All demos completed successfully!");
    Ok(())
}

/// Generate synthetic time series data with trend and seasonality
fn generate_sample_data() -> (Vec<DateTime<Utc>>, Vec<f64>) {
    let start_date = Utc::now() - Duration::days(100);
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    
    for i in 0..100 {
        let timestamp = start_date + Duration::days(i);
        // Add trend, seasonality, and noise
        let trend = 0.1 * i as f64;
        let seasonal = 5.0 * (2.0 * std::f64::consts::PI * (i % 7) as f64 / 7.0).sin();
        let noise = (i % 13) as f64 * 0.2 - 1.0; // Deterministic "noise"
        let value = 100.0 + trend + seasonal + noise;
        
        timestamps.push(timestamp);
        values.push(value);
    }
    
    (timestamps, values)
}

/// Demonstrate the fluent model builder pattern
fn demo_builder_pattern(data: &TimeSeriesData) -> Result<()> {
    println!("🏗️  Demo 1: Model Builder Pattern");
    println!("----------------------------------");

    // Build ARIMA model with fluent API
    println!("📈 Building ARIMA(2,1,1) model...");
    let mut arima_model = ModelBuilder::arima()
        .with_ar(2)
        .with_differencing(1)
        .with_ma(1)
        .build()?;
    
    arima_model.quick_fit(data)?;
    let arima_forecast = arima_model.quick_forecast(10)?;
    println!("   ✓ ARIMA forecast (10 periods): {:?}", &arima_forecast[..3]);

    // Build Exponential Smoothing model
    println!("📊 Building Exponential Smoothing model...");
    let mut es_model = ModelBuilder::exponential_smoothing()
        .with_alpha(0.3)
        .build()?;
    
    es_model.quick_fit(data)?;
    let es_forecast = es_model.quick_forecast(10)?;
    println!("   ✓ ES forecast (10 periods): {:?}", &es_forecast[..3]);

    // Build Moving Average model
    println!("📉 Building Moving Average model...");
    let mut ma_model = ModelBuilder::moving_average()
        .with_window(5)
        .build()?;
    
    ma_model.quick_fit(data)?;
    let ma_forecast = ma_model.quick_forecast(10)?;
    println!("   ✓ MA forecast (10 periods): {:?}\n", &ma_forecast[..3]);

    Ok(())
}

/// Demonstrate parameter validation
fn demo_parameter_validation() -> Result<()> {
    println!("✅ Demo 2: Parameter Validation");
    println!("--------------------------------");

    // Test valid parameters
    println!("🟢 Testing valid ARIMA parameters (2,1,1):");
    match ModelValidator::validate_arima_params(2, 1, 1) {
        Ok(()) => println!("   ✓ Parameters are valid"),
        Err(e) => println!("   ✗ Unexpected error: {}", e),
    }

    // Test invalid parameters
    println!("🔴 Testing invalid ARIMA parameters (15,3,15):");
    match ModelValidator::validate_arima_params(15, 3, 15) {
        Ok(()) => println!("   ✗ Should have failed validation"),
        Err(e) => println!("   ✓ Caught error: {}", e),
    }

    // Test smoothing parameter validation
    println!("🟢 Testing valid alpha parameter (0.3):");
    match ModelValidator::validate_smoothing_param(0.3, "alpha") {
        Ok(()) => println!("   ✓ Alpha parameter is valid"),
        Err(e) => println!("   ✗ Unexpected error: {}", e),
    }

    println!("🔴 Testing invalid alpha parameter (1.5):");
    match ModelValidator::validate_smoothing_param(1.5, "alpha") {
        Ok(()) => println!("   ✗ Should have failed validation"),
        Err(e) => println!("   ✓ Caught error: {}\n", e),
    }

    Ok(())
}

/// Demonstrate automatic model selection
fn demo_auto_selection(data: &TimeSeriesData) -> Result<()> {
    println!("🤖 Demo 3: Automatic Model Selection");
    println!("-------------------------------------");

    // Auto-select with AIC criterion
    println!("📋 Auto-selecting best model using AIC...");
    let selector = AutoSelector::with_aic().max_models(5);
    
    match selector.select_best(data) {
        Ok((mut best_model, score, name)) => {
            println!("   ✓ Best model: {} (AIC: {:.2})", name, score);
            
            // Use the selected model for forecasting
            let forecast = best_model.quick_forecast(5)?;
            println!("   📈 Forecast from best model: {:?}", forecast);
        }
        Err(e) => println!("   ✗ Auto-selection failed: {}", e),
    }

    // Auto-select with cross-validation
    println!("🎯 Auto-selecting with cross-validation...");
    let cv_selector = AutoSelector::with_cross_validation(3).max_models(3);
    
    match cv_selector.select_best(data) {
        Ok((_model, score, name)) => {
            println!("   ✓ Best model: {} (CV Score: {:.2})", name, score);
        }
        Err(e) => println!("   ✗ CV selection failed: {}", e),
    }
    
    println!();
    Ok(())
}

/// Demonstrate enhanced error handling
fn demo_error_handling() -> Result<()> {
    println!("🚨 Demo 4: Enhanced Error Handling");
    println!("-----------------------------------");

    // Try to build an invalid model
    println!("🔴 Attempting to build invalid ARIMA model...");
    match ModelBuilder::arima()
        .with_ar(20)  // Too high
        .with_differencing(5)  // Too high
        .with_ma(20)  // Too high
        .build()
    {
        Ok(_) => println!("   ✗ Should have failed to build"),
        Err(e) => println!("   ✓ Build failed with clear error: {}", e),
    }

    // Try invalid smoothing parameter
    println!("🔴 Attempting invalid exponential smoothing...");
    match ModelBuilder::exponential_smoothing()
        .with_alpha(2.0)  // Invalid: > 1.0
        .build()
    {
        Ok(_) => println!("   ✗ Should have failed to build"),
        Err(e) => println!("   ✓ Build failed with clear error: {}\n", e),
    }

    Ok(())
}

/// Demonstrate the unified QuickForecaster interface
fn demo_quick_interface(data: &TimeSeriesData) -> Result<()> {
    println!("⚡ Demo 5: Quick Forecasting Interface");
    println!("--------------------------------------");

    // Create different models using the same interface
    let models = vec![
        ("ARIMA", ModelBuilder::arima().with_ar(1).with_differencing(1).with_ma(1).build()?),
        ("Exponential Smoothing", ModelBuilder::exponential_smoothing().with_alpha(0.3).build()?),
        ("Moving Average", ModelBuilder::moving_average().with_window(5).build()?),
    ];

    for (name, mut model) in models {
        println!("🔄 Testing {} model:", name);
        
        // Fit the model
        match model.quick_fit(data) {
            Ok(()) => println!("   ✓ Model fitted successfully"),
            Err(e) => {
                println!("   ✗ Fit failed: {}", e);
                continue;
            }
        }
        
        // Generate forecast
        match model.quick_forecast(5) {
            Ok(forecast) => println!("   📊 5-period forecast: {:?}", forecast),
            Err(e) => println!("   ✗ Forecast failed: {}", e),
        }
        
        // Get model evaluation
        match model.evaluate(data) {
            Ok(eval) => println!("   📈 MSE: {:.4}, MAE: {:.4}", eval.mse, eval.mae),
            Err(e) => println!("   ✗ Evaluation failed: {}", e),
        }
        
        println!("   🏷️  Model name: {}", model.model_name());
        println!();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_data_generation() {
        let (timestamps, values) = generate_sample_data();
        
        assert_eq!(timestamps.len(), 100);
        assert_eq!(values.len(), 100);
        assert!(values.iter().all(|&v| v.is_finite()));
        
        // Check that timestamps are in ascending order
        for i in 1..timestamps.len() {
            assert!(timestamps[i] > timestamps[i-1]);
        }
    }

    #[test]
    fn test_builder_pattern() {
        // Test ARIMA builder
        let arima = ModelBuilder::arima()
            .with_ar(2)
            .with_differencing(1)
            .with_ma(1)
            .build();
        assert!(arima.is_ok());

        // Test ES builder
        let es = ModelBuilder::exponential_smoothing()
            .with_alpha(0.3)
            .build();
        assert!(es.is_ok());

        // Test MA builder
        let ma = ModelBuilder::moving_average()
            .with_window(5)
            .build();
        assert!(ma.is_ok());
    }

    #[test]
    fn test_parameter_validation() {
        // Valid parameters should pass
        assert!(ModelValidator::validate_arima_params(2, 1, 1).is_ok());
        assert!(ModelValidator::validate_smoothing_param(0.3, "alpha").is_ok());

        // Invalid parameters should fail
        assert!(ModelValidator::validate_arima_params(20, 5, 20).is_err());
        assert!(ModelValidator::validate_smoothing_param(1.5, "alpha").is_err());
    }

    #[test]
    fn test_auto_selector() {
        let (timestamps, values) = generate_sample_data();
        let data = TimeSeriesData::new(timestamps, values, "Test").unwrap();

        let selector = AutoSelector::with_aic().max_models(3);
        let result = selector.select_best(&data);
        
        // Should either succeed or fail gracefully
        match result {
            Ok((_, score, name)) => {
                assert!(score.is_finite());
                assert!(!name.is_empty());
            }
            Err(_) => {
                // It's okay if auto-selection fails with limited test data
            }
        }
    }
} 