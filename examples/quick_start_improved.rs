/*!
# OxiDiviner Quick Start - Improved API Demo

This example demonstrates the new accessibility improvements to OxiDiviner,
showcasing the unified API, builder pattern, validation utilities, and smart
model selection features.

Run with:
```bash
cargo run --example quick_start_improved
```
*/

use chrono::{Duration, Utc};
use oxidiviner::prelude::*;
use oxidiviner::{quick, ModelBuilder, AutoSelector};
use oxidiviner_core::{validation::ValidationUtils, ModelValidator};
use rand::Rng;

fn main() -> Result<()> {
    println!("🚀 OxiDiviner - Improved API Demo");
    println!("==================================\n");

    // 1. Create sample data with realistic patterns
    let data = create_realistic_sample_data(50)?;
    println!("📊 Created time series with {} data points", data.values.len());

    // 2. Demonstrate Quick API
    println!("\n🔥 Quick API Demonstrations:");
    demonstrate_quick_api(&data)?;

    // 3. Demonstrate Builder Pattern
    println!("\n🏗️  Builder Pattern Demonstrations:");
    demonstrate_builder_pattern(&data)?;

    // 4. Demonstrate Validation Utilities
    println!("\n✅ Validation Utilities:");
    demonstrate_validation_utilities(&data)?;

    // 5. Demonstrate Smart Model Selection
    println!("\n🧠 Smart Model Selection:");
    demonstrate_smart_selection(&data)?;

    // 6. Demonstrate Error Handling
    println!("\n❌ Error Handling Examples:");
    demonstrate_error_handling()?;

    println!("\n✨ Demo completed successfully!");
    Ok(())
}

/// Create realistic sample time series data with trend, seasonality, and noise
fn create_realistic_sample_data(n: usize) -> Result<TimeSeriesData> {
    let mut timestamps = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);

    let base_time = Utc::now() - Duration::days(n as i64);
    let mut rng = rand::rng();

    for i in 0..n {
        let timestamp = base_time + Duration::days(i as i64);

        // Create realistic pattern: trend + seasonality + noise
        let t = i as f64;
        let trend = 100.0 + 0.5 * t; // Slight upward trend
        let seasonal = 10.0 * (2.0 * std::f64::consts::PI * t / 7.0).sin(); // Weekly seasonality
        let noise = rng.random_range(-3.0..3.0); // Random noise

        let value = trend + seasonal + noise;

        timestamps.push(timestamp);
        values.push(value);
    }

    TimeSeriesData::new(timestamps, values, "realistic_sample")
}

/// Demonstrate the Quick API with simple one-line forecasting
fn demonstrate_quick_api(data: &TimeSeriesData) -> Result<()> {
    let forecast_horizon = 5;

    // Quick ARIMA forecasting
    println!("  📈 Quick ARIMA forecast:");
    let arima_forecast = quick::arima(data.clone(), forecast_horizon)?;
    println!("     ARIMA(1,1,1): {:?}", 
        arima_forecast.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());

    // Quick AR forecasting
    println!("  📊 Quick AR forecast:");
    let ar_forecast = quick::ar(data.clone(), forecast_horizon, Some(3))?;
    println!("     AR(3): {:?}", 
        ar_forecast.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());

    // Quick Moving Average
    println!("  📉 Quick Moving Average forecast:");
    let ma_forecast = quick::moving_average(data.clone(), forecast_horizon, Some(7))?;
    println!("     MA(7): {:?}", 
        ma_forecast.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());

    // Quick Exponential Smoothing
    println!("  📋 Quick Exponential Smoothing forecast:");
    let es_forecast = quick::exponential_smoothing(data.clone(), forecast_horizon, Some(0.3))?;
    println!("     ES(α=0.3): {:?}", 
        es_forecast.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());

    Ok(())
}

/// Demonstrate the builder pattern for model configuration
fn demonstrate_builder_pattern(data: &TimeSeriesData) -> Result<()> {
    // Build an ARIMA model with fluent interface
    println!("  🏗️  Building ARIMA(2,1,2) model:");
    let arima_config = ModelBuilder::arima()
        .with_ar(2)
        .with_differencing(1)
        .with_ma(2)
        .build_config();
    
    let arima_forecast = quick::forecast_with_config(data.clone(), 5, arima_config)?;
    println!("     Forecast: {:?}", 
        arima_forecast.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());

    // Build a custom exponential smoothing model
    println!("  🏗️  Building ES model with custom alpha:");
    let es_config = ModelBuilder::exponential_smoothing()
        .with_alpha(0.7)
        .with_parameter("custom_param", 42.0)
        .build_config();
    
    println!("     Model type: {}", es_config.model_type);
    println!("     Parameters: {:?}", es_config.parameters);

    // Build multiple model configurations
    println!("  🏗️  Building multiple model configurations:");
    let configs = vec![
        ModelBuilder::ar().with_ar(1).build_config(),
        ModelBuilder::ar().with_ar(2).build_config(),
        ModelBuilder::ar().with_ar(3).build_config(),
    ];

    for (i, config) in configs.iter().enumerate() {
        let forecast = quick::forecast_with_config(data.clone(), 3, config.clone())?;
        println!("     AR({}) forecast: {:?}", 
            i + 1, 
            forecast.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());
    }

    Ok(())
}

/// Demonstrate validation utilities for data splitting and accuracy metrics
fn demonstrate_validation_utilities(data: &TimeSeriesData) -> Result<()> {
    // Time series splitting
    println!("  📊 Time series data splitting:");
    let (train, test) = ValidationUtils::time_split(data, 0.8)?;
    println!("     Original: {} points, Train: {} points, Test: {} points", 
        data.values.len(), train.values.len(), test.values.len());

    // Cross-validation splits
    println!("  🔄 Time series cross-validation:");
    let cv_splits = ValidationUtils::time_series_cv(data, 3, Some(20))?;
    println!("     Created {} CV splits", cv_splits.len());
    for (i, (train_split, test_split)) in cv_splits.iter().enumerate() {
        println!("       Split {}: Train {} points, Test {} points", 
            i + 1, train_split.values.len(), test_split.values.len());
    }

    // Accuracy metrics demonstration
    println!("  📏 Accuracy metrics calculation:");
    
    // Generate a simple forecast for testing
    let simple_forecast = quick::moving_average(train.clone(), test.values.len(), Some(5))?;
    
    if simple_forecast.len() == test.values.len() {
        let metrics = ValidationUtils::accuracy_metrics(
            &test.values, 
            &simple_forecast, 
            None
        )?;
        
        println!("     MAE:  {:.4}", metrics.mae);
        println!("     RMSE: {:.4}", metrics.rmse);
        println!("     MAPE: {:.2}%", metrics.mape);
        println!("     R²:   {:.4}", metrics.r_squared);
        println!("     N:    {}", metrics.n_observations);
    }

    Ok(())
}

/// Demonstrate smart model selection with automatic best model detection
fn demonstrate_smart_selection(data: &TimeSeriesData) -> Result<()> {
    println!("  🧠 Automatic model selection:");
    
    let (best_forecast, best_model) = quick::auto_select(data.clone(), 7)?;
    
    println!("     Best model: {}", best_model);
    println!("     Forecast (7 periods): {:?}", 
        best_forecast.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());

    // Using AutoSelector with different criteria
    println!("  🔍 AutoSelector with custom criteria:");
    let selector = AutoSelector::with_aic()
        .add_candidate(ModelBuilder::ar().with_ar(4).build_config())
        .add_candidate(ModelBuilder::moving_average().with_window(10).build_config());
    
    println!("     Selection criteria: {:?}", selector.criteria());
    println!("     Number of candidates: {}", selector.candidates().len());

    Ok(())
}

/// Demonstrate comprehensive error handling and parameter validation
fn demonstrate_error_handling() -> Result<()> {
    println!("  ❌ Parameter validation examples:");

    // Valid parameter validation
    println!("  ✅ Valid ARIMA parameters (2,1,1):");
    match ModelValidator::validate_arima_params(2, 1, 1) {
        Ok(()) => println!("     Parameters are valid"),
        Err(e) => println!("     Error: {}", e),
    }

    // Invalid parameter examples
    println!("  ❌ Invalid ARIMA parameters (20,5,15):");
    match ModelValidator::validate_arima_params(20, 5, 15) {
        Ok(()) => println!("     Parameters are valid"),
        Err(e) => println!("     Error: {}", e),
    }

    println!("  ❌ Invalid exponential smoothing alpha (1.5):");
    match ModelValidator::validate_exponential_smoothing_params(1.5, None, None) {
        Ok(()) => println!("     Parameters are valid"),
        Err(e) => println!("     Error: {}", e),
    }

    println!("  ❌ Invalid forecast horizon:");
    match ModelValidator::validate_forecast_horizon(100, 50) {
        Ok(()) => println!("     Horizon is valid"),
        Err(e) => println!("     Error: {}", e),
    }

    // Try to forecast with insufficient data
    println!("  ❌ Insufficient data example:");
    let small_data = TimeSeriesData::new(
        vec![Utc::now()],
        vec![1.0],
        "insufficient"
    )?;

    match quick::arima(small_data, 5) {
        Ok(_) => println!("     Unexpectedly succeeded"),
        Err(e) => println!("     Expected error: {}", e),
    }

    Ok(())
} 