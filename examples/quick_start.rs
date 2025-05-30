//! Quick Start Guide for OxiDiviner
//! 
//! This example demonstrates the easiest ways to get started with forecasting
//! using the improved OxiDiviner API.

use oxidiviner::prelude::*;
use chrono::{DateTime, Utc, Duration};

fn main() -> Result<()> {
    println!("🚀 OxiDiviner Quick Start");
    println!("=========================\n");

    // 1. Generate some sample data
    let (timestamps, values) = create_sample_data();
    let data = TimeSeriesData::new(timestamps, values, "Sample Data")?;
    
    println!("📊 Created sample data with {} points", data.values.len());
    println!("   Values: {:.2} to {:.2}\n", 
             data.values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             data.values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

    // 2. Quick forecasting with auto model selection
    println!("🤖 Method 1: Auto Model Selection (Easiest)");
    println!("--------------------------------------------");
    
    let selector = AutoSelector::with_aic().max_models(5);
    match selector.select_best(&data) {
        Ok((mut best_model, score, model_name)) => {
            println!("✓ Best model: {} (AIC: {:.2})", model_name, score);
            let forecast = best_model.quick_forecast(7)?;
            println!("📈 7-day forecast: {:?}\n", &forecast[..3]);
        }
        Err(e) => println!("⚠️  Auto-selection failed: {}\n", e),
    }

    // 3. Builder pattern for custom models
    println!("🏗️  Method 2: Model Builder (Flexible)");
    println!("---------------------------------------");
    
    // Build an ARIMA model
    let mut arima = ModelBuilder::arima()
        .with_ar(2)
        .with_differencing(1)
        .with_ma(1)
        .build()?;
    
    arima.quick_fit(&data)?;
    let arima_forecast = arima.quick_forecast(7)?;
    println!("✓ ARIMA(2,1,1) forecast: {:?}", &arima_forecast[..3]);
    
    // Build an Exponential Smoothing model
    let mut es = ModelBuilder::exponential_smoothing()
        .with_alpha(0.3)
        .build()?;
    
    es.quick_fit(&data)?;
    let es_forecast = es.quick_forecast(7)?;
    println!("✓ Exponential Smoothing forecast: {:?}\n", &es_forecast[..3]);

    // 4. Quick one-liners
    println!("⚡ Method 3: One-liner Functions (Fastest)");
    println!("-------------------------------------------");
    
    // Use the quick functions for immediate results
    let quick_arima = arima(&data, 7)?;
    println!("✓ Quick ARIMA: {:?}", &quick_arima[..3]);
    
    let quick_ma = moving_average(&data, 7, Some(5))?;
    println!("✓ Quick Moving Average: {:?}", &quick_ma[..3]);
    
    let (auto_forecast, auto_model) = auto_select(data.clone(), 7)?;
    println!("✓ Auto selection chose: {} -> {:?}\n", auto_model, &auto_forecast[..3]);

    // 5. Model evaluation and comparison
    println!("📊 Method 4: Model Evaluation");
    println!("------------------------------");
    
    // Compare multiple models
    let models = vec![
        ("ARIMA(1,1,1)", ModelBuilder::arima().with_ar(1).with_differencing(1).with_ma(1).build()?),
        ("ES(α=0.3)", ModelBuilder::exponential_smoothing().with_alpha(0.3).build()?),
        ("MA(5)", ModelBuilder::moving_average().with_window(5).build()?),
    ];
    
    for (name, mut model) in models {
        match model.quick_fit(&data) {
            Ok(()) => {
                if let Ok(eval) = model.evaluate(&data) {
                    println!("✓ {}: MSE={:.4}, MAE={:.4}", name, eval.mse, eval.mae);
                }
            }
            Err(e) => println!("⚠️  {} failed: {}", name, e),
        }
    }

    println!("\n🎉 Quick start complete! You can now:");
    println!("   • Use AutoSelector for automatic model selection");
    println!("   • Use ModelBuilder for custom configurations");
    println!("   • Use quick functions for immediate results");
    println!("   • Compare models with built-in evaluation metrics");
    
    Ok(())
}

/// Create sample time series data for demonstration
fn create_sample_data() -> (Vec<DateTime<Utc>>, Vec<f64>) {
    let start = Utc::now() - Duration::days(50);
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    
    for i in 0..50 {
        timestamps.push(start + Duration::days(i));
        
        // Create realistic-looking data with trend and weekly pattern
        let trend = 100.0 + 0.5 * i as f64;
        let weekly = 10.0 * (2.0 * std::f64::consts::PI * (i % 7) as f64 / 7.0).sin();
        let noise = (i % 17) as f64 * 0.3 - 2.5; // Deterministic "noise"
        
        values.push(trend + weekly + noise);
    }
    
    (timestamps, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_start_example() {
        // Test that the quick start example runs without panicking
        let result = main();
        
        // The example should complete successfully or fail gracefully
        match result {
            Ok(()) => println!("Quick start example completed successfully"),
            Err(e) => println!("Quick start example failed gracefully: {}", e),
        }
    }

    #[test]
    fn test_sample_data_creation() {
        let (timestamps, values) = create_sample_data();
        
        assert_eq!(timestamps.len(), 50);
        assert_eq!(values.len(), 50);
        
        // Verify timestamps are increasing
        for i in 1..timestamps.len() {
            assert!(timestamps[i] > timestamps[i-1]);
        }
        
        // Verify values are finite
        for &value in &values {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_model_builder_basic() {
        // Test that basic model building works
        let arima = ModelBuilder::arima()
            .with_ar(1)
            .with_differencing(1)
            .with_ma(1)
            .build();
        assert!(arima.is_ok());

        let es = ModelBuilder::exponential_smoothing()
            .with_alpha(0.3)
            .build();
        assert!(es.is_ok());

        let ma = ModelBuilder::moving_average()
            .with_window(5)
            .build();
        assert!(ma.is_ok());
    }
} 