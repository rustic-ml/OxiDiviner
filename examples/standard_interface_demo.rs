//! Standard Interface Demo for OxiDiviner
//!
//! This example demonstrates the unified interface across different models
//! using the new API improvements.

use chrono::{Duration, Utc};
use oxidiviner::api::*;
use oxidiviner::prelude::*;
use rand::Rng;
use std::error::Error;

// Main function to demonstrate OxiDiviner's standard interface
fn main() -> std::result::Result<(), Box<dyn Error>> {
    println!("OxiDiviner - Standard Interface Demo");
    println!("====================================\n");

    // Create some demo data
    let data = create_sample_data(100, "test_series")?;

    // Demo 1: High-level Forecaster interface
    println!("=== Demo 1: High-level Forecaster Interface ===");
    demo_high_level_interface(&data)?;

    // Demo 2: Builder pattern interface
    println!("\n=== Demo 2: Builder Pattern Interface ===");
    demo_builder_interface(&data)?;

    // Demo 3: Model comparison
    println!("\n=== Demo 3: Model Comparison ===");
    demo_model_comparison(&data)?;

    Ok(())
}

fn demo_high_level_interface(data: &TimeSeriesData) -> Result<()> {
    println!("Using the high-level Forecaster interface...\n");

    // Simple exponential smoothing
    let forecaster = Forecaster::new().model(ModelType::SimpleES).alpha(0.3);
    let output = forecaster.forecast(data, 10)?;
    println!("📈 Simple ES forecast: {:?}", &output.forecast[..5]);

    // Moving Average
    let forecaster = Forecaster::new()
        .model(ModelType::MovingAverage)
        .ma_window(5);
    let output = forecaster.forecast(data, 10)?;
    println!("📈 MA(5) forecast: {:?}", &output.forecast[..5]);

    // ARIMA
    let forecaster = Forecaster::new()
        .model(ModelType::ARIMA)
        .arima_params(2, 1, 1);
    let output = forecaster.forecast(data, 10)?;
    println!("📈 ARIMA(2,1,1) forecast: {:?}", &output.forecast[..5]);

    Ok(())
}

fn demo_builder_interface(data: &TimeSeriesData) -> Result<()> {
    println!("Using the builder pattern interface...\n");

    // Simple Exponential Smoothing with builder
    let mut ses_model = ModelBuilder::exponential_smoothing()
        .with_alpha(0.3)
        .build()?;
    ses_model.quick_fit(data)?;
    let forecast = ses_model.quick_forecast(10)?;
    println!("🏗️  ES builder forecast: {:?}", &forecast[..5]);

    // Moving Average with builder
    let mut ma_model = ModelBuilder::moving_average().with_window(5).build()?;
    ma_model.quick_fit(data)?;
    let forecast = ma_model.quick_forecast(10)?;
    println!("🏗️  MA builder forecast: {:?}", &forecast[..5]);

    // ARIMA with builder
    let mut arima_model = ModelBuilder::arima()
        .with_ar(2)
        .with_differencing(1)
        .with_ma(1)
        .build()?;
    arima_model.quick_fit(data)?;
    let forecast = arima_model.quick_forecast(10)?;
    println!("🏗️  ARIMA builder forecast: {:?}", &forecast[..5]);

    Ok(())
}

fn demo_model_comparison(data: &TimeSeriesData) -> Result<()> {
    println!("Comparing different models...\n");

    let models = vec![
        (
            "Simple ES (α=0.3)",
            ModelBuilder::exponential_smoothing().with_alpha(0.3),
        ),
        (
            "Simple ES (α=0.7)",
            ModelBuilder::exponential_smoothing().with_alpha(0.7),
        ),
        ("MA(3)", ModelBuilder::moving_average().with_window(3)),
        ("MA(7)", ModelBuilder::moving_average().with_window(7)),
        (
            "ARIMA(1,1,1)",
            ModelBuilder::arima()
                .with_ar(1)
                .with_differencing(1)
                .with_ma(1),
        ),
    ];

    println!(
        "{:<15} {:<8} {:<8} {:<8} {:<8}",
        "Model", "MAE", "MSE", "RMSE", "MAPE"
    );
    println!("{}", "-".repeat(55));

    for (name, builder) in models {
        match builder.build() {
            Ok(mut model) => {
                if model.quick_fit(data).is_ok() {
                    if let Ok(evaluation) = model.evaluate(data) {
                        println!(
                            "{:<15} {:<8.3} {:<8.3} {:<8.3} {:<8.2}",
                            name, evaluation.mae, evaluation.mse, evaluation.rmse, evaluation.mape
                        );
                    }
                }
            }
            Err(_) => {
                println!("{:<15} Failed to build", name);
            }
        }
    }

    Ok(())
}

// Helper function to create sample time series data
fn create_sample_data(n: usize, name: &str) -> std::result::Result<TimeSeriesData, Box<dyn Error>> {
    println!("Creating sample data with {} points...", n);

    let mut timestamps = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);

    // Create a base timestamp
    let base_time = Utc::now();

    // Initialize random number generator
    let mut rng = rand::thread_rng();

    // Generate a sinusoidal pattern with some noise
    for i in 0..n {
        // Add i days to the base time
        let timestamp = base_time + Duration::days(i as i64);

        // Generate a value with sinusoidal pattern and noise
        let time = i as f64 / 10.0;
        let trend = 0.5 * time;
        let seasonal = 10.0 * (time * 0.5).sin();
        let noise = rng.random::<f64>() * 5.0 - 2.5;

        let value = 100.0 + trend + seasonal + noise;

        timestamps.push(timestamp);
        values.push(value);
    }

    // Create the TimeSeriesData struct
    Ok(TimeSeriesData::new(timestamps, values, name)?)
}
