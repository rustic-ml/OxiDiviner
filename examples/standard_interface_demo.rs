use chrono::{DateTime, TimeZone, Utc};
use oxidiviner::prelude::*;
use rand::Rng;
use std::error::Error;

// Main function to demonstrate OxiDiviner's standard interface
fn main() -> std::result::Result<(), Box<dyn Error>> {
    println!("OxiDiviner - Standard Interface Demo");
    println!("====================================\n");

    // Create some demo data
    let data = create_sample_data(100, "test_series")?;

    // Create and configure models
    println!("Creating and configuring models...");

    // Simple Exponential Smoothing model
    let mut ses_model = SESModel::new(0.3)?;

    // Moving Average model
    let mut ma_model = MAModel::new(5)?;

    // Autoregressive model (order 3)
    let mut ar_model = ARModel::new(3, true)?;

    // Fit models to data
    println!("\nFitting models to data...");

    ses_model.fit(&data)?;
    ma_model.fit(&data)?;
    ar_model.fit(&data)?;

    // Make predictions
    println!("\nMaking predictions...");

    let forecast_horizon = 10;
    let ses_output = ses_model.predict(forecast_horizon, None)?;
    let ma_output = ma_model.predict(forecast_horizon, None)?;
    let ar_output = ar_model.predict(forecast_horizon, None)?;

    // Print results
    println!("\nModel outputs:");

    print_model_output(&ses_output);
    print_model_output(&ma_output);
    print_model_output(&ar_output);

    Ok(())
}

// Helper function to print model output information
fn print_model_output(output: &ModelOutput) {
    println!("\nModel: {}", output.model_name);
    println!("Forecasts:");

    for (i, value) in output.forecasts.iter().enumerate() {
        println!("  t+{}: {:.2}", i + 1, value);
    }

    // Print evaluation metrics if available
    if let Some(eval) = &output.evaluation {
        println!("\nEvaluation metrics:");
        println!("  MAE:  {:.4}", eval.mae);
        println!("  RMSE: {:.4}", eval.rmse);
        println!("  MAPE: {:.2}%", eval.mape);
    }

    println!("");
}

// Helper function to create sample time series data
fn create_sample_data(n: usize, name: &str) -> std::result::Result<TimeSeriesData, Box<dyn Error>> {
    println!("Creating sample data with {} points...", n);

    let mut timestamps = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);

    // Create a base timestamp
    let base_time = chrono::Utc::now();

    // Initialize random number generator
    let mut rng = rand::thread_rng();

    // Generate a sinusoidal pattern with some noise
    for i in 0..n {
        // Add i days to the base time
        let timestamp = base_time + chrono::Duration::days(i as i64);

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
    Ok(TimeSeriesData {
        timestamps,
        values,
        name: name.to_string(),
    })
}
