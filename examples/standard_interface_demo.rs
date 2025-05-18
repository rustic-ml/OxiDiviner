use chrono::{DateTime, TimeZone, Utc};
use oxidiviner::prelude::*;
use oxidiviner::models::Forecaster;
use std::error::Error;

fn main() -> std::result::Result<(), Box<dyn Error>> {
    println!("OxiDiviner - Standardized Model Interface Demo");
    println!("==============================================\n");

    // Generate sample time series data
    let (time_series, _true_trend) = generate_demo_data();
    
    println!("Using standardized model interface for different models");
    println!("------------------------------------------------------\n");
    
    // Create models
    let mut ses_model = SESModel::new(0.3, None)?;
    let mut ma_model = MAModel::new(5)?;
    
    // Train both models on the data using the standard interface
    ses_model.fit(&time_series)?;
    ma_model.fit(&time_series)?;
    
    // Forecast with both models using the standard interface
    let horizon = 10;
    let ses_output = ses_model.predict(horizon, Some(&time_series))?;
    let ma_output = ma_model.predict(horizon, Some(&time_series))?;
    
    // Print forecasts using the standard output format
    println!("Forecasts from Simple Exponential Smoothing (SES):");
    println!("--------------------------------------------------");
    print_model_output(&ses_output);
    
    println!("\nForecasts from Moving Average (MA):");
    println!("----------------------------------");
    print_model_output(&ma_output);
    
    println!("\nComparing model evaluation metrics:");
    println!("----------------------------------");
    if let Some(ses_eval) = ses_output.evaluation {
        println!("SES Model ({}): MAE = {:.4}, RMSE = {:.4}", 
                 ses_eval.model_name, ses_eval.mae, ses_eval.rmse);
    }
    
    if let Some(ma_eval) = ma_output.evaluation {
        println!("MA Model ({}): MAE = {:.4}, RMSE = {:.4}", 
                 ma_eval.model_name, ma_eval.mae, ma_eval.rmse);
    }
    
    Ok(())
}

// Print the model output in a nicely formatted way
fn print_model_output(output: &oxidiviner::models::data::ModelOutput) {
    println!("Model: {}", output.model_name);
    println!("Forecast:");
    
    for (i, value) in output.forecasts.iter().enumerate() {
        println!("  t+{}: {:.4}", i+1, value);
    }
    
    if let Some(intervals) = &output.confidence_intervals {
        println!("Confidence Intervals:");
        for (i, (lower, upper)) in intervals.iter().enumerate() {
            println!("  t+{}: [{:.4}, {:.4}]", i+1, lower, upper);
        }
    }
    
    if !output.metadata.is_empty() {
        println!("Metadata:");
        for (key, value) in &output.metadata {
            println!("  {}: {}", key, value);
        }
    }
}

// Generate synthetic demo data
fn generate_demo_data() -> (TimeSeriesData, Vec<f64>) {
    let now = Utc::now();
    let n = 100;
    
    // Create timestamps (daily intervals)
    let timestamps: Vec<DateTime<Utc>> = (0..n)
        .map(|i| Utc.timestamp_opt(now.timestamp() + i as i64 * 86400, 0).unwrap())
        .collect();
    
    // Create a trend with some noise
    let mut values = Vec::with_capacity(n);
    let mut trend = Vec::with_capacity(n);
    
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for i in 0..n {
        // Trend component: linear trend
        let t = 10.0 + 0.5 * (i as f64);
        trend.push(t);
        
        // Add some noise
        let noise = rng.gen::<f64>() * 5.0 - 2.5;
        values.push(t + noise);
    }
    
    let time_series = TimeSeriesData::new(
        timestamps,
        values,
        "demo_data"
    ).unwrap();
    
    (time_series, trend)
} 