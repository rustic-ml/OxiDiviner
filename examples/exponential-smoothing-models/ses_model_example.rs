use chrono::{DateTime, TimeZone, Utc};
use oxidiviner_core::prelude::*;
use oxidiviner_core::models::Forecaster;
use std::error::Error;

fn main() -> std::result::Result<(), Box<dyn Error>> {
    println!("OxiDiviner - Simple Exponential Smoothing Model Example");
    println!("======================================================\n");

    // Generate sample time series data
    let time_series = generate_sample_data();
    
    // Split data into training and testing sets
    let (train_data, test_data) = time_series.train_test_split(0.8)?;
    
    println!("Data split: {} total points ({} train, {} test)",
             time_series.len(), train_data.len(), test_data.len());
    
    // Create a SES model with alpha = 0.3 (smoothing parameter)
    // None for target_column means it will use the default (Close price)
    let mut ses_model = SESModel::new(0.3, None)?;
    
    println!("\nTraining model: {}", ses_model.name());
    
    // Train the model using the standardized interface
    ses_model.fit(&train_data)?;
    
    // Forecast horizon - how many periods ahead to predict
    let horizon = 10;
    
    // Generate forecasts and evaluation metrics in a standardized format
    let output = ses_model.predict(horizon, Some(&test_data))?;
    
    // Print the forecasts
    println!("\nForecasts for the next {} periods:", horizon);
    for (i, value) in output.forecasts.iter().enumerate() {
        println!("  Period t+{}: {:.4}", i+1, value);
    }
    
    // Print evaluation metrics
    if let Some(eval) = output.evaluation {
        println!("\nModel Evaluation:");
        println!("  MAE: {:.4} (Mean Absolute Error)", eval.mae);
        println!("  MSE: {:.4} (Mean Squared Error)", eval.mse);
        println!("  RMSE: {:.4} (Root Mean Squared Error)", eval.rmse);
        println!("  MAPE: {:.4}% (Mean Absolute Percentage Error)", eval.mape);
    }
    
    // Demonstrate forecasting with different alpha values
    println!("\nComparing different alpha values:");
    
    let alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9];
    
    for &alpha in &alpha_values {
        let mut model = SESModel::new(alpha, None)?;
        model.fit(&train_data)?;
        let output = model.predict(horizon, Some(&test_data))?;
        
        if let Some(eval) = output.evaluation {
            println!("  Alpha = {:.1}: RMSE = {:.4}, MAE = {:.4}", 
                     alpha, eval.rmse, eval.mae);
        }
    }
    
    println!("\nNote on the alpha parameter:");
    println!("- Small alpha (closer to 0): More weight on historical data, smoother forecasts");
    println!("- Large alpha (closer to 1): More weight on recent data, responsive to changes");
    
    Ok(())
}

// Generate a sample time series with trend and noise
fn generate_sample_data() -> TimeSeriesData {
    let now = Utc::now();
    let n = 100;
    
    // Create timestamps (daily intervals)
    let timestamps: Vec<DateTime<Utc>> = (0..n)
        .map(|i| Utc.timestamp_opt(now.timestamp() + i as i64 * 86400, 0).unwrap())
        .collect();
    
    // Create values with a linear trend and some noise
    let mut values = Vec::with_capacity(n);
    
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for i in 0..n {
        // Linear trend with slope 0.5
        let trend = 10.0 + 0.5 * (i as f64);
        
        // Add random noise
        let noise = rng.gen::<f64>() * 5.0 - 2.5;
        values.push(trend + noise);
    }
    
    TimeSeriesData::new(
        timestamps,
        values,
        "sample_data"
    ).unwrap()
} 