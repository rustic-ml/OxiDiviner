use chrono::{DateTime, TimeZone, Utc};
use oxidiviner_core::prelude::*;
use oxidiviner_core::models::Forecaster;
use oxidiviner_core::models::exponential_smoothing::SESModel;
use std::error::Error;

fn main() -> std::result::Result<(), Box<dyn Error>> {
    println!("OxiDiviner - SES Parameter Tuning Example");
    println!("========================================\n");

    // Generate sample time series data
    let time_series = generate_sample_data();
    
    // Split data into training and testing sets (80% train, 20% test)
    let (train_data, test_data) = time_series.train_test_split(0.8)?;
    
    println!("Data split: {} total points ({} train, {} test)",
             time_series.len(), train_data.len(), test_data.len());
    
    // Set forecast horizon
    let horizon = 10;
    
    // Parameter tuning for Simple Exponential Smoothing
    println!("\nParameter Tuning for Simple Exponential Smoothing");
    println!("------------------------------------------------");
    println!("Testing different alpha values (smoothing parameter):");
    println!("Alpha    RMSE     MAE      MAPE");
    println!("--------------------------------");
    
    // Try different alpha values (smoothing parameters)
    let alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let mut best_alpha = 0.0;
    let mut best_rmse = f64::MAX;
    
    for &alpha in &alpha_values {
        // Create and train SES model with this alpha
        let mut model = SESModel::new(alpha, None)?;
        model.fit(&train_data)?;
        
        // Evaluate on test data
        let output = model.predict(horizon, Some(&test_data))?;
        
        if let Some(eval) = output.evaluation {
            println!("{:.1}     {:.4}  {:.4}  {:.4}%", 
                    alpha, eval.rmse, eval.mae, eval.mape);
                    
            // Track best parameter
            if eval.rmse < best_rmse {
                best_rmse = eval.rmse;
                best_alpha = alpha;
            }
        }
    }
    
    println!("\nBest alpha value: {:.1} (RMSE: {:.4})", best_alpha, best_rmse);
    
    // Create SES model with best alpha value
    let mut best_model = SESModel::new(best_alpha, None)?;
    best_model.fit(&train_data)?;
    
    // Generate forecasts for future periods
    let output = best_model.predict(horizon, None)?;
    
    // Print forecasts
    println!("\nForecasts for the next {} periods:", horizon);
    for (i, value) in output.forecasts.iter().enumerate() {
        println!("  Period t+{}: {:.4}", i+1, value);
    }
    
    println!("\nNote on the alpha parameter:");
    println!("- Small alpha (closer to 0): More weight on historical data, smoother forecasts");
    println!("- Large alpha (closer to 1): More weight on recent data, responsive to changes");
    println!("\nFor real applications, consider:");
    println!("1. Using cross-validation with multiple train/test splits");
    println!("2. Trying a finer grid of parameters around the best values");
    println!("3. Optimizing for the most appropriate metric for your specific application");
    
    Ok(())
}

// Generate a sample time series with trend and some noise
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
        let noise = rng.gen::<f64>() * 3.0 - 1.5;
        
        // Combined signal
        values.push(trend + noise);
    }
    
    TimeSeriesData::new(
        timestamps,
        values,
        "sample_data_with_trend"
    ).unwrap()
} 