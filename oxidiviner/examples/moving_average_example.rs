//! Moving Average Model Example
//!
//! This example demonstrates how to use Moving Average models for time series forecasting.
//! Moving Average models are simple yet effective for smoothing data and basic forecasting,
//! particularly useful for noisy time series or as baseline models.

use oxidiviner::prelude::*;
use oxidiviner::models::moving_average::MAModel;
use chrono::{Duration, Utc};

fn main() -> oxidiviner::Result<()> {
    println!("=== Moving Average Model Example ===\n");

    // Generate sample data with trend and noise
    let start_date = Utc::now() - Duration::days(60);
    let timestamps: Vec<DateTime<Utc>> = (0..60)
        .map(|i| start_date + Duration::days(i))
        .collect();
    
    // Create a time series with trend and noise
    let values: Vec<f64> = (0..60)
        .map(|i| {
            let trend = 50.0 + 0.3 * i as f64;
            let noise = (rand::random::<f64>() - 0.5) * 8.0;
            trend + noise
        })
        .collect();

    println!("Generated {} data points with trend and noise", values.len());
    println!("Data range: {:.2} to {:.2}", 
             values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

    // Create time series data
    let ts_data = TimeSeriesData::new(timestamps.clone(), values.clone(), "sample_series")?;

    // Split data into train/test (80/20 split)
    let split_idx = (values.len() as f64 * 0.8) as usize;
    let train_timestamps = timestamps[..split_idx].to_vec();
    let train_values = values[..split_idx].to_vec();
    let test_values = values[split_idx..].to_vec();

    let train_data = TimeSeriesData::new(train_timestamps, train_values, "train_data")?;

    println!("\nData split: {} training, {} testing observations", 
             train_data.len(), test_values.len());

    // Example 1: Basic Moving Average with window size 5
    println!("\n1. Basic Moving Average (Window = 5)");
    println!("====================================");

    let mut ma_5 = MAModel::new(5)?;
    println!("Created Moving Average model with window size 5");

    // Fit the model
    ma_5.fit(&train_data)?;
    println!("✓ Model fitted successfully");

    // Generate forecasts
    let forecast_5 = ma_5.forecast(test_values.len())?;
    println!("✓ Generated {} forecasts", forecast_5.len());

    // Evaluate the model
    let test_data = TimeSeriesData::new(
        timestamps[split_idx..].to_vec(),
        test_values.clone(),
        "test_data"
    )?;
    let evaluation_5 = ma_5.evaluate(&test_data)?;
    
    println!("Model Performance (Window = 5):");
    println!("  MAE:  {:.3}", evaluation_5.mae);
    println!("  RMSE: {:.3}", evaluation_5.rmse);
    println!("  MAPE: {:.2}%", evaluation_5.mape);

    // Example 2: Comparing different window sizes
    println!("\n2. Comparing Different Window Sizes");
    println!("===================================");

    let window_sizes = vec![3, 5, 7, 10, 15, 20];

    for window_size in window_sizes {
        match MAModel::new(window_size) {
            Ok(mut model) => {
                match model.fit(&train_data) {
                    Ok(_) => {
                        match model.forecast(test_values.len()) {
                            Ok(_forecast) => {
                                match model.evaluate(&test_data) {
                                    Ok(eval) => {
                                        println!("  Window {}: RMSE = {:.3}, MAE = {:.3}", 
                                                window_size, eval.rmse, eval.mae);
                                    }
                                    Err(_) => println!("  Window {}: Evaluation failed", window_size),
                                }
                            }
                            Err(_) => println!("  Window {}: Forecast failed", window_size),
                        }
                    }
                    Err(_) => println!("  Window {}: Fit failed", window_size),
                }
            }
            Err(_) => println!("  Window {}: Model creation failed", window_size),
        }
    }

    // Example 3: Forecasting future values
    println!("\n3. Future Forecasting");
    println!("=====================");

    // Use the full dataset to fit the model for future forecasting
    let mut final_model = MAModel::new(7)?; // Use optimal window size
    final_model.fit(&ts_data)?;

    // Forecast next 10 periods
    let future_forecast = final_model.forecast(10)?;
    println!("Next 10-period forecast (Window = 7):");
    for (i, &forecast_val) in future_forecast.iter().enumerate() {
        println!("  Period {}: {:.2}", i + 1, forecast_val);
    }

    // Example 4: Different scenarios - High vs Low noise
    println!("\n4. Performance with Different Noise Levels");
    println!("==========================================");

    // Low noise data
    let low_noise_values: Vec<f64> = (0..60)
        .map(|i| {
            let trend = 50.0 + 0.3 * i as f64;
            let noise = (rand::random::<f64>() - 0.5) * 2.0; // Lower noise
            trend + noise
        })
        .collect();

    let low_noise_data = TimeSeriesData::new(timestamps.clone(), low_noise_values, "low_noise")?;
    
    // High noise data  
    let high_noise_values: Vec<f64> = (0..60)
        .map(|i| {
            let trend = 50.0 + 0.3 * i as f64;
            let noise = (rand::random::<f64>() - 0.5) * 15.0; // Higher noise
            trend + noise
        })
        .collect();

    let high_noise_data = TimeSeriesData::new(timestamps.clone(), high_noise_values, "high_noise")?;

    println!("Testing optimal window sizes for different noise levels:");

    for (data, label) in &[(low_noise_data, "Low Noise"), (high_noise_data, "High Noise")] {
        let split_idx = (data.len() as f64 * 0.8) as usize;
        let train_part = TimeSeriesData::new(
            data.timestamps[..split_idx].to_vec(),
            data.values[..split_idx].to_vec(),
            &format!("{}_train", label)
        )?;
        let test_part = TimeSeriesData::new(
            data.timestamps[split_idx..].to_vec(),
            data.values[split_idx..].to_vec(),
            &format!("{}_test", label)
        )?;

        let mut best_rmse = f64::INFINITY;
        let mut best_window = 3;

        for window in [3, 5, 7, 10] {
            if let Ok(mut model) = MAModel::new(window) {
                if model.fit(&train_part).is_ok() {
                    if let Ok(eval) = model.evaluate(&test_part) {
                        if eval.rmse < best_rmse {
                            best_rmse = eval.rmse;
                            best_window = window;
                        }
                    }
                }
            }
        }

        println!("  {}: Best window = {}, RMSE = {:.3}", label, best_window, best_rmse);
    }

    // Example 5: Using the quick API for comparison
    println!("\n5. Quick API Comparison");
    println!("=======================");

    use oxidiviner::quick;

    // Default moving average
    let quick_forecast = quick::moving_average(ts_data.clone(), 10, None)?;
    println!("Quick API forecast (default window, first 5 values): {:?}", &quick_forecast[..5]);

    // Custom window size
    let quick_forecast_7 = quick::moving_average(ts_data.clone(), 10, Some(7))?;
    println!("Quick API forecast (window=7, first 5 values): {:?}", &quick_forecast_7[..5]);

    // Example 6: Practical applications
    println!("\n6. Practical Applications");
    println!("=========================");

    // Sales forecasting scenario
    let sales_data: Vec<f64> = vec![
        120.0, 135.0, 142.0, 128.0, 155.0, 163.0, 178.0, 145.0, 160.0, 172.0,
        185.0, 158.0, 195.0, 202.0, 180.0, 215.0, 225.0, 198.0, 240.0, 255.0
    ];

    let sales_timestamps: Vec<DateTime<Utc>> = (0..sales_data.len())
        .map(|i| start_date + Duration::days(i as i64))
        .collect();

    let sales_ts = TimeSeriesData::new(sales_timestamps, sales_data, "daily_sales")?;

    println!("Sales forecasting example:");
    
    // Test different windows for sales data
    for window in [3, 5, 7] {
        let mut sales_model = MAModel::new(window)?;
        sales_model.fit(&sales_ts)?;
        let sales_forecast = sales_model.forecast(5)?;
        println!("  Window {}: Next 5 days average sales = {:.1}", 
                window, sales_forecast.iter().sum::<f64>() / sales_forecast.len() as f64);
    }

    println!("\n=== Moving Average Example Complete ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moving_average_example() {
        let result = main();
        assert!(result.is_ok(), "Moving Average example should run successfully: {:?}", result);
    }
} 