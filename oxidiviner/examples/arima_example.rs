//! ARIMA Model Example
//!
//! This example demonstrates how to use ARIMA (AutoRegressive Integrated Moving Average) models
//! for time series forecasting. ARIMA models are suitable for non-stationary time series data
//! with trends and autocorrelation patterns.

use oxidiviner::prelude::*;
use oxidiviner::models::autoregressive::ARIMAModel;
use chrono::{Duration, Utc};

fn main() -> oxidiviner::Result<()> {
    println!("=== ARIMA Model Example ===\n");

    // Generate sample data with trend and noise
    let start_date = Utc::now() - Duration::days(100);
    let timestamps: Vec<DateTime<Utc>> = (0..100)
        .map(|i| start_date + Duration::days(i))
        .collect();
    
    // Create a time series with trend and some randomness
    let values: Vec<f64> = (0..100)
        .map(|i| {
            let trend = 100.0 + 0.5 * i as f64;
            let seasonal = 5.0 * (i as f64 * 2.0 * std::f64::consts::PI / 7.0).sin();
            let noise = (rand::random::<f64>() - 0.5) * 3.0;
            trend + seasonal + noise
        })
        .collect();

    println!("Generated {} data points with trend and weekly seasonality", values.len());
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

    // Example 1: Basic ARIMA(1,1,1) model
    println!("\n1. Basic ARIMA(1,1,1) Model");
    println!("===========================");

    let mut arima_111 = ARIMAModel::new(1, 1, 1, true)?;
    println!("Created ARIMA(1,1,1) with intercept");

    // Fit the model
    arima_111.fit(&train_data)?;
    println!("✓ Model fitted successfully");

    // Generate forecasts
    let forecast_111 = arima_111.forecast(test_values.len())?;
    println!("✓ Generated {} forecasts", forecast_111.len());

    // Evaluate the model
    let test_data = TimeSeriesData::new(
        timestamps[split_idx..].to_vec(),
        test_values.clone(),
        "test_data"
    )?;
    let evaluation_111 = arima_111.evaluate(&test_data)?;
    
    println!("Model Performance:");
    println!("  MAE:  {:.3}", evaluation_111.mae);
    println!("  RMSE: {:.3}", evaluation_111.rmse);
    println!("  MAPE: {:.2}%", evaluation_111.mape);

    // Example 2: Different ARIMA configurations
    println!("\n2. Comparing Different ARIMA Orders");
    println!("===================================");

    let arima_configs = vec![
        (1, 1, 0, "AR(1) with differencing"),
        (0, 1, 1, "MA(1) with differencing"),
        (2, 1, 1, "ARIMA(2,1,1)"),
        (1, 1, 2, "ARIMA(1,1,2)"),
        (2, 1, 2, "ARIMA(2,1,2)"),
    ];

    for (p, d, q, description) in arima_configs {
        match ARIMAModel::new(p, d, q, true) {
            Ok(mut model) => {
                match model.fit(&train_data) {
                    Ok(_) => {
                        match model.forecast(test_values.len()) {
                            Ok(forecast) => {
                                match model.evaluate(&test_data) {
                                    Ok(eval) => {
                                        println!("  {}: RMSE = {:.3}, MAE = {:.3}", 
                                                description, eval.rmse, eval.mae);
                                    }
                                    Err(_) => println!("  {}: Evaluation failed", description),
                                }
                            }
                            Err(_) => println!("  {}: Forecast failed", description),
                        }
                    }
                    Err(_) => println!("  {}: Fit failed", description),
                }
            }
            Err(_) => println!("  {}: Model creation failed", description),
        }
    }

    // Example 3: Forecasting future values
    println!("\n3. Future Forecasting");
    println!("=====================");

    // Use the full dataset to fit the model for future forecasting
    let mut final_model = ARIMAModel::new(1, 1, 1, true)?;
    final_model.fit(&ts_data)?;

    // Forecast next 14 days
    let future_forecast = final_model.forecast(14)?;
    println!("Next 14-day forecast:");
    for (i, &forecast_val) in future_forecast.iter().enumerate() {
        println!("  Day {}: {:.2}", i + 1, forecast_val);
    }

    // Example 4: Model diagnostics and information
    println!("\n4. Model Information");
    println!("====================");

    // Display model parameters if available
    if let Some(ar_coeffs) = final_model.ar_coefficients() {
        println!("AR coefficients: {:?}", ar_coeffs);
    }
    if let Some(ma_coeffs) = final_model.ma_coefficients() {
        println!("MA coefficients: {:?}", ma_coeffs);
    }

    // Example 5: Using the quick API for comparison
    println!("\n5. Quick API Comparison");
    println!("=======================");

    use oxidiviner::quick;

    let quick_forecast = quick::arima(ts_data.clone(), 14)?;
    println!("Quick API forecast (first 5 values): {:?}", &quick_forecast[..5]);

    let (auto_forecast, auto_model) = quick::auto_forecast(
        timestamps.clone(), 
        values.clone(), 
        14
    )?;
    println!("Auto-selected model: {}", auto_model);
    println!("Auto forecast (first 5 values): {:?}", &auto_forecast[..5]);

    println!("\n=== ARIMA Example Complete ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arima_example() {
        let result = main();
        assert!(result.is_ok(), "ARIMA example should run successfully: {:?}", result);
    }
} 