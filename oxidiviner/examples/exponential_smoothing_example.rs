//! Exponential Smoothing Models Example
//!
//! This example demonstrates the family of Exponential Smoothing models available in OxiDiviner:
//! - Simple Exponential Smoothing (SES): For level-only data
//! - Holt's Linear Method: For data with trend
//! - Holt-Winters: For data with trend and seasonality
//! These models are particularly effective for business forecasting and demand planning.

use oxidiviner::prelude::*;
use oxidiviner::models::exponential_smoothing::{SimpleESModel, HoltLinearModel, HoltWintersModel};
use chrono::{Duration, Utc};

fn main() -> oxidiviner::Result<()> {
    println!("=== Exponential Smoothing Models Example ===\n");

    // Generate different types of data for different models
    let start_date = Utc::now() - Duration::days(100);
    let timestamps: Vec<DateTime<Utc>> = (0..100)
        .map(|i| start_date + Duration::days(i))
        .collect();

    // Example 1: Simple Exponential Smoothing (Level-only data)
    println!("1. Simple Exponential Smoothing (SES)");
    println!("=====================================");

    // Generate level-only data (stationary around a mean with noise)
    let level_values: Vec<f64> = (0..100)
        .map(|_| {
            let level = 50.0;
            let noise = (rand::random::<f64>() - 0.5) * 6.0;
            level + noise
        })
        .collect();

    let level_data = TimeSeriesData::new(timestamps.clone(), level_values.clone(), "level_series")?;

    println!("Generated level-only data (mean ≈ 50, noise ± 3)");
    println!("Data range: {:.2} to {:.2}", 
             level_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             level_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

    // Split data for evaluation
    let split_idx = 80;
    let train_level = TimeSeriesData::new(
        timestamps[..split_idx].to_vec(),
        level_values[..split_idx].to_vec(),
        "level_train"
    )?;
    let test_level = TimeSeriesData::new(
        timestamps[split_idx..].to_vec(),
        level_values[split_idx..].to_vec(),
        "level_test"
    )?;

    // Test different alpha values for SES
    let alpha_values = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    println!("\nTesting different alpha values for SES:");

    for alpha in alpha_values {
        match SimpleESModel::new(alpha) {
            Ok(mut model) => {
                match model.fit(&train_level) {
                    Ok(_) => {
                        match model.evaluate(&test_level) {
                            Ok(eval) => {
                                println!("  Alpha {:.1}: RMSE = {:.3}, MAE = {:.3}", 
                                        alpha, eval.rmse, eval.mae);
                            }
                            Err(_) => println!("  Alpha {:.1}: Evaluation failed", alpha),
                        }
                    }
                    Err(_) => println!("  Alpha {:.1}: Fit failed", alpha),
                }
            }
            Err(_) => println!("  Alpha {:.1}: Model creation failed", alpha),
        }
    }

    // Use optimal alpha for forecasting
    let mut ses_model = SimpleESModel::new(0.3)?;
    ses_model.fit(&level_data)?;
    let ses_forecast = ses_model.forecast(10)?;
    println!("\nSES forecast (next 10 periods): {:?}", &ses_forecast[..5]);

    // Example 2: Holt's Linear Method (Trend data)
    println!("\n2. Holt's Linear Method (Trend)");
    println!("===============================");

    // Generate trending data
    let trend_values: Vec<f64> = (0..100)
        .map(|i| {
            let level = 30.0;
            let trend = 0.4 * i as f64;
            let noise = (rand::random::<f64>() - 0.5) * 4.0;
            level + trend + noise
        })
        .collect();

    let trend_data = TimeSeriesData::new(timestamps.clone(), trend_values.clone(), "trend_series")?;

    println!("Generated trending data (slope ≈ 0.4, noise ± 2)");
    println!("Data range: {:.2} to {:.2}", 
             trend_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             trend_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

    let train_trend = TimeSeriesData::new(
        timestamps[..split_idx].to_vec(),
        trend_values[..split_idx].to_vec(),
        "trend_train"
    )?;
    let test_trend = TimeSeriesData::new(
        timestamps[split_idx..].to_vec(),
        trend_values[split_idx..].to_vec(),
        "trend_test"
    )?;

    // Test different parameter combinations for Holt
    let holt_params = vec![
        (0.3, 0.1, "Conservative"),
        (0.5, 0.2, "Moderate"),
        (0.7, 0.3, "Aggressive"),
        (0.9, 0.1, "High alpha, low beta"),
        (0.3, 0.5, "Low alpha, high beta"),
    ];

    println!("\nTesting different parameter combinations for Holt:");

    for (alpha, beta, description) in holt_params {
        match HoltLinearModel::new(alpha, beta) {
            Ok(mut model) => {
                match model.fit(&train_trend) {
                    Ok(_) => {
                        match model.evaluate(&test_trend) {
                            Ok(eval) => {
                                println!("  {} (α={}, β={}): RMSE = {:.3}, MAE = {:.3}", 
                                        description, alpha, beta, eval.rmse, eval.mae);
                            }
                            Err(_) => println!("  {}: Evaluation failed", description),
                        }
                    }
                    Err(_) => println!("  {}: Fit failed", description),
                }
            }
            Err(_) => println!("  {}: Model creation failed", description),
        }
    }

    // Use optimal parameters for forecasting
    let mut holt_model = HoltLinearModel::new(0.5, 0.2)?;
    holt_model.fit(&trend_data)?;
    let holt_forecast = holt_model.forecast(10)?;
    println!("\nHolt forecast (next 10 periods): {:?}", &holt_forecast[..5]);

    // Example 3: Holt-Winters (Seasonal data)
    println!("\n3. Holt-Winters (Seasonal)");
    println!("==========================");

    // Generate seasonal data with trend
    let seasonal_values: Vec<f64> = (0..100)
        .map(|i| {
            let level = 40.0;
            let trend = 0.2 * i as f64;
            let seasonal = 8.0 * (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin(); // Monthly seasonality
            let noise = (rand::random::<f64>() - 0.5) * 3.0;
            level + trend + seasonal + noise
        })
        .collect();

    let seasonal_data = TimeSeriesData::new(timestamps.clone(), seasonal_values.clone(), "seasonal_series")?;

    println!("Generated seasonal data (12-period cycle, trend, noise)");
    println!("Data range: {:.2} to {:.2}", 
             seasonal_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             seasonal_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

    let train_seasonal = TimeSeriesData::new(
        timestamps[..split_idx].to_vec(),
        seasonal_values[..split_idx].to_vec(),
        "seasonal_train"
    )?;
    let test_seasonal = TimeSeriesData::new(
        timestamps[split_idx..].to_vec(),
        seasonal_values[split_idx..].to_vec(),
        "seasonal_test"
    )?;

    // Test Holt-Winters with different parameters
    let hw_params = vec![
        (0.3, 0.1, 0.1, "Conservative"),
        (0.5, 0.2, 0.2, "Moderate"),
        (0.7, 0.3, 0.3, "Aggressive"),
        (0.9, 0.1, 0.1, "High alpha"),
        (0.3, 0.5, 0.1, "High beta"),
        (0.3, 0.1, 0.5, "High gamma"),
    ];

    println!("\nTesting different parameter combinations for Holt-Winters:");

    for (alpha, beta, gamma, description) in hw_params {
        match HoltWintersModel::new(alpha, beta, gamma, 12) { // 12-period seasonality
            Ok(mut model) => {
                match model.fit(&train_seasonal) {
                    Ok(_) => {
                        match model.evaluate(&test_seasonal) {
                            Ok(eval) => {
                                println!("  {} (α={}, β={}, γ={}): RMSE = {:.3}, MAE = {:.3}", 
                                        description, alpha, beta, gamma, eval.rmse, eval.mae);
                            }
                            Err(_) => println!("  {}: Evaluation failed", description),
                        }
                    }
                    Err(_) => println!("  {}: Fit failed", description),
                }
            }
            Err(_) => println!("  {}: Model creation failed", description),
        }
    }

    // Use optimal parameters for forecasting
    let mut hw_model = HoltWintersModel::new(0.5, 0.2, 0.2, 12)?;
    hw_model.fit(&seasonal_data)?;
    let hw_forecast = hw_model.forecast(24)?; // Forecast 2 cycles
    println!("\nHolt-Winters forecast (next 24 periods, first 12): {:?}", &hw_forecast[..12]);

    // Example 4: Model Comparison
    println!("\n4. Model Comparison on Same Dataset");
    println!("===================================");

    // Test all models on the seasonal data to see which performs best
    println!("Comparing all ES models on seasonal data:");

    // SES
    if let Ok(mut ses) = SimpleESModel::new(0.3) {
        if ses.fit(&train_seasonal).is_ok() {
            if let Ok(eval) = ses.evaluate(&test_seasonal) {
                println!("  Simple ES:     RMSE = {:.3}, MAE = {:.3}", eval.rmse, eval.mae);
            }
        }
    }

    // Holt
    if let Ok(mut holt) = HoltLinearModel::new(0.5, 0.2) {
        if holt.fit(&train_seasonal).is_ok() {
            if let Ok(eval) = holt.evaluate(&test_seasonal) {
                println!("  Holt Linear:   RMSE = {:.3}, MAE = {:.3}", eval.rmse, eval.mae);
            }
        }
    }

    // Holt-Winters
    if let Ok(mut hw) = HoltWintersModel::new(0.5, 0.2, 0.2, 12) {
        if hw.fit(&train_seasonal).is_ok() {
            if let Ok(eval) = hw.evaluate(&test_seasonal) {
                println!("  Holt-Winters:  RMSE = {:.3}, MAE = {:.3}", eval.rmse, eval.mae);
            }
        }
    }

    // Example 5: Business Application - Sales Forecasting
    println!("\n5. Business Application - Sales Forecasting");
    println!("============================================");

    // Simulate monthly sales data with seasonality and growth
    let monthly_sales: Vec<f64> = (0..36) // 3 years of monthly data
        .map(|i| {
            let base_level = 1000.0;
            let growth = 20.0 * i as f64; // 20 units growth per month
            let seasonal_factor = match i % 12 {
                11 | 0 | 1 => 1.4,  // Holiday season boost
                5 | 6 | 7 => 1.2,   // Summer boost
                2 | 3 | 9 => 0.9,   // Slower months
                _ => 1.0,           // Normal months
            };
            let noise = (rand::random::<f64>() - 0.5) * 50.0;
            (base_level + growth) * seasonal_factor + noise
        })
        .collect();

    let sales_timestamps: Vec<DateTime<Utc>> = (0..36)
        .map(|i| start_date + Duration::days(i * 30)) // Monthly intervals
        .collect();

    let sales_data = TimeSeriesData::new(sales_timestamps, monthly_sales.clone(), "monthly_sales")?;

    println!("Monthly sales data generated (3 years, seasonal patterns)");
    println!("Average monthly sales: {:.0}", 
             monthly_sales.iter().sum::<f64>() / monthly_sales.len() as f64);

    // Forecast next 6 months using Holt-Winters
    let mut sales_model = HoltWintersModel::new(0.3, 0.1, 0.3, 12)?;
    sales_model.fit(&sales_data)?;
    let sales_forecast = sales_model.forecast(6)?;

    println!("Sales forecast for next 6 months:");
    for (i, &forecast) in sales_forecast.iter().enumerate() {
        println!("  Month {}: {:.0} units", i + 1, forecast);
    }

    let total_forecast = sales_forecast.iter().sum::<f64>();
    println!("Total forecast for next 6 months: {:.0} units", total_forecast);

    // Example 6: Quick API comparison
    println!("\n6. Quick API Comparison");
    println!("=======================");

    use oxidiviner::quick;

    let quick_es_forecast = quick::es_forecast(timestamps.clone(), seasonal_values.clone(), 10)?;
    println!("Quick API ES forecast (first 5 values): {:?}", &quick_es_forecast[..5]);

    println!("\n=== Exponential Smoothing Example Complete ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_smoothing_example() {
        let result = main();
        assert!(result.is_ok(), "Exponential Smoothing example should run successfully: {:?}", result);
    }
} 