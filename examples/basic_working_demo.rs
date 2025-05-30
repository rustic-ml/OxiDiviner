//! Basic Working Demo
//!
//! This example demonstrates the core OxiDiviner functionality using
//! the basic models directly, without the complex API layer.

use chrono::{DateTime, TimeZone, Utc};
use oxidiviner::core::{Result, TimeSeriesData};
use oxidiviner::models::autoregressive::{ARIMAModel, ARModel};
use oxidiviner::models::exponential_smoothing::SimpleESModel;
use oxidiviner::models::moving_average::MAModel;
use oxidiviner::quick;

fn main() -> Result<()> {
    println!("🚀 OxiDiviner Basic Working Demo");
    println!("==================================\n");

    // Create sample time series data
    let dates = generate_sample_dates(50);
    let values = generate_sample_data(50);
    let data = TimeSeriesData::new(dates, values, "sample_series")?;

    println!("📊 Sample data created with {} points", data.len());
    println!("First 5 values: {:?}", &data.values[0..5]);
    println!();

    // Demo 1: Quick functions
    demo_quick_functions(&data)?;

    // Demo 2: Direct model usage
    demo_direct_models(&data)?;

    // Demo 3: Model evaluation
    demo_model_evaluation(&data)?;

    println!("✅ All demos completed successfully!");
    Ok(())
}

fn demo_quick_functions(data: &TimeSeriesData) -> Result<()> {
    println!("🔧 Demo 1: Quick Functions");
    println!("--------------------------");

    // Quick ARIMA forecast (default parameters)
    let arima_forecast = quick::arima(data.clone(), 5)?;
    println!("ARIMA(1,1,1) forecast (5 periods): {:?}", arima_forecast);

    // Quick ARIMA forecast with custom parameters
    let arima_custom_forecast = quick::arima_with_config(data.clone(), 5, Some((2, 1, 1)))?;
    println!(
        "ARIMA(2,1,1) forecast (5 periods): {:?}",
        arima_custom_forecast
    );

    // Quick AR forecast
    let ar_forecast = quick::ar(data.clone(), 5, Some(3))?;
    println!("AR(3) forecast (5 periods): {:?}", ar_forecast);

    // Quick Exponential Smoothing forecast
    let es_forecast = quick::exponential_smoothing(data.clone(), 5, Some(0.3))?;
    println!("ES(α=0.3) forecast (5 periods): {:?}", es_forecast);

    // Quick Moving Average forecast
    let ma_forecast = quick::moving_average(data.clone(), 5, Some(5))?;
    println!("MA(5) forecast (5 periods): {:?}", ma_forecast);

    // Auto-select best model
    let (auto_forecast, best_model) = quick::auto_select(data.clone(), 5)?;
    println!(
        "Auto-selected {} forecast (5 periods): {:?}",
        best_model, auto_forecast
    );

    println!();
    Ok(())
}

fn demo_direct_models(data: &TimeSeriesData) -> Result<()> {
    println!("🎯 Demo 2: Direct Model Usage");
    println!("------------------------------");

    // AR Model
    let mut ar_model = ARModel::new(2, true)?;
    ar_model.fit(data)?;
    let ar_forecast = ar_model.forecast(3)?;
    println!("AR(2) model forecast: {:?}", ar_forecast);

    // ARIMA Model
    let mut arima_model = ARIMAModel::new(1, 1, 1, true)?;
    arima_model.fit(data)?;
    let arima_forecast = arima_model.forecast(3)?;
    println!("ARIMA(1,1,1) model forecast: {:?}", arima_forecast);

    // Simple Exponential Smoothing
    let mut es_model = SimpleESModel::new(0.2)?;
    es_model.fit(data)?;
    let es_forecast = es_model.forecast(3)?;
    println!("Simple ES(α=0.2) forecast: {:?}", es_forecast);

    // Moving Average
    let mut ma_model = MAModel::new(7)?;
    ma_model.fit(data)?;
    let ma_forecast = ma_model.forecast(3)?;
    println!("MA(7) forecast: {:?}", ma_forecast);

    println!();
    Ok(())
}

fn demo_model_evaluation(data: &TimeSeriesData) -> Result<()> {
    println!("📈 Demo 3: Model Evaluation");
    println!("----------------------------");

    // Split data for evaluation
    let split_point = data.len() * 3 / 4;
    let train_data = data.slice(0, split_point)?;
    let test_data = data.slice(split_point, data.len())?;

    println!("Training data: {} points", train_data.len());
    println!("Test data: {} points", test_data.len());

    // Evaluate AR model
    let mut ar_model = ARModel::new(2, true)?;
    ar_model.fit(&train_data)?;
    let ar_eval = ar_model.evaluate(&test_data)?;
    println!("\nAR(2) Evaluation:");
    println!("  MAE: {:.4}", ar_eval.mae);
    println!("  MSE: {:.4}", ar_eval.mse);
    println!("  RMSE: {:.4}", ar_eval.rmse);
    println!("  R²: {:.4}", ar_eval.r_squared);

    // Evaluate ARIMA model
    let mut arima_model = ARIMAModel::new(1, 1, 1, true)?;
    arima_model.fit(&train_data)?;
    let arima_eval = arima_model.evaluate(&test_data)?;
    println!("\nARIMA(1,1,1) Evaluation:");
    println!("  MAE: {:.4}", arima_eval.mae);
    println!("  MSE: {:.4}", arima_eval.mse);
    println!("  RMSE: {:.4}", arima_eval.rmse);
    println!("  R²: {:.4}", arima_eval.r_squared);

    // Evaluate ES model
    let mut es_model = SimpleESModel::new(0.3)?;
    es_model.fit(&train_data)?;
    let es_eval = es_model.evaluate(&test_data)?;
    println!("\nSimple ES(α=0.3) Evaluation:");
    println!("  MAE: {:.4}", es_eval.mae);
    println!("  MSE: {:.4}", es_eval.mse);
    println!("  RMSE: {:.4}", es_eval.rmse);
    println!("  R²: {:.4}", es_eval.r_squared);

    println!();
    Ok(())
}

fn generate_sample_dates(n: usize) -> Vec<DateTime<Utc>> {
    (0..n)
        .map(|i| {
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i as i64)
        })
        .collect()
}

fn generate_sample_data(n: usize) -> Vec<f64> {
    // Generate a simple trend + noise time series
    (0..n)
        .map(|i| {
            let trend = 100.0 + 0.5 * i as f64;
            let seasonal = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            let noise = (i as f64 * 0.1).sin() * 2.0;
            trend + seasonal + noise
        })
        .collect()
}
