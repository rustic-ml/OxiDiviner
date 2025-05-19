use oxidiviner_autoregressive::{ARModel, ARMAModel, ARIMAModel, SARIMAModel, VARModel};
use oxidiviner_core::{TimeSeriesData, Forecaster};
use chrono::{DateTime, Utc, Duration};
use std::collections::HashMap;
use std::f64::consts::PI;

fn main() {
    println!("OxiDiviner Autoregressive Models Demo");
    println!("=====================================\n");
    
    // Create sample data
    let now = Utc::now();
    let n = 100; // Number of data points
    
    // Generate timestamps (daily data)
    let timestamps: Vec<DateTime<Utc>> = (0..n)
        .map(|i| now + Duration::days(i as i64))
        .collect();
    
    // 1. Linear trend data for AR, ARMA, and ARIMA models
    let mut trend_values: Vec<f64> = (1..=n as i64).map(|i| i as f64).collect();
    
    // 2. Seasonal data for SARIMA
    let mut seasonal_values = Vec::with_capacity(n);
    for i in 0..n {
        let trend = i as f64 * 0.5; // Small upward trend
        let seasonal = 10.0 * (2.0 * PI * (i % 12) as f64 / 12.0).sin(); // Seasonal component
        seasonal_values.push(trend + seasonal);
    }
    
    // 3. Two related series for VAR
    let mut y1_values = trend_values.clone();
    let mut y2_values: Vec<f64> = trend_values.iter().map(|&x| 2.0 * x + 5.0).collect();
    
    // Create TimeSeriesData objects
    let trend_data = TimeSeriesData::new(timestamps.clone(), trend_values, "trend_series").unwrap();
    let seasonal_data = TimeSeriesData::new(timestamps.clone(), seasonal_values, "seasonal_series").unwrap();
    let y1_data = TimeSeriesData::new(timestamps.clone(), y1_values, "y1").unwrap();
    let y2_data = TimeSeriesData::new(timestamps.clone(), y2_values, "y2").unwrap();
    
    // Split data into training and test sets
    let train_size = 80;
    let train_trend = trend_data.slice(0, train_size).unwrap();
    let test_trend = trend_data.slice(train_size, n).unwrap();
    
    let train_seasonal = seasonal_data.slice(0, train_size).unwrap();
    let test_seasonal = seasonal_data.slice(train_size, n).unwrap();
    
    let train_y1 = y1_data.slice(0, train_size).unwrap();
    let test_y1 = y1_data.slice(train_size, n).unwrap();
    
    let train_y2 = y2_data.slice(0, train_size).unwrap();
    let test_y2 = y2_data.slice(train_size, n).unwrap();
    
    // Test horizon
    let horizon = n - train_size;
    
    // Demo 1: AR Model
    println!("\n1. Autoregressive (AR) Model");
    println!("---------------------------");
    let mut ar_model = ARModel::new(2, true).unwrap();
    ar_model.fit(&train_trend).unwrap();
    
    let ar_forecast = ar_model.forecast(horizon).unwrap();
    let ar_eval = ar_model.evaluate(&test_trend).unwrap();
    
    println!("AR(2) Evaluation:");
    println!("  MAE:   {:.4}", ar_eval.mae);
    println!("  RMSE:  {:.4}", ar_eval.rmse);
    println!("  MAPE:  {:.4}%", ar_eval.mape * 100.0);
    
    // Demo 2: ARMA Model
    println!("\n2. Autoregressive Moving Average (ARMA) Model");
    println!("-------------------------------------------");
    let mut arma_model = ARMAModel::new(2, 1, true).unwrap();
    arma_model.fit(&train_trend).unwrap();
    
    let arma_forecast = arma_model.forecast(horizon).unwrap();
    let arma_eval = arma_model.evaluate(&test_trend).unwrap();
    
    println!("ARMA(2,1) Evaluation:");
    println!("  MAE:   {:.4}", arma_eval.mae);
    println!("  RMSE:  {:.4}", arma_eval.rmse);
    println!("  MAPE:  {:.4}%", arma_eval.mape * 100.0);
    
    // Demo 3: ARIMA Model
    println!("\n3. Autoregressive Integrated Moving Average (ARIMA) Model");
    println!("------------------------------------------------------");
    let mut arima_model = ARIMAModel::new(1, 1, 0, true).unwrap();
    arima_model.fit(&train_trend).unwrap();
    
    let arima_forecast = arima_model.forecast(horizon).unwrap();
    let arima_eval = arima_model.evaluate(&test_trend).unwrap();
    
    println!("ARIMA(1,1,0) Evaluation:");
    println!("  MAE:   {:.4}", arima_eval.mae);
    println!("  RMSE:  {:.4}", arima_eval.rmse);
    println!("  MAPE:  {:.4}%", arima_eval.mape * 100.0);
    
    // Demo 4: SARIMA Model
    println!("\n4. Seasonal ARIMA (SARIMA) Model");
    println!("------------------------------");
    let mut sarima_model = SARIMAModel::new(1, 0, 0, 1, 1, 0, 12, true).unwrap();
    sarima_model.fit(&train_seasonal).unwrap();
    
    let sarima_forecast = sarima_model.forecast(horizon).unwrap();
    let sarima_eval = sarima_model.evaluate(&test_seasonal).unwrap();
    
    println!("SARIMA(1,0,0)(1,1,0)12 Evaluation:");
    println!("  MAE:   {:.4}", sarima_eval.mae);
    println!("  RMSE:  {:.4}", sarima_eval.rmse);
    println!("  MAPE:  {:.4}%", sarima_eval.mape * 100.0);
    
    // Demo 5: VAR Model
    println!("\n5. Vector Autoregression (VAR) Model");
    println!("----------------------------------");
    let variable_names = vec!["y1".to_string(), "y2".to_string()];
    let mut var_model = VARModel::new(1, variable_names, true).unwrap();
    
    let mut train_data_map = HashMap::new();
    train_data_map.insert("y1".to_string(), train_y1);
    train_data_map.insert("y2".to_string(), train_y2);
    
    let mut test_data_map = HashMap::new();
    test_data_map.insert("y1".to_string(), test_y1);
    test_data_map.insert("y2".to_string(), test_y2);
    
    var_model.fit_multiple(&train_data_map).unwrap();
    
    let var_forecasts = var_model.forecast_multiple(horizon).unwrap();
    let var_evals = var_model.evaluate_multiple(&test_data_map).unwrap();
    
    println!("VAR(1) Evaluation for y1:");
    println!("  MAE:   {:.4}", var_evals["y1"].mae);
    println!("  RMSE:  {:.4}", var_evals["y1"].rmse);
    println!("  MAPE:  {:.4}%", var_evals["y1"].mape * 100.0);
    
    println!("\nVAR(1) Evaluation for y2:");
    println!("  MAE:   {:.4}", var_evals["y2"].mae);
    println!("  RMSE:  {:.4}", var_evals["y2"].rmse);
    println!("  MAPE:  {:.4}%", var_evals["y2"].mape * 100.0);
    
    // Compare models on trend data
    println!("\nModel Comparison on Trend Data");
    println!("-----------------------------");
    println!("AR(2)       RMSE: {:.4}", ar_eval.rmse);
    println!("ARMA(2,1)   RMSE: {:.4}", arma_eval.rmse);
    println!("ARIMA(1,1,0) RMSE: {:.4}", arima_eval.rmse);
    
    println!("\nDemo complete!");
} 