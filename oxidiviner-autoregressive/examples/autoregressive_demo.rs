use oxidiviner_autoregressive::{ARIMAModel, ARMAModel, ARModel, SARIMAModel, VARModel};
use oxidiviner_core::TimeSeriesData;
use std::collections::HashMap;
use std::error::Error;
use std::f64::consts::PI;

fn main() -> std::result::Result<(), Box<dyn Error>> {
    println!("Autoregressive Models Example");
    println!("============================\n");

    // Generate some test data
    let n = 120;
    let train_size = 100;

    // Generate trend, seasonal, and two response variables
    let (trend_data, seasonal_data, y1_data, y2_data) = generate_test_data(n);

    // Split into training and test sets
    let train_trend = slice_time_series(&trend_data, 0, train_size);
    let test_trend = slice_time_series(&trend_data, train_size, n);

    let train_seasonal = slice_time_series(&seasonal_data, 0, train_size);
    let test_seasonal = slice_time_series(&seasonal_data, train_size, n);

    let train_y1 = slice_time_series(&y1_data, 0, train_size);
    let test_y1 = slice_time_series(&y1_data, train_size, n);

    let train_y2 = slice_time_series(&y2_data, 0, train_size);
    let test_y2 = slice_time_series(&y2_data, train_size, n);

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
    Ok(())
}

// Helper function to slice a time series
fn slice_time_series(data: &TimeSeriesData, start: usize, end: usize) -> TimeSeriesData {
    TimeSeriesData {
        timestamps: data.timestamps[start..end].to_vec(),
        values: data.values[start..end].to_vec(),
        name: data.name.clone(),
    }
}

// Generate test data for the AR models
fn generate_test_data(
    n: usize,
) -> (
    TimeSeriesData,
    TimeSeriesData,
    TimeSeriesData,
    TimeSeriesData,
) {
    let now = chrono::Utc::now();

    // Generate timestamps (daily data)
    let timestamps: Vec<chrono::DateTime<chrono::Utc>> = (0..n)
        .map(|i| now + chrono::Duration::days(i as i64))
        .collect();

    // 1. Linear trend data for AR, ARMA, and ARIMA models
    let trend_values: Vec<f64> = (1..=n as i64).map(|i| i as f64).collect();

    // 2. Seasonal data for SARIMA
    let mut seasonal_values = Vec::with_capacity(n);
    for i in 0..n {
        let trend = i as f64 * 0.5; // Small upward trend
        let seasonal = 10.0 * (2.0 * PI * (i % 12) as f64 / 12.0).sin(); // Seasonal component
        seasonal_values.push(trend + seasonal);
    }

    // 3. Two related series for VAR
    let y1_values = trend_values.clone();
    let y2_values: Vec<f64> = trend_values.iter().map(|&x| 2.0 * x + 5.0).collect();

    // Create TimeSeriesData objects
    let trend_data = TimeSeriesData {
        timestamps: timestamps.clone(),
        values: trend_values,
        name: "trend_series".to_string(),
    };

    let seasonal_data = TimeSeriesData {
        timestamps: timestamps.clone(),
        values: seasonal_values,
        name: "seasonal_series".to_string(),
    };

    let y1_data = TimeSeriesData {
        timestamps: timestamps.clone(),
        values: y1_values,
        name: "y1".to_string(),
    };

    let y2_data = TimeSeriesData {
        timestamps: timestamps.clone(),
        values: y2_values,
        name: "y2".to_string(),
    };

    (trend_data, seasonal_data, y1_data, y2_data)
}
