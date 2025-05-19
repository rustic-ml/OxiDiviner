use chrono::{DateTime, Duration, Utc};
use oxidiviner_autoregressive::{ARIMAModel, ARMAModel, ARModel, SARIMAModel, VARModel};
use oxidiviner_core::{ModelEvaluation, TimeSeriesData};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use std::f64::consts::PI;

fn main() {
    println!("OxiDiviner Autoregressive Models Demo");
    println!("=====================================\n");

    // Create sample data
    let now = Utc::now();
    let n = 100; // Number of data points

    // Generate timestamps (daily data)
    let timestamps: Vec<DateTime<Utc>> = (0..n).map(|i| now + Duration::days(i as i64)).collect();

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
    // Create a seeded RNG for reproducible results
    let mut rng = StdRng::seed_from_u64(42);

    // Add some random noise to break perfect correlation
    let y1_values: Vec<f64> = trend_values
        .iter()
        .map(|&x| x + rng.gen::<f64>() * 4.0 - 2.0)
        .collect();

    // Make y2 correlated but not perfectly collinear with y1
    let y2_values: Vec<f64> = y1_values
        .iter()
        .map(|&x| 2.0 * x + 5.0 + rng.gen::<f64>() * 6.0 - 3.0)
        .collect();

    // Create TimeSeriesData objects
    let trend_data = TimeSeriesData::new(timestamps.clone(), trend_values, "trend_series").unwrap();
    let seasonal_data =
        TimeSeriesData::new(timestamps.clone(), seasonal_values, "seasonal_series").unwrap();
    let y1_data = TimeSeriesData::new(timestamps.clone(), y1_values, "y1").unwrap();
    let y2_data = TimeSeriesData::new(timestamps.clone(), y2_values, "y2").unwrap();

    // Split data into training and test sets
    let train_size = 80;

    // Create training and test data using subset of the original data
    let train_trend_timestamps = timestamps[0..train_size].to_vec();
    let train_trend_values = trend_data.values[0..train_size].to_vec();
    let train_trend = TimeSeriesData::new(
        train_trend_timestamps.clone(),
        train_trend_values,
        "train_trend",
    )
    .unwrap();

    let test_trend_timestamps = timestamps[train_size..].to_vec();
    let test_trend_values = trend_data.values[train_size..].to_vec();
    let test_trend = TimeSeriesData::new(
        test_trend_timestamps.clone(),
        test_trend_values,
        "test_trend",
    )
    .unwrap();

    let train_seasonal_values = seasonal_data.values[0..train_size].to_vec();
    let train_seasonal = TimeSeriesData::new(
        train_trend_timestamps.clone(),
        train_seasonal_values,
        "train_seasonal",
    )
    .unwrap();

    let test_seasonal_values = seasonal_data.values[train_size..].to_vec();
    let test_seasonal = TimeSeriesData::new(
        test_trend_timestamps.clone(),
        test_seasonal_values,
        "test_seasonal",
    )
    .unwrap();

    let train_y1_values = y1_data.values[0..train_size].to_vec();
    let train_y1 =
        TimeSeriesData::new(train_trend_timestamps.clone(), train_y1_values, "train_y1").unwrap();

    let test_y1_values = y1_data.values[train_size..].to_vec();
    let test_y1 =
        TimeSeriesData::new(test_trend_timestamps.clone(), test_y1_values, "test_y1").unwrap();

    let train_y2_values = y2_data.values[0..train_size].to_vec();
    let train_y2 =
        TimeSeriesData::new(train_trend_timestamps.clone(), train_y2_values, "train_y2").unwrap();

    let test_y2_values = y2_data.values[train_size..].to_vec();
    let test_y2 =
        TimeSeriesData::new(test_trend_timestamps.clone(), test_y2_values, "test_y2").unwrap();

    // Test horizon
    let horizon = n - train_size;

    // Demo 1: AR Model
    println!("\n1. Autoregressive (AR) Model");
    println!("---------------------------");
    let mut ar_model = ARModel::new(2, true).unwrap();
    ar_model.fit(&train_trend).unwrap();

    let _ar_forecast = ar_model.forecast(horizon).unwrap();
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

    let _arma_forecast = arma_model.forecast(horizon).unwrap();
    let arma_eval = arma_model.evaluate(&test_trend).unwrap();

    println!("ARMA(2,1) Evaluation:");
    println!("  MAE:   {:.4}", arma_eval.mae);
    println!("  RMSE:  {:.4}", arma_eval.rmse);
    println!("  MAPE:  {:.4}%", arma_eval.mape * 100.0);

    // Create a modified dataset that's more suitable for ARIMA modeling
    // The key is to make it non-stationary (with trend) but with some randomness
    let arima_train_values: Vec<f64> = (0..train_size)
        .map(|i| {
            let trend = (i as f64) * 2.0;
            let noise = rng.gen::<f64>() * 10.0 - 5.0;
            trend + noise
        })
        .collect();

    let arima_test_values: Vec<f64> = (train_size..n)
        .map(|i| {
            let trend = (i as f64) * 2.0;
            let noise = rng.gen::<f64>() * 10.0 - 5.0;
            trend + noise
        })
        .collect();

    let arima_train = TimeSeriesData::new(
        train_trend_timestamps.clone(),
        arima_train_values,
        "arima_train",
    )
    .unwrap();
    let arima_test = TimeSeriesData::new(
        test_trend_timestamps.clone(),
        arima_test_values,
        "arima_test",
    )
    .unwrap();

    // Create suitable seasonal data for SARIMA
    let sarima_train_values: Vec<f64> = (0..train_size)
        .map(|i| {
            let trend = (i as f64) * 0.5;
            let seasonal = 10.0 * (2.0 * PI * (i % 12) as f64 / 12.0).sin();
            let noise = rng.gen::<f64>() * 4.0 - 2.0;
            trend + seasonal + noise
        })
        .collect();

    let sarima_test_values: Vec<f64> = (0..(n - train_size))
        .map(|i| {
            let j = i + train_size;
            let trend = (j as f64) * 0.5;
            let seasonal = 10.0 * (2.0 * PI * (j % 12) as f64 / 12.0).sin();
            let noise = rng.gen::<f64>() * 4.0 - 2.0;
            trend + seasonal + noise
        })
        .collect();

    let sarima_train = TimeSeriesData::new(
        train_trend_timestamps.clone(),
        sarima_train_values,
        "sarima_train",
    )
    .unwrap();
    let sarima_test = TimeSeriesData::new(
        test_trend_timestamps.clone(),
        sarima_test_values,
        "sarima_test",
    )
    .unwrap();

    // Demo 3: ARIMA Model
    println!("\n3. Autoregressive Integrated Moving Average (ARIMA) Model");
    println!("------------------------------------------------------");
    let mut arima_model = ARIMAModel::new(2, 1, 1, true).unwrap();
    arima_model.fit(&arima_train).unwrap();

    let _arima_forecast = arima_model.forecast(horizon).unwrap();
    let arima_eval = arima_model.evaluate(&arima_test).unwrap();

    println!("ARIMA(2,1,1) Evaluation:");
    println!("  MAE:   {:.4}", arima_eval.mae);
    println!("  RMSE:  {:.4}", arima_eval.rmse);
    println!("  MAPE:  {:.4}%", arima_eval.mape * 100.0);

    // Demo 4: SARIMA Model
    println!("\n4. Seasonal ARIMA (SARIMA) Model");
    println!("------------------------------");
    let mut sarima_model = SARIMAModel::new(1, 0, 0, 1, 1, 0, 12, true).unwrap();
    sarima_model.fit(&sarima_train).unwrap();

    let _sarima_forecast = sarima_model.forecast(horizon).unwrap();
    let sarima_eval = sarima_model.evaluate(&sarima_test).unwrap();

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

    let _var_forecasts = var_model.forecast_multiple(horizon).unwrap();
    let var_evals = var_model.evaluate_multiple(&test_data_map).unwrap();

    println!("VAR(1) Evaluation for y1:");
    println!("  MAE:   {:.4}", var_evals["y1"].mae);
    println!("  RMSE:  {:.4}", var_evals["y1"].rmse);
    println!("  MAPE:  {:.4}%", var_evals["y1"].mape * 100.0);

    println!("\nVAR(1) Evaluation for y2:");
    println!("  MAE:   {:.4}", var_evals["y2"].mae);
    println!("  RMSE:  {:.4}", var_evals["y2"].rmse);
    println!("  MAPE:  {:.4}%", var_evals["y2"].mape * 100.0);

    // Compare all models
    println!("\nModel Comparison (RMSE)");
    println!("----------------------");
    println!("AR(2)                 : {:.4}", ar_eval.rmse);
    println!("ARMA(2,1)             : {:.4}", arma_eval.rmse);
    println!("ARIMA(2,1,1)          : {:.4}", arima_eval.rmse);
    println!("SARIMA(1,0,0)(1,1,0)12: {:.4}", sarima_eval.rmse);
    println!("VAR(1) - y1           : {:.4}", var_evals["y1"].rmse);
    println!("VAR(1) - y2           : {:.4}", var_evals["y2"].rmse);

    println!("\nDemo complete!");
}
