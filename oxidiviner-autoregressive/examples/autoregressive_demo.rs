use chrono::{DateTime, Duration, Utc};
use oxidiviner_autoregressive::{ARIMAModel, ARMAModel, ARModel, SARIMAModel, VARModel};
use oxidiviner_core::TimeSeriesData;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use std::f64::consts::PI;

fn main() {
    println!("Autoregressive Models Example");
    println!("============================\n");

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
        .map(|&x| x + rng.random_range(0.0..1.0) * 4.0 - 2.0)
        .collect();

    // Make y2 correlated but not perfectly collinear with y1
    let y2_values: Vec<f64> = y1_values
        .iter()
        .map(|&x| 2.0 * x + 5.0 + rng.random_range(0.0..1.0) * 6.0 - 3.0)
        .collect();

    // Create TimeSeriesData objects
    let trend_data = match TimeSeriesData::new(timestamps.clone(), trend_values, "trend_series") {
        Ok(data) => data,
        Err(e) => {
            println!("Error creating trend data: {}", e);
            return;
        }
    };

    let _seasonal_data =
        match TimeSeriesData::new(timestamps.clone(), seasonal_values, "seasonal_series") {
            Ok(data) => data,
            Err(e) => {
                println!("Error creating seasonal data: {}", e);
                return;
            }
        };

    let y1_data = match TimeSeriesData::new(timestamps.clone(), y1_values, "y1") {
        Ok(data) => data,
        Err(e) => {
            println!("Error creating y1 data: {}", e);
            return;
        }
    };

    let y2_data = match TimeSeriesData::new(timestamps.clone(), y2_values, "y2") {
        Ok(data) => data,
        Err(e) => {
            println!("Error creating y2 data: {}", e);
            return;
        }
    };

    // Split data into training and test sets
    let train_size = 80;

    // Create training and test data using subset of the original data
    let train_trend_timestamps = timestamps[0..train_size].to_vec();
    let train_trend_values = trend_data.values[0..train_size].to_vec();
    let train_trend = match TimeSeriesData::new(
        train_trend_timestamps.clone(),
        train_trend_values,
        "train_trend",
    ) {
        Ok(data) => data,
        Err(e) => {
            println!("Error creating train trend data: {}", e);
            return;
        }
    };

    let test_trend_timestamps = timestamps[train_size..].to_vec();
    let test_trend_values = trend_data.values[train_size..].to_vec();
    let test_trend = match TimeSeriesData::new(
        test_trend_timestamps.clone(),
        test_trend_values,
        "test_trend",
    ) {
        Ok(data) => data,
        Err(e) => {
            println!("Error creating test trend data: {}", e);
            return;
        }
    };

    let train_y1_values = y1_data.values[0..train_size].to_vec();
    let train_y1 =
        match TimeSeriesData::new(train_trend_timestamps.clone(), train_y1_values, "train_y1") {
            Ok(data) => data,
            Err(e) => {
                println!("Error creating train y1 data: {}", e);
                return;
            }
        };

    let test_y1_values = y1_data.values[train_size..].to_vec();
    let test_y1 =
        match TimeSeriesData::new(test_trend_timestamps.clone(), test_y1_values, "test_y1") {
            Ok(data) => data,
            Err(e) => {
                println!("Error creating test y1 data: {}", e);
                return;
            }
        };

    let train_y2_values = y2_data.values[0..train_size].to_vec();
    let train_y2 =
        match TimeSeriesData::new(train_trend_timestamps.clone(), train_y2_values, "train_y2") {
            Ok(data) => data,
            Err(e) => {
                println!("Error creating train y2 data: {}", e);
                return;
            }
        };

    let test_y2_values = y2_data.values[train_size..].to_vec();
    let test_y2 =
        match TimeSeriesData::new(test_trend_timestamps.clone(), test_y2_values, "test_y2") {
            Ok(data) => data,
            Err(e) => {
                println!("Error creating test y2 data: {}", e);
                return;
            }
        };

    // Test horizon
    let horizon = n - train_size;

    // Demo 1: AR Model
    println!("\n1. Autoregressive (AR) Model");
    println!("---------------------------");
    let mut ar_model = match ARModel::new(2, true) {
        Ok(model) => model,
        Err(e) => {
            println!("Error creating AR model: {}", e);
            return;
        }
    };

    if let Err(e) = ar_model.fit(&train_trend) {
        println!("Error fitting AR model: {}", e);
        return;
    }

    let _ar_forecast = match ar_model.forecast(horizon) {
        Ok(forecast) => forecast,
        Err(e) => {
            println!("Error forecasting with AR model: {}", e);
            return;
        }
    };

    let ar_eval = match ar_model.evaluate(&test_trend) {
        Ok(eval) => eval,
        Err(e) => {
            println!("Error evaluating AR model: {}", e);
            return;
        }
    };

    println!("AR(2) Evaluation:");
    println!("  MAE:   {:.4}", ar_eval.mae);
    println!("  RMSE:  {:.4}", ar_eval.rmse);
    println!("  MAPE:  {:.4}%", ar_eval.mape * 100.0);

    // Demo 2: ARMA Model
    println!("\n2. Autoregressive Moving Average (ARMA) Model");
    println!("-------------------------------------------");
    let mut arma_model = match ARMAModel::new(2, 1, true) {
        Ok(model) => model,
        Err(e) => {
            println!("Error creating ARMA model: {}", e);
            return;
        }
    };

    if let Err(e) = arma_model.fit(&train_trend) {
        println!("Error fitting ARMA model: {}", e);
        return;
    }

    let _arma_forecast = match arma_model.forecast(horizon) {
        Ok(forecast) => forecast,
        Err(e) => {
            println!("Error forecasting with ARMA model: {}", e);
            return;
        }
    };

    let arma_eval = match arma_model.evaluate(&test_trend) {
        Ok(eval) => eval,
        Err(e) => {
            println!("Error evaluating ARMA model: {}", e);
            return;
        }
    };

    println!("ARMA(2,1) Evaluation:");
    println!("  MAE:   {:.4}", arma_eval.mae);
    println!("  RMSE:  {:.4}", arma_eval.rmse);
    println!("  MAPE:  {:.4}%", arma_eval.mape * 100.0);

    // Create a modified dataset that's more suitable for ARIMA modeling
    // The key is to make it non-stationary (with trend) but with some randomness
    let arima_train_values: Vec<f64> = (0..train_size)
        .map(|i| {
            let trend = (i as f64) * 2.0;
            let noise = rng.random_range(-5.0..5.0);
            trend + noise
        })
        .collect();

    let arima_test_values: Vec<f64> = (train_size..n)
        .map(|i| {
            let trend = (i as f64) * 2.0;
            let noise = rng.random_range(-5.0..5.0);
            trend + noise
        })
        .collect();

    let arima_train = match TimeSeriesData::new(
        train_trend_timestamps.clone(),
        arima_train_values,
        "arima_train",
    ) {
        Ok(data) => data,
        Err(e) => {
            println!("Error creating ARIMA training data: {}", e);
            return;
        }
    };

    let arima_test = match TimeSeriesData::new(
        test_trend_timestamps.clone(),
        arima_test_values,
        "arima_test",
    ) {
        Ok(data) => data,
        Err(e) => {
            println!("Error creating ARIMA test data: {}", e);
            return;
        }
    };

    // Create suitable seasonal data for SARIMA
    let sarima_train_values: Vec<f64> = (0..train_size)
        .map(|i| {
            let trend = (i as f64) * 0.5;
            let seasonal = 10.0 * (2.0 * PI * (i % 12) as f64 / 12.0).sin();
            let noise = rng.random_range(-2.0..2.0);
            trend + seasonal + noise
        })
        .collect();

    let sarima_test_values: Vec<f64> = (0..(n - train_size))
        .map(|i| {
            let j = i + train_size;
            let trend = (j as f64) * 0.5;
            let seasonal = 10.0 * (2.0 * PI * (j % 12) as f64 / 12.0).sin();
            let noise = rng.random_range(-2.0..2.0);
            trend + seasonal + noise
        })
        .collect();

    let sarima_train = match TimeSeriesData::new(
        train_trend_timestamps.clone(),
        sarima_train_values,
        "sarima_train",
    ) {
        Ok(data) => data,
        Err(e) => {
            println!("Error creating SARIMA training data: {}", e);
            return;
        }
    };

    let sarima_test = match TimeSeriesData::new(
        test_trend_timestamps.clone(),
        sarima_test_values,
        "sarima_test",
    ) {
        Ok(data) => data,
        Err(e) => {
            println!("Error creating SARIMA test data: {}", e);
            return;
        }
    };

    // Demo 3: ARIMA Model
    println!("\n3. Autoregressive Integrated Moving Average (ARIMA) Model");
    println!("------------------------------------------------------");
    let mut arima_model = match ARIMAModel::new(1, 1, 1, true) {
        Ok(model) => model,
        Err(e) => {
            println!("Error creating ARIMA model: {}", e);
            return;
        }
    };

    match arima_model.fit(&arima_train) {
        Ok(_) => match arima_model.forecast(horizon) {
            Ok(_arima_forecast) => match arima_model.evaluate(&arima_test) {
                Ok(arima_eval) => {
                    println!("ARIMA(1,1,1) Evaluation:");
                    println!("  MAE:   {:.4}", arima_eval.mae);
                    println!("  RMSE:  {:.4}", arima_eval.rmse);
                    println!("  MAPE:  {:.4}%", arima_eval.mape * 100.0);
                }
                Err(e) => println!("Error evaluating ARIMA model: {}", e),
            },
            Err(e) => println!("Error forecasting with ARIMA model: {}", e),
        },
        Err(e) => println!("Error fitting ARIMA model: {}", e),
    }

    // Demo 4: SARIMA Model
    println!("\n4. Seasonal ARIMA (SARIMA) Model");
    println!("------------------------------");
    let mut sarima_model = match SARIMAModel::new(0, 1, 1, 0, 1, 1, 12, true) {
        Ok(model) => model,
        Err(e) => {
            println!("Error creating SARIMA model: {}", e);
            return;
        }
    };

    match sarima_model.fit(&sarima_train) {
        Ok(_) => match sarima_model.forecast(horizon) {
            Ok(_sarima_forecast) => match sarima_model.evaluate(&sarima_test) {
                Ok(sarima_eval) => {
                    println!("SARIMA(0,1,1)(0,1,1)12 Evaluation:");
                    println!("  MAE:   {:.4}", sarima_eval.mae);
                    println!("  RMSE:  {:.4}", sarima_eval.rmse);
                    println!("  MAPE:  {:.4}%", sarima_eval.mape * 100.0);
                }
                Err(e) => println!("Error evaluating SARIMA model: {}", e),
            },
            Err(e) => println!("Error forecasting with SARIMA model: {}", e),
        },
        Err(e) => println!("Error fitting SARIMA model: {}", e),
    }

    // Demo 5: VAR Model
    println!("\n5. Vector Autoregression (VAR) Model");
    println!("----------------------------------");
    let variable_names = vec!["y1".to_string(), "y2".to_string()];
    let mut var_model = match VARModel::new(1, variable_names, true) {
        Ok(model) => model,
        Err(e) => {
            println!("Error creating VAR model: {}", e);
            return;
        }
    };

    let mut train_data_map = HashMap::new();
    train_data_map.insert("y1".to_string(), train_y1);
    train_data_map.insert("y2".to_string(), train_y2);

    let mut test_data_map = HashMap::new();
    test_data_map.insert("y1".to_string(), test_y1);
    test_data_map.insert("y2".to_string(), test_y2);

    if let Err(e) = var_model.fit_multiple(&train_data_map) {
        println!("Error fitting VAR model: {}", e);
        return;
    }

    let _var_forecasts = match var_model.forecast_multiple(horizon) {
        Ok(forecasts) => forecasts,
        Err(e) => {
            println!("Error forecasting with VAR model: {}", e);
            return;
        }
    };

    match var_model.evaluate_multiple(&test_data_map) {
        Ok(var_evals) => {
            println!("VAR(1) Evaluation for y1:");
            println!("  MAE:   {:.4}", var_evals["y1"].mae);
            println!("  RMSE:  {:.4}", var_evals["y1"].rmse);
            println!("  MAPE:  {:.4}%", var_evals["y1"].mape * 100.0);

            println!("\nVAR(1) Evaluation for y2:");
            println!("  MAE:   {:.4}", var_evals["y2"].mae);
            println!("  RMSE:  {:.4}", var_evals["y2"].rmse);
            println!("  MAPE:  {:.4}%", var_evals["y2"].mape * 100.0);
        }
        Err(e) => println!("Error evaluating VAR model: {}", e),
    }
}
