use chrono::{DateTime, Duration, Utc};
use oxidiviner::prelude::*;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

// Helper function to generate test data with trend, seasonality, and noise
fn generate_test_data(
    n: usize,
    with_trend: bool,
    with_seasonality: bool,
    seed: u64,
) -> TimeSeriesData {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let now = Utc::now();

    let mut timestamps = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);

    for i in 0..n {
        // Create timestamp (daily intervals)
        let timestamp = now + Duration::days(i as i64);

        // Calculate value components
        let trend_component = if with_trend { 0.5 * i as f64 } else { 0.0 };
        let seasonal_component = if with_seasonality {
            10.0 * ((i % 7) as f64 / 7.0 * 2.0 * std::f64::consts::PI).sin()
        } else {
            0.0
        };
        let noise = rng.gen::<f64>() * 5.0 - 2.5;

        let value = 100.0 + trend_component + seasonal_component + noise;

        timestamps.push(timestamp);
        values.push(value);
    }

    TimeSeriesData {
        timestamps,
        values,
        name: "test_data".to_string(),
    }
}

#[test]
fn test_moving_average_model() {
    // Generate test data (100 points, with trend but no seasonality)
    let data = generate_test_data(100, true, false, 42);

    // Split into training and test sets
    let train_size = 80;
    let train_data = TimeSeriesData {
        timestamps: data.timestamps[0..train_size].to_vec(),
        values: data.values[0..train_size].to_vec(),
        name: "train_data".to_string(),
    };

    let test_data = TimeSeriesData {
        timestamps: data.timestamps[train_size..].to_vec(),
        values: data.values[train_size..].to_vec(),
        name: "test_data".to_string(),
    };

    // Create and fit the model
    let mut ma_model = MAModel::new(7).expect("Failed to create MA model");
    ma_model.fit(&train_data).expect("Failed to fit MA model");

    // Make predictions
    let forecast_horizon = test_data.len();
    let output = ma_model
        .predict(forecast_horizon, Some(&test_data))
        .expect("Failed to make predictions");

    // Verify the output
    assert_eq!(output.forecasts.len(), forecast_horizon);
    assert!(output.evaluation.is_some());

    // Check that the MAE and RMSE are reasonable
    let eval = output.evaluation.unwrap();
    assert!(
        eval.mae > 0.0 && eval.mae < 30.0,
        "MAE should be reasonable, got {}",
        eval.mae
    );
    assert!(
        eval.rmse > 0.0 && eval.rmse < 40.0,
        "RMSE should be reasonable, got {}",
        eval.rmse
    );
}

#[test]
fn test_simple_exponential_smoothing_model() {
    // Generate test data (100 points, no trend, no seasonality)
    let data = generate_test_data(100, false, false, 43);

    // Split into training and test sets
    let train_size = 80;
    let train_data = TimeSeriesData {
        timestamps: data.timestamps[0..train_size].to_vec(),
        values: data.values[0..train_size].to_vec(),
        name: "train_data".to_string(),
    };

    let test_data = TimeSeriesData {
        timestamps: data.timestamps[train_size..].to_vec(),
        values: data.values[train_size..].to_vec(),
        name: "test_data".to_string(),
    };

    // Create and fit the model
    let mut ses_model = SESModel::new(0.3).expect("Failed to create SES model");
    ses_model.fit(&train_data).expect("Failed to fit SES model");

    // Make predictions
    let forecast_horizon = test_data.len();
    let output = ses_model
        .predict(forecast_horizon, Some(&test_data))
        .expect("Failed to make predictions");

    // Verify the output
    assert_eq!(output.forecasts.len(), forecast_horizon);
    assert!(output.evaluation.is_some());

    // Check that the MAE and RMSE are reasonable
    let eval = output.evaluation.unwrap();
    assert!(
        eval.mae > 0.0 && eval.mae < 20.0,
        "MAE should be reasonable, got {}",
        eval.mae
    );
    assert!(
        eval.rmse > 0.0 && eval.rmse < 25.0,
        "RMSE should be reasonable, got {}",
        eval.rmse
    );
}

#[test]
fn test_holt_winters_model() {
    // Generate test data (100 points, with trend and seasonality)
    let data = generate_test_data(100, true, true, 44);

    // Split into training and test sets
    let train_size = 80;
    let train_data = TimeSeriesData {
        timestamps: data.timestamps[0..train_size].to_vec(),
        values: data.values[0..train_size].to_vec(),
        name: "train_data".to_string(),
    };

    let test_data = TimeSeriesData {
        timestamps: data.timestamps[train_size..].to_vec(),
        values: data.values[train_size..].to_vec(),
        name: "test_data".to_string(),
    };

    // Create and fit the model
    let mut hw_model =
        HoltWintersModel::new(0.2, 0.1, 0.1, 7).expect("Failed to create Holt-Winters model");
    hw_model
        .fit(&train_data)
        .expect("Failed to fit Holt-Winters model");

    // Make predictions
    let forecast_horizon = test_data.len();
    let output = hw_model
        .predict(forecast_horizon, Some(&test_data))
        .expect("Failed to make predictions");

    // Verify the output
    assert_eq!(output.forecasts.len(), forecast_horizon);
    assert!(output.evaluation.is_some());

    // Check that the MAE and RMSE are reasonable
    let eval = output.evaluation.unwrap();
    assert!(
        eval.mae > 0.0 && eval.mae < 25.0,
        "MAE should be reasonable, got {}",
        eval.mae
    );
    assert!(
        eval.rmse > 0.0 && eval.rmse < 30.0,
        "RMSE should be reasonable, got {}",
        eval.rmse
    );
}

#[test]
fn test_autoregressive_model() {
    // Generate test data (100 points, with trend, no seasonality)
    let data = generate_test_data(100, true, false, 45);

    // Split into training and test sets
    let train_size = 80;
    let train_data = TimeSeriesData {
        timestamps: data.timestamps[0..train_size].to_vec(),
        values: data.values[0..train_size].to_vec(),
        name: "train_data".to_string(),
    };

    let test_data = TimeSeriesData {
        timestamps: data.timestamps[train_size..].to_vec(),
        values: data.values[train_size..].to_vec(),
        name: "test_data".to_string(),
    };

    // Create and fit the model
    let mut ar_model = ARModel::new(3, true).expect("Failed to create AR model");
    ar_model.fit(&train_data).expect("Failed to fit AR model");

    // Make predictions
    let forecast_horizon = test_data.len();
    let output = ar_model
        .predict(forecast_horizon, Some(&test_data))
        .expect("Failed to make predictions");

    // Verify the output
    assert_eq!(output.forecasts.len(), forecast_horizon);
    assert!(output.evaluation.is_some());

    // Check that the MAE and RMSE are reasonable
    let eval = output.evaluation.unwrap();
    assert!(eval.mae > 0.0, "MAE should be positive, got {}", eval.mae);
    assert!(
        eval.rmse > 0.0,
        "RMSE should be positive, got {}",
        eval.rmse
    );
}

#[test]
fn test_ohlcv_data() {
    // Create OHLCV data
    let timestamps = vec![
        Utc::now(),
        Utc::now() + Duration::days(1),
        Utc::now() + Duration::days(2),
        Utc::now() + Duration::days(3),
        Utc::now() + Duration::days(4),
    ];

    let open = vec![100.0, 101.0, 102.0, 103.0, 104.0];
    let high = vec![105.0, 106.0, 107.0, 108.0, 109.0];
    let low = vec![95.0, 96.0, 97.0, 98.0, 99.0];
    let close = vec![102.0, 103.0, 104.0, 105.0, 106.0];
    let volume = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0];

    let ohlcv_data = OHLCVData {
        symbol: "AAPL".to_string(),
        timestamps: timestamps.clone(),
        open: open.clone(),
        high: high.clone(),
        low: low.clone(),
        close: close.clone(),
        volume: volume.clone(),
        adjusted_close: None,
    };

    // Test conversion to time series (using close prices)
    let time_series = ohlcv_data.to_time_series(false);

    assert_eq!(time_series.timestamps, timestamps);
    assert_eq!(time_series.values, close);
    assert!(time_series.name.contains("AAPL"));

    // Test with adjusted close (should use regular close since adjusted_close is None)
    let time_series_adjusted = ohlcv_data.to_time_series(true);
    assert_eq!(time_series_adjusted.values, close);
}

#[test]
fn test_ensemble_forecasting() {
    // Generate test data (100 points, with trend and seasonality)
    let data = generate_test_data(100, true, true, 46);

    // Split into training and test sets
    let train_size = 80;
    let train_data = TimeSeriesData {
        timestamps: data.timestamps[0..train_size].to_vec(),
        values: data.values[0..train_size].to_vec(),
        name: "train_data".to_string(),
    };

    let test_data = TimeSeriesData {
        timestamps: data.timestamps[train_size..].to_vec(),
        values: data.values[train_size..].to_vec(),
        name: "test_data".to_string(),
    };

    let forecast_horizon = test_data.len();

    // Create and fit multiple models
    let mut ma_model = MAModel::new(7).expect("Failed to create MA model");
    ma_model.fit(&train_data).expect("Failed to fit MA model");

    let mut ses_model = SESModel::new(0.3).expect("Failed to create SES model");
    ses_model.fit(&train_data).expect("Failed to fit SES model");

    let mut hw_model =
        HoltWintersModel::new(0.2, 0.1, 0.1, 7).expect("Failed to create Holt-Winters model");
    hw_model
        .fit(&train_data)
        .expect("Failed to fit Holt-Winters model");

    // Generate forecasts
    let ma_output = ma_model
        .predict(forecast_horizon, Some(&test_data))
        .expect("Failed to make MA predictions");
    let ses_output = ses_model
        .predict(forecast_horizon, Some(&test_data))
        .expect("Failed to make SES predictions");
    let hw_output = hw_model
        .predict(forecast_horizon, Some(&test_data))
        .expect("Failed to make Holt-Winters predictions");

    // Create ensemble forecast (average of all models)
    let mut ensemble_forecast = vec![0.0; forecast_horizon];
    for i in 0..forecast_horizon {
        ensemble_forecast[i] =
            (ma_output.forecasts[i] + ses_output.forecasts[i] + hw_output.forecasts[i]) / 3.0;
    }

    // Calculate ensemble error metrics
    let ensemble_mae = calculate_mae(&test_data.values, &ensemble_forecast);
    let ensemble_rmse = calculate_rmse(&test_data.values, &ensemble_forecast);

    // Verify ensemble forecasting
    assert!(
        ensemble_mae > 0.0,
        "Ensemble MAE should be positive, got {}",
        ensemble_mae
    );
    assert!(
        ensemble_rmse > 0.0,
        "Ensemble RMSE should be positive, got {}",
        ensemble_rmse
    );

    // Check if ensemble performs reasonably compared to individual models
    let ma_mae = ma_output.evaluation.as_ref().unwrap().mae;
    let ses_mae = ses_output.evaluation.as_ref().unwrap().mae;
    let hw_mae = hw_output.evaluation.as_ref().unwrap().mae;

    // The ensemble should be at least better than the worst model
    let worst_model_mae = ma_mae.max(ses_mae).max(hw_mae);
    assert!(
        ensemble_mae <= worst_model_mae * 1.2,
        "Ensemble should not be much worse than the worst model"
    );
}

// Helper functions for ensemble metrics calculation
fn calculate_mae(actual: &[f64], forecast: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..actual.len() {
        sum += (actual[i] - forecast[i]).abs();
    }
    sum / actual.len() as f64
}

fn calculate_rmse(actual: &[f64], forecast: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..actual.len() {
        sum += (actual[i] - forecast[i]).powi(2);
    }
    (sum / actual.len() as f64).sqrt()
}
