#![allow(clippy::needless_range_loop)]

use chrono::{Duration, Utc};
use oxidiviner::prelude::*;
use rand::{Rng, SeedableRng};

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
    let mut ses_model = SimpleESModel::new(0.3).expect("Failed to create SES model");
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

    let mut ses_model = SimpleESModel::new(0.3).expect("Failed to create SES model");
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
    let mse = actual
        .iter()
        .zip(forecast.iter())
        .map(|(a, f)| (a - f).powi(2))
        .sum::<f64>()
        / actual.len() as f64;
    mse.sqrt()
}

// ================================
// Enhanced API Integration Tests
// ================================

#[test]
fn test_quick_api_arima() {
    use oxidiviner::core::validation::ValidationUtils;
    use oxidiviner::quick;

    let data = generate_test_data(60, true, false, 46);
    let (train, test) = ValidationUtils::time_split(&data, 0.8).unwrap();

    // Test basic ARIMA forecast
    let forecast = quick::arima(train.clone(), test.values.len()).unwrap();
    assert_eq!(forecast.len(), test.values.len());

    // Test ARIMA with custom config
    let forecast_custom =
        quick::arima_with_config(train, test.values.len(), Some((2, 1, 1))).unwrap();
    assert_eq!(forecast_custom.len(), test.values.len());
}

#[test]
fn test_quick_api_moving_average() {
    use oxidiviner::core::validation::ValidationUtils;
    use oxidiviner::quick;

    let data = generate_test_data(60, true, false, 47);
    let (train, test) = ValidationUtils::time_split(&data, 0.8).unwrap();

    // Test MA with different windows
    for window in [3, 5, 7] {
        let forecast =
            quick::moving_average(train.clone(), test.values.len(), Some(window)).unwrap();
        assert_eq!(forecast.len(), test.values.len());
    }

    // Test default window
    let forecast_default = quick::moving_average(train, test.values.len(), None).unwrap();
    assert_eq!(forecast_default.len(), test.values.len());
}

#[test]
fn test_builder_pattern() {
    use oxidiviner::ModelBuilder;

    // Test ARIMA builder
    let arima_config = ModelBuilder::arima()
        .with_ar(2)
        .with_differencing(1)
        .with_ma(1)
        .build_config();

    assert_eq!(arima_config.model_type, "ARIMA");
    assert_eq!(arima_config.parameters.get("p"), Some(&2.0));
    assert_eq!(arima_config.parameters.get("d"), Some(&1.0));
    assert_eq!(arima_config.parameters.get("q"), Some(&1.0));

    // Test MA builder
    let ma_config = ModelBuilder::moving_average().with_window(7).build_config();

    assert_eq!(ma_config.model_type, "MA");
    assert_eq!(ma_config.parameters.get("window"), Some(&7.0));
}

#[test]
fn test_model_validator() {
    use oxidiviner::ModelValidator;

    // Valid ARIMA parameters
    assert!(ModelValidator::validate_arima_params(2, 1, 1).is_ok());
    assert!(ModelValidator::validate_arima_params(1, 0, 1).is_ok());

    // Invalid ARIMA parameters
    assert!(ModelValidator::validate_arima_params(15, 3, 10).is_err());
    assert!(ModelValidator::validate_arima_params(0, 0, 0).is_err());

    // Valid MA parameters
    assert!(ModelValidator::validate_ma_params(1).is_ok());
    assert!(ModelValidator::validate_ma_params(10).is_ok());

    // Invalid MA parameters
    assert!(ModelValidator::validate_ma_params(0).is_err());
}

#[test]
fn test_validation_utilities() {
    use oxidiviner::core::validation::ValidationUtils;

    let data = generate_test_data(100, true, false, 48);

    // Test time split
    let (train, test) = ValidationUtils::time_split(&data, 0.8).unwrap();
    assert_eq!(train.values.len(), 80);
    assert_eq!(test.values.len(), 20);

    // Test edge cases
    assert!(ValidationUtils::time_split(&data, 0.0).is_err());
    assert!(ValidationUtils::time_split(&data, 1.0).is_err());

    // Test time series CV
    let splits = ValidationUtils::time_series_cv(&data, 3, Some(30)).unwrap();
    assert_eq!(splits.len(), 3);

    for (train, test) in splits {
        assert!(train.values.len() >= 30);
        assert!(!test.values.is_empty());
    }
}

#[test]
fn test_accuracy_metrics() {
    use oxidiviner::core::validation::ValidationUtils;

    let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let predicted = vec![1.1, 2.1, 2.9, 4.1, 4.9];

    let metrics = ValidationUtils::accuracy_metrics(&actual, &predicted, None).unwrap();

    assert!(metrics.mae > 0.0);
    assert!(metrics.rmse > 0.0);
    assert!(metrics.mape > 0.0);
    assert!(metrics.smape > 0.0);
    assert!(metrics.r_squared > 0.0);
    assert_eq!(metrics.n_observations, 5);

    // Test with baseline for MASE - use a naive baseline with different values
    let baseline = vec![0.9, 1.9, 2.9, 3.9, 4.9]; // Previous period forecast
    let metrics_with_baseline =
        ValidationUtils::accuracy_metrics(&actual, &predicted, Some(&baseline)).unwrap();
    assert!(metrics_with_baseline.mase.is_some());
}

#[test]
fn test_auto_selector() {
    use oxidiviner::{AutoSelector, SelectionCriteria};

    // Test creating auto selectors with different criteria
    let aic_selector = AutoSelector::with_aic();
    matches!(aic_selector.criteria(), SelectionCriteria::AIC);

    let bic_selector = AutoSelector::with_bic();
    matches!(bic_selector.criteria(), SelectionCriteria::BIC);

    let cv_selector = AutoSelector::with_cross_validation(3);
    matches!(
        cv_selector.criteria(),
        SelectionCriteria::CrossValidation { folds: 3 }
    );

    // Test adding custom candidates
    let custom_config = oxidiviner::ModelBuilder::ar().with_ar(5).build_config();
    let selector_with_custom = AutoSelector::with_aic().add_candidate(custom_config);

    assert!(!selector_with_custom.candidates().is_empty());
}

#[test]
fn test_auto_select_integration() {
    use oxidiviner::quick;

    let data = generate_test_data(60, true, false, 49);

    // Test auto select
    let (forecast, best_model) = quick::auto_select(data, 10).unwrap();
    assert_eq!(forecast.len(), 10);
    assert!(!best_model.is_empty());

    println!("Auto-selected model: {}", best_model);
    println!("Forecast length: {}", forecast.len());
}

#[test]
fn test_builder_with_quick_api() {
    use oxidiviner::core::validation::ValidationUtils;
    use oxidiviner::{quick, ModelBuilder};

    let data = generate_test_data(60, true, false, 50);
    let (train, test) = ValidationUtils::time_split(&data, 0.8).unwrap();

    let config = ModelBuilder::arima()
        .with_ar(1)
        .with_differencing(1)
        .with_ma(1)
        .build_config();

    let forecast = quick::forecast_with_config(train, test.values.len(), config).unwrap();
    assert_eq!(forecast.len(), test.values.len());
}

#[test]
fn test_parameter_validation_edge_cases() {
    use oxidiviner::{quick, ModelValidator};

    let data = generate_test_data(2, false, false, 51); // Very small dataset - should fail for ARIMA

    // Should fail with insufficient data for ARIMA(1,1,1) which needs p+d+1 = 3 points minimum
    assert!(quick::arima(data.clone(), 2).is_err());

    // Test minimum data validation
    assert!(ModelValidator::validate_min_data_length(5, 10, "ARIMA").is_err());
    assert!(ModelValidator::validate_min_data_length(15, 10, "ARIMA").is_ok());

    // Test forecast horizon validation
    assert!(ModelValidator::validate_forecast_horizon(20, 5).is_err());
    assert!(ModelValidator::validate_forecast_horizon(5, 20).is_ok());
}

#[test]
fn test_comprehensive_enhanced_workflow() {
    use oxidiviner::core::validation::ValidationUtils;
    use oxidiviner::{quick, ModelBuilder};

    // Create realistic test scenario
    let data = generate_test_data(100, true, false, 52);

    // 1. Split data for validation
    let (train, test) = ValidationUtils::time_split(&data, 0.8).unwrap();

    // 2. Use auto selection to find best model
    let (auto_forecast, best_model) = quick::auto_select(train.clone(), test.values.len()).unwrap();

    // 3. Validate the forecast
    let metrics = ValidationUtils::accuracy_metrics(&test.values, &auto_forecast, None).unwrap();

    // 4. Use builder pattern to create custom models
    let custom_config = ModelBuilder::arima()
        .with_ar(2)
        .with_differencing(1)
        .with_ma(1)
        .build_config();

    let custom_forecast =
        quick::forecast_with_config(train, test.values.len(), custom_config).unwrap();

    // 5. Compare results
    let custom_metrics =
        ValidationUtils::accuracy_metrics(&test.values, &custom_forecast, None).unwrap();

    println!("Auto-selected model: {}", best_model);
    println!("Auto model MAE: {:.3}", metrics.mae);
    println!("Custom model MAE: {:.3}", custom_metrics.mae);

    // Both should produce valid forecasts
    assert_eq!(auto_forecast.len(), test.values.len());
    assert_eq!(custom_forecast.len(), test.values.len());

    // Basic sanity checks on metrics
    assert!(metrics.mae > 0.0);
    assert!(metrics.rmse > 0.0);
    assert!(custom_metrics.mae > 0.0);
    assert!(custom_metrics.rmse > 0.0);
}
