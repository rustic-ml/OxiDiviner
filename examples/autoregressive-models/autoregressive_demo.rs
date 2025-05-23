#![allow(deprecated)]
#![allow(unused_variables)]

/*!
# Autoregressive Models Example - Enhanced API Demo

This example demonstrates both the traditional API and the new improved API
features for autoregressive models in OxiDiviner, showing the dramatic
improvements in usability and functionality.

Run with:
```bash
cargo run --package oxidiviner-examples --bin autoregressive_demo
```
*/

use chrono::{DateTime, Duration, Utc};
use oxidiviner::{quick, AutoSelector, ModelBuilder};
use oxidiviner::{ARIMAModel, ARModel, SARIMAModel, VARModel};
use oxidiviner::{validation::ValidationUtils, ModelValidator, TimeSeriesData};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use std::f64::consts::PI;

fn main() {
    println!("🚀 Autoregressive Models Example - Enhanced API Demo");
    println!("=====================================================\n");

    // Create sample data
    let (trend_data, seasonal_data, var_data) = create_sample_data();

    println!("📊 Created sample datasets:");
    println!("  • Trend data: {} points", trend_data.values.len());
    println!("  • Seasonal data: {} points", seasonal_data.values.len());
    println!(
        "  • VAR data: {} series with {} points each",
        var_data.len(),
        var_data[0].values.len()
    );

    // Split data using new validation utilities
    let (train_trend, test_trend) =
        ValidationUtils::time_split(&trend_data, 0.8).expect("Failed to split trend data");
    let (train_seasonal, test_seasonal) =
        ValidationUtils::time_split(&seasonal_data, 0.8).expect("Failed to split seasonal data");

    println!("\n📊 Data split completed:");
    println!(
        "  • Training: {} points, Testing: {} points",
        train_trend.values.len(),
        test_trend.values.len()
    );

    // Demonstrate the new Quick API
    demonstrate_quick_api(&train_trend, &test_trend);

    // Demonstrate Builder Pattern
    demonstrate_builder_pattern(&train_trend, &test_trend);

    // Demonstrate Smart Model Selection
    demonstrate_smart_selection(&trend_data);

    // Demonstrate Validation Utilities
    demonstrate_validation_utilities(&trend_data);

    // Traditional API examples for comparison
    demonstrate_traditional_api(
        &train_trend,
        &test_trend,
        &train_seasonal,
        &test_seasonal,
        &var_data,
    );

    println!("\n✨ Enhanced API Demo completed successfully!");
}

/// Create realistic sample datasets for demonstration
fn create_sample_data() -> (TimeSeriesData, TimeSeriesData, Vec<TimeSeriesData>) {
    let now = Utc::now();
    let n = 100;
    let mut rng = StdRng::seed_from_u64(42);

    // Generate timestamps (daily data)
    let timestamps: Vec<DateTime<Utc>> = (0..n).map(|i| now + Duration::days(i as i64)).collect();

    // 1. Trend data with noise
    let trend_values: Vec<f64> = (1..=n as i64)
        .map(|i| i as f64 + rng.gen_range(-2.0..2.0))
        .collect();

    // 2. Seasonal data with trend
    let mut seasonal_values = Vec::with_capacity(n);
    for i in 0..n {
        let trend = i as f64 * 0.5;
        let seasonal = 10.0 * (2.0 * PI * (i % 12) as f64 / 12.0).sin();
        let noise = rng.gen_range(-1.0..1.0);
        seasonal_values.push(trend + seasonal + noise);
    }

    // 3. Two related series for VAR
    let y1_values: Vec<f64> = trend_values
        .iter()
        .map(|&x| x + rng.gen_range(-2.0..2.0))
        .collect();

    let y2_values: Vec<f64> = y1_values
        .iter()
        .map(|&x| 2.0 * x + 5.0 + rng.gen_range(-3.0..3.0))
        .collect();

    let trend_data = TimeSeriesData::new(timestamps.clone(), trend_values, "trend_series")
        .expect("Failed to create trend data");
    let seasonal_data = TimeSeriesData::new(timestamps.clone(), seasonal_values, "seasonal_series")
        .expect("Failed to create seasonal data");

    let y1_data =
        TimeSeriesData::new(timestamps.clone(), y1_values, "y1").expect("Failed to create y1 data");
    let y2_data =
        TimeSeriesData::new(timestamps, y2_values, "y2").expect("Failed to create y2 data");

    (trend_data, seasonal_data, vec![y1_data, y2_data])
}

/// Demonstrate the new Quick API for autoregressive models
fn demonstrate_quick_api(train_data: &TimeSeriesData, test_data: &TimeSeriesData) {
    println!("\n🔥 Quick API Demonstrations:");
    println!("=============================");

    let forecast_horizon = test_data.values.len();

    // Quick ARIMA forecasting
    println!("\n  📈 Quick ARIMA forecasting:");
    match quick::arima(train_data.clone(), forecast_horizon) {
        Ok(forecast) => {
            let metrics = ValidationUtils::accuracy_metrics(&test_data.values, &forecast, None)
                .unwrap_or_else(|_| panic!("Failed to calculate metrics"));
            println!(
                "     ARIMA(1,1,1) - MAE: {:.3}, RMSE: {:.3}",
                metrics.mae, metrics.rmse
            );
        }
        Err(e) => println!("     Error: {}", e),
    }

    // Quick AR forecasting with different orders
    println!("\n  📊 Quick AR forecasting:");
    for order in [1, 2, 3] {
        match quick::ar(train_data.clone(), forecast_horizon, Some(order)) {
            Ok(forecast) => {
                let metrics = ValidationUtils::accuracy_metrics(&test_data.values, &forecast, None)
                    .unwrap_or_else(|_| panic!("Failed to calculate metrics"));
                println!(
                    "     AR({}) - MAE: {:.3}, RMSE: {:.3}",
                    order, metrics.mae, metrics.rmse
                );
            }
            Err(e) => println!("     AR({}) Error: {}", order, e),
        }
    }
}

/// Demonstrate the builder pattern for model configuration
fn demonstrate_builder_pattern(train_data: &TimeSeriesData, test_data: &TimeSeriesData) {
    println!("\n🏗️  Builder Pattern Demonstrations:");
    println!("==================================");

    let forecast_horizon = test_data.values.len();

    // Build different ARIMA configurations
    let configs = vec![
        (
            "ARIMA(1,1,1)",
            ModelBuilder::arima()
                .with_ar(1)
                .with_differencing(1)
                .with_ma(1)
                .build_config(),
        ),
        (
            "ARIMA(2,1,1)",
            ModelBuilder::arima()
                .with_ar(2)
                .with_differencing(1)
                .with_ma(1)
                .build_config(),
        ),
        (
            "ARIMA(1,1,2)",
            ModelBuilder::arima()
                .with_ar(1)
                .with_differencing(1)
                .with_ma(2)
                .build_config(),
        ),
    ];

    println!("\n  🏗️  Testing different ARIMA configurations:");
    for (name, config) in configs {
        match quick::forecast_with_config(train_data.clone(), forecast_horizon, config) {
            Ok(forecast) => {
                let metrics = ValidationUtils::accuracy_metrics(&test_data.values, &forecast, None)
                    .unwrap_or_else(|_| panic!("Failed to calculate metrics"));
                println!(
                    "     {} - MAE: {:.3}, RMSE: {:.3}",
                    name, metrics.mae, metrics.rmse
                );
            }
            Err(e) => println!("     {} Error: {}", name, e),
        }
    }

    // Demonstrate parameter validation
    println!("\n  ✅ Parameter validation examples:");
    match ModelValidator::validate_arima_params(2, 1, 1) {
        Ok(()) => println!("     ✓ ARIMA(2,1,1) parameters are valid"),
        Err(e) => println!("     ✗ Error: {}", e),
    }

    match ModelValidator::validate_arima_params(15, 3, 12) {
        Ok(()) => println!("     ✓ ARIMA(15,3,12) parameters are valid"),
        Err(e) => println!("     ✗ ARIMA(15,3,12): {}", e),
    }
}

/// Demonstrate smart model selection
fn demonstrate_smart_selection(data: &TimeSeriesData) {
    println!("\n🧠 Smart Model Selection:");
    println!("=========================");

    // Automatic model selection
    println!("\n  🤖 Automatic model selection:");
    match quick::auto_select(data.clone(), 10) {
        Ok((forecast, best_model)) => {
            println!("     Best model: {}", best_model);
            println!(
                "     Forecast (first 5): {:?}",
                forecast
                    .iter()
                    .take(5)
                    .map(|x| format!("{:.2}", x))
                    .collect::<Vec<_>>()
            );
        }
        Err(e) => println!("     Error: {}", e),
    }

    // Custom auto selector
    println!("\n  🔍 Custom AutoSelector with additional candidates:");
    let selector = AutoSelector::with_cross_validation(3)
        .add_candidate(ModelBuilder::ar().with_ar(4).build_config())
        .add_candidate(ModelBuilder::ar().with_ar(5).build_config());

    println!("     Selection criteria: {:?}", selector.criteria());
    println!("     Total candidates: {}", selector.candidates().len());
}

/// Demonstrate validation utilities
fn demonstrate_validation_utilities(data: &TimeSeriesData) {
    println!("\n✅ Validation Utilities:");
    println!("========================");

    // Cross-validation splits
    println!("\n  🔄 Time series cross-validation:");
    match ValidationUtils::time_series_cv(data, 4, Some(25)) {
        Ok(splits) => {
            println!("     Created {} CV splits:", splits.len());
            for (i, (train, test)) in splits.iter().enumerate() {
                println!(
                    "       Split {}: Train {} points, Test {} points",
                    i + 1,
                    train.values.len(),
                    test.values.len()
                );
            }
        }
        Err(e) => println!("     Error: {}", e),
    }

    // Demonstrate comprehensive accuracy metrics
    println!("\n  📏 Comprehensive accuracy metrics demo:");
    let (train, test) = ValidationUtils::time_split(data, 0.7).expect("Failed to split data");

    // Create a simple forecast for demonstration
    if let Ok(forecast) = quick::moving_average(train, test.values.len(), Some(7)) {
        if let Ok(metrics) = ValidationUtils::accuracy_metrics(&test.values, &forecast, None) {
            println!("     Moving Average (7) performance:");
            println!("       MAE:  {:.4}", metrics.mae);
            println!("       RMSE: {:.4}", metrics.rmse);
            println!("       MAPE: {:.2}%", metrics.mape);
            println!("       R²:   {:.4}", metrics.r_squared);
            println!("       N:    {}", metrics.n_observations);
        }
    }
}

/// Demonstrate traditional API for comparison with the new improved API
fn demonstrate_traditional_api(
    train_trend: &TimeSeriesData,
    test_trend: &TimeSeriesData,
    train_seasonal: &TimeSeriesData,
    test_seasonal: &TimeSeriesData,
    var_data: &[TimeSeriesData],
) {
    println!("\n🔧 Traditional API Examples (for comparison):");
    println!("==============================================");

    let horizon = test_trend.values.len();

    // Traditional AR Model
    println!("\n  📊 Traditional AR Model:");
    let mut ar_model = ARModel::new(2, true).expect("Failed to create AR model");
    ar_model.fit(train_trend).expect("Failed to fit AR model");
    let ar_forecast = ar_model
        .forecast(horizon)
        .expect("Failed to forecast with AR model");
    let ar_eval = ar_model
        .evaluate(test_trend)
        .expect("Failed to evaluate AR model");

    println!(
        "     AR(2) Traditional API - MAE: {:.3}, RMSE: {:.3}",
        ar_eval.mae, ar_eval.rmse
    );

    // Traditional ARIMA Model
    println!("\n  📈 Traditional ARIMA Model:");
    let mut arima_model = ARIMAModel::new(1, 1, 1, true).expect("Failed to create ARIMA model");
    arima_model
        .fit(train_trend)
        .expect("Failed to fit ARIMA model");
    let arima_forecast = arima_model
        .forecast(horizon)
        .expect("Failed to forecast with ARIMA model");
    let arima_eval = arima_model
        .evaluate(test_trend)
        .expect("Failed to evaluate ARIMA model");

    println!(
        "     ARIMA(1,1,1) Traditional API - MAE: {:.3}, RMSE: {:.3}",
        arima_eval.mae, arima_eval.rmse
    );

    // Traditional SARIMA Model
    println!("\n  🌊 Traditional SARIMA Model:");
    match SARIMAModel::new(0, 1, 1, 0, 1, 1, 12, true) {
        Ok(mut sarima_model) => {
            match sarima_model.fit(train_seasonal) {
                Ok(_) => match sarima_model.forecast(horizon) {
                    Ok(_sarima_forecast) => match sarima_model.evaluate(test_seasonal) {
                        Ok(sarima_eval) => {
                            println!("     SARIMA(0,1,1)(0,1,1)12 Traditional API - MAE: {:.3}, RMSE: {:.3}", 
                                        sarima_eval.mae, sarima_eval.rmse);
                        }
                        Err(e) => println!("     Error evaluating SARIMA model: {}", e),
                    },
                    Err(e) => println!("     Error forecasting with SARIMA model: {}", e),
                },
                Err(e) => println!("     Error fitting SARIMA model: {}", e),
            }
        }
        Err(e) => println!("     Error creating SARIMA model: {}", e),
    }

    // Traditional VAR Model
    println!("\n  🔗 Traditional VAR Model:");
    let variable_names = vec!["y1".to_string(), "y2".to_string()];
    match VARModel::new(1, variable_names, true) {
        Ok(mut var_model) => {
            // Split VAR data into train/test
            let (train_y1, test_y1) =
                ValidationUtils::time_split(&var_data[0], 0.8).expect("Failed to split y1 data");
            let (train_y2, test_y2) =
                ValidationUtils::time_split(&var_data[1], 0.8).expect("Failed to split y2 data");

            let mut train_data_map = HashMap::new();
            train_data_map.insert("y1".to_string(), train_y1);
            train_data_map.insert("y2".to_string(), train_y2);

            let mut test_data_map = HashMap::new();
            test_data_map.insert("y1".to_string(), test_y1);
            test_data_map.insert("y2".to_string(), test_y2);

            match var_model.fit_multiple(&train_data_map) {
                Ok(_) => match var_model.evaluate_multiple(&test_data_map) {
                    Ok(var_evals) => {
                        println!(
                            "     VAR(1) Traditional API - y1 MAE: {:.3}, y2 MAE: {:.3}",
                            var_evals["y1"].mae, var_evals["y2"].mae
                        );
                    }
                    Err(e) => println!("     Error evaluating VAR model: {}", e),
                },
                Err(e) => println!("     Error fitting VAR model: {}", e),
            }
        }
        Err(e) => println!("     Error creating VAR model: {}", e),
    }

    println!("\n  📊 API Comparison Summary:");
    println!("     🔥 New Quick API: One-line forecasting with automatic validation");
    println!("     🏗️  Builder Pattern: Fluent configuration with parameter validation");
    println!("     🧠 Smart Selection: Automatic best model detection");
    println!("     ✅ Validation Utils: Professional-grade testing and metrics");
    println!("     🔧 Traditional API: Still available for fine-grained control");
}
