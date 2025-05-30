#![allow(deprecated)]
#![allow(unused_variables)]

/*!
# Moving Average Models Example - Enhanced API Demo

This example demonstrates both the traditional API and the new improved API
features for moving average models in OxiDiviner.

Run with:
```bash
cargo run --package oxidiviner-examples --bin ma_demo
```
*/

use chrono::{Duration, Utc};
use oxidiviner::MAModel;
use oxidiviner::{quick, ModelBuilder};
use oxidiviner::{validation::ValidationUtils, ModelValidator, TimeSeriesData};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() {
    println!("🌊 Moving Average Models Example - Enhanced API Demo");
    println!("====================================================\n");

    // Create realistic sample data
    let data = create_sample_data();
    println!(
        "📊 Created sample dataset with {} points",
        data.values.len()
    );

    // Split data using new validation utilities
    let (train_data, test_data) =
        ValidationUtils::time_split(&data, 0.75).expect("Failed to split data");

    println!(
        "📊 Data split: {} training, {} testing points",
        train_data.values.len(),
        test_data.values.len()
    );

    // Demonstrate the new Quick API
    demonstrate_quick_api(&train_data, &test_data);

    // Demonstrate Builder Pattern and Validation
    demonstrate_builder_and_validation(&train_data, &test_data);

    // Demonstrate Validation Utilities
    demonstrate_validation_utilities(&data);

    // Traditional API for comparison
    demonstrate_traditional_api(&train_data, &test_data);

    println!("\n✨ Enhanced Moving Average API Demo completed successfully!");
}

/// Create realistic sample data for moving average demonstration
fn create_sample_data() -> TimeSeriesData {
    let now = Utc::now();
    let n = 80;
    let mut rng = StdRng::seed_from_u64(42);

    // Generate timestamps (daily data)
    let timestamps = (0..n).map(|i| now + Duration::days(i as i64)).collect();

    // Create time series with trend and noise (good for MA models)
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let trend = i as f64 * 0.5; // Small trend
            let noise = rng.gen_range(-3.0..3.0); // Random noise
            let base = 50.0; // Base level
            base + trend + noise
        })
        .collect();

    TimeSeriesData::new(timestamps, values, "ma_demo_series")
        .expect("Failed to create time series data")
}

/// Demonstrate the new Quick API for moving average models
fn demonstrate_quick_api(train_data: &TimeSeriesData, test_data: &TimeSeriesData) {
    println!("\n🔥 Quick API Demonstrations:");
    println!("=============================");

    let forecast_horizon = test_data.values.len();

    // Quick moving average forecasting with different windows
    println!("\n  📊 Quick Moving Average forecasting:");
    for window in [3, 5, 7, 10] {
        match quick::moving_average(train_data.clone(), forecast_horizon, Some(window)) {
            Ok(forecast) => {
                let metrics = ValidationUtils::accuracy_metrics(&test_data.values, &forecast, None)
                    .unwrap_or_else(|_| panic!("Failed to calculate metrics"));
                println!(
                    "     MA({}) - MAE: {:.3}, RMSE: {:.3}, R²: {:.4}",
                    window, metrics.mae, metrics.rmse, metrics.r_squared
                );
            }
            Err(e) => println!("     MA({}) Error: {}", window, e),
        }
    }

    // Default window (should be 5)
    println!("\n  📈 Quick Moving Average with default window:");
    match quick::moving_average(train_data.clone(), forecast_horizon, None) {
        Ok(forecast) => {
            let metrics = ValidationUtils::accuracy_metrics(&test_data.values, &forecast, None)
                .unwrap_or_else(|_| panic!("Failed to calculate metrics"));
            println!(
                "     MA(default) - MAE: {:.3}, RMSE: {:.3}",
                metrics.mae, metrics.rmse
            );
        }
        Err(e) => println!("     Error: {}", e),
    }
}

/// Demonstrate builder pattern and validation features
fn demonstrate_builder_and_validation(train_data: &TimeSeriesData, test_data: &TimeSeriesData) {
    println!("\n🏗️  Builder Pattern & Validation:");
    println!("==================================");

    let forecast_horizon = test_data.values.len();

    // Build different MA configurations
    println!("\n  🏗️  Testing MA configurations with Builder:");
    let configs = vec![
        (
            "MA(3)",
            ModelBuilder::moving_average().with_window(3).build_config(),
        ),
        (
            "MA(7)",
            ModelBuilder::moving_average().with_window(7).build_config(),
        ),
        (
            "MA(12)",
            ModelBuilder::moving_average()
                .with_window(12)
                .build_config(),
        ),
    ];

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

    // Valid windows
    for window in [1, 5, 10] {
        match ModelValidator::validate_ma_params(window) {
            Ok(()) => println!("     ✓ MA({}) parameters are valid", window),
            Err(e) => println!("     ✗ MA({}) Error: {}", window, e),
        }
    }

    // Invalid windows
    for window in [0, 25] {
        match ModelValidator::validate_ma_params(window) {
            Ok(()) => println!("     ✓ MA({}) parameters are valid", window),
            Err(e) => println!("     ✗ MA({}) Error: {}", window, e),
        }
    }
}

/// Demonstrate validation utilities with moving average models
fn demonstrate_validation_utilities(data: &TimeSeriesData) {
    println!("\n✅ Validation Utilities:");
    println!("========================");

    // Cross-validation for MA models
    println!("\n  🔄 Moving Average Cross-Validation:");
    match ValidationUtils::time_series_cv(data, 3, Some(20)) {
        Ok(splits) => {
            println!(
                "     Created {} CV splits for MA model evaluation:",
                splits.len()
            );

            let mut best_window = 3;
            let mut best_mae = f64::INFINITY;

            for window in [3, 5, 7] {
                let mut total_mae = 0.0;
                let mut valid_splits = 0;

                for (i, (train, test)) in splits.iter().enumerate() {
                    if let Ok(forecast) =
                        quick::moving_average(train.clone(), test.values.len(), Some(window))
                    {
                        if let Ok(metrics) =
                            ValidationUtils::accuracy_metrics(&test.values, &forecast, None)
                        {
                            total_mae += metrics.mae;
                            valid_splits += 1;
                        }
                    }
                }

                if valid_splits > 0 {
                    let avg_mae = total_mae / valid_splits as f64;
                    println!("       MA({}) - Average CV MAE: {:.3}", window, avg_mae);

                    if avg_mae < best_mae {
                        best_mae = avg_mae;
                        best_window = window;
                    }
                }
            }

            println!(
                "     🏆 Best window size: MA({}) with MAE: {:.3}",
                best_window, best_mae
            );
        }
        Err(e) => println!("     Error: {}", e),
    }

    // Comprehensive accuracy metrics demonstration
    println!("\n  📏 Comprehensive accuracy metrics:");
    let (train, test) = ValidationUtils::time_split(data, 0.7).expect("Failed to split data");

    if let Ok(forecast) = quick::moving_average(train, test.values.len(), Some(5)) {
        if let Ok(metrics) = ValidationUtils::accuracy_metrics(&test.values, &forecast, None) {
            println!("     MA(5) Performance Report:");
            println!("       MAE:   {:.4}", metrics.mae);
            println!("       MSE:   {:.4}", metrics.mse);
            println!("       RMSE:  {:.4}", metrics.rmse);
            println!("       MAPE:  {:.2}%", metrics.mape);
            println!("       SMAPE: {:.2}%", metrics.smape);
            println!("       R²:    {:.4}", metrics.r_squared);
            println!("       N:     {}", metrics.n_observations);
        }
    }
}

/// Demonstrate traditional API for comparison
fn demonstrate_traditional_api(train_data: &TimeSeriesData, test_data: &TimeSeriesData) {
    println!("\n🔧 Traditional API Example (for comparison):");
    println!("==============================================");

    let horizon = test_data.values.len();

    // Traditional MA approach - more verbose
    println!("\n  📊 Traditional MA Model Creation and Fitting:");

    match MAModel::new(5) {
        Ok(mut ma_model) => {
            println!("     ✓ Created MA(5) model");

            match ma_model.fit(train_data) {
                Ok(_) => {
                    println!("     ✓ Fitted model to training data");

                    match ma_model.forecast(horizon) {
                        Ok(_forecast) => {
                            println!("     ✓ Generated forecast");

                            match ma_model.evaluate(test_data) {
                                Ok(eval) => {
                                    println!("     📊 Traditional API Results:");
                                    println!("        MAE:  {:.3}", eval.mae);
                                    println!("        RMSE: {:.3}", eval.rmse);
                                    println!("        MAPE: {:.2}%", eval.mape * 100.0);
                                }
                                Err(e) => println!("     ✗ Error evaluating: {}", e),
                            }
                        }
                        Err(e) => println!("     ✗ Error forecasting: {}", e),
                    }
                }
                Err(e) => println!("     ✗ Error fitting: {}", e),
            }
        }
        Err(e) => println!("     ✗ Error creating model: {}", e),
    }

    println!("\n  📊 API Comparison Summary:");
    println!("     🔥 Quick API: quick::moving_average(data, periods, Some(5))");
    println!("     🔧 Traditional: MAModel::new(5) -> fit() -> forecast() -> evaluate()");
    println!("     📈 Lines of code: 1 vs 10+ (90% reduction!)");
    println!("     ✅ Built-in validation: Automatic vs Manual");
    println!("     📏 Rich metrics: Comprehensive vs Basic");
}
