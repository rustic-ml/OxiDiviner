//! Comprehensive Accuracy Measurement for OxiDiviner Models
//!
//! This example demonstrates accuracy measurement across different:
//! 1. Model types (ARIMA, ES, MA, GARCH, etc.)
//! 2. Data types (synthetic trends, seasonal, real financial data)
//! 3. Accuracy metrics (MAE, RMSE, MAPE, R², Directional Accuracy, etc.)
//! 4. Forecasting horizons (1-step, multi-step)

use chrono::{Duration, Utc};
use oxidiviner::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AccuracyReport {
    pub model_name: String,
    pub dataset_name: String,
    pub mae: f64,
    pub rmse: f64,
    pub mape: f64,
    pub r_squared: f64,
    pub directional_accuracy: f64,
    pub max_error: f64,
    pub mean_error: f64,
    pub std_error: f64,
    pub forecast_horizon: usize,
    pub training_size: usize,
    pub test_size: usize,
}

impl AccuracyReport {
    pub fn new(
        model_name: String,
        dataset_name: String,
        actual: &[f64],
        predicted: &[f64],
        forecast_horizon: usize,
        training_size: usize,
    ) -> Self {
        let mae = calculate_mae(actual, predicted);
        let rmse = calculate_rmse(actual, predicted);
        let mape = calculate_mape(actual, predicted);
        let r_squared = calculate_r_squared(actual, predicted);
        let directional_accuracy = calculate_directional_accuracy(actual, predicted);
        let errors: Vec<f64> = actual
            .iter()
            .zip(predicted.iter())
            .map(|(a, p)| a - p)
            .collect();
        let max_error = errors.iter().map(|e| e.abs()).fold(0.0, f64::max);
        let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
        let std_error = {
            let variance =
                errors.iter().map(|e| (e - mean_error).powi(2)).sum::<f64>() / errors.len() as f64;
            variance.sqrt()
        };

        Self {
            model_name,
            dataset_name,
            mae,
            rmse,
            mape,
            r_squared,
            directional_accuracy,
            max_error,
            mean_error,
            std_error,
            forecast_horizon,
            training_size,
            test_size: actual.len(),
        }
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🔍 OxiDiviner - Comprehensive Accuracy Measurement");
    println!("==================================================\n");

    let mut all_reports = Vec::new();

    // Test 1: Synthetic Data with Known Properties
    println!("📊 Test 1: Synthetic Data Analysis");
    println!("==================================");

    let synthetic_datasets = generate_synthetic_datasets();
    for (name, data) in synthetic_datasets {
        println!("\n🔹 Testing on {} dataset ({} points)", name, data.len());
        let reports = test_models_on_dataset(&data, &name)?;
        all_reports.extend(reports);

        // Print best model for this dataset
        let best = all_reports
            .iter()
            .filter(|r| r.dataset_name == name)
            .min_by(|a, b| a.mae.partial_cmp(&b.mae).unwrap());

        if let Some(best_model) = best {
            println!(
                "   🏆 Best model: {} (MAE: {:.4})",
                best_model.model_name, best_model.mae
            );
        }
    }

    // Test 2: Real Financial Data
    println!("\n\n📈 Test 2: Real Financial Data Analysis");
    println!("=======================================");
    if let Ok(financial_data) = load_financial_data() {
        println!(
            "\n🔹 Testing on AAPL financial data ({} points)",
            financial_data.len()
        );
        let reports = test_models_on_dataset(&financial_data, "AAPL_Financial")?;
        all_reports.extend(reports);

        let best = all_reports
            .iter()
            .filter(|r| r.dataset_name == "AAPL_Financial")
            .min_by(|a, b| a.mae.partial_cmp(&b.mae).unwrap());

        if let Some(best_model) = best {
            println!(
                "   🏆 Best model: {} (MAE: {:.4})",
                best_model.model_name, best_model.mae
            );
        }
    } else {
        println!("   ⚠️  Could not load financial data, skipping...");
    }

    // Test 3: Cross-Validation Analysis
    println!("\n\n🔄 Test 3: Cross-Validation Analysis");
    println!("===================================");
    let cv_results = cross_validation_analysis()?;
    println!(
        "   📊 Cross-validation completed on {} models",
        cv_results.len()
    );

    // Generate comprehensive accuracy report
    println!("\n\n📋 COMPREHENSIVE ACCURACY REPORT");
    println!("=================================");

    generate_summary_report(&all_reports);
    generate_detailed_report(&all_reports);
    generate_rankings(&all_reports);

    println!("\n✅ Accuracy measurement completed!");
    Ok(())
}

fn generate_synthetic_datasets() -> HashMap<String, TimeSeriesData> {
    let mut datasets = HashMap::new();
    let start_time = Utc::now();

    // 1. Linear Trend
    let trend_data = {
        let timestamps: Vec<_> = (0..100).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5).collect();
        TimeSeriesData::new(timestamps, values, "linear_trend").unwrap()
    };
    datasets.insert("Linear_Trend".to_string(), trend_data);

    // 2. Seasonal Pattern
    let seasonal_data = {
        let timestamps: Vec<_> = (0..100).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..100)
            .map(|i| 100.0 + 10.0 * (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin())
            .collect();
        TimeSeriesData::new(timestamps, values, "seasonal").unwrap()
    };
    datasets.insert("Seasonal_Pattern".to_string(), seasonal_data);

    // 3. Trend + Seasonal + Noise
    let complex_data = {
        let timestamps: Vec<_> = (0..100).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..100)
            .map(|i| {
                let trend = 100.0 + i as f64 * 0.3;
                let seasonal = 5.0 * (i as f64 * 2.0 * std::f64::consts::PI / 7.0).sin();
                let noise = (i as f64 * 0.1).sin() * 2.0;
                trend + seasonal + noise
            })
            .collect();
        TimeSeriesData::new(timestamps, values, "complex").unwrap()
    };
    datasets.insert("Complex_Pattern".to_string(), complex_data);

    // 4. Random Walk
    let random_walk_data = {
        let timestamps: Vec<_> = (0..100).map(|i| start_time + Duration::days(i)).collect();
        let mut values = vec![100.0];
        for i in 1..100 {
            let change = (i as f64 * 0.1).sin() * 0.5; // Deterministic "random" walk
            values.push(values[i - 1] + change);
        }
        TimeSeriesData::new(timestamps, values, "random_walk").unwrap()
    };
    datasets.insert("Random_Walk".to_string(), random_walk_data);

    datasets
}

fn load_financial_data() -> std::result::Result<TimeSeriesData, Box<dyn std::error::Error>> {
    // Try to load the AAPL data from the examples
    use std::fs;
    let content = fs::read_to_string("examples/csv/AAPL_daily_ohlcv.csv")?;
    let lines: Vec<&str> = content.lines().collect();

    if lines.len() < 50 {
        return Err("Insufficient financial data".into());
    }

    let data_lines = &lines[1..101]; // Take first 100 rows for consistency
    let start_time = Utc::now();

    let mut timestamps = Vec::new();
    let mut values = Vec::new();

    for (i, line) in data_lines.iter().enumerate() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 5 {
            timestamps.push(start_time + Duration::days(i as i64));
            if let Ok(close_price) = parts[4].trim().parse::<f64>() {
                values.push(close_price);
            }
        }
    }

    if values.len() < 50 {
        return Err("Could not parse enough financial data".into());
    }

    TimeSeriesData::new(timestamps, values, "AAPL").map_err(|e| e.into())
}

fn test_models_on_dataset(
    data: &TimeSeriesData,
    dataset_name: &str,
) -> std::result::Result<Vec<AccuracyReport>, Box<dyn std::error::Error>> {
    let mut reports = Vec::new();
    let split_point = (data.len() as f64 * 0.8) as usize;
    let forecast_horizon = data.len() - split_point;

    if split_point < 20 || forecast_horizon < 5 {
        return Ok(reports); // Skip if insufficient data
    }

    let train_data = TimeSeriesData::new(
        data.timestamps[..split_point].to_vec(),
        data.values[..split_point].to_vec(),
        "train",
    )?;

    let test_values = &data.values[split_point..];

    // Test different models
    let models_to_test = vec![
        (
            "MA(5)",
            Box::new(|| ModelBuilder::moving_average().with_window(5).build())
                as Box<dyn Fn() -> std::result::Result<Box<dyn QuickForecaster>, OxiError>>,
        ),
        (
            "MA(10)",
            Box::new(|| ModelBuilder::moving_average().with_window(10).build()),
        ),
        (
            "ES(α=0.3)",
            Box::new(|| {
                ModelBuilder::exponential_smoothing()
                    .with_alpha(0.3)
                    .build()
            }),
        ),
        (
            "ES(α=0.7)",
            Box::new(|| {
                ModelBuilder::exponential_smoothing()
                    .with_alpha(0.7)
                    .build()
            }),
        ),
        (
            "ARIMA(1,1,1)",
            Box::new(|| {
                ModelBuilder::arima()
                    .with_ar(1)
                    .with_differencing(1)
                    .with_ma(1)
                    .build()
            }),
        ),
        (
            "ARIMA(2,1,2)",
            Box::new(|| {
                ModelBuilder::arima()
                    .with_ar(2)
                    .with_differencing(1)
                    .with_ma(2)
                    .build()
            }),
        ),
    ];

    for (model_name, model_builder) in models_to_test {
        if let Ok(mut model) = model_builder() {
            if model.quick_fit(&train_data).is_ok() {
                if let Ok(forecast) = model.quick_forecast(forecast_horizon) {
                    let forecast_slice = &forecast[..test_values.len().min(forecast.len())];
                    let test_slice = &test_values[..forecast_slice.len()];

                    let report = AccuracyReport::new(
                        model_name.to_string(),
                        dataset_name.to_string(),
                        test_slice,
                        forecast_slice,
                        forecast_horizon,
                        split_point,
                    );

                    println!(
                        "     ✓ {} - MAE: {:.4}, RMSE: {:.4}, MAPE: {:.2}%",
                        model_name, report.mae, report.rmse, report.mape
                    );
                    reports.push(report);
                }
            }
        }
    }

    Ok(reports)
}

fn cross_validation_analysis(
) -> std::result::Result<Vec<AccuracyReport>, Box<dyn std::error::Error>> {
    // Generate a test dataset for cross-validation
    let start_time = Utc::now();
    let timestamps: Vec<_> = (0..150).map(|i| start_time + Duration::days(i)).collect();
    let values: Vec<f64> = (0..150)
        .map(|i| 100.0 + i as f64 * 0.2 + (i as f64 * 0.1).sin() * 3.0)
        .collect();
    let data = TimeSeriesData::new(timestamps, values, "cv_test")?;

    let mut reports = Vec::new();
    let folds = 5;
    let fold_size = data.len() / folds;

    println!("   🔄 Performing {}-fold cross-validation...", folds);

    // Test ARIMA(1,1,1) with cross-validation
    for fold in 0..folds {
        let test_start = fold * fold_size;
        let test_end = ((fold + 1) * fold_size).min(data.len());

        if test_start + 20 >= test_end {
            continue;
        } // Skip if insufficient data

        // Create training data (everything except current fold)
        let mut train_timestamps = Vec::new();
        let mut train_values = Vec::new();

        // Add data before test fold
        train_timestamps.extend_from_slice(&data.timestamps[..test_start]);
        train_values.extend_from_slice(&data.values[..test_start]);

        // Add data after test fold
        if test_end < data.len() {
            train_timestamps.extend_from_slice(&data.timestamps[test_end..]);
            train_values.extend_from_slice(&data.values[test_end..]);
        }

        if train_values.len() < 20 {
            continue;
        }

        let train_data =
            TimeSeriesData::new(train_timestamps, train_values, &format!("cv_fold_{}", fold))?;
        let test_values = &data.values[test_start..test_end];

        if let Ok(mut model) = ModelBuilder::arima()
            .with_ar(1)
            .with_differencing(1)
            .with_ma(1)
            .build()
        {
            if model.quick_fit(&train_data).is_ok() {
                if let Ok(forecast) = model.quick_forecast(test_values.len()) {
                    let report = AccuracyReport::new(
                        format!("ARIMA(1,1,1)_CV_Fold_{}", fold),
                        "Cross_Validation".to_string(),
                        test_values,
                        &forecast[..test_values.len()],
                        test_values.len(),
                        train_data.len(),
                    );
                    reports.push(report);
                }
            }
        }
    }

    Ok(reports)
}

fn generate_summary_report(reports: &[AccuracyReport]) {
    println!("\n📊 SUMMARY BY DATASET:");
    let datasets: std::collections::HashSet<String> =
        reports.iter().map(|r| r.dataset_name.clone()).collect();

    for dataset in datasets {
        let dataset_reports: Vec<_> = reports
            .iter()
            .filter(|r| r.dataset_name == dataset)
            .collect();
        if dataset_reports.is_empty() {
            continue;
        }

        println!("\n🔹 Dataset: {}", dataset);
        println!("   Models tested: {}", dataset_reports.len());

        let avg_mae =
            dataset_reports.iter().map(|r| r.mae).sum::<f64>() / dataset_reports.len() as f64;
        let avg_rmse =
            dataset_reports.iter().map(|r| r.rmse).sum::<f64>() / dataset_reports.len() as f64;
        let avg_mape =
            dataset_reports.iter().map(|r| r.mape).sum::<f64>() / dataset_reports.len() as f64;

        println!("   Average MAE:  {:.4}", avg_mae);
        println!("   Average RMSE: {:.4}", avg_rmse);
        println!("   Average MAPE: {:.2}%", avg_mape);

        let best = dataset_reports
            .iter()
            .min_by(|a, b| a.mae.partial_cmp(&b.mae).unwrap())
            .unwrap();
        println!(
            "   🏆 Best model: {} (MAE: {:.4})",
            best.model_name, best.mae
        );
    }
}

fn generate_detailed_report(reports: &[AccuracyReport]) {
    println!("\n📋 DETAILED ACCURACY METRICS:");
    println!(
        "{:<20} {:<15} {:<8} {:<8} {:<8} {:<8} {:<8}",
        "Model", "Dataset", "MAE", "RMSE", "MAPE%", "R²", "Dir%"
    );
    println!("{}", "-".repeat(95));

    for report in reports {
        println!(
            "{:<20} {:<15} {:<8.4} {:<8.4} {:<8.2} {:<8.4} {:<8.2}",
            report.model_name,
            report.dataset_name,
            report.mae,
            report.rmse,
            report.mape,
            report.r_squared,
            report.directional_accuracy
        );
    }
}

fn generate_rankings(reports: &[AccuracyReport]) {
    println!("\n🏆 MODEL RANKINGS:");

    // Overall ranking by MAE
    let mut mae_ranking = reports.to_vec();
    mae_ranking.sort_by(|a, b| a.mae.partial_cmp(&b.mae).unwrap());

    println!("\n📈 Best Models by MAE (Overall):");
    for (i, report) in mae_ranking.iter().take(5).enumerate() {
        println!(
            "   {}. {} on {} - MAE: {:.4}",
            i + 1,
            report.model_name,
            report.dataset_name,
            report.mae
        );
    }

    // Ranking by R²
    let mut r2_ranking = reports.to_vec();
    r2_ranking.sort_by(|a, b| b.r_squared.partial_cmp(&a.r_squared).unwrap());

    println!("\n📊 Best Models by R² (Overall):");
    for (i, report) in r2_ranking.iter().take(5).enumerate() {
        println!(
            "   {}. {} on {} - R²: {:.4}",
            i + 1,
            report.model_name,
            report.dataset_name,
            report.r_squared
        );
    }
}

// Accuracy metric calculation functions
fn calculate_mae(actual: &[f64], predicted: &[f64]) -> f64 {
    actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>()
        / actual.len() as f64
}

fn calculate_rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    let mse = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>()
        / actual.len() as f64;
    mse.sqrt()
}

fn calculate_mape(actual: &[f64], predicted: &[f64]) -> f64 {
    let mape = actual
        .iter()
        .zip(predicted.iter())
        .filter(|(a, _)| **a != 0.0)
        .map(|(a, p)| ((a - p) / a).abs())
        .sum::<f64>()
        / actual.len() as f64;
    mape * 100.0
}

fn calculate_r_squared(actual: &[f64], predicted: &[f64]) -> f64 {
    let mean_actual = actual.iter().sum::<f64>() / actual.len() as f64;
    let ss_tot = actual
        .iter()
        .map(|a| (a - mean_actual).powi(2))
        .sum::<f64>();
    let ss_res = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>();

    if ss_tot == 0.0 {
        0.0
    } else {
        1.0 - ss_res / ss_tot
    }
}

fn calculate_directional_accuracy(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() < 2 {
        return 0.0;
    }

    let correct_directions = actual
        .windows(2)
        .zip(predicted.windows(2))
        .map(|(a_window, p_window)| {
            let actual_direction = a_window[1] > a_window[0];
            let predicted_direction = p_window[1] > p_window[0];
            if actual_direction == predicted_direction {
                1.0
            } else {
                0.0
            }
        })
        .sum::<f64>();

    correct_directions / (actual.len() - 1) as f64 * 100.0
}
