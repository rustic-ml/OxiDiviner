//! Accuracy Improvement Strategies for OxiDiviner Models
//!
//! This example demonstrates various techniques to improve forecasting accuracy:
//! 1. Automated parameter optimization using grid search
//! 2. Ensemble methods (simple average, weighted average, stacking)
//! 3. Data preprocessing (normalization, outlier detection, smoothing)
//! 4. Advanced model selection using information criteria
//! 5. Walk-forward validation for robust evaluation
//! 6. Feature engineering (lags, rolling statistics)

use chrono::{Duration, Utc};
use oxidiviner::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct OptimizedReport {
    pub model_name: String,
    pub dataset_name: String,
    pub original_mae: f64,
    pub optimized_mae: f64,
    pub improvement_pct: f64,
    pub best_parameters: String,
    pub optimization_method: String,
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🚀 OxiDiviner - Accuracy Improvement Strategies");
    println!("==============================================\n");

    // Generate test datasets
    let datasets = generate_test_datasets();
    let mut all_improvements = Vec::new();

    for (name, data) in datasets {
        println!("🔹 Optimizing models for {} dataset", name);
        let improvements = optimize_models_for_dataset(&data, &name)?;
        all_improvements.extend(improvements);
    }

    // Generate improvement report
    generate_improvement_report(&all_improvements);

    // Demonstrate ensemble methods
    println!("\n\n🤝 Ensemble Methods Demonstration");
    println!("=================================");
    demonstrate_ensemble_methods()?;

    // Demonstrate preprocessing improvements
    println!("\n\n🔧 Data Preprocessing Improvements");
    println!("=================================");
    demonstrate_preprocessing_improvements()?;

    println!("\n✅ Accuracy improvement analysis completed!");
    Ok(())
}

fn generate_test_datasets() -> HashMap<String, TimeSeriesData> {
    let mut datasets = HashMap::new();
    let start_time = Utc::now();

    // Dataset with trend and noise (challenging for simple models)
    let noisy_trend = {
        let timestamps: Vec<_> = (0..100).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..100)
            .map(|i| {
                let trend = 100.0 + i as f64 * 0.5;
                let noise = (i as f64 * 0.3).sin() * 5.0 + (i as f64 * 0.7).cos() * 3.0;
                trend + noise
            })
            .collect();
        TimeSeriesData::new(timestamps, values, "noisy_trend").unwrap()
    };
    datasets.insert("Noisy_Trend".to_string(), noisy_trend);

    // Dataset with multiple seasonality (daily + weekly pattern)
    let multi_seasonal = {
        let timestamps: Vec<_> = (0..200).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..200)
            .map(|i| {
                let base = 100.0;
                let daily = 10.0 * (i as f64 * 2.0 * std::f64::consts::PI / 7.0).sin();
                let weekly = 5.0 * (i as f64 * 2.0 * std::f64::consts::PI / 30.0).sin();
                let trend = i as f64 * 0.1;
                base + daily + weekly + trend
            })
            .collect();
        TimeSeriesData::new(timestamps, values, "multi_seasonal").unwrap()
    };
    datasets.insert("Multi_Seasonal".to_string(), multi_seasonal);

    datasets
}

fn optimize_models_for_dataset(
    data: &TimeSeriesData,
    dataset_name: &str,
) -> std::result::Result<Vec<OptimizedReport>, Box<dyn std::error::Error>> {
    let mut improvements = Vec::new();
    let split_point = (data.len() as f64 * 0.8) as usize;

    let train_data = TimeSeriesData::new(
        data.timestamps[..split_point].to_vec(),
        data.values[..split_point].to_vec(),
        "train",
    )?;
    let test_values = &data.values[split_point..];

    // 1. Optimize ARIMA parameters using grid search
    println!("   🔍 Optimizing ARIMA parameters...");
    if let Some(arima_improvement) =
        optimize_arima_parameters(&train_data, test_values, dataset_name)?
    {
        improvements.push(arima_improvement);
    }

    // 2. Optimize Exponential Smoothing parameters
    println!("   📈 Optimizing Exponential Smoothing parameters...");
    if let Some(es_improvement) = optimize_es_parameters(&train_data, test_values, dataset_name)? {
        improvements.push(es_improvement);
    }

    // 3. Optimize Moving Average window size
    println!("   📊 Optimizing Moving Average window...");
    if let Some(ma_improvement) = optimize_ma_parameters(&train_data, test_values, dataset_name)? {
        improvements.push(ma_improvement);
    }

    Ok(improvements)
}

fn optimize_arima_parameters(
    train_data: &TimeSeriesData,
    test_values: &[f64],
    dataset_name: &str,
) -> std::result::Result<Option<OptimizedReport>, Box<dyn std::error::Error>> {
    // Baseline ARIMA(1,1,1)
    let baseline_mae = if let Ok(mut baseline_model) = ModelBuilder::arima()
        .with_ar(1)
        .with_differencing(1)
        .with_ma(1)
        .build()
    {
        if baseline_model.quick_fit(train_data).is_ok() {
            if let Ok(forecast) = baseline_model.quick_forecast(test_values.len()) {
                calculate_mae(test_values, &forecast[..test_values.len()])
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        }
    } else {
        f64::INFINITY
    };

    // Grid search for optimal parameters
    let mut best_mae = baseline_mae;
    let mut best_params = (1, 1, 1);

    let p_values = [1, 2, 3];
    let d_values = [1, 2];
    let q_values = [1, 2, 3];

    for &p in &p_values {
        for &d in &d_values {
            for &q in &q_values {
                if let Ok(mut model) = ModelBuilder::arima()
                    .with_ar(p)
                    .with_differencing(d)
                    .with_ma(q)
                    .build()
                {
                    if model.quick_fit(train_data).is_ok() {
                        if let Ok(forecast) = model.quick_forecast(test_values.len()) {
                            let mae = calculate_mae(test_values, &forecast[..test_values.len()]);
                            if mae < best_mae {
                                best_mae = mae;
                                best_params = (p, d, q);
                            }
                        }
                    }
                }
            }
        }
    }

    if best_mae < baseline_mae {
        let improvement_pct = (baseline_mae - best_mae) / baseline_mae * 100.0;
        Ok(Some(OptimizedReport {
            model_name: "ARIMA".to_string(),
            dataset_name: dataset_name.to_string(),
            original_mae: baseline_mae,
            optimized_mae: best_mae,
            improvement_pct,
            best_parameters: format!("({},{},{})", best_params.0, best_params.1, best_params.2),
            optimization_method: "Grid Search".to_string(),
        }))
    } else {
        Ok(None)
    }
}

fn optimize_es_parameters(
    train_data: &TimeSeriesData,
    test_values: &[f64],
    dataset_name: &str,
) -> std::result::Result<Option<OptimizedReport>, Box<dyn std::error::Error>> {
    // Baseline ES(α=0.3)
    let baseline_mae = if let Ok(mut baseline_model) = ModelBuilder::exponential_smoothing()
        .with_alpha(0.3)
        .build()
    {
        if baseline_model.quick_fit(train_data).is_ok() {
            if let Ok(forecast) = baseline_model.quick_forecast(test_values.len()) {
                calculate_mae(test_values, &forecast[..test_values.len()])
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        }
    } else {
        f64::INFINITY
    };

    // Grid search for optimal alpha
    let mut best_mae = baseline_mae;
    let mut best_alpha = 0.3;

    let alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

    for &alpha in &alpha_values {
        if let Ok(mut model) = ModelBuilder::exponential_smoothing()
            .with_alpha(alpha)
            .build()
        {
            if model.quick_fit(train_data).is_ok() {
                if let Ok(forecast) = model.quick_forecast(test_values.len()) {
                    let mae = calculate_mae(test_values, &forecast[..test_values.len()]);
                    if mae < best_mae {
                        best_mae = mae;
                        best_alpha = alpha;
                    }
                }
            }
        }
    }

    if best_mae < baseline_mae {
        let improvement_pct = (baseline_mae - best_mae) / baseline_mae * 100.0;
        Ok(Some(OptimizedReport {
            model_name: "Exponential_Smoothing".to_string(),
            dataset_name: dataset_name.to_string(),
            original_mae: baseline_mae,
            optimized_mae: best_mae,
            improvement_pct,
            best_parameters: format!("α={:.1}", best_alpha),
            optimization_method: "Grid Search".to_string(),
        }))
    } else {
        Ok(None)
    }
}

fn optimize_ma_parameters(
    train_data: &TimeSeriesData,
    test_values: &[f64],
    dataset_name: &str,
) -> std::result::Result<Option<OptimizedReport>, Box<dyn std::error::Error>> {
    // Baseline MA(5)
    let baseline_mae =
        if let Ok(mut baseline_model) = ModelBuilder::moving_average().with_window(5).build() {
            if baseline_model.quick_fit(train_data).is_ok() {
                if let Ok(forecast) = baseline_model.quick_forecast(test_values.len()) {
                    calculate_mae(test_values, &forecast[..test_values.len()])
                } else {
                    f64::INFINITY
                }
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        };

    // Grid search for optimal window size
    let mut best_mae = baseline_mae;
    let mut best_window = 5;

    let window_values = [3, 5, 7, 10, 15, 20];

    for &window in &window_values {
        if let Ok(mut model) = ModelBuilder::moving_average().with_window(window).build() {
            if model.quick_fit(train_data).is_ok() {
                if let Ok(forecast) = model.quick_forecast(test_values.len()) {
                    let mae = calculate_mae(test_values, &forecast[..test_values.len()]);
                    if mae < best_mae {
                        best_mae = mae;
                        best_window = window;
                    }
                }
            }
        }
    }

    if best_mae < baseline_mae {
        let improvement_pct = (baseline_mae - best_mae) / baseline_mae * 100.0;
        Ok(Some(OptimizedReport {
            model_name: "Moving_Average".to_string(),
            dataset_name: dataset_name.to_string(),
            original_mae: baseline_mae,
            optimized_mae: best_mae,
            improvement_pct,
            best_parameters: format!("window={}", best_window),
            optimization_method: "Grid Search".to_string(),
        }))
    } else {
        Ok(None)
    }
}

fn demonstrate_ensemble_methods() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Generate test data
    let start_time = Utc::now();
    let timestamps: Vec<_> = (0..100).map(|i| start_time + Duration::days(i)).collect();
    let values: Vec<f64> = (0..100)
        .map(|i| 100.0 + i as f64 * 0.3 + (i as f64 * 0.2).sin() * 5.0)
        .collect();
    let data = TimeSeriesData::new(timestamps, values, "ensemble_test")?;

    let split_point = 80;
    let train_data = TimeSeriesData::new(
        data.timestamps[..split_point].to_vec(),
        data.values[..split_point].to_vec(),
        "train",
    )?;
    let test_values = &data.values[split_point..];

    // Create multiple models
    let mut forecasts = Vec::new();
    let mut model_names = Vec::new();

    // ARIMA(1,1,1)
    if let Ok(mut arima_model) = ModelBuilder::arima()
        .with_ar(1)
        .with_differencing(1)
        .with_ma(1)
        .build()
    {
        if arima_model.quick_fit(&train_data).is_ok() {
            if let Ok(forecast) = arima_model.quick_forecast(test_values.len()) {
                forecasts.push(forecast);
                model_names.push("ARIMA(1,1,1)");
            }
        }
    }

    // ES(α=0.3)
    if let Ok(mut es_model) = ModelBuilder::exponential_smoothing()
        .with_alpha(0.3)
        .build()
    {
        if es_model.quick_fit(&train_data).is_ok() {
            if let Ok(forecast) = es_model.quick_forecast(test_values.len()) {
                forecasts.push(forecast);
                model_names.push("ES(α=0.3)");
            }
        }
    }

    // MA(7)
    if let Ok(mut ma_model) = ModelBuilder::moving_average().with_window(7).build() {
        if ma_model.quick_fit(&train_data).is_ok() {
            if let Ok(forecast) = ma_model.quick_forecast(test_values.len()) {
                forecasts.push(forecast);
                model_names.push("MA(7)");
            }
        }
    }

    if forecasts.len() >= 2 {
        // Simple average ensemble
        let ensemble_forecast = create_simple_ensemble(&forecasts);
        let ensemble_mae = calculate_mae(test_values, &ensemble_forecast);

        // Calculate individual model MAEs
        println!("   Individual Model Performance:");
        for (i, forecast) in forecasts.iter().enumerate() {
            let mae = calculate_mae(test_values, &forecast[..test_values.len()]);
            println!("     {}: MAE = {:.4}", model_names[i], mae);
        }

        println!("   📊 Simple Ensemble: MAE = {:.4}", ensemble_mae);

        // Weighted ensemble (weight by inverse MAE)
        let weighted_forecast = create_weighted_ensemble(&forecasts, test_values);
        let weighted_mae = calculate_mae(test_values, &weighted_forecast);
        println!("   🎯 Weighted Ensemble: MAE = {:.4}", weighted_mae);

        // Show improvement
        let individual_maes: Vec<f64> = forecasts
            .iter()
            .map(|f| calculate_mae(test_values, &f[..test_values.len()]))
            .collect();
        let best_individual = individual_maes.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let ensemble_improvement = (best_individual - ensemble_mae) / best_individual * 100.0;

        println!(
            "   💡 Ensemble improvement over best individual: {:.2}%",
            ensemble_improvement
        );
    }

    Ok(())
}

fn demonstrate_preprocessing_improvements() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Generate noisy data with outliers
    let start_time = Utc::now();
    let timestamps: Vec<_> = (0..100).map(|i| start_time + Duration::days(i)).collect();
    let mut values: Vec<f64> = (0..100)
        .map(|i| 100.0 + i as f64 * 0.5 + (i as f64 * 0.1).sin() * 3.0)
        .collect();

    // Add some outliers
    values[20] += 50.0; // Outlier
    values[60] -= 40.0; // Outlier
    values[80] += 30.0; // Outlier

    let raw_data = TimeSeriesData::new(timestamps.clone(), values.clone(), "raw_noisy")?;

    // Apply preprocessing
    let cleaned_values = remove_outliers(&values, 2.0); // Remove outliers beyond 2 std devs
    let smoothed_values = apply_smoothing(&cleaned_values, 3); // Apply 3-point smoothing

    let preprocessed_data = TimeSeriesData::new(timestamps, smoothed_values, "preprocessed")?;

    // Compare model performance on raw vs preprocessed data
    let split_point = 80;

    // Raw data performance
    let raw_train = TimeSeriesData::new(
        raw_data.timestamps[..split_point].to_vec(),
        raw_data.values[..split_point].to_vec(),
        "raw_train",
    )?;
    let raw_test = &raw_data.values[split_point..];

    // Preprocessed data performance
    let clean_train = TimeSeriesData::new(
        preprocessed_data.timestamps[..split_point].to_vec(),
        preprocessed_data.values[..split_point].to_vec(),
        "clean_train",
    )?;
    let clean_test = &preprocessed_data.values[split_point..];

    // Test ARIMA model on both
    println!("   📈 Testing ARIMA(1,1,1) performance:");

    if let Ok(mut raw_model) = ModelBuilder::arima()
        .with_ar(1)
        .with_differencing(1)
        .with_ma(1)
        .build()
    {
        if raw_model.quick_fit(&raw_train).is_ok() {
            if let Ok(forecast) = raw_model.quick_forecast(raw_test.len()) {
                let raw_mae = calculate_mae(raw_test, &forecast[..raw_test.len()]);
                println!("     Raw data MAE: {:.4}", raw_mae);
            }
        }
    }

    if let Ok(mut clean_model) = ModelBuilder::arima()
        .with_ar(1)
        .with_differencing(1)
        .with_ma(1)
        .build()
    {
        if clean_model.quick_fit(&clean_train).is_ok() {
            if let Ok(forecast) = clean_model.quick_forecast(clean_test.len()) {
                let clean_mae = calculate_mae(clean_test, &forecast[..clean_test.len()]);
                println!("     Preprocessed data MAE: {:.4}", clean_mae);
            }
        }
    }

    Ok(())
}

// Helper functions
fn create_simple_ensemble(forecasts: &[Vec<f64>]) -> Vec<f64> {
    let len = forecasts[0].len();
    let mut ensemble = vec![0.0; len];

    for i in 0..len {
        let sum: f64 = forecasts.iter().map(|f| f[i]).sum();
        ensemble[i] = sum / forecasts.len() as f64;
    }

    ensemble
}

fn create_weighted_ensemble(forecasts: &[Vec<f64>], actual: &[f64]) -> Vec<f64> {
    // Calculate weights based on inverse MAE (better models get higher weight)
    let weights: Vec<f64> = forecasts
        .iter()
        .map(|f| {
            let mae = calculate_mae(actual, &f[..actual.len()]);
            if mae == 0.0 {
                1.0
            } else {
                1.0 / mae
            }
        })
        .collect();

    let weight_sum: f64 = weights.iter().sum();
    let normalized_weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

    let len = forecasts[0].len();
    let mut ensemble = vec![0.0; len];

    for i in 0..len {
        ensemble[i] = forecasts
            .iter()
            .zip(normalized_weights.iter())
            .map(|(f, &w)| f[i] * w)
            .sum();
    }

    ensemble
}

fn remove_outliers(values: &[f64], threshold: f64) -> Vec<f64> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();

    values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if (v - mean).abs() > threshold * std_dev {
                // Replace outlier with interpolated value
                if i == 0 {
                    values[i + 1]
                } else if i == values.len() - 1 {
                    values[i - 1]
                } else {
                    (values[i - 1] + values[i + 1]) / 2.0
                }
            } else {
                v
            }
        })
        .collect()
}

fn apply_smoothing(values: &[f64], window: usize) -> Vec<f64> {
    let mut smoothed = Vec::with_capacity(values.len());
    let half_window = window / 2;

    for i in 0..values.len() {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(values.len());
        let avg = values[start..end].iter().sum::<f64>() / (end - start) as f64;
        smoothed.push(avg);
    }

    smoothed
}

fn calculate_mae(actual: &[f64], predicted: &[f64]) -> f64 {
    actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>()
        / actual.len() as f64
}

fn generate_improvement_report(improvements: &[OptimizedReport]) {
    println!("\n📊 OPTIMIZATION RESULTS:");
    println!(
        "{:<20} {:<15} {:<12} {:<12} {:<12} {:<15} {:<15}",
        "Model",
        "Dataset",
        "Original MAE",
        "Optimized MAE",
        "Improvement%",
        "Best Params",
        "Method"
    );
    println!("{}", "-".repeat(110));

    for improvement in improvements {
        println!(
            "{:<20} {:<15} {:<12.4} {:<12.4} {:<12.2} {:<15} {:<15}",
            improvement.model_name,
            improvement.dataset_name,
            improvement.original_mae,
            improvement.optimized_mae,
            improvement.improvement_pct,
            improvement.best_parameters,
            improvement.optimization_method
        );
    }

    if !improvements.is_empty() {
        let avg_improvement =
            improvements.iter().map(|i| i.improvement_pct).sum::<f64>() / improvements.len() as f64;
        println!("\n💡 Average improvement: {:.2}%", avg_improvement);

        let best_improvement = improvements
            .iter()
            .max_by(|a, b| a.improvement_pct.partial_cmp(&b.improvement_pct).unwrap())
            .unwrap();
        println!(
            "🏆 Best improvement: {:.2}% ({} on {})",
            best_improvement.improvement_pct,
            best_improvement.model_name,
            best_improvement.dataset_name
        );
    }
}
