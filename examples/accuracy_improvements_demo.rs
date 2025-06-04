//! Comprehensive Accuracy Improvements Demo
//!
//! This example demonstrates all the accuracy improvement features:
//! 1. Enhanced edge case handling for model stability
//! 2. Ensemble methods for improved robustness
//! 3. Parameter optimization for better accuracy
//! 4. Best practice recommendations

use chrono::{Duration, Utc};
use oxidiviner::ensemble::{EnsembleBuilder, EnsembleMethod};
use oxidiviner::optimization::{OptimizationMethod, OptimizationMetric, OptimizerBuilder};
use oxidiviner::prelude::*;
use std::collections::HashMap;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🚀 OxiDiviner - Comprehensive Accuracy Improvements Demo");
    println!("=======================================================\n");

    // Generate different types of challenging datasets
    let datasets = generate_challenging_datasets();

    for (name, data) in datasets {
        println!("📊 Testing on {} dataset ({} points)", name, data.len());
        println!("-------------------------------------------");

        // Step 1: Test baseline models
        println!("1️⃣ Baseline Model Performance:");
        let baseline_results = test_baseline_models(&data)?;
        print_baseline_results(&baseline_results);

        // Step 2: Apply parameter optimization
        println!("\n2️⃣ Parameter Optimization:");
        let optimized_results = apply_parameter_optimization(&data)?;
        print_optimization_results(&optimized_results);

        // Step 3: Create ensemble forecasts
        println!("\n3️⃣ Ensemble Methods:");
        let ensemble_results = create_ensemble_forecasts(&data, &optimized_results)?;
        print_ensemble_results(&ensemble_results);

        // Step 4: Show improvement summary
        println!("\n4️⃣ Improvement Summary:");
        show_improvement_summary(&baseline_results, &optimized_results, &ensemble_results);

        println!("\n{}\n", "=".repeat(60));
    }

    // Demonstrate best practice recommendations
    demonstrate_best_practices()?;

    println!("✅ Accuracy improvements demo completed!");
    Ok(())
}

/// Generate datasets with different challenging characteristics
fn generate_challenging_datasets() -> HashMap<String, TimeSeriesData> {
    let mut datasets = HashMap::new();
    let start_time = Utc::now();

    // 1. Near-constant data (edge case for ARIMA)
    let constant_data = {
        let timestamps: Vec<_> = (0..50).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.001).sin()).collect();
        TimeSeriesData::new(timestamps, values, "constant").unwrap()
    };
    datasets.insert("Near_Constant_Data".to_string(), constant_data);

    // 2. High volatility data
    let volatile_data = {
        let timestamps: Vec<_> = (0..100).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 20.0 + (i as f64 * 0.3).cos() * 15.0)
            .collect();
        TimeSeriesData::new(timestamps, values, "volatile").unwrap()
    };
    datasets.insert("High_Volatility_Data".to_string(), volatile_data);

    // 3. Complex seasonal data
    let seasonal_data = {
        let timestamps: Vec<_> = (0..150).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..150)
            .map(|i| {
                let trend = 100.0 + i as f64 * 0.2;
                let seasonal1 = 10.0 * (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin();
                let seasonal2 = 5.0 * (i as f64 * 2.0 * std::f64::consts::PI / 7.0).cos();
                let noise = (i as f64 * 0.7).sin() * 3.0;
                trend + seasonal1 + seasonal2 + noise
            })
            .collect();
        TimeSeriesData::new(timestamps, values, "seasonal").unwrap()
    };
    datasets.insert("Complex_Seasonal_Data".to_string(), seasonal_data);

    datasets
}

/// Test baseline model performance
fn test_baseline_models(
    data: &TimeSeriesData,
) -> std::result::Result<HashMap<String, ModelResult>, Box<dyn std::error::Error>> {
    let mut results = HashMap::new();
    let split_point = (data.len() as f64 * 0.8) as usize;

    let train_data = TimeSeriesData::new(
        data.timestamps[..split_point].to_vec(),
        data.values[..split_point].to_vec(),
        "train",
    )?;
    let test_values = &data.values[split_point..];

    // Test ARIMA(1,1,1) - enhanced with edge case handling
    if let Ok(mut arima_model) = ARIMAModel::new(1, 1, 1, true) {
        if let Ok(()) = arima_model.fit(&train_data) {
            if let Ok(forecast) = arima_model.forecast(test_values.len()) {
                let mae = calculate_mae(test_values, &forecast[..test_values.len()]);
                results.insert("ARIMA(1,1,1)".to_string(), ModelResult { mae, forecast });
            }
        }
    }

    // Test Simple ES
    if let Ok(mut es_model) = SimpleESModel::new(0.3) {
        if let Ok(()) = es_model.fit(&train_data) {
            if let Ok(forecast) = es_model.forecast(test_values.len()) {
                let mae = calculate_mae(test_values, &forecast[..test_values.len()]);
                results.insert("SimpleES(0.3)".to_string(), ModelResult { mae, forecast });
            }
        }
    }

    // Test Moving Average
    if let Ok(mut ma_model) = MAModel::new(5) {
        if let Ok(()) = ma_model.fit(&train_data) {
            if let Ok(forecast) = ma_model.forecast(test_values.len()) {
                let mae = calculate_mae(test_values, &forecast[..test_values.len()]);
                results.insert("MA(5)".to_string(), ModelResult { mae, forecast });
            }
        }
    }

    Ok(results)
}

/// Apply parameter optimization to improve accuracy
fn apply_parameter_optimization(
    data: &TimeSeriesData,
) -> std::result::Result<HashMap<String, ModelResult>, Box<dyn std::error::Error>> {
    let mut results = HashMap::new();
    let split_point = (data.len() as f64 * 0.8) as usize;

    let train_data = TimeSeriesData::new(
        data.timestamps[..split_point].to_vec(),
        data.values[..split_point].to_vec(),
        "train",
    )?;
    let test_values = &data.values[split_point..];

    // Optimize ARIMA parameters
    let optimizer = OptimizerBuilder::new()
        .method(OptimizationMethod::GridSearch)
        .metric(OptimizationMetric::MAE)
        .max_evaluations(20)
        .build();

    if let Ok(arima_opt) = optimizer.optimize_arima(&train_data, 3, 2, 3) {
        let p = arima_opt.best_parameters["p"] as usize;
        let d = arima_opt.best_parameters["d"] as usize;
        let q = arima_opt.best_parameters["q"] as usize;

        if let Ok(mut model) = ARIMAModel::new(p, d, q, true) {
            if let Ok(()) = model.fit(&train_data) {
                if let Ok(forecast) = model.forecast(test_values.len()) {
                    let mae = calculate_mae(test_values, &forecast[..test_values.len()]);
                    results.insert(
                        format!("Optimized_ARIMA({},{},{})", p, d, q),
                        ModelResult { mae, forecast },
                    );
                }
            }
        }
    }

    // Optimize ES parameters
    if let Ok(es_opt) = optimizer.optimize_exponential_smoothing(&train_data) {
        let alpha = es_opt.best_parameters["alpha"];

        if let Ok(mut model) = SimpleESModel::new(alpha) {
            if let Ok(()) = model.fit(&train_data) {
                if let Ok(forecast) = model.forecast(test_values.len()) {
                    let mae = calculate_mae(test_values, &forecast[..test_values.len()]);
                    results.insert(
                        format!("Optimized_ES(α={:.2})", alpha),
                        ModelResult { mae, forecast },
                    );
                }
            }
        }
    }

    // Optimize MA parameters
    if let Ok(ma_opt) = optimizer.optimize_moving_average(&train_data, 15) {
        let window = ma_opt.best_parameters["window"] as usize;

        if let Ok(mut model) = MAModel::new(window) {
            if let Ok(()) = model.fit(&train_data) {
                if let Ok(forecast) = model.forecast(test_values.len()) {
                    let mae = calculate_mae(test_values, &forecast[..test_values.len()]);
                    results.insert(
                        format!("Optimized_MA({})", window),
                        ModelResult { mae, forecast },
                    );
                }
            }
        }
    }

    Ok(results)
}

/// Create ensemble forecasts for improved robustness
fn create_ensemble_forecasts(
    data: &TimeSeriesData,
    optimized_models: &HashMap<String, ModelResult>,
) -> std::result::Result<HashMap<String, ModelResult>, Box<dyn std::error::Error>> {
    let mut results = HashMap::new();
    let test_values = &data.values[(data.len() as f64 * 0.8) as usize..];

    if optimized_models.len() >= 2 {
        // Simple Average Ensemble
        let simple_ensemble = EnsembleBuilder::new(EnsembleMethod::SimpleAverage);
        let mut builder = simple_ensemble;

        for (name, result) in optimized_models {
            builder = builder.add_model_forecast(name.clone(), result.forecast.clone());
        }

        if let Ok(ensemble) = builder.build() {
            if let Some(forecast) = ensemble.get_forecast() {
                let mae = calculate_mae(test_values, &forecast[..test_values.len()]);
                results.insert(
                    "Simple_Ensemble".to_string(),
                    ModelResult {
                        mae,
                        forecast: forecast.clone(),
                    },
                );
            }
        }

        // Weighted Ensemble (weight by inverse MAE)
        let weighted_ensemble = EnsembleBuilder::new(EnsembleMethod::WeightedAverage);
        let mut builder = weighted_ensemble;

        for (name, result) in optimized_models {
            let weight = if result.mae > 0.0 {
                1.0 / result.mae
            } else {
                1.0
            };
            builder = builder.add_weighted_forecast(name.clone(), result.forecast.clone(), weight);
        }

        if let Ok(ensemble) = builder.build() {
            if let Some(forecast) = ensemble.get_forecast() {
                let mae = calculate_mae(test_values, &forecast[..test_values.len()]);
                results.insert(
                    "Weighted_Ensemble".to_string(),
                    ModelResult {
                        mae,
                        forecast: forecast.clone(),
                    },
                );
            }
        }

        // Median Ensemble (robust to outliers)
        let median_ensemble = EnsembleBuilder::new(EnsembleMethod::Median);
        let mut builder = median_ensemble;

        for (name, result) in optimized_models {
            builder = builder.add_model_forecast(name.clone(), result.forecast.clone());
        }

        if let Ok(ensemble) = builder.build() {
            if let Some(forecast) = ensemble.get_forecast() {
                let mae = calculate_mae(test_values, &forecast[..test_values.len()]);
                results.insert(
                    "Median_Ensemble".to_string(),
                    ModelResult {
                        mae,
                        forecast: forecast.clone(),
                    },
                );
            }
        }
    }

    Ok(results)
}

/// Print baseline results
fn print_baseline_results(results: &HashMap<String, ModelResult>) {
    for (name, result) in results {
        println!("   {}: MAE = {:.4}", name, result.mae);
    }
}

/// Print optimization results
fn print_optimization_results(results: &HashMap<String, ModelResult>) {
    for (name, result) in results {
        println!("   {}: MAE = {:.4}", name, result.mae);
    }
}

/// Print ensemble results
fn print_ensemble_results(results: &HashMap<String, ModelResult>) {
    for (name, result) in results {
        println!("   {}: MAE = {:.4}", name, result.mae);
    }
}

/// Show improvement summary
fn show_improvement_summary(
    baseline: &HashMap<String, ModelResult>,
    optimized: &HashMap<String, ModelResult>,
    ensemble: &HashMap<String, ModelResult>,
) {
    let best_baseline = baseline
        .values()
        .map(|r| r.mae)
        .fold(f64::INFINITY, f64::min);
    let best_optimized = optimized
        .values()
        .map(|r| r.mae)
        .fold(f64::INFINITY, f64::min);
    let best_ensemble = ensemble
        .values()
        .map(|r| r.mae)
        .fold(f64::INFINITY, f64::min);

    println!("   Best Baseline MAE: {:.4}", best_baseline);
    println!("   Best Optimized MAE: {:.4}", best_optimized);
    println!("   Best Ensemble MAE: {:.4}", best_ensemble);

    if best_baseline > 0.0 {
        let opt_improvement = (best_baseline - best_optimized) / best_baseline * 100.0;
        let ens_improvement = (best_baseline - best_ensemble) / best_baseline * 100.0;

        println!("   💡 Optimization improvement: {:.1}%", opt_improvement);
        println!("   🤝 Ensemble improvement: {:.1}%", ens_improvement);
    }
}

/// Demonstrate best practice recommendations
fn demonstrate_best_practices() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("📋 BEST PRACTICE RECOMMENDATIONS");
    println!("================================");

    println!("\n1️⃣ Model Selection Guidelines:");
    println!("   • Use Holt's Linear Method for trending data");
    println!("   • Use ARIMA(2,1,2) for complex autoregressive patterns");
    println!("   • Use Simple ES for stable, level data");
    println!("   • Use ensembles when accuracy is critical");

    println!("\n2️⃣ Edge Case Handling:");
    println!("   • Enhanced ARIMA with robust coefficient validation");
    println!("   • Automatic fallback mechanisms for singular matrices");
    println!("   • Stationarity enforcement and regularization");

    println!("\n3️⃣ Parameter Optimization:");
    println!("   • Use grid search for systematic exploration");
    println!("   • Limit evaluation budget (20-50 evaluations)");
    println!("   • Choose appropriate metrics (MAE for robustness)");

    println!("\n4️⃣ Ensemble Methods:");
    println!("   • Simple average for balanced performance");
    println!("   • Weighted average for known model quality");
    println!("   • Median ensemble for outlier robustness");

    println!("\n5️⃣ Validation Strategy:");
    println!("   • Use time series cross-validation");
    println!("   • Maintain temporal order in splits");
    println!("   • Test on multiple datasets");

    Ok(())
}

/// Helper structure for model results
#[derive(Debug, Clone)]
struct ModelResult {
    mae: f64,
    forecast: Vec<f64>,
}

/// Calculate MAE helper function
fn calculate_mae(actual: &[f64], forecast: &[f64]) -> f64 {
    let len = actual.len().min(forecast.len());
    let sum: f64 = actual[..len]
        .iter()
        .zip(forecast[..len].iter())
        .map(|(a, f)| (a - f).abs())
        .sum();
    sum / len as f64
}
