//! Performance benchmarks for accuracy improvements
//!
//! This benchmark measures the performance and accuracy of:
//! 1. Enhanced ARIMA models with edge case handling
//! 2. Parameter optimization
//! 3. Ensemble methods
//! 4. Overall improvement vs baseline models

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use chrono::{Duration, Utc};
use oxidiviner::prelude::*;
use oxidiviner::ensemble::{EnsembleBuilder, EnsembleMethod};
use oxidiviner::optimization::{OptimizerBuilder, OptimizationMethod, OptimizationMetric};

fn generate_test_datasets() -> Vec<(&'static str, TimeSeriesData)> {
    let start_time = Utc::now();
    let mut datasets = Vec::new();

    // 1. Trending data
    let trending_data = {
        let timestamps: Vec<_> = (0..100).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5 + (i as f64 * 0.1).sin()).collect();
        TimeSeriesData::new(timestamps, values, "trending").unwrap()
    };
    datasets.push(("trending", trending_data));

    // 2. Seasonal data
    let seasonal_data = {
        let timestamps: Vec<_> = (0..120).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..120)
            .map(|i| {
                let trend = 100.0 + i as f64 * 0.1;
                let seasonal = 15.0 * (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin();
                let noise = (i as f64 * 0.3).sin() * 2.0;
                trend + seasonal + noise
            })
            .collect();
        TimeSeriesData::new(timestamps, values, "seasonal").unwrap()
    };
    datasets.push(("seasonal", seasonal_data));

    // 3. Volatile data
    let volatile_data = {
        let timestamps: Vec<_> = (0..80).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..80)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 20.0 + (i as f64 * 0.3).cos() * 15.0)
            .collect();
        TimeSeriesData::new(timestamps, values, "volatile").unwrap()
    };
    datasets.push(("volatile", volatile_data));

    datasets
}

fn calculate_mae(actual: &[f64], forecast: &[f64]) -> f64 {
    let len = actual.len().min(forecast.len());
    let sum: f64 = actual[..len]
        .iter()
        .zip(forecast[..len].iter())
        .map(|(a, f)| (a - f).abs())
        .sum();
    sum / len as f64
}

/// Benchmark baseline model performance
fn benchmark_baseline_models(c: &mut Criterion) {
    let datasets = generate_test_datasets();
    
    let mut group = c.benchmark_group("baseline_models");
    
    for (name, data) in datasets {
        let split_point = (data.len() as f64 * 0.8) as usize;
        let train_data = TimeSeriesData::new(
            data.timestamps[..split_point].to_vec(),
            data.values[..split_point].to_vec(),
            "train",
        ).unwrap();

        // Benchmark ARIMA(1,1,1)
        group.bench_with_input(
            BenchmarkId::new("ARIMA_baseline", name),
            &train_data,
            |b, train_data| {
                b.iter(|| {
                    let mut model = ARIMAModel::new(1, 1, 1, false).unwrap();
                    let _ = model.fit(black_box(train_data));
                    let _ = model.forecast(black_box(10));
                })
            },
        );

        // Benchmark Enhanced ARIMA(1,1,1) with edge case handling
        group.bench_with_input(
            BenchmarkId::new("ARIMA_enhanced", name),
            &train_data,
            |b, train_data| {
                b.iter(|| {
                    let mut model = ARIMAModel::new(1, 1, 1, true).unwrap();
                    let _ = model.fit(black_box(train_data));
                    let _ = model.forecast(black_box(10));
                })
            },
        );

        // Benchmark Simple ES
        group.bench_with_input(
            BenchmarkId::new("SimpleES", name),
            &train_data,
            |b, train_data| {
                b.iter(|| {
                    let mut model = SimpleESModel::new(0.3).unwrap();
                    let _ = model.fit(black_box(train_data));
                    let _ = model.forecast(black_box(10));
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark parameter optimization
fn benchmark_optimization(c: &mut Criterion) {
    let datasets = generate_test_datasets();
    
    let mut group = c.benchmark_group("optimization");
    group.sample_size(10); // Reduce sample size for expensive operations
    
    for (name, data) in datasets {
        let split_point = (data.len() as f64 * 0.8) as usize;
        let train_data = TimeSeriesData::new(
            data.timestamps[..split_point].to_vec(),
            data.values[..split_point].to_vec(),
            "train",
        ).unwrap();

        // Benchmark ARIMA optimization
        group.bench_with_input(
            BenchmarkId::new("ARIMA_optimization", name),
            &train_data,
            |b, train_data| {
                b.iter(|| {
                    let optimizer = OptimizerBuilder::new()
                        .method(OptimizationMethod::GridSearch)
                        .metric(OptimizationMetric::MAE)
                        .max_evaluations(10)
                        .build();
                    let _ = optimizer.optimize_arima(black_box(train_data), 2, 1, 2);
                })
            },
        );

        // Benchmark ES optimization
        group.bench_with_input(
            BenchmarkId::new("ES_optimization", name),
            &train_data,
            |b, train_data| {
                b.iter(|| {
                    let optimizer = OptimizerBuilder::new()
                        .max_evaluations(5)
                        .build();
                    let _ = optimizer.optimize_exponential_smoothing(black_box(train_data));
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark ensemble methods
fn benchmark_ensemble_methods(c: &mut Criterion) {
    // Create sample forecasts for ensemble testing
    let forecast1 = vec![100.0, 101.0, 102.0, 103.0, 104.0];
    let forecast2 = vec![99.0, 100.5, 101.5, 102.5, 103.5];
    let forecast3 = vec![101.0, 102.0, 103.0, 104.0, 105.0];
    
    let mut group = c.benchmark_group("ensemble_methods");
    
    // Benchmark Simple Average
    group.bench_function("simple_average", |b| {
        b.iter(|| {
            let ensemble = EnsembleBuilder::new(EnsembleMethod::SimpleAverage)
                .add_model_forecast("Model1".to_string(), black_box(forecast1.clone()))
                .add_model_forecast("Model2".to_string(), black_box(forecast2.clone()))
                .add_model_forecast("Model3".to_string(), black_box(forecast3.clone()))
                .build();
            let _ = ensemble;
        })
    });

    // Benchmark Weighted Average
    group.bench_function("weighted_average", |b| {
        b.iter(|| {
            let ensemble = EnsembleBuilder::new(EnsembleMethod::WeightedAverage)
                .add_weighted_forecast("Model1".to_string(), black_box(forecast1.clone()), 0.5)
                .add_weighted_forecast("Model2".to_string(), black_box(forecast2.clone()), 0.3)
                .add_weighted_forecast("Model3".to_string(), black_box(forecast3.clone()), 0.2)
                .build();
            let _ = ensemble;
        })
    });

    // Benchmark Median Ensemble
    group.bench_function("median_ensemble", |b| {
        b.iter(|| {
            let ensemble = EnsembleBuilder::new(EnsembleMethod::Median)
                .add_model_forecast("Model1".to_string(), black_box(forecast1.clone()))
                .add_model_forecast("Model2".to_string(), black_box(forecast2.clone()))
                .add_model_forecast("Model3".to_string(), black_box(forecast3.clone()))
                .build();
            let _ = ensemble;
        })
    });
    
    group.finish();
}

/// Benchmark accuracy improvements end-to-end
fn benchmark_accuracy_improvements(c: &mut Criterion) {
    let datasets = generate_test_datasets();
    
    let mut group = c.benchmark_group("accuracy_improvements");
    group.sample_size(10); // Reduce sample size for expensive operations
    
    for (name, data) in datasets {
        let split_point = (data.len() as f64 * 0.8) as usize;
        let train_data = TimeSeriesData::new(
            data.timestamps[..split_point].to_vec(),
            data.values[..split_point].to_vec(),
            "train",
        ).unwrap();
        let test_values = &data.values[split_point..];

        // Benchmark full pipeline: baseline -> optimization -> ensemble
        group.bench_with_input(
            BenchmarkId::new("full_pipeline", name),
            &(&train_data, test_values),
            |b, (train_data, test_values)| {
                b.iter(|| {
                    // 1. Baseline model
                    let mut baseline_arima = ARIMAModel::new(1, 1, 1, true).unwrap();
                    let _ = baseline_arima.fit(black_box(train_data));
                    let baseline_forecast = baseline_arima.forecast(test_values.len()).unwrap_or_default();
                    let baseline_mae = calculate_mae(test_values, &baseline_forecast);

                    // 2. Optimization
                    let optimizer = OptimizerBuilder::new()
                        .max_evaluations(5)
                        .build();
                    if let Ok(opt_result) = optimizer.optimize_arima(train_data, 2, 1, 2) {
                        let p = opt_result.best_parameters["p"] as usize;
                        let d = opt_result.best_parameters["d"] as usize;
                        let q = opt_result.best_parameters["q"] as usize;
                        
                        let mut opt_model = ARIMAModel::new(p, d, q, true).unwrap();
                        let _ = opt_model.fit(train_data);
                        let opt_forecast = opt_model.forecast(test_values.len()).unwrap_or_default();

                        // 3. Ensemble
                        if !baseline_forecast.is_empty() && !opt_forecast.is_empty() {
                            let ensemble = EnsembleBuilder::new(EnsembleMethod::SimpleAverage)
                                .add_model_forecast("Baseline".to_string(), black_box(baseline_forecast))
                                .add_model_forecast("Optimized".to_string(), black_box(opt_forecast))
                                .build();
                            let _ = ensemble;
                        }
                    }

                    black_box(baseline_mae)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage and allocation patterns
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let start_time = Utc::now();
    let timestamps: Vec<_> = (0..1000).map(|i| start_time + Duration::days(i)).collect();
    let values: Vec<f64> = (0..1000).map(|i| 100.0 + i as f64 * 0.1).collect();
    let large_data = TimeSeriesData::new(timestamps, values, "large").unwrap();
    
    let mut group = c.benchmark_group("memory_efficiency");
    
    // Test memory allocation for large datasets
    group.bench_function("large_dataset_arima", |b| {
        b.iter(|| {
            let mut model = ARIMAModel::new(2, 1, 2, true).unwrap();
            let _ = model.fit(black_box(&large_data));
            let _ = model.forecast(black_box(50));
        })
    });

    group.bench_function("large_dataset_optimization", |b| {
        b.iter(|| {
            let optimizer = OptimizerBuilder::new()
                .max_evaluations(3)
                .build();
            let _ = optimizer.optimize_arima(black_box(&large_data), 2, 1, 2);
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_baseline_models,
    benchmark_optimization,
    benchmark_ensemble_methods,
    benchmark_accuracy_improvements,
    benchmark_memory_efficiency
);
criterion_main!(benches); 