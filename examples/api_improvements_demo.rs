//! Comprehensive demo of OxiDiviner API improvements
//!
//! This example demonstrates all the enhanced API features:
//! 1. Unified forecasting interface
//! 2. Builder pattern for fluent model construction
//! 3. Automatic model selection with various criteria
//! 4. Parameter validation and error handling
//! 5. Confidence intervals and model evaluation

use chrono::{Duration, Utc};
use oxidiviner::prelude::*;
use oxidiviner::{api::*, builder::*};

fn main() -> Result<()> {
    println!("🔮 OxiDiviner API Improvements Demo\n");

    // Generate sample time series data
    let start_time = Utc::now();
    let timestamps: Vec<_> = (0..50).map(|i| start_time + Duration::days(i)).collect();

    // Create a trend + seasonal + noise pattern
    let values: Vec<f64> = (0..50)
        .map(|i| {
            let trend = 100.0 + i as f64 * 0.5;
            let seasonal = 10.0 * (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin();
            let noise = (i as f64 * 0.1).sin() * 2.0;
            trend + seasonal + noise
        })
        .collect();

    let data = TimeSeriesData::new(timestamps, values, "demo_series")?;
    println!("📊 Created time series with {} data points", data.len());

    // Demo 1: High-level Forecaster API
    println!("\n=== Demo 1: High-level Forecaster API ===");
    demo_high_level_api(&data)?;

    // Demo 2: Builder Pattern API
    println!("\n=== Demo 2: Builder Pattern API ===");
    demo_builder_pattern(&data)?;

    // Demo 3: Automatic Model Selection
    println!("\n=== Demo 3: Automatic Model Selection ===");
    demo_auto_selection(&data)?;

    // Demo 4: Model Comparison
    println!("\n=== Demo 4: Model Comparison ===");
    demo_model_comparison(&data)?;

    println!("\n✅ All demos completed successfully!");
    Ok(())
}

fn demo_high_level_api(data: &TimeSeriesData) -> Result<()> {
    println!("Using the high-level Forecaster interface...");

    // Simple forecasting with defaults
    let forecaster = Forecaster::new();
    let output = forecaster.forecast(data, 10)?;
    println!(
        "📈 Auto forecast ({}): {:?}",
        output.model_used,
        &output.forecast[..3]
    );

    // Configured forecasting
    let forecaster = Forecaster::new()
        .model(ModelType::ARIMA)
        .arima_params(2, 1, 1);
    let output = forecaster.forecast(data, 10)?;
    println!("📈 ARIMA(2,1,1) forecast: {:?}", &output.forecast[..3]);

    // Using ForecastBuilder
    let forecaster = ForecastBuilder::new().arima(1, 1, 2).build();
    let output = forecaster.forecast(data, 10)?;
    println!(
        "📈 Builder ARIMA(1,1,2) forecast: {:?}",
        &output.forecast[..3]
    );

    Ok(())
}

fn demo_builder_pattern(data: &TimeSeriesData) -> Result<()> {
    println!("Using the fluent builder pattern...");

    // ARIMA model with builder
    let mut arima_model = ModelBuilder::arima()
        .with_ar(2)
        .with_differencing(1)
        .with_ma(1)
        .build()?;

    arima_model.quick_fit(data)?;
    let forecast = arima_model.quick_forecast(5)?;
    println!("🏗️  ARIMA builder forecast: {:?}", forecast);

    // Exponential Smoothing with builder
    let mut es_model = ModelBuilder::exponential_smoothing()
        .with_alpha(0.3)
        .build()?;

    es_model.quick_fit(data)?;
    let forecast = es_model.quick_forecast(5)?;
    println!("🏗️  ES builder forecast: {:?}", forecast);

    // Moving Average with builder
    let mut ma_model = ModelBuilder::moving_average().with_window(7).build()?;

    ma_model.quick_fit(data)?;
    let forecast = ma_model.quick_forecast(5)?;
    println!("🏗️  MA builder forecast: {:?}", forecast);

    Ok(())
}

fn demo_auto_selection(data: &TimeSeriesData) -> Result<()> {
    println!("Demonstrating automatic model selection...");

    // AIC-based selection
    let selector = AutoSelector::with_aic().max_models(5);
    let (mut best_model, score, name) = selector.select_best(data)?;
    let forecast = best_model.quick_forecast(5)?;
    println!("🎯 AIC best model: {} (score: {:.2})", name, score);
    println!("   Forecast: {:?}", forecast);

    // BIC-based selection
    let selector = AutoSelector::with_bic().max_models(5);
    let (mut best_model, score, name) = selector.select_best(data)?;
    let forecast = best_model.quick_forecast(5)?;
    println!("🎯 BIC best model: {} (score: {:.2})", name, score);
    println!("   Forecast: {:?}", forecast);

    // Cross-validation based selection
    let selector = AutoSelector::with_cross_validation(3).max_models(5);
    let (mut best_model, score, name) = selector.select_best(data)?;
    let forecast = best_model.quick_forecast(5)?;
    println!("🎯 CV best model: {} (score: {:.2})", name, score);
    println!("   Forecast: {:?}", forecast);

    // Out-of-sample validation
    let selector = AutoSelector::with_out_of_sample(0.2).max_models(5);
    let (mut best_model, score, name) = selector.select_best(data)?;
    let forecast = best_model.quick_forecast(5)?;
    println!(
        "🎯 Out-of-sample best model: {} (score: {:.2})",
        name, score
    );
    println!("   Forecast: {:?}", forecast);

    Ok(())
}

fn demo_model_comparison(data: &TimeSeriesData) -> Result<()> {
    println!("Comparing different models...");

    let models = vec![
        (
            "ARIMA(1,1,1)",
            ModelBuilder::arima()
                .with_ar(1)
                .with_differencing(1)
                .with_ma(1),
        ),
        (
            "ARIMA(2,1,2)",
            ModelBuilder::arima()
                .with_ar(2)
                .with_differencing(1)
                .with_ma(2),
        ),
        (
            "ES(α=0.3)",
            ModelBuilder::exponential_smoothing().with_alpha(0.3),
        ),
        (
            "ES(α=0.7)",
            ModelBuilder::exponential_smoothing().with_alpha(0.7),
        ),
        ("MA(5)", ModelBuilder::moving_average().with_window(5)),
        ("MA(10)", ModelBuilder::moving_average().with_window(10)),
    ];

    println!("📊 Model Performance Comparison:");
    println!(
        "{:<15} {:<8} {:<8} {:<8} {:<8}",
        "Model", "MAE", "MSE", "RMSE", "MAPE"
    );
    println!("{}", "-".repeat(55));

    for (name, builder) in models {
        match builder.build() {
            Ok(mut model) => {
                if model.quick_fit(data).is_ok() {
                    if let Ok(evaluation) = model.evaluate(data) {
                        println!(
                            "{:<15} {:<8.2} {:<8.2} {:<8.2} {:<8.2}",
                            name, evaluation.mae, evaluation.mse, evaluation.rmse, evaluation.mape
                        );
                    }
                }
            }
            Err(_) => {
                println!("{:<15} Failed to build", name);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_improvements() {
        // Create test data
        let start_time = Utc::now();
        let timestamps: Vec<_> = (0..30).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let data = TimeSeriesData::new(timestamps, values, "test").unwrap();

        // Test high-level API
        let forecaster = Forecaster::new();
        let output = forecaster.forecast(&data, 5).unwrap();
        assert_eq!(output.forecast.len(), 5);

        // Test builder pattern
        let mut model = ModelBuilder::arima()
            .with_ar(1)
            .with_differencing(1)
            .with_ma(1)
            .build()
            .unwrap();

        model.quick_fit(&data).unwrap();
        let forecast = model.quick_forecast(5).unwrap();
        assert_eq!(forecast.len(), 5);

        // Test auto selection
        let selector = AutoSelector::with_aic().max_models(3);
        let (mut best_model, _score, _name) = selector.select_best(&data).unwrap();
        let forecast = best_model.quick_forecast(5).unwrap();
        assert_eq!(forecast.len(), 5);
    }

    #[test]
    fn test_model_evaluation() {
        let start_time = Utc::now();
        let timestamps: Vec<_> = (0..20).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let data = TimeSeriesData::new(timestamps, values, "test").unwrap();

        let mut model = ModelBuilder::moving_average()
            .with_window(3)
            .build()
            .unwrap();

        model.quick_fit(&data).unwrap();
        let evaluation = model.evaluate(&data).unwrap();

        assert!(evaluation.mae >= 0.0);
        assert!(evaluation.mse >= 0.0);
        assert!(evaluation.rmse >= 0.0);
    }
}
