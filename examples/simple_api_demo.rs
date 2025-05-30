//! Simple demo of OxiDiviner API improvements
//!
//! This example demonstrates the key API improvements:
//! 1. Unified forecasting interface
//! 2. Builder pattern for fluent model construction
//! 3. Basic model comparison

use chrono::{Duration, Utc};
use oxidiviner::api::*;
use oxidiviner::prelude::*;

fn main() -> Result<()> {
    println!("🔮 OxiDiviner Simple API Demo\n");

    // Generate simple time series data
    let start_time = Utc::now();
    let timestamps: Vec<_> = (0..30).map(|i| start_time + Duration::days(i)).collect();

    // Create a simple trend pattern
    let values: Vec<f64> = (0..30)
        .map(|i| 100.0 + i as f64 * 0.5 + (i as f64 * 0.1).sin())
        .collect();

    let data = TimeSeriesData::new(timestamps, values, "demo_series")?;
    println!("📊 Created time series with {} data points", data.len());

    // Demo 1: High-level Forecaster API
    println!("\n=== Demo 1: High-level Forecaster API ===");
    demo_high_level_api(&data)?;

    // Demo 2: Builder Pattern API
    println!("\n=== Demo 2: Builder Pattern API ===");
    demo_builder_pattern(&data)?;

    // Demo 3: Model Comparison
    println!("\n=== Demo 3: Model Comparison ===");
    demo_model_comparison(&data)?;

    println!("\n✅ Demo completed successfully!");
    Ok(())
}

fn demo_high_level_api(data: &TimeSeriesData) -> Result<()> {
    println!("Using the high-level Forecaster interface...");

    // Simple forecasting with defaults
    let forecaster = Forecaster::new();
    let output = forecaster.forecast(data, 5)?;
    println!(
        "📈 Auto forecast ({}): {:?}",
        output.model_used,
        &output.forecast[..3]
    );

    // Configured ARIMA forecasting
    let forecaster = Forecaster::new()
        .model(ModelType::ARIMA)
        .arima_params(1, 1, 1);
    let output = forecaster.forecast(data, 5)?;
    println!("📈 ARIMA(1,1,1) forecast: {:?}", &output.forecast[..3]);

    // Moving Average forecasting
    let forecaster = Forecaster::new()
        .model(ModelType::MovingAverage)
        .ma_window(5);
    let output = forecaster.forecast(data, 5)?;
    println!("📈 MA(5) forecast: {:?}", &output.forecast[..3]);

    // Using ForecastBuilder
    let forecaster = ForecastBuilder::new().simple_es(0.3).build();
    let output = forecaster.forecast(data, 5)?;
    println!("📈 Builder ES(α=0.3) forecast: {:?}", &output.forecast[..3]);

    Ok(())
}

fn demo_builder_pattern(data: &TimeSeriesData) -> Result<()> {
    println!("Using the fluent builder pattern...");

    // ARIMA model with builder
    let mut arima_model = ModelBuilder::arima()
        .with_ar(1)
        .with_differencing(1)
        .with_ma(1)
        .build()?;

    arima_model.quick_fit(data)?;
    let forecast = arima_model.quick_forecast(5)?;
    println!("🏗️  ARIMA builder forecast: {:?}", &forecast[..3]);

    // Exponential Smoothing with builder
    let mut es_model = ModelBuilder::exponential_smoothing()
        .with_alpha(0.3)
        .build()?;

    es_model.quick_fit(data)?;
    let forecast = es_model.quick_forecast(5)?;
    println!("🏗️  ES builder forecast: {:?}", &forecast[..3]);

    // Moving Average with builder
    let mut ma_model = ModelBuilder::moving_average().with_window(5).build()?;

    ma_model.quick_fit(data)?;
    let forecast = ma_model.quick_forecast(5)?;
    println!("🏗️  MA builder forecast: {:?}", &forecast[..3]);

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
            "ES(α=0.3)",
            ModelBuilder::exponential_smoothing().with_alpha(0.3),
        ),
        (
            "ES(α=0.7)",
            ModelBuilder::exponential_smoothing().with_alpha(0.7),
        ),
        ("MA(3)", ModelBuilder::moving_average().with_window(3)),
        ("MA(7)", ModelBuilder::moving_average().with_window(7)),
    ];

    println!("📊 Model Performance Comparison:");
    println!(
        "{:<12} {:<8} {:<8} {:<8} {:<10}",
        "Model", "MAE", "MSE", "RMSE", "Forecast"
    );
    println!("{}", "-".repeat(60));

    for (name, builder) in models {
        match builder.build() {
            Ok(mut model) => {
                if model.quick_fit(data).is_ok() {
                    if let Ok(evaluation) = model.evaluate(data) {
                        if let Ok(forecast) = model.quick_forecast(3) {
                            println!(
                                "{:<12} {:<8.2} {:<8.2} {:<8.2} {:?}",
                                name,
                                evaluation.mae,
                                evaluation.mse,
                                evaluation.rmse,
                                &forecast[..2]
                            );
                        }
                    }
                }
            }
            Err(e) => {
                println!("{:<12} Failed: {:?}", name, e);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_api() {
        // Create test data
        let start_time = Utc::now();
        let timestamps: Vec<_> = (0..20).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let data = TimeSeriesData::new(timestamps, values, "test").unwrap();

        // Test high-level API
        let forecaster = Forecaster::new();
        let output = forecaster.forecast(&data, 5).unwrap();
        assert_eq!(output.forecast.len(), 5);

        // Test builder pattern
        let mut model = ModelBuilder::moving_average()
            .with_window(3)
            .build()
            .unwrap();

        model.quick_fit(&data).unwrap();
        let forecast = model.quick_forecast(5).unwrap();
        assert_eq!(forecast.len(), 5);
    }
}
