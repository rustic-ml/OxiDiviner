//! Comprehensive demo of OxiDiviner's enhanced API modules
//!
//! This example demonstrates the usage of all enhanced API modules:
//! - financial: Specialized financial time series analysis
//! - api: High-level unified interface
//! - quick: One-line utility functions
//! - batch: Batch processing for multiple time series

use chrono::{DateTime, Duration, Utc};
use oxidiviner::prelude::*;
use oxidiviner::{api, batch, quick, FinancialTimeSeries};

fn main() -> oxidiviner::Result<()> {
    println!("=== OxiDiviner Enhanced API Demo ===\n");

    // Generate sample data
    let start_date = Utc::now() - Duration::days(30);
    let timestamps: Vec<DateTime<Utc>> = (0..30).map(|i| start_date + Duration::days(i)).collect();
    let prices: Vec<f64> = (0..30)
        .map(|i| 100.0 + (i as f64) * 2.0 + (i as f64 * 0.1).sin() * 5.0)
        .collect();

    // 1. QUICK MODULE DEMO
    println!("1. QUICK MODULE - One-line forecasting");
    println!("=====================================");

    // Quick ARIMA forecast
    let arima_forecast = quick::arima_forecast(timestamps.clone(), prices.clone(), 5)?;
    println!("ARIMA forecast: {:?}", arima_forecast);

    // Quick exponential smoothing
    let es_forecast = quick::es_forecast(timestamps.clone(), prices.clone(), 5)?;
    println!("ES forecast: {:?}", es_forecast);

    // Quick moving average
    let ma_forecast = quick::ma_forecast(timestamps.clone(), prices.clone(), 5)?;
    println!("MA forecast: {:?}", ma_forecast);

    // Automatic model selection
    let (auto_forecast, model_used) = quick::auto_forecast(timestamps.clone(), prices.clone(), 5)?;
    println!("Auto forecast using {}: {:?}", model_used, auto_forecast);

    // Compare all models
    let model_comparison = quick::compare_models(timestamps.clone(), prices.clone(), 5)?;
    println!("Model comparison:");
    for (model_name, forecast) in &model_comparison {
        println!("  {}: {:?}", model_name, &forecast[..3]); // Show first 3 values
    }

    println!();

    // 2. FINANCIAL MODULE DEMO
    println!("2. FINANCIAL MODULE - Financial-specific analysis");
    println!("================================================");

    let financial_ts =
        FinancialTimeSeries::from_prices(timestamps.clone(), prices.clone(), "DEMO_STOCK")?;

    println!("Symbol: {}", financial_ts.symbol());

    // Calculate returns
    let simple_returns = financial_ts.simple_returns()?;
    println!("Simple returns (first 5): {:?}", &simple_returns[..5]);

    let log_returns = financial_ts.log_returns()?;
    println!("Log returns (first 5): {:?}", &log_returns[..5]);

    // Automatic financial forecasting
    let (fin_forecast, fin_model) = financial_ts.auto_forecast(5)?;
    println!(
        "Financial auto forecast using {}: {:?}",
        fin_model, fin_forecast
    );

    // Compare models for financial data
    let fin_comparison = financial_ts.compare_models(5)?;
    println!("Financial model comparison:");
    for result in &fin_comparison.results {
        println!("  {}: {:?}", result.name, &result.forecast[..3]);
    }
    if let Some(best) = fin_comparison.best() {
        println!("  Best model: {}", best.name);
    }

    println!();

    // 3. API MODULE DEMO
    println!("3. API MODULE - High-level unified interface");
    println!("============================================");

    // Create a time series data object
    let ts_data = TimeSeriesData::new(timestamps.clone(), prices.clone(), "demo_series")?;

    // Use the high-level API with builder pattern
    let forecaster = api::ForecastBuilder::new().arima(2, 1, 2).build();

    let api_result = forecaster.forecast(&ts_data, 5)?;
    println!(
        "API forecast using {}: {:?}",
        api_result.model_used, api_result.forecast
    );

    // Auto-selection API
    let auto_forecaster = api::ForecastBuilder::new().auto().build();

    let auto_result = auto_forecaster.forecast(&ts_data, 5)?;
    println!(
        "API auto forecast using {}: {:?}",
        auto_result.model_used, auto_result.forecast
    );

    // Direct forecaster creation
    let direct_forecaster = api::Forecaster::new()
        .model(api::ModelType::SimpleES)
        .alpha(0.4);

    let direct_result = direct_forecaster.forecast(&ts_data, 5)?;
    println!(
        "Direct API forecast using {}: {:?}",
        direct_result.model_used, direct_result.forecast
    );

    println!();

    // 4. BATCH MODULE DEMO
    println!("4. BATCH MODULE - Multiple time series processing");
    println!("=================================================");

    // Create multiple time series for batch processing
    let mut batch_data = Vec::new();

    for i in 0..3 {
        let series_name = format!("series_{}", i + 1);
        let series_values: Vec<f64> = (0..25)
            .map(|j| {
                100.0
                    + (i as f64 * 10.0)
                    + (j as f64)
                    + (j as f64 * 0.1 * (i + 1) as f64).sin() * 3.0
            })
            .collect();
        let series_timestamps: Vec<DateTime<Utc>> =
            (0..25).map(|j| start_date + Duration::days(j)).collect();

        batch_data.push((series_name, series_timestamps, series_values));
    }

    // Create batch time series
    let batch_ts = batch::BatchTimeSeries::from_data_arrays(batch_data)?;

    // Get batch summary
    let summary = batch_ts.summary();
    println!("Batch summary:");
    println!("  Total series: {}", summary.total_series);
    println!("  Average length: {:.1}", summary.avg_length);
    println!(
        "  Length range: {} - {}",
        summary.min_length, summary.max_length
    );
    println!("  Series names: {:?}", summary.series_names);

    // Perform batch forecasting with default config
    let batch_results = batch_ts.forecast(5)?;
    println!("\nBatch forecasting results:");
    for (name, forecast) in &batch_results.forecasts {
        let unknown_model = "Unknown".to_string();
        let model = batch_results
            .models_used
            .get(name)
            .unwrap_or(&unknown_model);
        println!("  {}: {} -> {:?}", name, model, &forecast[..3]);
    }

    // Show any errors
    if !batch_results.errors.is_empty() {
        println!("\nBatch errors:");
        for (name, error) in &batch_results.errors {
            println!("  {}: {}", name, error);
        }
    }

    // Custom batch configuration
    let custom_config = batch::BatchConfig {
        forecast_periods: 3,
        parallel: false, // Sequential for this demo
        model_type: Some(batch::BatchModelType::ARIMA { p: 1, d: 1, q: 1 }),
        continue_on_error: true,
    };

    let custom_results = batch_ts.forecast_with_config(&custom_config)?;
    println!("\nCustom batch results (ARIMA only):");
    for (name, forecast) in &custom_results.forecasts {
        println!("  {}: {:?}", name, forecast);
    }

    // Export results for easier processing
    let exported = batch_ts.export_results(&batch_results);
    println!("\nExported results:");
    for (name, result) in &exported {
        println!(
            "  {}: success={}, model={:?}",
            name,
            result.success,
            result.model_used.as_deref().unwrap_or("None")
        );
    }

    println!();

    // 5. INTEGRATION DEMO
    println!("5. INTEGRATION DEMO - Combining all modules");
    println!("===========================================");

    // Start with quick module for initial exploration
    let (initial_forecast, initial_model) = quick::values_only_forecast(prices.clone(), 3)?;
    println!(
        "Initial exploration with quick module: {} -> {:?}",
        initial_model, initial_forecast
    );

    // Move to financial module for specialized analysis
    let financial_series =
        FinancialTimeSeries::from_prices(timestamps.clone(), prices.clone(), "INTEGRATED")?;
    let returns = financial_series.simple_returns()?;
    println!("Financial analysis: {} returns calculated", returns.len());

    // Use API module for detailed configuration
    let precise_forecaster = api::ForecastBuilder::new().arima(1, 1, 2).build();
    let ts_data = TimeSeriesData::new(timestamps.clone(), prices.clone(), "integrated")?;
    let precise_result = precise_forecaster.forecast(&ts_data, 3)?;
    println!(
        "Precise API forecast: {} -> {:?}",
        precise_result.model_used, precise_result.forecast
    );

    // Scale up with batch module
    let mut integrated_batch = batch::BatchTimeSeries::new();
    integrated_batch.add_from_data("original".to_string(), timestamps.clone(), prices.clone())?;

    // Add a modified series
    let modified_prices: Vec<f64> = prices.iter().map(|p| p * 1.1).collect();
    integrated_batch.add_from_data("modified".to_string(), timestamps.clone(), modified_prices)?;

    let final_results = integrated_batch.forecast(3)?;
    println!(
        "Final batch results: {} series processed",
        final_results.forecasts.len()
    );

    println!("\n=== Demo Complete ===");
    println!("All enhanced API modules working successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_api_demo() {
        // This test ensures the demo runs without panicking
        let result = main();
        assert!(result.is_ok(), "Demo should run successfully: {:?}", result);
    }
}
