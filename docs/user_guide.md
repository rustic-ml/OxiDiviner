# OxiDiviner User Guide

Welcome to the comprehensive user guide for OxiDiviner! This guide will walk you through all the features and capabilities of the library.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Core Concepts](#core-concepts)
3. [Data Structures](#data-structures)
4. [Quick API](#quick-api)
5. [Forecasting Models](#forecasting-models)
6. [Model Validation](#model-validation)
7. [Batch Processing](#batch-processing)
8. [Financial Applications](#financial-applications)
9. [Advanced Usage](#advanced-usage)
10. [Best Practices](#best-practices)

## Installation and Setup

### Basic Installation

Add OxiDiviner to your `Cargo.toml`:

```toml
[dependencies]
oxidiviner = "0.4.2"
```

### Import Patterns

```rust
// Most common - imports all main types
use oxidiviner::prelude::*;

// Specific modules
use oxidiviner::{quick, api, batch};
use oxidiviner::models::autoregressive::ARIMAModel;

// Quick API only
use oxidiviner::quick;
```

## Core Concepts

### Forecaster Trait

All models in OxiDiviner implement the `Forecaster` trait:

```rust
pub trait Forecaster {
    fn name(&self) -> &str;
    fn fit(&mut self, data: &TimeSeriesData) -> Result<()>;
    fn forecast(&self, horizon: usize) -> Result<Vec<f64>>;
    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation>;
    fn predict(&self, horizon: usize, test_data: Option<&TimeSeriesData>) -> Result<ModelOutput>;
}
```

This ensures a consistent interface across all models.

### Error Handling

OxiDiviner uses comprehensive error types:

```rust
use oxidiviner::core::OxiError;

match result {
    Ok(forecast) => println!("Success: {:?}", forecast),
    Err(OxiError::InvalidParameter(msg)) => println!("Invalid parameter: {}", msg),
    Err(OxiError::ModelError(msg)) => println!("Model error: {}", msg),
    Err(OxiError::DataError(msg)) => println!("Data error: {}", msg),
    // ... other error types
}
```

## Data Structures

### TimeSeriesData

The core data structure for univariate time series:

```rust
use oxidiviner::prelude::*;
use chrono::{Duration, Utc};

// Create timestamps
let start = Utc::now();
let timestamps: Vec<_> = (0..100)
    .map(|i| start + Duration::days(i))
    .collect();

// Create values (example: linear trend with noise)
let values: Vec<f64> = (0..100)
    .map(|i| 10.0 + 0.5 * i as f64 + rand::random::<f64>() * 5.0)
    .collect();

// Create time series
let ts = TimeSeriesData::new(timestamps, values, "my_series")?;
```

### OHLCVData

For financial time series data:

```rust
use oxidiviner::prelude::*;

let ohlcv = OHLCVData {
    symbol: "AAPL".to_string(),
    timestamps: timestamps,
    open: open_prices,
    high: high_prices,
    low: low_prices,
    close: close_prices,
    volume: volumes,
    adjusted_close: Some(adjusted_closes), // Optional
};

// Convert to time series using close prices
let ts = ohlcv.to_time_series(false); // false = use close, true = use adjusted_close
```

## Quick API

The Quick API provides one-line forecasting functions:

### ARIMA Forecasting

```rust
use oxidiviner::quick;

// Default ARIMA(1,1,1)
let forecast = quick::arima(data.clone(), 10)?;

// Custom parameters ARIMA(2,1,2)
let forecast = quick::arima_with_config(data, 10, Some((2, 1, 2)))?;
```

### Moving Average

```rust
// Default window size (5)
let forecast = quick::moving_average(data.clone(), 10, None)?;

// Custom window size
let forecast = quick::moving_average(data, 10, Some(7))?;
```

### Auto Model Selection

```rust
// Automatically selects best model
let (forecast, model_name) = quick::auto_select(data, 10)?;
println!("Selected model: {}", model_name);
```

### Model Comparison

```rust
// Compare all available models
let comparisons = quick::compare_models(timestamps, values, 10)?;
for (model_name, forecast) in comparisons {
    println!("{}: {:?}", model_name, &forecast[..3]);
}
```

## Forecasting Models

### ARIMA Models

ARIMA (AutoRegressive Integrated Moving Average) models:

```rust
use oxidiviner::models::autoregressive::ARIMAModel;

// Create ARIMA(p, d, q) model
let mut model = ARIMAModel::new(2, 1, 1, true)?; // p=2, d=1, q=1, with_intercept=true
model.fit(&data)?;

let forecasts = model.forecast(10)?;
let evaluation = model.evaluate(&test_data)?;

// Access model coefficients
if let Some(ar_coef) = model.ar_coefficients() {
    println!("AR coefficients: {:?}", ar_coef);
}
```

### Exponential Smoothing

#### Simple Exponential Smoothing

```rust
use oxidiviner::models::exponential_smoothing::SimpleESModel;

let mut model = SimpleESModel::new(0.3)?; // alpha = 0.3
model.fit(&data)?;
let forecast = model.forecast(10)?;
```

#### Holt's Linear Method

```rust
use oxidiviner::models::exponential_smoothing::HoltLinearModel;

let mut model = HoltLinearModel::new(0.3, 0.1)?; // alpha = 0.3, beta = 0.1
model.fit(&data)?;
let forecast = model.forecast(10)?;
```

#### Holt-Winters Seasonal

```rust
use oxidiviner::models::exponential_smoothing::HoltWintersModel;

// For seasonal data (e.g., monthly data with yearly seasonality)
let mut model = HoltWintersModel::new(0.3, 0.1, 0.1, 12)?; // period = 12
model.fit(&data)?;
let forecast = model.forecast(24)?; // Forecast 2 years
```

### GARCH Models

For volatility modeling:

```rust
use oxidiviner::models::garch::GARCHModel;

// GARCH(1,1) model
let mut model = GARCHModel::new(1, 1, None)?;
model.fit(&returns_data)?; // Use returns, not prices

let volatility_forecast = model.forecast(10)?;
```

### VAR Models

For multivariate time series:

```rust
use oxidiviner::models::autoregressive::VARModel;
use std::collections::HashMap;

// Create VAR model with 2 variables
let variable_names = vec!["gdp".to_string(), "inflation".to_string()];
let mut model = VARModel::new(2, variable_names, true)?; // lag=2, with_intercept=true

// Prepare data
let mut data_map = HashMap::new();
data_map.insert("gdp".to_string(), gdp_data);
data_map.insert("inflation".to_string(), inflation_data);

// Fit and forecast
model.fit_multiple(&data_map)?;
let forecasts = model.forecast_multiple(10)?; // Returns HashMap<String, Vec<f64>>
```

## Model Validation

### Time Series Split

```rust
use oxidiviner::core::validation::ValidationUtils;

// Split data into train/test (80/20)
let (train, test) = ValidationUtils::time_split(&data, 0.8)?;
```

### Cross-Validation

```rust
// Time series cross-validation with 5 folds, minimum 50 observations per fold
let cv_splits = ValidationUtils::time_series_cv(&data, 5, Some(50))?;

let mut mae_scores = Vec::new();
for (train, test) in cv_splits {
    let mut model = ARIMAModel::new(1, 1, 1, true)?;
    model.fit(&train)?;
    let forecast = model.forecast(test.len())?;
    
    let metrics = ValidationUtils::accuracy_metrics(&test.values, &forecast, None)?;
    mae_scores.push(metrics.mae);
}

let avg_mae = mae_scores.iter().sum::<f64>() / mae_scores.len() as f64;
println!("Average CV MAE: {:.3}", avg_mae);
```

### Accuracy Metrics

```rust
// Calculate comprehensive accuracy metrics
let metrics = ValidationUtils::accuracy_metrics(&actual, &predicted, Some(&baseline))?;

println!("MAE:  {:.3}", metrics.mae);
println!("RMSE: {:.3}", metrics.rmse);
println!("MAPE: {:.3}%", metrics.mape);
println!("SMAPE: {:.3}%", metrics.smape);
println!("R²:   {:.3}", metrics.r_squared);
if let Some(mase) = metrics.mase {
    println!("MASE: {:.3}", mase);
}
```

## Batch Processing

Process multiple time series simultaneously:

```rust
use oxidiviner::batch::BatchProcessor;
use std::collections::HashMap;

// Create batch processor
let processor = BatchProcessor::new();

// Prepare multiple time series
let mut series_map = HashMap::new();
series_map.insert("sales_region_1".to_string(), sales_data_1);
series_map.insert("sales_region_2".to_string(), sales_data_2);
series_map.insert("inventory".to_string(), inventory_data);

// Auto forecast all series
let results = processor.auto_forecast_multiple(series_map, 30)?;

// Process results
for (name, result) in &results.forecasts {
    println!("Series {}: {} using {}", 
        name, 
        result.len(), 
        results.models_used.get(name).unwrap_or(&"Unknown".to_string())
    );
}

// Export results
let exported = processor.export_results(&results)?;
```

## Financial Applications

### Stock Price Forecasting

```rust
use oxidiviner::prelude::*;

// Load daily stock prices
let stock_data = OHLCVData { /* ... */ };
let price_series = stock_data.to_time_series(false); // Use close prices

// Forecast prices
let mut model = ARIMAModel::new(1, 1, 1, true)?;
model.fit(&price_series)?;
let price_forecast = model.forecast(30)?; // 30-day forecast
```

### Volatility Modeling

```rust
use oxidiviner::models::garch::GARCHModel;

// Calculate returns from prices
let returns = financial::calculate_returns(&prices, false)?; // Simple returns
let returns_ts = TimeSeriesData::new(timestamps[1..].to_vec(), returns, "returns")?;

// Fit GARCH model for volatility
let mut garch = GARCHModel::new(1, 1, None)?;
garch.fit(&returns_ts)?;
let volatility = garch.forecast(30)?;
```

### Portfolio Analysis

```rust
use std::collections::HashMap;

// Multiple asset data
let mut assets = HashMap::new();
assets.insert("AAPL".to_string(), aapl_data);
assets.insert("GOOGL".to_string(), googl_data);
assets.insert("MSFT".to_string(), msft_data);

// Batch forecast all assets
let processor = BatchProcessor::new();
let results = processor.auto_forecast_multiple(assets, 30)?;

// Analyze results
for (symbol, forecast) in &results.forecasts {
    let expected_return = forecast.iter().sum::<f64>() / forecast.len() as f64;
    println!("{}: Expected 30-day average: {:.2}", symbol, expected_return);
}
```

## Advanced Usage

### Custom Model Configuration

```rust
use oxidiviner::ModelBuilder;

// Build complex model configurations
let config = ModelBuilder::arima()
    .with_ar(3)
    .with_differencing(2)
    .with_ma(2)
    .build_config();

let forecast = quick::forecast_with_config(data, 10, config)?;
```

### Model Selection with Custom Criteria

```rust
use oxidiviner::{AutoSelector, SelectionCriteria};

// Create auto selector with cross-validation
let selector = AutoSelector::with_cross_validation(5);

// Add custom candidates
let custom_config = ModelBuilder::arima()
    .with_ar(5)
    .with_differencing(1)
    .with_ma(3)
    .build_config();

let selector = selector.add_candidate(custom_config);

// Select best model (requires implementing the selection logic)
// This functionality would be expanded in future versions
```

### Parallel Processing

```rust
use rayon::prelude::*;

// Parallel batch processing (if using rayon)
let results: Vec<_> = series_map
    .par_iter()
    .map(|(name, data)| {
        let forecast = quick::auto_select(data.clone(), 30).unwrap();
        (name.clone(), forecast)
    })
    .collect();
```

## Best Practices

### Data Preparation

1. **Check for missing values**: Handle NaN or infinite values before fitting
2. **Stationarity**: Use differencing for non-stationary data
3. **Outliers**: Consider outlier detection and treatment
4. **Frequency**: Ensure consistent time intervals

```rust
// Check for missing values
let has_nan = data.values.iter().any(|&x| x.is_nan() || x.is_infinite());
if has_nan {
    // Handle missing values
    println!("Warning: Data contains NaN or infinite values");
}
```

### Model Selection

1. **Start simple**: Begin with simple models (MA, SES) before complex ones
2. **Use cross-validation**: Don't rely on in-sample fit alone
3. **Consider domain knowledge**: Financial data often needs volatility models
4. **Ensemble methods**: Combine multiple models for robustness

```rust
// Ensemble example
let ma_forecast = quick::moving_average(data.clone(), 10, Some(5))?;
let es_forecast = quick::es_forecast(timestamps.clone(), values.clone(), 10)?;
let arima_forecast = quick::arima(data, 10)?;

// Simple average ensemble
let ensemble: Vec<f64> = (0..10)
    .map(|i| (ma_forecast[i] + es_forecast[i] + arima_forecast[i]) / 3.0)
    .collect();
```

### Performance Tips

1. **Batch processing**: Use BatchProcessor for multiple series
2. **Model reuse**: Fit once, forecast multiple horizons
3. **Appropriate model complexity**: Don't overfit with too many parameters

### Error Handling

```rust
use oxidiviner::core::OxiError;

fn safe_forecast(data: TimeSeriesData) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    match quick::arima(data.clone(), 10) {
        Ok(forecast) => Ok(forecast),
        Err(OxiError::ARInsufficientData { actual, expected }) => {
            println!("Not enough data: need {}, got {}", expected, actual);
            // Fallback to simpler model
            quick::moving_average(data, 10, Some(3)).map_err(|e| e.into())
        }
        Err(e) => Err(e.into()),
    }
}
```

## Conclusion

OxiDiviner provides a comprehensive toolkit for time series forecasting in Rust. From simple one-line forecasts to complex multi-model ensemble approaches, the library scales with your needs.

For more examples and advanced usage patterns, check out the [examples directory](../examples/) and the [API documentation](https://docs.rs/oxidiviner).

---

**Happy forecasting! 📈** 