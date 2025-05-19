# OxiDiviner

![OxiDiviner Logo](OxiDiviner_250px.JPG)

[![Crates.io](https://img.shields.io/crates/v/oxidiviner.svg)](https://crates.io/crates/oxidiviner)
[![Documentation](https://docs.rs/oxidiviner/badge.svg)](https://docs.rs/oxidiviner)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rustic-ml/OxiDiviner/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-rustic--ml/OxiDiviner-8da0cb?logo=github)](https://github.com/rustic-ml/OxiDiviner)

OxiDiviner is a Rust library for time series analysis and forecasting, designed to be a comprehensive toolkit for traders and data scientists.

## Standardized Model Interface

OxiDiviner provides a consistent, standardized interface for all forecasting models through the `Forecaster` trait. This standardization ensures that:

1. Every model has **one public entry point** via the `predict()` method
2. All models return a **standardized output format** via the `ModelOutput` struct

### Benefits of the Standardized Interface

- **Consistency**: Use any model with the same method calls and output format
- **Interoperability**: Easily switch between models or compare their performance
- **Extensibility**: Add new models that seamlessly integrate with the existing ecosystem
- **Accessibility**: Learn one interface to use all available models

### Example Usage

```rust
use chrono::{DateTime, TimeZone, Utc};
use oxidiviner::prelude::*;
use oxidiviner::models::Forecaster;

// Create and train different model types
let mut ses_model = SESModel::new(0.3, None)?;
let mut ma_model = MAModel::new(5)?;
    
// Train models using the standardized interface
ses_model.fit(&time_series)?;
ma_model.fit(&time_series)?;
    
// Forecast using the standardized interface  
let horizon = 10;
let ses_output = ses_model.predict(horizon, Some(&time_series))?;
let ma_output = ma_model.predict(horizon, Some(&time_series))?;

// Both outputs have the same format and can be processed uniformly
println!("SES forecast: {:?}", ses_output.forecasts);
println!("MA forecast: {:?}", ma_output.forecasts);

// Access evaluation metrics in a standardized way
if let Some(ses_eval) = ses_output.evaluation {
    println!("SES Model MAE: {}", ses_eval.mae);
}

if let Some(ma_eval) = ma_output.evaluation {
    println!("MA Model MAE: {}", ma_eval.mae);
}
```

## Features

- Time series forecasting models including:
  - Simple Exponential Smoothing (SES)
  - Moving Average (MA)
  - Exponential Smoothing models with trend and seasonality
  - Specialized models for daily and minute-level financial data
  - More models coming soon!

- Standardized model interface via the `Forecaster` trait
- Consistent output format via the `ModelOutput` struct
- Comprehensive model evaluation metrics
- Utilities for data loading, preprocessing, and visualization

## Getting Started

Add OxiDiviner to your Cargo.toml:

```toml
[dependencies]
oxidiviner = "0.3.3"
```

See the examples directory for complete usage examples.

## Features

- **Data Handling**: Import and manage OHLCV (Open-High-Low-Close-Volume) stock data
- **Time Series Processing**: Tools for manipulating and analyzing time series data
- **Forecasting Models**: Implementation of various statistical forecasting methods
  - Error-Trend-Seasonality (ETS) models
  - Simple Exponential Smoothing
  - Holt's Linear Trend
  - Holt-Winters Seasonal Model
- **Time-Based Models**: Specialized models for different time granularities
  - `DailyETSModel`: Optimized for daily financial data
  - `MinuteETSModel`: Designed for high-frequency data with aggregation support
- **Flexible Analysis**: Support for both daily and minute-level data
- **Model Evaluation**: Tools for assessing forecast accuracy and model fit

## Installation

Add OxiDiviner to your Cargo.toml:

```toml
[dependencies]
oxdiviner = "0.3.3"
```

## Usage Examples

### Basic Time Series Analysis

```rust
use chrono::Utc;
use oxdiviner::{OHLCVData, TimeSeriesData};

// Create or load OHLCV data
let data = OHLCVData::new("AAPL");

// Or load from a CSV file (with `polars_integration` feature)
#[cfg(feature = "polars_integration")]
let data = OHLCVData::from_csv("path/to/aapl_data.csv").unwrap();

// Convert to a time series for analysis
let time_series = data.to_time_series(false);  // false = use regular close, not adjusted

// Split into training and testing sets
let (train, test) = time_series.train_test_split(0.8).unwrap();
```

### Using Daily ETS Models

The `DailyETSModel` provides specialized functionality for daily financial data:

```rust
use oxdiviner::models::ets::{ETSComponent, DailyETSModel};

// Create a Holt-Winters seasonal model with weekly seasonality
let mut model = DailyETSModel::holt_winters_additive(
    0.3,                     // alpha (level smoothing)
    0.1,                     // beta (trend smoothing)
    0.1,                     // gamma (seasonal smoothing)
    7,                       // Seasonal period (7 = weekly)
    None,                    // Target column (None = use Close price)
).unwrap();

// Or create with full control over all parameters
let mut custom_model = DailyETSModel::new(
    ETSComponent::Additive,  // Error type
    ETSComponent::Additive,  // Trend type
    ETSComponent::Additive,  // Seasonal type
    0.3,                     // alpha (level smoothing)
    Some(0.1),               // beta (trend smoothing)
    Some(0.1),               // gamma (seasonal smoothing)
    None,                    // phi (damping factor, None = no damping)
    Some(7),                 // Seasonal period (7 = weekly)
    None,                    // Target column (None = use Close price)
).unwrap();

// Fit the model to data
model.fit(&daily_data).unwrap();

// Generate forecasts
let horizon = 30;  // 30-day forecast
let forecasts = model.forecast(horizon).unwrap();

// Evaluate the model
let evaluation = model.evaluate(&test_data).unwrap();
println!("MAE: {:.4}", evaluation.mae);
println!("RMSE: {:.4}", evaluation.rmse);
println!("MAPE: {:.2}%", evaluation.mape);

// Get fitted values
if let Some(fitted) = model.fitted_values() {
    println!("Last fitted value: {:.4}", fitted.last().unwrap());
}
```

### Working with Minute Data

The `MinuteETSModel` is designed specifically for high-frequency data and includes unique features like data aggregation:

```rust
use oxdiviner::models::ets::{ETSComponent, MinuteETSModel};

// Create a simple model with 5-minute aggregation
let mut simple_model = MinuteETSModel::simple(
    0.4,                     // alpha (higher for minute data to adapt faster)
    None,                    // Default to Close price
    Some(5),                 // 5-minute aggregation (reduces noise)
).unwrap();

// Create a model with hourly seasonality pattern
let mut hourly_model = MinuteETSModel::holt_winters_additive(
    0.4,                     // alpha 
    0.1,                     // beta
    0.1,                     // gamma
    60,                      // Seasonal period = 60 minutes (hourly pattern)
    None,                    // Default to Close price
    Some(5),                 // 5-minute aggregation
).unwrap();

// Fit and forecast as with daily models
hourly_model.fit(&minute_data).unwrap();
let forecasts = hourly_model.forecast(12).unwrap();  // Forecast the next hour
```

#### Benefits of the MinuteETSModel

- **Data Aggregation**: Reduce noise by converting 1-minute data to higher timeframes (e.g., 5-minute, 15-minute)
- **Appropriate Defaults**: Higher smoothing parameters better suited for high-frequency data
- **Suitable Seasonality**: Optimized for intraday patterns (e.g., hourly cycles)
- **Performance**: Efficiently handles large volumes of minute-level data

## Running the Demo

The package includes a comprehensive demo that showcases ETS models with detailed interpretations:

```bash
cargo run --example ets_demo -- AAPL daily
```

Options:
- First argument: Stock ticker (e.g., AAPL, MSFT) or SYNTHETIC for synthetic data
- Second argument: Data type - 'daily' or 'minute'

## Features

OxiDiviner has several optional features that can be enabled:

- `plotting`: Enables visualization capabilities using the `plotters` crate
- `polars_integration`: Adds support for loading data from CSV and Parquet files
- `ndarray_support`: Adds integration with the `ndarray` library for numerical computing

Enable features in your Cargo.toml:

```toml
[dependencies]
oxdiviner = { version = "0.3.3", features = ["plotting", "polars_integration"] }
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 