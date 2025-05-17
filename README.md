# OxiDiviner

OxiDiviner is a Rust library for time series analysis and forecasting, specializing in financial market data. It provides tools for analyzing stock price data, detecting patterns, and making predictions using various statistical models.

## Features

- **Data Handling**: Import and manage OHLCV (Open-High-Low-Close-Volume) stock data
- **Time Series Processing**: Tools for manipulating and analyzing time series data
- **Forecasting Models**: Implementation of various statistical forecasting methods
  - Error-Trend-Seasonality (ETS) models
  - Simple Exponential Smoothing
  - Holt's Linear Trend
  - Holt-Winters Seasonal Model
- **Flexible Analysis**: Support for both daily and minute-level data
- **Model Evaluation**: Tools for assessing forecast accuracy and model fit

## Installation

Add OxiDiviner to your Cargo.toml:

```toml
[dependencies]
oxdiviner = "0.1.0"
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

### Using ETS Models

```rust
use oxdiviner::models::ets::{ETSComponent, DailyETSModel};

// Create a Holt-Winters seasonal model
let mut model = DailyETSModel::new(
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
model.fit(&train_data).unwrap();

// Generate forecasts
let horizon = 30;  // 30-day forecast
let forecasts = model.forecast(horizon).unwrap();

// Evaluate the model
let evaluation = model.evaluate(&test_data).unwrap();
println!("MAE: {:.4}", evaluation.mae);
println!("RMSE: {:.4}", evaluation.rmse);
println!("MAPE: {:.2}%", evaluation.mape);
```

### Working with Minute Data

```rust
use oxdiviner::models::ets::{ETSComponent, MinuteETSModel};

// Create a model for minute-level data with 5-minute aggregation
let mut model = MinuteETSModel::new(
    ETSComponent::Additive,  // Error type
    ETSComponent::None,      // No trend
    ETSComponent::Additive,  // Additive seasonality
    0.3,                     // alpha
    None,                    // No beta (no trend)
    Some(0.1),               // gamma
    None,                    // No phi (no damping)
    Some(60),                // Seasonal period = 60 minutes (hourly pattern)
    None,                    // Default to Close price
    Some(5),                 // 5-minute aggregation
).unwrap();

// Fit and forecast as with daily models
model.fit(&minute_data).unwrap();
let forecasts = model.forecast(12).unwrap();  // Next hour forecast (12 x 5 minutes)
```

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
oxdiviner = { version = "0.1.0", features = ["plotting", "polars_integration"] }
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 