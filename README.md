# OxiDiviner

![OxiDiviner Logo](https://raw.githubusercontent.com/rustic-ml/OxiDiviner/main/OxiDiviner_250px.JPG)

[![Crates.io](https://img.shields.io/crates/v/oxidiviner.svg)](https://crates.io/crates/oxidiviner)
[![Documentation](https://docs.rs/oxidiviner/badge.svg)](https://docs.rs/oxidiviner)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rustic-ml/OxiDiviner/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-rustic--ml/OxiDiviner-8da0cb?logo=github)](https://github.com/rustic-ml/OxiDiviner)

OxiDiviner is a Rust library for time series analysis and forecasting, designed to be a comprehensive toolkit for traders and data scientists.

## The OxiDiviner: A Prophetic Approach to Forecasting

In ancient times, diviners were revered for their ability to predict the future through careful observation of patterns and signs. The OxiDiviner carries this tradition into the modern era, but instead of reading animal entrails or celestial alignments, it reads the patterns hidden within time series data.

Like the oracles of old, OxiDiviner draws upon time-tested methods to glimpse what lies beyond the horizon. It combines the wisdom of traditional statistical forecasting techniques with the precision and efficiency of Rust:

- **Moving Average models** act as the diviner's smoothing stone, revealing the underlying currents beneath chaotic market waters
- **Exponential Smoothing methods** serve as the weighted scales, balancing recent observations against historical patterns
- **Autoregressive models** function as the prophet's memory, recalling how past values influence future ones
- **ARIMA and SARIMA models** operate as the sage's differencing tools, finding stability in seemingly unstable trends
- **GARCH models** work as the seer's volatility crystal, detecting clusters of turbulence in financial markets

OxiDiviner doesn't rely on mysticism but on mathematical rigor. Each model is carefully implemented with statistical precision, providing not just forecasts but confidence intervals, error metrics, and diagnostic information. This approach allows traders and analysts to make decisions with both intuition and evidence—the hallmarks of true divination.

Whether you're tracking stock prices, predicting sales, or analyzing any time-dependent data, OxiDiviner offers the tools to see beyond the present moment into possible futures, all while maintaining the rational foundation of established forecasting methodologies.

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

- **Time series forecasting models** including:
  - Simple Exponential Smoothing (SES)
  - Moving Average (MA)
  - Exponential Smoothing models with trend and seasonality
  - Autoregressive (AR), ARIMA, and SARIMA models
  - Vector Autoregression (VAR) for multivariate time series
  - GARCH models and variants (GJR-GARCH, EGARCH, GARCH-M) for volatility forecasting
  - Specialized models for daily and minute-level financial data

- **Data handling**: Import and manage OHLCV (Open-High-Low-Close-Volume) stock data
- **Time series processing**: Tools for manipulating and analyzing time series data
- **Standardized model interface** via the `Forecaster` trait
- **Convenient prelude module** for easy importing of commonly used types and models
- **Consistent output format** via the `ModelOutput` struct
- **Comprehensive model evaluation metrics** (MAE, MSE, RMSE, MAPE, SMAPE)
- **Utilities for data loading, preprocessing, and visualization**

## Getting Started

Add OxiDiviner to your Cargo.toml:

```toml
[dependencies]
oxidiviner = "0.4.0"
```

See the examples directory for complete usage examples.

## Simplified Imports

OxiDiviner offers multiple ways to import functionality, making it easier to use the library based on your preferences:

### Option 1: Using the Prelude Module

The prelude module provides access to commonly used types and models in a single import:

```rust
use oxidiviner::prelude::*;

fn main() -> Result<()> {
    // Create a time series
    let data = TimeSeriesData::new(dates, values, "example_series")?;
    
    // Use models directly
    let mut model = ARModel::new(1, true)?;
    model.fit(&data)?;
    
    // Access metrics functions
    let mae_value = mae(&actual, &predicted);
    
    Ok(())
}
```

### Option 2: Direct Model Imports from Root

For a clean, concise approach, you can import models directly from the root:

```rust
use oxidiviner::{TimeSeriesData, Forecaster, ARModel, HoltWintersModel, MAModel};

fn main() -> oxidiviner::Result<()> {
    // Now you can use models directly without nested imports
    let mut ma_model = MAModel::new(3)?;
    let mut arima = oxidiviner::ARIMAModel::new(1, 1, 1, true)?;
    
    Ok(())
}
```

### Option 3: Direct Access to Subcrates

For more advanced usage or when working with specific components:

```rust
// Direct access to subcrates
use oxidiviner::moving_average::MAModel;
use oxidiviner::exponential_smoothing::HoltWintersModel;
use oxidiviner::autoregressive::ARIMAModel;
use oxidiviner::math::metrics::rmse;

fn main() {
    // Use types directly from the subcrates
}
```

## Usage Examples

### Basic Time Series Analysis

```rust
use chrono::Utc;
use oxidiviner::{OHLCVData, TimeSeriesData};

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

### Forecasting Examples

#### Moving Average (MA) Model

Ideal for smoothing short-term fluctuations and identifying longer-term trends:

```rust
use oxidiviner::prelude::*;
use chrono::{Utc, TimeZone};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample time series data
    let dates = (0..20).map(|i| Utc.with_ymd_and_hms(2023, 1, i+1, 0, 0, 0).unwrap()).collect();
    let values = vec![10.5, 11.2, 10.8, 11.5, 12.0, 11.8, 12.3, 13.0, 12.5, 12.8, 
                      13.2, 13.5, 13.0, 14.0, 14.5, 14.2, 14.8, 15.0, 15.5, 15.2];
    let data = TimeSeriesData::new(dates, values, "sample_data")?;
    
    // Create and fit an MA model with window size 3
    let mut ma_model = MAModel::new(3)?;
    ma_model.fit(&data)?;
    
    // Generate forecast for next 5 periods
    let forecast = ma_model.forecast(5)?;
    println!("MA(3) forecast: {:?}", forecast);
    
    Ok(())
}
```

#### Simple Exponential Smoothing (SES) Model

Appropriate for data without clear trend or seasonality:

```rust
use oxidiviner::prelude::*;
use chrono::{Utc, TimeZone};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample time series data
    let dates = (0..20).map(|i| Utc.with_ymd_and_hms(2023, 1, i+1, 0, 0, 0).unwrap()).collect();
    let values = vec![10.5, 11.2, 10.8, 11.5, 12.0, 11.8, 12.3, 13.0, 12.5, 12.8, 
                      13.2, 13.5, 13.0, 14.0, 14.5, 14.2, 14.8, 15.0, 15.5, 15.2];
    let data = TimeSeriesData::new(dates, values, "sample_data")?;
    
    // Create and fit a Simple Exponential Smoothing model with α=0.3
    let mut ses_model = SESModel::new(0.3, None)?;
    ses_model.fit(&data)?;
    
    // Generate forecast for next 5 periods
    let forecast = ses_model.forecast(5)?;
    println!("SES forecast: {:?}", forecast);
    
    Ok(())
}
```

#### Holt-Winters Model (Triple Exponential Smoothing)

For data with both trend and seasonality:

```rust
use oxidiviner::prelude::*;
use oxidiviner::models::exponential_smoothing::HoltWintersModel;
use chrono::{Utc, TimeZone};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample time series with seasonality (period=7 days)
    let dates = (0..28).map(|i| Utc.with_ymd_and_hms(2023, 1, i+1, 0, 0, 0).unwrap()).collect();
    
    // Values with weekly seasonality
    let values = vec![
        10.0, 11.2, 12.5, 14.0, 13.0, 11.5, 10.2,  // Week 1
        11.0, 12.2, 13.5, 15.0, 14.0, 12.5, 11.2,  // Week 2
        12.0, 13.2, 14.5, 16.0, 15.0, 13.5, 12.2,  // Week 3
        13.0, 14.2, 15.5, 17.0, 16.0, 14.5, 13.2   // Week 4
    ];
    
    let data = TimeSeriesData::new(dates, values, "seasonal_data")?;
    
    // Create and fit a Holt-Winters model
    // Parameters: α (level), β (trend), γ (seasonal), seasonal_period
    let mut hw_model = HoltWintersModel::new(0.2, 0.1, 0.1, 7)?;
    hw_model.fit(&data)?;
    
    // Generate forecast for next 7 days
    let forecast = hw_model.forecast(7)?;
    println!("Holt-Winters forecast: {:?}", forecast);
    
    Ok(())
}
```

#### Autoregressive (AR) Model

For data where values depend on previous values:

```rust
use oxidiviner::prelude::*;
use chrono::{Utc, TimeZone};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample time series data
    let dates = (0..30).map(|i| Utc.with_ymd_and_hms(2023, 1, i+1, 0, 0, 0).unwrap()).collect();
    
    // Create an AR(2) process with some noise
    let mut values = Vec::with_capacity(30);
    values.push(1.0);
    values.push(1.2);
    
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    for i in 2..30 {
        // AR(2) process: y_t = 0.7*y_{t-1} - 0.3*y_{t-2} + noise
        let next_val = 0.7 * values[i-1] - 0.3 * values[i-2] + rng.gen::<f64>() * 0.5 - 0.25;
        values.push(next_val);
    }
    
    let data = TimeSeriesData::new(dates, values, "ar_data")?;
    
    // Create and fit an AR(2) model
    let mut ar_model = ARModel::new(2)?;
    ar_model.fit(&data)?;
    
    // Generate forecast for next 5 periods
    let forecast = ar_model.forecast(5)?;
    println!("AR(2) forecast: {:?}", forecast);
    
    Ok(())
}
```

#### ARIMA Model

For data that requires differencing to become stationary:

```rust
use oxidiviner::prelude::*;
use chrono::{Utc, TimeZone};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample time series with trend (non-stationary)
    let dates = (0..30).map(|i| Utc.with_ymd_and_hms(2023, 1, i+1, 0, 0, 0).unwrap()).collect();
    
    // Create a trending series with AR components
    let mut values = Vec::with_capacity(30);
    values.push(10.0);
    values.push(10.4);
    
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    for i in 2..30 {
        // Trend + AR(1) process
        let trend = 0.2 * (i as f64);  // Linear trend
        let ar_component = 0.7 * (values[i-1] - trend);
        let next_val = trend + ar_component + rng.gen::<f64>() * 0.5 - 0.25;
        values.push(next_val);
    }
    
    let data = TimeSeriesData::new(dates, values, "arima_data")?;
    
    // Create and fit an ARIMA(1,1,0) model
    // Parameters: p (AR order), d (differencing), q (MA order)
    let mut arima_model = ARIMAModel::new(1, 1, 0)?;
    arima_model.fit(&data)?;
    
    // Generate forecast for next 5 periods
    let forecast = arima_model.forecast(5)?;
    println!("ARIMA(1,1,0) forecast: {:?}", forecast);
    
    Ok(())
}
```

#### GARCH Model for Volatility Forecasting

For financial time series where volatility clustering is important:

```rust
use oxidiviner::prelude::*;
use chrono::{Utc, TimeZone};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Sample financial returns data (percentage changes)
    let returns = vec![
        0.03, 0.02, 0.01, -0.02, -0.03, -0.01, 0.02, 0.03, 0.02, 0.01,
        -0.05, -0.06, -0.03, -0.04, -0.02, 0.01, 0.02, 0.01, 0.03, 0.04,
        0.02, 0.01, -0.01, -0.02, -0.01, 0.01, 0.02, 0.01, -0.03, -0.02
    ];
    
    // Create a GARCH(1,1) model
    let mut garch_model = GARCHModel::new(1, 1, None)?;
    
    // Fit the model to the returns data
    garch_model.fit(&returns, None)?;
    
    // Forecast volatility for the next 5 periods
    let volatility_forecast = garch_model.forecast_variance(5)?;
    println!("GARCH(1,1) volatility forecast: {:?}", volatility_forecast);
    
    Ok(())
}
```

### Combining Multiple Models for Improved Forecasting

OxiDiviner's standardized interface makes it easy to combine multiple models for ensemble forecasting, which often produces more robust predictions than any single model:

```rust
use oxidiviner::prelude::*;
use chrono::{Utc, TimeZone};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample time series data
    let dates = (0..30).map(|i| Utc.with_ymd_and_hms(2023, 1, i+1, 0, 0, 0).unwrap()).collect();
    let values = vec![
        10.5, 11.2, 10.8, 11.5, 12.0, 11.8, 12.3, 13.0, 12.5, 12.8, 
        13.2, 13.5, 13.0, 14.0, 14.5, 14.2, 14.8, 15.0, 15.5, 15.2,
        15.8, 16.1, 15.9, 16.3, 16.5, 16.0, 16.6, 17.0, 17.2, 17.5
    ];
    
    // Split into training and test sets
    let train_data = TimeSeriesData::new(
        dates[0..25].to_vec(),
        values[0..25].to_vec(),
        "train_data"
    )?;
    
    let test_data = TimeSeriesData::new(
        dates[25..30].to_vec(),
        values[25..30].to_vec(),
        "test_data"
    )?;
    
    // Create and fit multiple models
    let mut models: HashMap<String, Box<dyn Forecaster>> = HashMap::new();
    
    // Moving Average model
    let mut ma_model = MAModel::new(3)?;
    ma_model.fit(&train_data)?;
    models.insert("MA(3)".to_string(), Box::new(ma_model));
    
    // Simple Exponential Smoothing model
    let mut ses_model = SESModel::new(0.3, None)?;
    ses_model.fit(&train_data)?;
    models.insert("SES".to_string(), Box::new(ses_model));
    
    // AR model
    let mut ar_model = ARModel::new(2)?;
    ar_model.fit(&train_data)?;
    models.insert("AR(2)".to_string(), Box::new(ar_model));
    
    // Generate forecasts and evaluate each model
    let horizon = 5; // Same as test data length
    let mut forecasts = HashMap::new();
    let mut weights = HashMap::new();
    let mut total_weight = 0.0;
    
    println!("Individual model forecasts and evaluation:");
    for (name, model) in &models {
        let output = model.predict(horizon, Some(&test_data))?;
        
        if let Some(eval) = &output.evaluation {
            println!("{} - MAE: {:.4}, RMSE: {:.4}", name, eval.mae, eval.rmse);
            
            // Use inverse of MAE as weight (lower error = higher weight)
            let weight = 1.0 / eval.mae;
            weights.insert(name.clone(), weight);
            total_weight += weight;
        }
        
        forecasts.insert(name.clone(), output.forecasts);
    }
    
    // Calculate weighted ensemble forecast
    let mut ensemble_forecast = vec![0.0; horizon];
    
    for (name, model_forecast) in &forecasts {
        let weight = weights.get(name).unwrap() / total_weight;
        
        for i in 0..horizon {
            ensemble_forecast[i] += model_forecast[i] * weight;
        }
    }
    
    println!("\nEnsemble forecast (weighted average):");
    for i in 0..horizon {
        println!("t+{}: {:.4}", i+1, ensemble_forecast[i]);
    }
    
    // Calculate ensemble error metrics
    let mae = oxidiviner::math::metrics::mean_absolute_error(&test_data.values, &ensemble_forecast);
    let rmse = oxidiviner::math::metrics::root_mean_squared_error(&test_data.values, &ensemble_forecast);
    
    println!("\nEnsemble performance - MAE: {:.4}, RMSE: {:.4}", mae, rmse);
    
    Ok(())
}
```

### Using Daily ETS Models

The `DailyETSModel` provides specialized functionality for daily financial data:

```rust
use oxidiviner::models::ets::{ETSComponent, DailyETSModel};

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
use oxidiviner::models::ets::{ETSComponent, MinuteETSModel};

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
cd examples && cargo run --bin ets_demo -- SYNTHETIC daily
```

Options:
- First argument: Stock ticker (e.g., AAPL, MSFT) or SYNTHETIC for synthetic data
- Second argument: Data type - 'daily' or 'minute'

## Feature Options

OxiDiviner has several optional features that can be enabled:

- `plotting`: Enables visualization capabilities using the `plotters` crate
- `polars_integration`: Adds support for loading data from CSV and Parquet files
- `ndarray_support`: Adds integration with the `ndarray` library for numerical computing

Enable features in your Cargo.toml:

```toml
[dependencies]
oxidiviner = { version = "0.4.0", features = ["plotting", "polars_integration"] }
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 