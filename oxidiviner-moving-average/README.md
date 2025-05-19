# OxiDiviner Moving Average

[![Crates.io](https://img.shields.io/crates/v/oxidiviner-moving-average.svg)](https://crates.io/crates/oxidiviner-moving-average)
[![Documentation](https://docs.rs/oxidiviner-moving-average/badge.svg)](https://docs.rs/oxidiviner-moving-average)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Moving average models for time series forecasting in the OxiDiviner ecosystem.

## Overview

This crate provides implementations of Moving Average (MA) models for time series analysis and forecasting. MA models are useful for capturing short-term dependencies in time series data and are widely used in finance, economics, and signal processing.

## Features

- Implementation of MA(q) models
- Parameter estimation using method of moments
- Forecasting with confidence intervals
- Model selection based on information criteria
- Integration with the OxiDiviner forecasting ecosystem

## Model Definition

The Moving Average model MA(q) is defined as:

```
Y_t = μ + ε_t + θ_1*ε_{t-1} + θ_2*ε_{t-2} + ... + θ_q*ε_{t-q}
```

Where:
- Y_t is the time series value at time t
- μ is the mean of the process
- ε_t is white noise at time t
- θ_i are the model parameters
- q is the order of the moving average process

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxidiviner-moving-average = "0.1.0"
oxidiviner-core = "0.1.0"
```

### Example

```rust
use oxidiviner_core::{TimeSeriesData, Forecaster};
use oxidiviner_moving_average::MAModel;
use chrono::{Utc, TimeZone};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample time series data
    let dates = (0..10).map(|i| Utc.timestamp_opt(1609459200 + i * 86400, 0).unwrap()).collect();
    let values = vec![1.0, 1.2, 1.1, 1.3, 1.4, 1.3, 1.5, 1.4, 1.6, 1.5];
    let data = TimeSeriesData::new(dates, values)?;
    
    // Create an MA(2) model
    let mut model = MAModel::new(2)?;
    
    // Fit the model to the data
    model.fit(&data)?;
    
    // Display the fitted model
    println!("Model: {}", model);
    
    // Generate forecasts for the next 5 time steps
    let forecasts = model.forecast(5)?;
    println!("Forecasts: {:?}", forecasts);
    
    // Get a complete model output including evaluation metrics
    let output = model.predict(5, None)?;
    
    Ok(())
}
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details. 