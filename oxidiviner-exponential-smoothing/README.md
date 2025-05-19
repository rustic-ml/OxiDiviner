# OxiDiviner Exponential Smoothing

[![Crates.io](https://img.shields.io/crates/v/oxidiviner-exponential-smoothing.svg)](https://crates.io/crates/oxidiviner-exponential-smoothing)
[![Documentation](https://docs.rs/oxidiviner-exponential-smoothing/badge.svg)](https://docs.rs/oxidiviner-exponential-smoothing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive exponential smoothing models for time series forecasting.

## Overview

This crate provides a family of exponential smoothing models for time series forecasting. Exponential smoothing methods are widely used for modeling and forecasting trends and seasonal patterns in time series data.

## Models

- **Simple Exponential Smoothing** - For series without trend or seasonality
- **Holt Linear** - For series with trend but no seasonality
- **Holt-Winters** - For series with both trend and seasonality
- **Damped Trend** - For series with dampened trend
- **ETS (Error, Trend, Seasonal)** - State space model for exponential smoothing

## Features

- Multiple smoothing methods for different types of time series
- Automatic parameter optimization
- Additive and multiplicative trend/seasonal components
- Confidence intervals for forecasts
- Seasonality detection and handling
- Model selection based on information criteria

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxidiviner-exponential-smoothing = "0.1.0"
oxidiviner-core = "0.1.0" 
```

### Example

```rust
use oxidiviner_core::{TimeSeriesData, Forecaster};
use oxidiviner_exponential_smoothing::{SimpleESModel, HoltWintersModel};
use chrono::{Utc, TimeZone};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample time series data with seasonal pattern
    let dates = (0..24).map(|i| Utc.timestamp_opt(1609459200 + i * 86400, 0).unwrap()).collect();
    let values = vec![
        10.0, 12.0, 14.0, 16.0, 13.0, 11.0, 9.0,  // Week 1
        11.0, 13.0, 15.0, 17.0, 14.0, 12.0, 10.0, // Week 2
        12.0, 14.0, 16.0, 18.0, 15.0, 13.0, 11.0, // Week 3
        13.0, 15.0, 17.0                          // Week 4 (partial)
    ];
    let data = TimeSeriesData::new(dates, values)?;
    
    // Simple exponential smoothing
    let mut simple_model = SimpleESModel::new(0.3)?;
    simple_model.fit(&data)?;
    
    // Holt-Winters with seasonality
    let mut hw_model = HoltWintersModel::new(0.2, 0.1, 0.3, 7, true)?;
    hw_model.fit(&data)?;
    
    // Generate forecasts
    let simple_forecast = simple_model.forecast(7)?;
    let hw_forecast = hw_model.forecast(7)?;
    
    println!("Simple ES Forecast: {:?}", simple_forecast);
    println!("Holt-Winters Forecast: {:?}", hw_forecast);
    
    Ok(())
}
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details. 