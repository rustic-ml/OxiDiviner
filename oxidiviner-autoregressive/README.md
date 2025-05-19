# OxiDiviner Autoregressive Models

[![Crates.io](https://img.shields.io/crates/v/oxidiviner-autoregressive.svg)](https://crates.io/crates/oxidiviner-autoregressive)
[![Documentation](https://docs.rs/oxidiviner-autoregressive/badge.svg)](https://docs.rs/oxidiviner-autoregressive)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Autoregressive time series models for econometric forecasting and analysis.

## Overview

This crate provides a comprehensive collection of autoregressive models for time series analysis and forecasting. These models are fundamental tools in econometrics, finance, and time series forecasting, allowing for the modeling of both stationary and non-stationary time series.

## Models

- **AR** - Autoregressive model for stationary time series
- **ARMA** - Autoregressive Moving Average for stationary time series with both AR and MA components
- **ARIMA** - Autoregressive Integrated Moving Average for non-stationary time series
- **SARIMA** - Seasonal ARIMA for time series with seasonal patterns
- **VAR** - Vector Autoregression for multivariate time series

## Features

- Parameter estimation using maximum likelihood and OLS methods
- Order selection using information criteria (AIC, BIC)
- Stationarity and invertibility checking
- Forecasting with confidence intervals
- Diagnostic checking and residual analysis
- Season detection and handling

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxidiviner-autoregressive = "0.1.0"
oxidiviner-core = "0.1.0"
```

### Example

```rust
use oxidiviner_core::{TimeSeriesData, Forecaster};
use oxidiviner_autoregressive::{ARIMAModel, ARModel};
use chrono::{Utc, TimeZone};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample time series data
    let dates = (0..20).map(|i| Utc.timestamp_opt(1609459200 + i * 86400, 0).unwrap()).collect();
    let values = vec![
        10.2, 10.8, 11.4, 11.9, 12.3, 12.9, 13.2, 13.6, 13.9, 14.3,
        14.5, 14.9, 15.1, 15.4, 15.6, 15.8, 16.0, 16.1, 16.3, 16.4
    ];
    let data = TimeSeriesData::new(dates, values)?;
    
    // Create an AR(2) model
    let mut ar_model = ARModel::new(2)?;
    ar_model.fit(&data)?;
    
    // Create an ARIMA(1,1,1) model
    let mut arima_model = ARIMAModel::new(1, 1, 1)?;
    arima_model.fit(&data)?;
    
    // Generate forecasts
    let ar_forecast = ar_model.forecast(5)?;
    let arima_forecast = arima_model.forecast(5)?;
    
    println!("AR(2) Forecast: {:?}", ar_forecast);
    println!("ARIMA(1,1,1) Forecast: {:?}", arima_forecast);
    
    Ok(())
}
```

## Theory

Autoregressive models assume that the current value of a time series depends on its previous values plus an error term:

- **AR(p)**: Y_t = c + φ₁Y_{t-1} + φ₂Y_{t-2} + ... + φ_pY_{t-p} + ε_t
- **ARMA(p,q)**: Y_t = c + ΣφᵢY_{t-i} + Σθⱼε_{t-j} + ε_t
- **ARIMA(p,d,q)**: Applies ARMA(p,q) to dth difference of Y_t

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details. 