# OxiDiviner Math

[![Crates.io](https://img.shields.io/crates/v/oxidiviner-math.svg)](https://crates.io/crates/oxidiviner-math)
[![Documentation](https://docs.rs/oxidiviner-math/badge.svg)](https://docs.rs/oxidiviner-math)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Mathematical utilities for time series analysis and statistical forecasting.

## Overview

This crate provides essential mathematical functions, statistical tools, and metrics for time series analysis in the OxiDiviner ecosystem. It serves as a foundation for implementing various forecasting algorithms and data transformations.

## Features

- Data transformations (differencing, log transformations, scaling)
- Statistical functions (autocorrelation, partial autocorrelation)
- Forecasting accuracy metrics (MAE, MSE, RMSE, MAPE, etc.)
- Time series stationarity tests
- Numerical optimization utilities

## Modules

- `transforms` - Data transformation functions for time series preprocessing
- `statistics` - Statistical functions for time series analysis
- `metrics` - Performance and accuracy metrics for evaluating forecasts

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxidiviner-math = "0.1.0"
```

### Example

```rust
use oxidiviner_math::transforms::{difference, log_transform};
use oxidiviner_math::metrics::{mean_absolute_error, mean_squared_error};

fn main() {
    // Original time series data
    let data = vec![10.5, 11.2, 10.8, 11.5, 12.0, 12.5, 12.2, 12.8];
    
    // Apply transformations
    let log_data = log_transform(&data);
    let diff_data = difference(&data, 1);
    
    println!("Original data: {:?}", data);
    println!("Log-transformed: {:?}", log_data);
    println!("First difference: {:?}", diff_data);
    
    // Calculate forecast accuracy metrics
    let actual = vec![10.0, 11.0, 9.5, 10.5];
    let predicted = vec![9.8, 10.5, 9.7, 10.2];
    
    let mae = mean_absolute_error(&actual, &predicted);
    let mse = mean_squared_error(&actual, &predicted);
    
    println!("MAE: {:.4}, MSE: {:.4}", mae, mse);
}
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details. 