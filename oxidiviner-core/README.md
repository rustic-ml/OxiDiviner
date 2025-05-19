# OxiDiviner Core

[![Crates.io](https://img.shields.io/crates/v/oxidiviner-core.svg)](https://crates.io/crates/oxidiviner-core)
[![Documentation](https://docs.rs/oxidiviner-core/badge.svg)](https://docs.rs/oxidiviner-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The core components and traits for the OxiDiviner time series forecasting library.

## Overview

This crate provides the fundamental building blocks for time series analysis and forecasting in the OxiDiviner ecosystem. It defines the common data structures, traits, and error handling mechanisms used by all OxiDiviner forecasting models.

## Features

- `TimeSeriesData` - Flexible container for time series data with timestamps
- `OHLCVData` - Specialized container for financial time series (Open-High-Low-Close-Volume)
- `Forecaster` trait - Common interface for all forecasting models
- Standardized model evaluation metrics
- Common error types and results

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxidiviner-core = "0.1.0"
```

### Example

```rust
use oxidiviner_core::{TimeSeriesData, Forecaster, ModelOutput};
use chrono::{DateTime, Utc};

// Implementing a custom forecasting model
struct SimpleMovingAverage {
    name: String,
    window_size: usize,
    values: Vec<f64>,
}

impl SimpleMovingAverage {
    fn new(window_size: usize) -> Self {
        Self {
            name: format!("SMA({})", window_size),
            window_size,
            values: Vec::new(),
        }
    }
}

impl Forecaster for SimpleMovingAverage {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn fit(&mut self, data: &TimeSeriesData) -> oxidiviner_core::Result<()> {
        self.values = data.values().to_vec();
        Ok(())
    }
    
    fn forecast(&self, horizon: usize) -> oxidiviner_core::Result<Vec<f64>> {
        if self.values.is_empty() {
            return Err(oxidiviner_core::OxiError::NotFitted);
        }
        
        // Use last window_size values to calculate the average
        let start = self.values.len().saturating_sub(self.window_size);
        let window = &self.values[start..];
        let avg = window.iter().sum::<f64>() / window.len() as f64;
        
        // Return the average for each forecasted point
        Ok(vec![avg; horizon])
    }
    
    fn evaluate(&self, test_data: &TimeSeriesData) -> oxidiviner_core::Result<oxidiviner_core::ModelEvaluation> {
        // Implementation omitted for brevity
        unimplemented!()
    }
}
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details. 