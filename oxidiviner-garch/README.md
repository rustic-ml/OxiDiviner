# OxiDiviner GARCH

[![Crates.io](https://img.shields.io/crates/v/oxidiviner-garch.svg)](https://crates.io/crates/oxidiviner-garch)
[![Documentation](https://docs.rs/oxidiviner-garch/badge.svg)](https://docs.rs/oxidiviner-garch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Rust implementation of Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models for time series analysis and volatility forecasting.

## Overview

GARCH models are widely used for analyzing and forecasting volatility in financial time series data. This crate provides a robust implementation of GARCH(p,q) models, suitable for financial analysis, risk management, and econometric modeling.

## Features

- Full implementation of GARCH(p,q) models
- Parameter estimation and model fitting
- Volatility forecasting
- Model diagnostics and statistical metrics
- Support for time-indexed data

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxidiviner-garch = "0.1.0"
```

### Example

```rust
use oxidiviner_garch::GARCHModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a GARCH(1,1) model
    let mut model = GARCHModel::new(1, 1, None)?;
    
    // Sample time series data
    let data = vec![0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.4, -0.2, 0.1];
    
    // Fit the model
    model.fit(&data, None)?;
    
    // Display model parameters
    println!("{}", model);
    
    // Forecast future volatility (5 steps ahead)
    let forecast = model.forecast_variance(5)?;
    println!("Volatility forecast: {:?}", forecast);
    
    Ok(())
}
```

## Model Definition

The GARCH(p,q) model is defined as:

```
y_t = μ + ε_t
ε_t = σ_t * z_t, where z_t ~ N(0,1)
σ²_t = ω + Σ(i=1 to p) α_i * ε²_{t-i} + Σ(j=1 to q) β_j * σ²_{t-j}
```

Where:
- p is the order of the ARCH terms (ε²)
- q is the order of the GARCH terms (σ²)
- ω is the constant term (omega)
- α_i are the ARCH parameters
- β_j are the GARCH parameters

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details. 