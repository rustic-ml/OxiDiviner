# OxiDiviner Examples

This directory contains various examples demonstrating how to use the OxiDiviner library for time series analysis and forecasting.

## Running Examples

You can run any example using Cargo:

```bash
# From the project root directory
cargo run --example <example_name>

# Or from the examples directory
cargo run --bin <example_name>
```

## Examples Overview

### Main Examples

- **standard_interface_demo.rs** - Demonstrates the standardized forecasting interface used across all models
- **ohlcv_forecasting_example.rs** - Shows how to work with OHLCV (Open-High-Low-Close-Volume) financial data

### Model-Specific Examples

#### Exponential Smoothing Models

- **ses_demo.rs** - Simple Exponential Smoothing basic usage
- **ses_model_example.rs** - More comprehensive SES model example
- **ses_parameter_tuning.rs** - How to tune SES model parameters
- **holt_demo.rs** - Holt's Linear method for trend data
- **holt_winters_demo.rs** - Holt-Winters method for seasonal data
- **es_models_comparison.rs** - Comparison of different exponential smoothing models
- **es_parameter_tuning.rs** - Tuning parameters for various ES models
- **ets_demo.rs** - Generic ETS (Error-Trend-Seasonal) framework
- **ets_model_demo.rs** - Comprehensive ETS model examples

#### Autoregressive Models

- **autoregressive_demo.rs** - Demonstrates AR, ARMA, ARIMA, SARIMA, and VAR models
- **ma_demo.rs** - Moving Average model demonstration

#### GARCH Models

- **basic_garch_example.rs** - Basic GARCH model for volatility forecasting
- **stock_volatility_analysis.rs** - GARCH models applied to stock market volatility

### Data Handling

- **ohlcv-handling/** - Examples for working with financial time series data

## Advanced Usage

For more advanced examples and practical applications, please refer to the documentation at [docs.rs/oxidiviner](https://docs.rs/oxidiviner).

## Note on Deprecated Methods

Some examples contain deprecated Rand crate function calls which will be updated in a future release. This does not affect the functionality of the OxiDiviner library itself. 