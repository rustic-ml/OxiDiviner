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
- **ses_model_example.rs** - More comprehensive SES model demonstration
- **holt_demo.rs** - Holt's Linear Trend model usage
- **holt_winters_demo.rs** - Holt-Winters Seasonal model usage
- **es_models_comparison.rs** - Comparison of different exponential smoothing models
- **es_parameter_tuning.rs** - Parameter tuning for exponential smoothing models
- **ses_parameter_tuning.rs** - Parameter tuning specifically for SES models
- **ets_demo.rs** - Error-Trend-Seasonal (ETS) model usage
- **ets_model_demo.rs** - More comprehensive ETS model demonstration
- **ets_model_complete.rs** - Complete example of ETS model usage

#### Autoregressive Models

- **autoregressive_demo.rs** - Demonstrates AR, ARMA, ARIMA, SARIMA, and VAR models
- **ma_demo.rs** - Moving Average (MA) model usage

#### GARCH Models

- **basic_garch_example.rs** - Basic GARCH model usage
- **stock_volatility_analysis.rs** - Using GARCH for stock volatility analysis

### OHLCV Data Handling

- **ohlcv_data_processor.rs** - Example of processing OHLCV data

### Notes

Some examples require additional data files that are not included in the repository:

1. **ohlcv_data_processor.rs** - Requires CSV files in the examples/csv directory
2. **simple_forecast.rs** - Requires CSV files and uses an older API structure
3. **ets_simple_demo.rs** - A simplified demonstration that is not meant to be run fully

To run these examples, you would need to provide the appropriate data files or update the code to work with your own data.

## Advanced Usage

For more advanced examples and practical applications, please refer to the documentation at [docs.rs/oxidiviner](https://docs.rs/oxidiviner).

## Note on Deprecated Methods

Some examples contain deprecated Rand crate function calls which will be updated in a future release. This does not affect the functionality of the OxiDiviner library itself. 