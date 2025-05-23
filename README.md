# 🔮 OxiDiviner

**Advanced Time Series Forecasting Library for Rust**

[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-93%2B%25-brightgreen)](CODE_COVERAGE_REPORT.md)

OxiDiviner is a comprehensive, production-ready time series forecasting library built in Rust. It provides a wide range of statistical models and algorithms for predicting future values in time series data.

## ✨ **Features**

### 🎯 **Core Forecasting Models**
- **ARIMA** (AutoRegressive Integrated Moving Average) with seasonal support
- **Moving Averages** with configurable windows and optimization
- **Exponential Smoothing** (Simple, Holt, Holt-Winters)
- **AutoRegressive** models (AR, ARMA, VAR, SARIMA)
- **GARCH** models for volatility forecasting
- **ETS** (Error, Trend, Seasonal) state space models

### 🚀 **API Design**
- **Quick API**: One-line forecasting for rapid prototyping
- **Enhanced API**: Full control with builder patterns and configurations
- **Batch Processing**: Multi-series forecasting with parallel execution
- **Validation**: Comprehensive model validation and accuracy metrics

### 📊 **Analysis Tools**
- **Accuracy Metrics**: MAE, MSE, RMSE, MAPE, sMAPE
- **Model Comparison**: Automatic model selection and benchmarking
- **Cross-Validation**: Time series-aware validation techniques
- **Statistical Analysis**: Autocorrelation, stationarity testing

## 🚀 **Quick Start**

Add OxiDiviner to your `Cargo.toml`:

```toml
[dependencies]
oxidiviner = "0.4.2"
```

### Simple Forecasting

```rust
use oxidiviner::quick;

// Generate sample data
let data: Vec<f64> = (0..50).map(|i| i as f64 + 10.0).collect();

// One-line forecast
let forecast = quick::forecast(&data, 10)?;
println!("Next 10 values: {:?}", forecast);

// Model-specific forecasting
let arima_forecast = quick::arima(data.clone(), 5)?;
let ma_forecast = quick::moving_average(data.clone(), 5)?;
```

### Advanced Usage

```rust
use oxidiviner::{
    models::moving_average::MAModel,
    core::data::TimeSeriesData,
    enhanced::{ForecastConfig, ModelBuilder},
    math::metrics::calculate_rmse,
};

// Create time series data
let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let data = TimeSeriesData::new(values, None)?;

// Enhanced API with configuration
let config = ForecastConfig::new()
    .forecast_horizon(10)
    .confidence_intervals(true)
    .validation_split(0.2);

let forecast = ModelBuilder::moving_average()
    .window_size(3)
    .build()?
    .forecast_with_config(&data, &config)?;

println!("Forecast: {:?}", forecast.values);
println!("Confidence: {:?}", forecast.confidence_intervals);
```

## 📦 **Installation & Setup**

```bash
# Add to existing project
cargo add oxidiviner

# Clone and test
git clone https://github.com/yourusername/oxidiviner
cd oxidiviner/oxidiviner
cargo test
cargo run --example arima_example
```

## 📖 **Examples**

Comprehensive examples are available in the [`examples/`](examples/) directory:

| Example | Description | Key Features |
|---------|-------------|--------------|
| **[ARIMA Example](examples/arima_example.rs)** | ARIMA modeling and forecasting | Parameter estimation, model comparison |
| **[Moving Average](examples/moving_average_example.rs)** | Window-based smoothing | Optimal window selection, noise analysis |
| **[Exponential Smoothing](examples/exponential_smoothing_example.rs)** | Business forecasting | Sales prediction, seasonal patterns |
| **[GARCH Example](examples/garch_example.rs)** | Volatility modeling | Financial risk analysis, VaR calculation |
| **[Quick API Test](examples/quick_test.rs)** | Rapid prototyping | One-line forecasting |

### Running Examples

```bash
# Run specific example
cargo run --example arima_example

# Run all examples (from examples directory)
for example in examples/*.rs; do
    cargo run --example $(basename "$example" .rs)
done
```

## 🧪 **Testing**

OxiDiviner includes comprehensive test coverage:

```bash
# Unit tests
cargo test --lib

# Integration tests
cargo test --test integration_tests

# Run examples
cargo test --examples

# Coverage analysis
./run_coverage.sh
```

**Test Coverage**: 93.4% with 127 passing tests across all modules.

## 📈 **Model Performance**

| Model Type | Use Cases | Accuracy | Speed |
|------------|-----------|----------|-------|
| **ARIMA** | Trending data, seasonal patterns | High | Medium |
| **Moving Average** | Smoothing, baseline forecasting | Medium | High |
| **Exponential Smoothing** | Business metrics, sales | High | High |
| **GARCH** | Financial volatility | High | Medium |
| **AutoRegressive** | Pattern recognition | High | Medium |

## 🏗️ **Architecture**

OxiDiviner uses a unified single-crate architecture:

```
oxidiviner/
├── src/
│   ├── core/           # Core data structures and validation
│   ├── math/           # Mathematical functions and metrics
│   ├── models/         # Forecasting model implementations
│   ├── quick.rs        # High-level API
│   ├── enhanced.rs     # Advanced configuration API
│   └── batch.rs        # Multi-series processing
├── examples/           # Comprehensive examples
├── tests/             # Integration tests
└── docs/              # Documentation
```

## 🔧 **Configuration**

### Model Parameters

```rust
// ARIMA configuration
let arima = ARIMAModel::new(2, 1, 1, true)?;  // (p, d, q, intercept)

// Moving Average with custom window
let ma = MAModel::new(7)?;  // 7-day moving average

// Exponential Smoothing parameters
let ses = SimpleESModel::new(0.3)?;  // α = 0.3
```

### Validation and Metrics

```rust
use oxidiviner::{core::validation::*, math::metrics::*};

// Time series cross-validation
let cv_results = time_series_cv(&data, &model, 5)?;

// Calculate accuracy metrics
let mae = calculate_mae(&actual, &predicted);
let rmse = calculate_rmse(&actual, &predicted);
let mape = calculate_mape(&actual, &predicted);
```

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/yourusername/oxidiviner
cd oxidiviner/oxidiviner
cargo build
cargo test
```

## 📋 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 **Research & References**

OxiDiviner implements well-established forecasting algorithms based on:

- Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control
- Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity

## 📞 **Support**

- **Documentation**: [API Docs](https://docs.rs/oxidiviner)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/oxidiviner/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/oxidiviner/discussions)

---

**Built with ❤️ in Rust** 🦀 