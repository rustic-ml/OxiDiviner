# 🚀 OxiDiviner - Enhanced Time Series Forecasting

[![Crates.io](https://img.shields.io/crates/v/oxidiviner.svg)](https://crates.io/crates/oxidiviner)
[![Documentation](https://docs.rs/oxidiviner/badge.svg)](https://docs.rs/oxidiviner)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/your-username/oxidiviner/actions/workflows/test.yml/badge.svg)](https://github.com/your-username/oxidiviner/actions/workflows/test.yml)

A powerful, modern Rust library for time series analysis and forecasting with dramatically improved usability and production-ready features.

## ✨ What's New in v0.4.1 - Enhanced API

### 🔥 90% Code Reduction with Quick API

**Before (Traditional API):**
```rust
let mut model = ARIMAModel::new(2, 1, 1, true)?;
model.fit(&train_data)?;
let forecast = model.forecast(10)?;
let evaluation = model.evaluate(&test_data)?;
```

**After (Enhanced API):**
```rust
// One-line forecasting with automatic validation
let forecast = quick::arima(train_data, 10)?;

// Or with auto model selection
let (forecast, best_model) = quick::auto_select(data, 10)?;
```

### 🏗️ Fluent Builder Pattern

```rust
// Configure models with fluent interface
let config = ModelBuilder::arima()
    .with_ar(2)
    .with_differencing(1)
    .with_ma(1)
    .build_config();

let forecast = quick::forecast_with_config(data, 10, config)?;
```

### 🧠 Smart Model Selection

```rust
// Automatic best model detection with cross-validation
let selector = AutoSelector::with_cross_validation(3)
    .add_candidate(ModelBuilder::ar().with_ar(4).build_config())
    .add_candidate(ModelBuilder::ma().with_window(7).build_config());

let (forecast, best_model) = quick::auto_select(data, 10)?;
println!("Best model: {}", best_model); // e.g., "ARIMA(2,1,1)"
```

### ✅ Professional Validation Utilities

```rust
// Time series cross-validation
let splits = ValidationUtils::time_series_cv(&data, 5, Some(30))?;

// Comprehensive accuracy metrics
let metrics = ValidationUtils::accuracy_metrics(&actual, &predicted, None)?;
println!("MAE: {:.3}, RMSE: {:.3}, R²: {:.3}", 
    metrics.mae, metrics.rmse, metrics.r_squared);
```

### 🛡️ Automatic Parameter Validation

```rust
// Built-in parameter validation with helpful messages
ModelValidator::validate_arima_params(15, 3, 12)?;
// Error: "AR order (p) too high (max 10). Consider using a simpler model."
```

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
oxidiviner = "0.4.1"
chrono = "0.4"
```

### Simple Forecasting

```rust
use oxidiviner::{quick, TimeSeriesData};
use chrono::{Utc, Duration};

// Create sample data
let timestamps: Vec<_> = (0..100)
    .map(|i| Utc::now() + Duration::days(i))
    .collect();
let values: Vec<f64> = (0..100).map(|i| i as f64 + noise).collect();
let data = TimeSeriesData::new(timestamps, values, "my_series")?;

// One-line forecasting
let forecast = quick::arima(data.clone(), 10)?;

// Auto model selection
let (best_forecast, model_name) = quick::auto_select(data, 10)?;
println!("Best model: {}", model_name);
```

### Advanced Workflow

```rust
use oxidiviner::{quick, ModelBuilder, AutoSelector};
use oxidiviner_core::validation::ValidationUtils;

// 1. Data preparation and validation
let (train, test) = ValidationUtils::time_split(&data, 0.8)?;

// 2. Multiple model comparison
let configs = vec![
    ModelBuilder::arima().with_ar(1).with_differencing(1).with_ma(1).build_config(),
    ModelBuilder::arima().with_ar(2).with_differencing(1).with_ma(1).build_config(),
    ModelBuilder::ar().with_ar(3).build_config(),
];

// 3. Automated model selection
let (forecast, best_model) = quick::auto_select(train.clone(), test.values.len())?;

// 4. Comprehensive evaluation
let metrics = ValidationUtils::accuracy_metrics(&test.values, &forecast, None)?;
println!("Model: {}, MAE: {:.3}, R²: {:.3}", best_model, metrics.mae, metrics.r_squared);
```

## 📊 Available Models

### Autoregressive Models
- **AR(p)** - Autoregressive models
- **ARIMA(p,d,q)** - AutoRegressive Integrated Moving Average
- **SARIMA(p,d,q)(P,D,Q)s** - Seasonal ARIMA
- **VAR** - Vector Autoregression

### Exponential Smoothing
- **Simple ES** - Simple Exponential Smoothing
- **Holt Linear** - Double Exponential Smoothing
- **Holt-Winters** - Triple Exponential Smoothing
- **ETS** - Error, Trend, Seasonal models

### Other Models
- **Moving Average** - Simple and weighted moving averages
- **GARCH** - Volatility modeling for financial data

## 🏗️ Architecture

OxiDiviner uses a modern monorepo architecture:

```
oxidiviner/           # Main published crate
├── oxidiviner-core/     # Core traits and utilities
├── oxidiviner-autoregressive/  # AR, ARIMA, SARIMA models
├── oxidiviner-exponential-smoothing/  # ES models
├── oxidiviner-garch/    # Volatility models
├── oxidiviner-moving-average/  # MA models
├── oxidiviner-math/     # Mathematical utilities
└── examples/         # Comprehensive examples
```

## 📈 Performance & Features

### Enhanced API Benefits
- **90% fewer lines** of code for common tasks
- **Automatic parameter validation** prevents errors
- **Smart model selection** finds optimal models
- **Professional metrics** with comprehensive reporting
- **Time series CV** for robust validation
- **Fluent builder pattern** for readable configuration

### Traditional Rust Performance
- Zero-cost abstractions
- Memory-safe operations
- Parallel processing support
- SIMD optimizations where applicable

## 📚 Examples

Comprehensive examples are provided in the `examples/` directory:

- **Quick Start Enhanced** - New API overview
- **Autoregressive Demo** - AR, ARIMA, SARIMA models
- **Moving Average Demo** - MA models with validation
- **Traditional vs Enhanced** - API comparison
- **OHLCV Forecasting** - Financial data analysis
- **Model Comparison** - Automated model selection

Run examples:
```bash
cargo run --package oxidiviner-examples --bin quick_start_improved
cargo run --package oxidiviner-examples --bin autoregressive_demo
```

## 🔬 Testing

Comprehensive test suite with 90+ tests:

```bash
# Run all tests
cargo test --workspace

# Run enhanced API tests
cargo test test_enhanced --package oxidiviner

# Run specific examples
cargo run --example quick_start_improved
```

## 🛠️ Development

### Building from Source

```bash
git clone https://github.com/your-username/oxidiviner
cd oxidiviner
cargo build --release
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Python's `statsmodels` and R's forecasting packages
- Built with modern Rust best practices
- Designed for production use in quantitative finance and data science

## 📞 Support

- 📖 [Documentation](https://docs.rs/oxidiviner)
- 🐛 [Issue Tracker](https://github.com/your-username/oxidiviner/issues)
- 💬 [Discussions](https://github.com/your-username/oxidiviner/discussions)
- 📧 Email: your-email@example.com

---

**OxiDiviner**: Transforming time series forecasting in Rust with modern, accessible APIs and production-ready features. 🚀 