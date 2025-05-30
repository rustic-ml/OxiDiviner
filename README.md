# 🔮 OxiDiviner

![OxiDiviner Logo](OxiDiviner_250px.JPG)

[![Crates.io](https://img.shields.io/crates/v/oxidiviner.svg)](https://crates.io/crates/oxidiviner)
[![Documentation](https://docs.rs/oxidiviner/badge.svg)](https://docs.rs/oxidiviner)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rustic-ml/OxiDiviner/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-rustic--ml/OxiDiviner-8da0cb?logo=github)](https://github.com/rustic-ml/OxiDiviner)

> **The Rust Oracle for Time Series Forecasting** 🦀

OxiDiviner is a comprehensive, production-ready time series analysis and forecasting library built in Rust. Named after the fusion of **"Oxi"** (from oxidation/Rust) and **"Diviner"** (one who foresees the future), it empowers developers to predict time series patterns with the safety, speed, and elegance of Rust.

Whether you're forecasting financial markets, predicting business metrics, or analyzing sensor data, OxiDiviner provides the tools you need with multiple API layers designed for different expertise levels - from quick one-liners to advanced statistical modeling.

## ✨ **Why OxiDiviner?**

- **🚀 Performance**: Rust's zero-cost abstractions for high-speed forecasting
- **🛡️ Safety**: Memory-safe implementations with comprehensive error handling  
- **🎯 Accuracy**: Battle-tested statistical models with rigorous validation
- **🔄 Flexibility**: Multiple API layers from simple to sophisticated
- **📊 Comprehensive**: Complete toolkit from data prep to model evaluation
- **🧪 Production-Ready**: Extensively tested with 140+ passing tests

## 🎯 **Core Features**

### **Forecasting Models**
- **ARIMA** (AutoRegressive Integrated Moving Average) with seasonal support
- **Moving Averages** with adaptive window optimization
- **Exponential Smoothing** (Simple, Holt's Linear, Holt-Winters)  
- **AutoRegressive** models (AR, ARMA, VAR, SARIMA)
- **GARCH** models for volatility and risk forecasting
- **ETS** (Error, Trend, Seasonal) state space models

### **API Design Philosophy**
- **Quick API**: One-line forecasting for rapid prototyping
- **Builder Pattern**: Fluent configuration for complex models
- **Unified Interface**: Consistent `predict()` method across all models
- **Batch Processing**: Multi-series forecasting with parallel execution

### **Analysis & Validation**
- **Comprehensive Metrics**: MAE, MSE, RMSE, MAPE, sMAPE, AIC, BIC
- **Cross-Validation**: Time series-aware validation techniques
- **Model Selection**: Automatic best-model identification  
- **Data Quality**: Stationarity testing, missing value handling

## 🚀 **Quick Start**

Add OxiDiviner to your `Cargo.toml`:

```toml
[dependencies]
oxidiviner = "0.4.2"
```

### **30-Second Example**

```rust
use oxidiviner::prelude::*;
use oxidiviner::quick;

// Quick forecasting - one line!
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let forecast = quick::auto_select(data, 5)?;
println!("Next 5 values: {:?}", forecast.0);
```

## 📚 **API Guide: Choose Your Level**

OxiDiviner provides three API levels to match your expertise and use case:

### **🏃 Level 1: Quick API** (Beginner-Friendly)
*Perfect for rapid prototyping and simple forecasting tasks*

```rust
use oxidiviner::quick;

let data = vec![10.0, 12.0, 13.0, 12.0, 15.0, 16.0, 18.0];

// Automatic model selection
let (forecast, model_used) = quick::auto_select(data.clone(), 3)?;
println!("Best model: {}, Forecast: {:?}", model_used, forecast);

// Specific models
let arima_forecast = quick::arima(data.clone(), 3)?;
let ma_forecast = quick::moving_average(data.clone(), 3, Some(5))?;
let es_forecast = quick::exponential_smoothing(data, 3, Some(0.3))?;
```

### **🏗️ Level 2: Builder Pattern API** (Recommended)
*Ideal for production use with full control and configuration*

```rust
use oxidiviner::prelude::*;

// Create time series data
let timestamps = (0..30).map(|i| Utc::now() + Duration::days(i)).collect();
let values: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 + (i as f64 * 0.1).sin() * 5.0).collect();
let data = TimeSeriesData::new(timestamps, values, "my_series")?;

// Split data for validation
let (train, test) = ValidationUtils::time_split(&data, 0.8)?;

// Build and configure model
let config = ModelBuilder::arima()
    .with_ar(2)
    .with_differencing(1)
    .with_ma(1)
    .build_config();

// Forecast with configuration
let forecast = quick::forecast_with_config(train, test.len(), config)?;

// Evaluate accuracy
let metrics = ValidationUtils::accuracy_metrics(&test.values, &forecast, None)?;
println!("MAE: {:.3}, RMSE: {:.3}", metrics.mae, metrics.rmse);
```

### **⚙️ Level 3: Direct Model API** (Advanced)
*For fine-grained control and custom implementations*

```rust
use oxidiviner::prelude::*;

// Create and configure model directly
let mut model = HoltWintersModel::new(0.3, 0.1, 0.1, 7)?; // α, β, γ, season_length

// Fit to training data
model.fit(&train_data)?;

// Generate predictions with evaluation
let output = model.predict(forecast_horizon, Some(&test_data))?;

// Access detailed results
println!("Model: {}", output.model_name);
println!("Forecasts: {:?}", output.forecasts);
if let Some(eval) = output.evaluation {
    println!("MAE: {:.4}, RMSE: {:.4}, MAPE: {:.2}%", eval.mae, eval.rmse, eval.mape);
}
```

## 📊 **Working with Data**

### **TimeSeriesData Creation**

```rust
use oxidiviner::prelude::*;
use chrono::{Duration, Utc};

// From vectors with timestamps
let timestamps = (0..10).map(|i| Utc::now() + Duration::days(i)).collect();
let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let data = TimeSeriesData::new(timestamps, values, "my_series")?;

// From OHLCV financial data
let ohlcv = OHLCVData {
    symbol: "AAPL".to_string(),
    timestamps: timestamps,
    open: vec![100.0; 10],
    high: vec![105.0; 10], 
    low: vec![95.0; 10],
    close: vec![102.0; 10],
    volume: vec![1000.0; 10],
    adjusted_close: None,
};
let ts_data = ohlcv.to_time_series(false); // Use close prices
```

### **Data Validation & Preparation**

```rust
use oxidiviner::core::validation::ValidationUtils;

// Split for train/test
let (train, test) = ValidationUtils::time_split(&data, 0.8)?;

// Cross-validation
let cv_splits = ValidationUtils::time_series_cv(&data, 5, Some(20))?;

// Check data quality
let is_valid = ModelValidator::validate_minimum_data(data.len(), 10, "ARIMA")?;
```

## 🛠️ **Model Configuration**

### **Available Models & Parameters**

| Model | Constructor | Key Parameters | Use Case |
|-------|-------------|----------------|-----------|
| **SimpleESModel** | `new(alpha)` | α ∈ (0,1) | Stable series, no trend |
| **HoltLinearModel** | `new(alpha, beta)` | α,β ∈ (0,1) | Linear trend |
| **HoltWintersModel** | `new(alpha, beta, gamma, period)` | α,β,γ ∈ (0,1), period > 1 | Seasonal patterns |
| **ARIMAModel** | `new(p, d, q, intercept)` | p,d,q ≥ 0 | Complex temporal dependencies |
| **MAModel** | `new(window)` | window > 0 | Smoothing, noise reduction |
| **GARCHModel** | `new(p, q, mean_model)` | p,q ≥ 1 | Volatility modeling |

### **Parameter Validation**

```rust
use oxidiviner::ModelValidator;

// Validate before model creation
ModelValidator::validate_arima_params(2, 1, 1)?;
ModelValidator::validate_exponential_smoothing_params(0.3, Some(0.1), Some(0.1))?;
ModelValidator::validate_ma_params(7)?;
```

## 📈 **Model Evaluation & Selection**

### **Automatic Model Selection**

```rust
use oxidiviner::{AutoSelector, SelectionCriteria};

// Different selection criteria
let aic_selector = AutoSelector::with_aic();
let bic_selector = AutoSelector::with_bic(); 
let cv_selector = AutoSelector::with_cross_validation(5);

// Add custom models to comparison
let custom_config = ModelBuilder::ar().with_ar(3).build_config();
let selector = AutoSelector::with_aic().add_candidate(custom_config);
```

### **Accuracy Metrics**

```rust
// Get comprehensive metrics
let metrics = ValidationUtils::accuracy_metrics(&actual, &predicted, None)?;

println!("MAE:  {:.4}", metrics.mae);       // Mean Absolute Error
println!("RMSE: {:.4}", metrics.rmse);      // Root Mean Square Error  
println!("MAPE: {:.2}%", metrics.mape);     // Mean Absolute Percentage Error
println!("R²:   {:.4}", metrics.r_squared); // Coefficient of Determination
```

## 🔄 **Error Handling**

OxiDiviner uses comprehensive error handling with the `Result<T>` type:

```rust
use oxidiviner::prelude::*;

match quick::arima(data, 5) {
    Ok(forecast) => println!("Success: {:?}", forecast),
    Err(OxiError::InvalidParameter(msg)) => eprintln!("Parameter error: {}", msg),
    Err(OxiError::InsufficientData(msg)) => eprintln!("Data error: {}", msg),
    Err(OxiError::ModelFitError(msg)) => eprintln!("Fitting error: {}", msg),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## 📦 **Examples**

OxiDiviner provides extensive examples in two locations:

### **🚀 Main Library Examples** 
*Run from root directory with `cargo run --example <name>`*

| Example | Description | Command |
|---------|-------------|---------|
| **[Quick Test](oxidiviner/examples/quick_test.rs)** | Basic API functionality test | `cargo run --example quick_test` |
| **[Enhanced API Demo](oxidiviner/examples/enhanced_api_demo.rs)** | All API levels demonstrated | `cargo run --example enhanced_api_demo` |
| **[Exponential Smoothing](oxidiviner/examples/exponential_smoothing_example.rs)** | SES, Holt, Holt-Winters models | `cargo run --example exponential_smoothing_example` |
| **[Moving Average](oxidiviner/examples/moving_average_example.rs)** | MA models with window tuning | `cargo run --example moving_average_example` |
| **[ARIMA Models](oxidiviner/examples/arima_example.rs)** | ARIMA forecasting & evaluation | `cargo run --example arima_example` |
| **[AR Models](oxidiviner/examples/ar_example.rs)** | AutoRegressive model comparison | `cargo run --example ar_example` |
| **[GARCH Models](oxidiviner/examples/garch_example.rs)** | Volatility modeling & risk analysis | `cargo run --example garch_example` |

### **🏗️ Comprehensive Examples**
*Run from examples directory with `cd examples && cargo run --bin <name>`*

#### **📊 Getting Started & API Demos**
| Example | Description | Command |
|---------|-------------|---------|
| **[Quick Start Improved](examples/quick_start_improved.rs)** | Complete API showcase | `cargo run --bin quick_start_improved` |
| **[API Improvements Demo](examples/api_improvements_demo.rs)** | Enhanced API features | `cargo run --bin api_improvements_demo` |
| **[Simple API Demo](examples/simple_api_demo.rs)** | Basic usage patterns | `cargo run --bin simple_api_demo` |
| **[Standard Interface Demo](examples/standard_interface_demo.rs)** | Traditional API usage | `cargo run --bin standard_interface_demo` |

#### **📈 Financial Data & OHLCV**
| Example | Description | Command |
|---------|-------------|---------|
| **[OHLCV Forecasting](examples/ohlcv_forecasting_example.rs)** | Stock price forecasting | `cargo run --bin ohlcv_forecasting_example` |
| **[OHLCV Data Processor](examples/ohlcv-handling/data_processor.rs)** | Financial data handling | `cargo run --bin ohlcv_data_processor` |

#### **📊 Exponential Smoothing Models**
| Example | Description | Command |
|---------|-------------|---------|
| **[SES Demo](examples/exponential-smoothing-models/ses_demo.rs)** | Simple Exponential Smoothing | `cargo run --bin ses_demo` |
| **[SES Model Example](examples/exponential-smoothing-models/ses_model_example.rs)** | SES implementation details | `cargo run --bin ses_model_example` |
| **[SES Parameter Tuning](examples/exponential-smoothing-models/ses_parameter_tuning.rs)** | Alpha parameter optimization | `cargo run --bin ses_parameter_tuning` |
| **[Holt Demo](examples/exponential-smoothing-models/holt_demo.rs)** | Holt Linear Trend model | `cargo run --bin holt_demo` |
| **[Holt-Winters Demo](examples/exponential-smoothing-models/holt_winters_demo.rs)** | Seasonal forecasting | `cargo run --bin holt_winters_demo` |
| **[ETS Demo](examples/exponential-smoothing-models/ets_demo.rs)** | Error-Trend-Seasonal models | `cargo run --bin ets_demo` |
| **[ETS Model Complete](examples/exponential-smoothing-models/ets_model_complete.rs)** | Comprehensive ETS example | `cargo run --bin ets_model_complete` |
| **[ETS Model Demo](examples/exponential-smoothing-models/ets_model_demo.rs)** | Basic ETS functionality | `cargo run --bin ets_model_demo` |
| **[ES Models Comparison](examples/exponential-smoothing-models/es_models_comparison.rs)** | Trading strategy guide | `cargo run --bin es_models_comparison` |
| **[ES Parameter Tuning](examples/exponential-smoothing-models/es_parameter_tuning.rs)** | Parameter optimization | `cargo run --bin es_parameter_tuning` |

#### **🔄 AutoRegressive Models**
| Example | Description | Command |
|---------|-------------|---------|
| **[AutoRegressive Demo](examples/autoregressive-models/autoregressive_demo.rs)** | AR, ARIMA, SARIMA, VAR | `cargo run --bin autoregressive_demo` |
| **[Moving Average Demo](examples/autoregressive-models/ma_demo.rs)** | MA model implementation | `cargo run --bin ma_demo` |

#### **📉 GARCH & Volatility Models**
| Example | Description | Command |
|---------|-------------|---------|
| **[Basic GARCH Example](examples/garch-models/basic_garch_example.rs)** | GARCH, GJR-GARCH, EGARCH | `cargo run --bin basic_garch_example` |
| **[Stock Volatility Analysis](examples/garch-models/stock_volatility_analysis.rs)** | Risk management & VaR | `cargo run --bin stock_volatility_analysis` |

#### **🛠️ Development & Utilities**
| Example | Description | Command |
|---------|-------------|---------|
| **[Simple Demo](examples/simple_demo.rs)** | Minimal working example | `cargo run --bin simple_demo` |
| **[Basic Working Demo](examples/basic_working_demo.rs)** | Core functionality demo | `cargo run --bin basic_working_demo` |
| **[Simple Forecast Script](examples/scripts/simple_forecast.rs)** | Standalone forecasting | `cargo run --bin simple_forecast` |

### **📊 Example Categories Summary**
- **🎯 Quick Start**: 7 main library examples + 4 API demos  
- **💰 Financial**: 2 OHLCV and financial data examples
- **📈 Exponential Smoothing**: 10 ES model variants and comparisons
- **🔄 AutoRegressive**: 2 AR/ARIMA/SARIMA examples  
- **📉 GARCH**: 2 volatility and risk modeling examples
- **🛠️ Utilities**: 3 development and utility examples

**Total: 30+ comprehensive examples covering every forecasting scenario**

## 🧪 **Testing**

OxiDiviner has extensive test coverage:

```bash
# Run all tests
cargo test

# Run specific test suites
cargo test --test integration_tests
cargo test --test prelude_tests

# Run with coverage
cargo test --all-features

# Test examples
cargo test --examples
```

**Current Status**: 140+ tests passing with comprehensive coverage across all modules.

## 🏗️ **Architecture**

```
oxidiviner/
├── src/
│   ├── core/           # Core data structures & validation
│   │   ├── data.rs     # TimeSeriesData, OHLCVData
│   │   ├── validation.rs # ValidationUtils, accuracy metrics
│   │   └── mod.rs      # ModelOutput, traits, builders
│   ├── models/         # Forecasting implementations
│   │   ├── autoregressive/    # ARIMA, AR, ARMA, VAR
│   │   ├── exponential_smoothing/ # SES, Holt, Holt-Winters, ETS
│   │   ├── moving_average/    # Moving average models
│   │   └── garch/            # GARCH family models
│   ├── math/           # Mathematical functions
│   ├── api.rs          # Builder patterns & model wrappers  
│   ├── quick.rs        # One-line convenience functions
│   ├── batch.rs        # Multi-series processing
│   └── lib.rs          # Public API exports
└── examples/           # Comprehensive usage examples
```

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/rustic-ml/OxiDiviner.git
cd OxiDiviner
cargo build
cargo test
cargo run --example quick_start_improved
```

## 📋 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 **Research & References**

OxiDiviner implements well-established forecasting algorithms based on:

- Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*
- Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*
- Bollerslev, T. (1986). *Generalized autoregressive conditional heteroskedasticity*
- Brown, R. G. (1963). *Smoothing, Forecasting and Prediction of Discrete Time Series*

## 📞 **Support & Community**

- **📖 Documentation**: [docs.rs/oxidiviner](https://docs.rs/oxidiviner)
- **💡 Examples**: [examples/](examples/) directory
- **🐛 Issues**: [GitHub Issues](https://github.com/rustic-ml/OxiDiviner/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/rustic-ml/OxiDiviner/discussions)

---

**Built with ❤️ in Rust** 🦀 | **Predicting the Future, One Timestamp at a Time** ⏰ 