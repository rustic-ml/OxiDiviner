# OxiDiviner Examples

This directory contains comprehensive examples demonstrating all the forecasting models and features available in OxiDiviner.

## 🚀 Quick Start

Run any example using:
```bash
cargo run --example <example_name>
```

## 📊 Available Examples

### Core Forecasting Models

#### 1. **ARIMA Models** (`arima_example.rs`)
```bash
cargo run --example arima_example
```
**Demonstrates:**
- ARIMA(p,d,q) model configuration and fitting
- Comparing different ARIMA orders
- Parameter estimation and model diagnostics
- Future forecasting and evaluation
- Integration with Quick API

**Best for:** Non-stationary time series with trends and autocorrelation

---

#### 2. **Moving Average Models** (`moving_average_example.rs`)
```bash
cargo run --example moving_average_example
```
**Demonstrates:**
- Simple Moving Average with different window sizes
- Performance comparison across noise levels
- Practical sales forecasting applications
- Window size optimization

**Best for:** Smoothing noisy data and baseline forecasting

---

#### 3. **Exponential Smoothing Models** (`exponential_smoothing_example.rs`)
```bash
cargo run --example exponential_smoothing_example
```
**Demonstrates:**
- Simple Exponential Smoothing (SES) for level-only data
- Holt's Linear Method for trending data
- Holt-Winters for seasonal data with trend
- Parameter tuning and model comparison
- Business forecasting applications

**Best for:** Business forecasting, demand planning, seasonal data

---

#### 4. **AutoRegressive Models** (`ar_example.rs`)
```bash
cargo run --example ar_example
```
**Demonstrates:**
- AR(p) model fitting and forecasting
- Order selection and comparison
- Coefficient interpretation
- Autocorrelation modeling

**Best for:** Data with linear autocorrelation patterns

---

#### 5. **GARCH Models** (`garch_example.rs`)
```bash
cargo run --example garch_example
```
**Demonstrates:**
- GARCH(p,q) volatility modeling
- Financial risk management applications
- Value at Risk (VaR) calculations
- Options pricing with GARCH volatility
- Stress testing scenarios
- Model diagnostics

**Best for:** Financial time series, volatility forecasting, risk management

---

### High-Level APIs

#### 6. **Enhanced API Demo** (`enhanced_api_demo.rs`)
```bash
cargo run --example enhanced_api_demo
```
**Demonstrates:**
- Unified API for multiple models
- Batch processing capabilities
- Builder pattern usage
- Model comparison and selection
- Financial data handling

---

#### 7. **Quick API Test** (`quick_test.rs`)
```bash
cargo run --example quick_test
```
**Demonstrates:**
- One-line forecasting functions
- Auto model selection
- Rapid prototyping capabilities

---

## 🎯 Example Categories

### By Use Case

| **Financial Analysis** | **Business Forecasting** | **Research & Development** |
|----------------------|--------------------------|---------------------------|
| `garch_example` | `exponential_smoothing_example` | `arima_example` |
| `enhanced_api_demo` | `moving_average_example` | `ar_example` |
| | `quick_test` | |

### By Complexity Level

| **Beginner** | **Intermediate** | **Advanced** |
|-------------|-----------------|-------------|
| `quick_test` | `moving_average_example` | `garch_example` |
| `moving_average_example` | `exponential_smoothing_example` | `enhanced_api_demo` |
| | `arima_example` | |
| | `ar_example` | |

### By Data Type

| **Stationary Data** | **Trending Data** | **Seasonal Data** | **Financial Returns** |
|-------------------|------------------|------------------|---------------------|
| `ar_example` | `arima_example` | `exponential_smoothing_example` | `garch_example` |
| `moving_average_example` | `exponential_smoothing_example` | | `enhanced_api_demo` |

## 🔧 Running Examples

### Prerequisites
```bash
# Ensure you're in the oxidiviner directory
cd oxidiviner

# Build the project (optional, examples will build automatically)
cargo build
```

### Run All Examples
```bash
# Test all examples to ensure they work
cargo run --example arima_example
cargo run --example moving_average_example
cargo run --example exponential_smoothing_example
cargo run --example ar_example
cargo run --example garch_example
cargo run --example enhanced_api_demo
cargo run --example quick_test
```

### Run with Release Optimization
```bash
cargo run --example <example_name> --release
```

## 📚 Learning Path

### 1. **Start Here: Quick API**
Begin with `quick_test` to understand the basic concepts and one-line forecasting.

### 2. **Learn Core Models**
Progress through:
1. `moving_average_example` - Understand smoothing
2. `ar_example` - Learn autocorrelation modeling
3. `arima_example` - Master comprehensive time series modeling
4. `exponential_smoothing_example` - Explore business forecasting

### 3. **Advanced Applications**
1. `garch_example` - Financial volatility modeling
2. `enhanced_api_demo` - Production-ready workflows

## 🎨 Customizing Examples

Each example is self-contained and can be modified to:

- **Change data generation parameters** - Modify noise levels, trends, seasonality
- **Adjust model parameters** - Experiment with different orders and configurations
- **Add new evaluation metrics** - Extend the analysis with custom metrics
- **Integrate real data** - Replace synthetic data with your own datasets

## 📊 Example Output

Each example provides:
- **Model performance metrics** (MAE, RMSE, MAPE)
- **Parameter estimates** and diagnostics
- **Forecasts** with confidence intervals where applicable
- **Comparative analysis** across different model configurations
- **Practical insights** and recommendations

## 🔍 Troubleshooting

### Common Issues

1. **Compilation Errors**
   ```bash
   cargo clean
   cargo build
   ```

2. **Missing Dependencies**
   - All dependencies are included in `Cargo.toml`
   - Run `cargo update` if needed

3. **Performance Issues**
   - Use `--release` flag for faster execution
   - Reduce data size in examples if needed

### Getting Help

- Check the main [README.md](../README.md) for general usage
- Review the [User Guide](../docs/user_guide.md) for detailed documentation
- Examine the source code of examples for implementation details

## 🚀 Next Steps

After exploring the examples:

1. **Read the User Guide** - `docs/user_guide.md`
2. **Check the API Documentation** - `cargo doc --open`
3. **Try with your own data** - Adapt examples to your use case
4. **Explore advanced features** - Batch processing, model ensembles, custom metrics

---

**Happy Forecasting! 📈** 