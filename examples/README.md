# OxiDiviner Examples

This directory contains **comprehensive examples** demonstrating how to use the OxiDiviner library for time series analysis and forecasting, including **advanced regime-switching models** and **institutional-grade financial modeling**.

## Running Examples

You can run any example using Cargo:

```bash
# From the project root directory
cargo run --example <example_name>

# Or from the examples directory
cargo run --bin <example_name>
```

## Examples Overview

### 🚀 **Enhanced Regime-Switching Models**

- **✨ enhanced_regime_switching_demo.rs** - **COMPREHENSIVE**: Multivariate regime detection and higher-order dependencies
  - Multivariate regime detection across multiple assets (stocks, bonds, commodities)
  - Portfolio regime analysis with risk metrics and correlation switching
  - Higher-order dependencies and complex temporal patterns
  - Duration-dependent models and regime persistence analysis
  - Cross-asset correlation regime analysis with crisis vs normal market detection
  - Model comparison and selection framework

### 🎯 **Core Examples**

- **quick_start.rs** - **START HERE**: Fastest way to get started with OxiDiviner
- **standard_interface_demo.rs** - Demonstrates the standardized forecasting interface
- **api_improvements_demo.rs** - Advanced API features and builder patterns
- **quick_start_improved.rs** - Enhanced quick start with validation and comparison

### 💼 **Financial Models**

- **heston_stochastic_volatility_demo.rs** - Gold-standard volatility modeling with mean reversion
- **sabr_volatility_demo.rs** - Industry-standard FX and rates volatility surface modeling

### 🔧 **Advanced Features**

- **optimization_demo.rs** - Advanced parameter optimization techniques (Bayesian, Genetic, Simulated Annealing)
- **advanced_diagnostics_demo.rs** - Comprehensive model diagnostics and validation
- **accuracy_measurement.rs** - Accuracy measurement and validation frameworks
- **accuracy_improvements.rs** - Ensemble methods and accuracy enhancement techniques

### 📊 **Data Handling**

- **ohlcv_forecasting_example.rs** - Working with OHLCV (Open-High-Low-Close-Volume) financial data

### 📁 **Specialized Examples**

#### Exponential Smoothing Models (`exponential-smoothing-models/`)
- **ses_demo.rs** - Simple Exponential Smoothing
- **holt_demo.rs** - Holt's Linear Trend model
- **holt_winters_demo.rs** - Holt-Winters seasonal model
- **ets_model_complete.rs** - Complete ETS model implementation
- **es_models_comparison.rs** - Comparison of different ES models

#### Autoregressive Models (`autoregressive-models/`)
- **autoregressive_demo.rs** - Comprehensive AR/ARIMA/SARIMA demonstrations
- **var_demo.rs** - Vector Autoregression models
- **vecm_demo.rs** - Vector Error Correction models

#### GARCH Models (`garch-models/`)
- **basic_garch_example.rs** - Standard GARCH volatility modeling
- **advanced_garch_demo.rs** - Advanced GARCH variants (GJR, EGARCH, GARCH-M)
- **stock_volatility_analysis.rs** - Real-world stock volatility analysis

#### OHLCV Data Handling (`ohlcv-handling/`)
- **data_processor.rs** - OHLCV data processing and transformation

## 🎯 **Recommended Learning Path**

1. **Start with**: `quick_start.rs` - Get familiar with basic concepts
2. **Learn APIs**: `api_improvements_demo.rs` - Understand builder patterns and advanced features
3. **Financial Models**: `heston_stochastic_volatility_demo.rs` - Professional financial modeling
4. **Advanced Features**: `enhanced_regime_switching_demo.rs` - State-of-the-art regime detection
5. **Optimization**: `optimization_demo.rs` - Parameter optimization techniques

## 🔧 **Example Categories**

- **🟢 Beginner**: quick_start.rs, standard_interface_demo.rs
- **🟡 Intermediate**: api_improvements_demo.rs, heston_stochastic_volatility_demo.rs
- **🔴 Advanced**: enhanced_regime_switching_demo.rs, optimization_demo.rs

All examples are self-contained and include comprehensive documentation and explanations. 