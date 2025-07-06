# OxiDiviner User Guide

Welcome to the comprehensive user guide for OxiDiviner! This guide will walk you through all the features and capabilities of the library.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Core Concepts](#core-concepts)
3. [Data Structures](#data-structures)
4. [Quick API](#quick-api)
5. [Time Series Models](#time-series-models)
6. [Financial Models](#financial-models)
7. [Regime Detection](#regime-detection)
8. [Dependency Modeling](#dependency-modeling)
9. [State Space Models](#state-space-models)
10. [Model Validation](#model-validation)
11. [Best Practices](#best-practices)

## Installation and Setup

### Basic Installation

Add OxiDiviner to your `Cargo.toml`:

```toml
[dependencies]
oxidiviner = "1.0.0"
```

### Import Patterns

```rust
// Most common - imports all main types
use oxidiviner::prelude::*;

// Specific modules
use oxidiviner::{quick, api, batch};
use oxidiviner::models::autoregressive::ARIMAModel;

// Quick API only
use oxidiviner::quick;
```

## Core Concepts

### Forecaster Trait

All models in OxiDiviner implement the `Forecaster` trait:

```rust
pub trait Forecaster {
    fn name(&self) -> &str;
    fn fit(&mut self, data: &TimeSeriesData) -> Result<()>;
    fn forecast(&self, horizon: usize) -> Result<Vec<f64>>;
    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation>;
    fn predict(&self, horizon: usize, test_data: Option<&TimeSeriesData>) -> Result<ModelOutput>;
}
```

This ensures a consistent interface across all models.

### Error Handling

OxiDiviner uses comprehensive error types:

```rust
use oxidiviner::core::OxiError;

match result {
    Ok(forecast) => println!("Success: {:?}", forecast),
    Err(OxiError::InvalidParameter(msg)) => println!("Invalid parameter: {}", msg),
    Err(OxiError::ModelError(msg)) => println!("Model error: {}", msg),
    Err(OxiError::DataError(msg)) => println!("Data error: {}", msg),
    // ... other error types
}
```

## Data Structures

### TimeSeriesData

The core data structure for univariate time series:

```rust
use oxidiviner::prelude::*;
use chrono::{Duration, Utc};

// Create timestamps
let start = Utc::now();
let timestamps: Vec<_> = (0..100)
    .map(|i| start + Duration::days(i))
    .collect();

// Create values (example: linear trend with noise)
let values: Vec<f64> = (0..100)
    .map(|i| 10.0 + 0.5 * i as f64 + rand::random::<f64>() * 5.0)
    .collect();

// Create time series
let ts = TimeSeriesData::new(timestamps, values, "my_series")?;
```

### OHLCVData

For financial time series data:

```rust
use oxidiviner::prelude::*;

let ohlcv = OHLCVData {
    symbol: "AAPL".to_string(),
    timestamps: timestamps,
    open: open_prices,
    high: high_prices,
    low: low_prices,
    close: close_prices,
    volume: volumes,
    adjusted_close: Some(adjusted_closes), // Optional
};

// Convert to time series using close prices
let ts = ohlcv.to_time_series(false); // false = use close, true = use adjusted_close
```

## Quick API

The Quick API provides one-line forecasting functions:

### ARIMA Forecasting

```rust
use oxidiviner::quick;

// Default ARIMA(1,1,1)
let forecast = quick::arima(data.clone(), 10)?;

// Custom parameters ARIMA(2,1,2)
let forecast = quick::arima_with_config(data, 10, Some((2, 1, 2)))?;
```

### Moving Average

```rust
// Default window size (5)
let forecast = quick::moving_average(data.clone(), 10, None)?;

// Custom window size
let forecast = quick::moving_average(data, 10, Some(7))?;
```

### Auto Model Selection

```rust
// Automatically selects best model
let (forecast, model_name) = quick::auto_select(data, 10)?;
println!("Selected model: {}", model_name);
```

### Model Comparison

```rust
// Compare all available models
let comparisons = quick::compare_models(timestamps, values, 10)?;
for (model_name, forecast) in comparisons {
    println!("{}: {:?}", model_name, &forecast[..3]);
}
```

## Time Series Models

### ARIMA Family

#### Basic AR Model

```rust
use oxidiviner::models::autoregressive::ARModel;

// Create AR(p) model
let mut model = ARModel::new(2, true)?; // p=2, with_intercept=true
model.fit(&data)?;

let forecasts = model.forecast(10)?;
let evaluation = model.evaluate(&test_data)?;

// Access model coefficients
if let Some(ar_coef) = model.ar_coefficients() {
    println!("AR coefficients: {:?}", ar_coef);
}
```

#### ARMA Model

```rust
use oxidiviner::models::autoregressive::ARMAModel;

// Create ARMA(p,q) model
let mut model = ARMAModel::new(2, 1, true)?; // p=2, q=1, with_intercept=true
model.fit(&data)?;
let forecast = model.forecast(10)?;
```

#### ARIMA Model

```rust
use oxidiviner::models::autoregressive::ARIMAModel;

// Create ARIMA(p,d,q) model
let mut model = ARIMAModel::new(2, 1, 1, true)?; // p=2, d=1, q=1, with_intercept=true
model.fit(&data)?;
let forecast = model.forecast(10)?;
```

#### SARIMA Model

```rust
use oxidiviner::models::autoregressive::SARIMAModel;

// Create SARIMA(p,d,q)(P,D,Q)s model
let mut model = SARIMAModel::new(1, 1, 1, 1, 1, 1, 12, true)?; // Yearly seasonality
model.fit(&data)?;
let forecast = model.forecast(24)?; // Forecast 2 years ahead
```

#### VAR Model

```rust
use oxidiviner::models::autoregressive::VARModel;

// Create VAR model with 2 variables
let variable_names = vec!["gdp".to_string(), "inflation".to_string()];
let mut model = VARModel::new(2, variable_names, true)?; // lag=2, with_intercept=true
model.fit(&multivariate_data)?;
let forecast = model.forecast(10)?;
```

### Exponential Smoothing

#### Simple Exponential Smoothing

```rust
use oxidiviner::models::exponential_smoothing::SimpleESModel;

let mut model = SimpleESModel::new(0.3)?; // alpha = 0.3
model.fit(&data)?;
let forecast = model.forecast(10)?;
```

#### Holt's Linear Method

```rust
use oxidiviner::models::exponential_smoothing::HoltLinearModel;

let mut model = HoltLinearModel::new(0.3, 0.1)?; // alpha = 0.3, beta = 0.1
model.fit(&data)?;
let forecast = model.forecast(10)?;
```

#### Holt-Winters Seasonal

```rust
use oxidiviner::models::exponential_smoothing::HoltWintersModel;

// For seasonal data (e.g., monthly data with yearly seasonality)
let mut model = HoltWintersModel::new(0.3, 0.1, 0.1, 12)?; // period = 12
model.fit(&data)?;
let forecast = model.forecast(24)?; // Forecast 2 years
```

#### ETS Model

```rust
use oxidiviner::models::exponential_smoothing::ETSModel;

// Create ETS model with automatic component selection
let mut model = ETSModel::new()?;
model.fit(&data)?;
let forecast = model.forecast(10)?;

// Access components
println!("Error type: {:?}", model.error_type());
println!("Trend type: {:?}", model.trend_type());
println!("Seasonal type: {:?}", model.seasonal_type());
```

### GARCH Models

#### Basic GARCH

```rust
use oxidiviner::models::garch::GARCHModel;

// GARCH(1,1) model
let mut model = GARCHModel::new(1, 1, None)?;
model.fit(&returns_data)?; // Use returns, not prices

let volatility_forecast = model.forecast(10)?;
```

#### EGARCH

```rust
use oxidiviner::models::garch::EGARCHModel;

// EGARCH(1,1) for asymmetric volatility
let mut model = EGARCHModel::new(1, 1)?;
model.fit(&returns_data)?;
let forecast = model.forecast(10)?;
```

#### GJR-GARCH

```rust
use oxidiviner::models::garch::GJRGARCHModel;

// GJR-GARCH(1,1) for leverage effects
let mut model = GJRGARCHModel::new(1, 1)?;
model.fit(&returns_data)?;
let forecast = model.forecast(10)?;
```

#### GARCH-M

```rust
use oxidiviner::models::garch::{GARCHMModel, RiskPremiumType};

// GARCH-M with variance risk premium
let mut model = GARCHMModel::new(1, 1, RiskPremiumType::Variance)?;
model.fit(&returns_data)?;
let forecast = model.forecast(10)?;
```

## Financial Models

### Jump Diffusion Models

#### Merton Jump Diffusion

```rust
use oxidiviner::models::financial::MertonJumpDiffusionModel;

let mut model = MertonJumpDiffusionModel::new(
    0.05,  // drift
    0.2,   // volatility
    2.0,   // jump intensity
    -0.01, // jump mean
    0.05,  // jump volatility
)?;

model.fit(&price_data)?;
let forecast = model.forecast(10)?;

// Access jump events
if let Some(jumps) = model.detected_jumps() {
    for jump in jumps {
        println!("Jump at t={}: size={}", jump.time, jump.size);
    }
}
```

#### Kou Jump Diffusion

```rust
use oxidiviner::models::financial::KouJumpDiffusionModel;

let mut model = KouJumpDiffusionModel::new(
    0.05,  // drift
    0.2,   // volatility
    2.0,   // jump intensity
    0.3,   // prob of upward jump
    50.0,  // upward mean
    25.0,  // downward mean
)?;

model.fit(&price_data)?;
let forecast = model.forecast(10)?;
```

### Stochastic Volatility Models

#### Heston Model

```rust
use oxidiviner::models::financial::HestonStochasticVolatilityModel;

let mut model = HestonStochasticVolatilityModel::new(
    0.04,  // long-run variance
    2.0,   // mean reversion speed
    0.3,   // vol of vol
    -0.7,  // correlation
    0.04,  // initial variance
)?;

model.fit(&price_data)?;
let forecast = model.forecast(10)?;

// Generate sample paths
let paths = model.simulate_paths(100, 252)?; // 100 paths, 252 days each
```

#### SABR Model

```rust
use oxidiviner::models::financial::SABRVolatilityModel;

let mut model = SABRVolatilityModel::new(
    100.0, // initial forward
    0.2,   // initial volatility
    0.4,   // vol of vol
    0.6,   // beta (backbone parameter)
    -0.3,  // correlation
    1.0/252.0, // daily time step
)?;

model.fit(&forward_data)?;
let forecast = model.forecast(10)?;

// Get volatility surface points
let surface_point = model.implied_volatility(95.0, 0.5)?; // strike=95, T=0.5
```

## Regime Detection

### Basic Markov Switching

```rust
use oxidiviner::models::regime_switching::MarkovSwitchingModel;

// Two-regime model (e.g., bull/bear)
let mut model = MarkovSwitchingModel::new(2)?;
model.fit(&returns_data)?;

// Get regime probabilities
let probs = model.regime_probabilities()?;
let current_regime = model.most_likely_regime()?;

// Forecast with regime switching
let forecast = model.forecast(10)?;
```

### Multivariate Regime Switching

```rust
use oxidiviner::models::regime_switching::MultivariateMarkovSwitchingModel;

// Create model for multiple assets
let assets = vec!["SPY".to_string(), "TLT".to_string(), "GLD".to_string()];
let mut model = MultivariateMarkovSwitchingModel::new(2, assets)?;

// Fit model
model.fit(&multivariate_data)?;

// Get cross-asset regime analysis
let analysis = model.portfolio_regime_analysis()?;
println!("Current regime: {}", analysis.current_regime);
println!("Regime correlation matrix: {:?}", analysis.regime_correlations);
```

### Higher-Order Regime Switching

```rust
use oxidiviner::models::regime_switching::HigherOrderMarkovModel;

// Create second-order model
let mut model = HigherOrderMarkovModel::new(2, 2)?; // 2 regimes, order 2
model.fit(&data)?;

// Get transition probabilities conditional on past regimes
let prob = model.transition_probability(1, 0, 1)?; // P(S_t=1|S_{t-1}=0,S_{t-2}=1)
```

## Dependency Modeling

### Gaussian Copula

```rust
use oxidiviner::models::copula::GaussianCopulaModel;

// Create model for 3 assets
let mut model = GaussianCopulaModel::new(3)?;
model.fit(&multivariate_returns)?;

// Generate joint scenarios
let scenarios = model.simulate(1000)?; // 1000 scenarios
```

### Student's t-Copula

```rust
use oxidiviner::models::copula::TCopulaModel;

// Create t-copula with 5 degrees of freedom
let mut model = TCopulaModel::new(3, 5.0)?;
model.fit(&multivariate_returns)?;

// Get tail dependence coefficients
let tail_dep = model.tail_dependence_coefficient(0, 1)?;
```

### Archimedean Copula

```rust
use oxidiviner::models::copula::{ArchimedeanCopulaModel, ArchimedeanType};

// Create Clayton copula for lower tail dependence
let mut model = ArchimedeanCopulaModel::new(2, ArchimedeanType::Clayton)?;
model.fit(&bivariate_returns)?;

// Generate dependent uniform variates
let uniforms = model.generate_uniforms(1000)?;
```

## State Space Models

### Kalman Filter

```rust
use oxidiviner::models::state_space::KalmanFilter;

// Create local level model
let mut kf = KalmanFilter::new_local_level(0.1, 0.1)?;
kf.fit(&data)?;

// Get filtered states
let states = kf.filtered_states()?;
let state_covs = kf.filtered_state_covariances()?;

// Forecast with uncertainty
let (forecast, forecast_cov) = kf.forecast_with_covariance(10)?;
```

## Model Validation

### Cross-Validation

```rust
use oxidiviner::core::validation::cross_validate;

let cv_results = cross_validate(&model, &data, 5, 10)?; // 5-fold CV, horizon=10
println!("CV MAE: {}", cv_results.mae);
println!("CV RMSE: {}", cv_results.rmse);
```

### Backtesting

```rust
use oxidiviner::core::validation::backtest;

let bt_results = backtest(&model, &data, 252, 10)?; // 1-year window, horizon=10
println!("Hit rate: {}%", bt_results.hit_rate * 100.0);
println!("Directional accuracy: {}%", bt_results.directional_accuracy * 100.0);
```

### Diagnostics

```rust
use oxidiviner::core::diagnostics::ModelDiagnostics;

let diag = ModelDiagnostics::new(&model, &data)?;
println!("AIC: {}", diag.aic());
println!("BIC: {}", diag.bic());
println!("Log-likelihood: {}", diag.log_likelihood());
```

## Best Practices

### Model Selection

1. Start with simple models (AR, MA, ES) and gradually increase complexity
2. Use information criteria (AIC, BIC) for model order selection
3. Consider ensemble methods for improved robustness
4. Validate models using both in-sample and out-of-sample metrics

### Data Preprocessing

1. Check for and handle missing values
2. Ensure data stationarity when required
3. Scale/normalize data appropriately
4. Split data properly for training and testing

### Performance Optimization

1. Use appropriate batch sizes for large datasets
2. Leverage parallel processing when available
3. Monitor memory usage with large datasets
4. Cache intermediate results when beneficial

### Production Deployment

1. Implement proper error handling
2. Log model performance metrics
3. Set up monitoring for model drift
4. Plan for regular model retraining

For more detailed examples and advanced usage patterns, check the `examples/` directory in the repository. 