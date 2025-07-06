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

#### Holt-Winters Method

```rust
use oxidiviner::models::exponential_smoothing::HoltWintersModel;

let mut model = HoltWintersModel::new(0.3, 0.1, 0.1, 12)?; // alpha, beta, gamma, seasonal_period
model.fit(&data)?;
let forecast = model.forecast(24)?; // Forecast 2 seasonal periods ahead
```

#### ETS Model

```rust
use oxidiviner::models::exponential_smoothing::ETSModel;

let mut model = ETSModel::new()?;
model.fit(&data)?;
let forecast = model.forecast(10)?;
```

### GARCH Models

#### Basic GARCH

```rust
use oxidiviner::models::garch::GARCHModel;

let mut model = GARCHModel::new(1, 1, None)?; // p=1, q=1, no mean model
model.fit(&returns)?;
let volatility_forecast = model.forecast(10)?;
```

#### EGARCH

```rust
use oxidiviner::models::garch::EGARCHModel;

let mut model = EGARCHModel::new(1, 1, None)?;
model.fit(&returns)?;
let volatility_forecast = model.forecast(10)?;
```

## Financial Models

### Jump Diffusion

#### Merton Jump Diffusion

```rust
use oxidiviner::models::financial::MertonJumpDiffusionModel;

let mut model = MertonJumpDiffusionModel::new(
    0.05,  // drift
    0.2,   // diffusion volatility
    2.0,   // jump intensity
    -0.01, // jump mean
    0.05   // jump volatility
)?;
model.fit(&returns)?;
let forecast = model.forecast(10)?;
```

#### Kou Jump Diffusion

```rust
use oxidiviner::models::financial::KouJumpDiffusionModel;

let mut model = KouJumpDiffusionModel::new(
    0.05,  // drift
    0.2,   // diffusion volatility
    2.0,   // jump intensity
    0.3,   // up probability
    1.5,   // up magnitude
    2.0    // down magnitude
)?;
model.fit(&returns)?;
let forecast = model.forecast(10)?;
```

### Stochastic Volatility

#### Heston Model

```rust
use oxidiviner::models::financial::HestonStochasticVolatilityModel;

let mut model = HestonStochasticVolatilityModel::new(
    0.04,  // long-run variance
    2.0,   // mean reversion speed
    0.3,   // vol of vol
    -0.5,  // correlation
    0.04   // initial variance
)?;
model.fit(&returns)?;
let forecast = model.forecast(10)?;
```

#### SABR Model

```rust
use oxidiviner::models::financial::SABRVolatilityModel;

let mut model = SABRVolatilityModel::new(
    forward,
    0.2,   // alpha (initial vol)
    0.4,   // beta (CEV parameter)
    0.6,   // nu (vol of vol)
    -0.3,  // rho (correlation)
    1.0/252.0 // dt (time step)
)?;
model.fit(&returns)?;
let implied_vol = model.implied_volatility(strike, expiry)?;
```

## Regime Detection

### Markov Switching

```rust
use oxidiviner::models::regime_switching::MarkovSwitchingModel;

let mut model = MarkovSwitchingModel::new(2)?; // 2 regimes
model.fit(&returns)?;
let regime = model.most_likely_regime()?;
let regime_probs = model.regime_probabilities()?;
```

### Multivariate Markov Switching

```rust
use oxidiviner::models::regime_switching::MultivariateMarkovSwitchingModel;

let mut model = MultivariateMarkovSwitchingModel::new(2, asset_names)?;
model.fit(&returns)?;
let regimes = model.most_likely_regimes()?;
let transition_matrix = model.transition_matrix()?;
```

## Dependency Modeling

### Copulas

#### Gaussian Copula

```rust
use oxidiviner::models::copula::GaussianCopulaModel;

let mut model = GaussianCopulaModel::new(2)?; // 2 variables
model.fit(&returns)?;
let correlation = model.correlation_matrix()?;
let simulated = model.simulate(1000)?;
```

#### Student's t Copula

```rust
use oxidiviner::models::copula::TCopulaModel;

let mut model = TCopulaModel::new(2, 5.0)?; // 2 variables, 5 degrees of freedom
model.fit(&returns)?;
let tail_dependence = model.tail_dependence_coefficient(0, 1)?;
```

## State Space Models

### Kalman Filter

```rust
use oxidiviner::models::state_space::KalmanFilter;

let mut kf = KalmanFilter::new(
    state_dim,
    obs_dim,
    initial_state,
    initial_covariance,
    transition_matrix,
    observation_matrix,
    process_noise,
    observation_noise
)?;

// Update with new observation
kf.update(&observation)?;

// Get current state estimate
let state = kf.state_estimate()?;
let covariance = kf.state_covariance()?;
```

## Model Validation

### Cross Validation

```rust
use oxidiviner::validation::cross_validate;

let cv_results = cross_validate(&model, &data, 5, 10)?;
println!("CV MSE: {}", cv_results.mse);
println!("CV MAE: {}", cv_results.mae);
println!("CV MAPE: {}", cv_results.mape);
```

### Backtesting

```rust
use oxidiviner::validation::backtest;

let bt_results = backtest(&model, &data, 252, 10)?;
println!("Hit Rate: {}%", bt_results.hit_rate * 100.0);
println!("Profit Factor: {}", bt_results.profit_factor);
```

## Best Practices

1. **Data Preprocessing**
   - Always check for missing values
   - Consider scaling/normalization
   - Handle outliers appropriately
   - Check for stationarity

2. **Model Selection**
   - Start with simple models
   - Use information criteria (AIC, BIC)
   - Consider ensemble methods
   - Validate out-of-sample

3. **Performance Optimization**
   - Use appropriate batch sizes
   - Consider parallel processing
   - Monitor memory usage
   - Profile critical sections

4. **Production Deployment**
   - Implement proper error handling
   - Set up monitoring and logging
   - Plan for model updates
   - Consider scaling requirements

## Conclusion

OxiDiviner provides a comprehensive toolkit for time series forecasting in Rust. From simple one-line forecasts to complex multi-model ensemble approaches, the library scales with your needs.

For more examples and advanced usage patterns, check out the [examples directory](../examples/) and the [API documentation](https://docs.rs/oxidiviner).

---

**Happy forecasting! 📈** 