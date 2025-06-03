# 🔮 OxiDiviner

![OxiDiviner Logo](https://raw.githubusercontent.com/rustic-ml/OxiDiviner/main/OxiDiviner_250px.JPG)

[![Crates.io](https://img.shields.io/crates/v/oxidiviner.svg)](https://crates.io/crates/oxidiviner)
[![Documentation](https://docs.rs/oxidiviner/badge.svg)](https://docs.rs/oxidiviner)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rustic-ml/OxiDiviner/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-rustic--ml/OxiDiviner-8da0cb?logo=github)](https://github.com/rustic-ml/OxiDiviner)
[![X (Twitter)](https://img.shields.io/badge/follow-@CelsisDurham-1DA1F2?logo=x&logoColor=white)](https://x.com/CelsisDurham)

> **The Complete Rust Oracle for Time Series Forecasting** 🦀

OxiDiviner is a **comprehensive, production-ready time series analysis and forecasting library** built in Rust. Named after the fusion of **"Oxi"** (from oxidation/Rust) and **"Diviner"** (one who foresees the future), it empowers developers to predict time series patterns with the safety, speed, and elegance of Rust.

Whether you're forecasting financial markets, predicting business metrics, or analyzing sensor data, OxiDiviner provides **institutional-grade tools** with multiple API layers designed for different expertise levels - from quick one-liners to advanced statistical modeling.

## ✨ **Why OxiDiviner?**

- **🚀 Performance**: Rust's zero-cost abstractions for high-speed forecasting
- **🛡️ Safety**: Memory-safe implementations with comprehensive error handling  
- **🎯 Accuracy**: Battle-tested statistical models with rigorous validation
- **🔄 Flexibility**: Multiple API layers from simple to sophisticated
- **📊 Complete**: **100% feature-complete** toolkit from data prep to model evaluation
- **🧪 Production-Ready**: Extensively tested with **240+ passing tests**
- **✨ Institutional Grade**: Professional quantitative finance capabilities
- **🌟 Advanced**: State-of-the-art regime-switching and multivariate models

## 🎯 **Core Features**

### **📊 Traditional Time Series Models**
- **ARIMA** (AutoRegressive Integrated Moving Average) with seasonal support
- **Moving Averages** with adaptive window optimization
- **Exponential Smoothing** (Simple, Holt's Linear, Holt-Winters)  
- **AutoRegressive** models (AR, ARMA, VAR, SARIMA)
- **GARCH** models for volatility and risk forecasting
- **ETS** (Error, Trend, Seasonal) state space models

### **🚀 Advanced Forecasting Models**
- **Kalman Filters** - State-space models for dynamic forecasting with hidden states
  - Local Level Model (random walk with noise)
  - Local Linear Trend Model (level + trend dynamics)
  - Seasonal Model (level + trend + seasonal components)
- **Markov Regime-Switching** - Capture different market states and behavioral regimes
  - **Univariate Models**: Two-regime, three-regime, and N-regime models
  - **✨ NEW: Multivariate Models**: Cross-asset regime detection with portfolio analysis
  - **✨ NEW: Higher-Order Models**: Second and third-order Markov dependencies
  - **✨ NEW: Duration-Dependent Models**: Regime persistence effects modeling
  - **✨ NEW: Regime-Switching AR**: Autoregressive dynamics with regime changes
- **Vector Error Correction Models (VECM)** - Cointegration-based forecasting
  - Long-run equilibrium relationships
  - Error correction mechanisms
  - Multi-variate forecasting with cointegrated series
- **Threshold Autoregressive (TAR)** - Non-linear regime-dependent models
  - Threshold detection and regime switching
  - Regime-dependent autoregressive dynamics
  - Momentum vs. mean-reversion modeling
- **STL Decomposition** - Seasonal-Trend decomposition using Loess
  - Trend, seasonal, and remainder component extraction
  - Seasonal strength and trend strength metrics
  - Component-based forecasting with explicit seasonality
- **Copula Models** - Dependency structure modeling

### **💼 Financial Models**
- **Jump-Diffusion Models** - Capture sudden market shocks and crashes
  - **Merton Jump-Diffusion**: Industry standard with Gaussian jumps for crash modeling
  - **Kou Jump-Diffusion**: Asymmetric double-exponential jumps for realistic tail modeling
- **Stochastic Volatility Models** - Advanced volatility modeling for derivatives pricing
  - **Heston Model**: Gold standard with square-root volatility and mean reversion
  - **SABR Model**: Industry standard for FX and rates volatility surface modeling
- **Risk Management** - Comprehensive risk assessment capabilities
  - Monte Carlo Value-at-Risk (VaR) calculations with realistic distributions
  - Options pricing with stochastic volatility and jump risk
  - Volatility surface generation and smile/skew analysis
  - Stress testing and scenario analysis for portfolio management

### **✨ NEW: Enhanced Regime-Switching Capabilities**
- **Multivariate Regime Detection**: Cross-asset regime analysis with correlation switching
- **Portfolio Regime Analysis**: Risk metrics, diversification ratios, and asset allocation
- **Higher-Order Dependencies**: Complex temporal pattern recognition beyond first-order Markov
- **Duration-Dependent Models**: Regime persistence effects and "regime fatigue" modeling
- **Advanced Demonstrations**: Comprehensive examples with 6+ analysis scenarios

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
- **Model Diagnostics**: Innovation analysis, Ljung-Box testing, confidence intervals

## 🚀 **Quick Start**

Add OxiDiviner to your `Cargo.toml`:

```toml
[dependencies]
oxidiviner = "0.4.4"
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

### **🚀 Advanced Models Examples**

#### **State-Space Models (Kalman Filters)**
```rust
use oxidiviner::models::state_space::kalman_filter::KalmanFilter;

// Local level model (random walk + noise)
let mut kalman = KalmanFilter::local_level(1.0, 0.5)?;
kalman.fit(&data)?;
let forecasts = kalman.forecast(10)?;

// With confidence intervals
let (point_forecasts, lower_bounds, upper_bounds) = 
    kalman.forecast_with_intervals(10, 0.95)?;

// Access state estimates
if let Some(state) = kalman.get_state() {
    println!("Current level estimate: {:.3}", state[0]);
}
```

#### **Regime-Switching Models**
```rust
use oxidiviner::models::regime_switching::markov_switching::MarkovSwitchingModel;

// Two-regime model for bull/bear markets
let mut markov = MarkovSwitchingModel::two_regime(Some(1000), Some(1e-6));
markov.fit(&market_data)?;

// Classify current market regime
let (current_regime, probability) = markov.classify_current_regime()?;
println!("Current regime: {} (prob: {:.3})", current_regime, probability);

// Get regime parameters
if let Some((means, std_devs)) = markov.get_regime_parameters() {
    for (i, (mean, std)) in means.iter().zip(std_devs.iter()).enumerate() {
        println!("Regime {}: μ={:.3}, σ={:.3}", i, mean, std);
    }
}
```

#### **Cointegration Models (VECM)**
```rust
use oxidiviner::models::cointegration::vecm::VECMModel;

// Vector Error Correction Model for pairs trading
let mut vecm = VECMModel::new(1, 2, true, false)?; // 1 cointegrating relation, lag=2
vecm.fit_multiple(&[series1, series2])?;
let forecasts = vecm.forecast_multiple(10)?;

// Check cointegrating relationships
if let Some(coint_vectors) = vecm.get_cointegrating_vectors() {
    println!("Cointegrating vector: {:?}", coint_vectors[0]);
}

// Error correction terms
if let Some(ect) = vecm.get_error_correction_terms() {
    let latest_error = ect[0].last().unwrap();
    println!("Current deviation from equilibrium: {:.3}", latest_error);
}
```

#### **Non-linear Models (TAR)**
```rust
use oxidiviner::models::nonlinear::tar::TARModel;

// Threshold Autoregressive model
let mut tar = TARModel::new(vec![2, 3], 1)?; // AR(2) and AR(3) regimes, delay=1
tar.fit(&data)?;

// Get threshold and regime analysis
if let Some(threshold) = tar.get_threshold() {
    println!("Estimated threshold: {:.3}", threshold);
}

if let Some(regime_seq) = tar.get_regime_sequence() {
    let regime_0_pct = regime_seq.iter().filter(|&&x| x == 0).count() as f64 
                      / regime_seq.len() as f64 * 100.0;
    println!("Time in regime 0: {:.1}%", regime_0_pct);
}
```

#### **Decomposition Models (STL)**
```rust
use oxidiviner::models::decomposition::stl::STLModel;

// STL decomposition for seasonal data
let mut stl = STLModel::new(12, Some(7), Some(21))?; // Monthly seasonality
stl.fit(&seasonal_data)?;

// Get decomposed components
if let Some((trend, seasonal, remainder)) = stl.get_components() {
    println!("Latest trend: {:.3}", trend.last().unwrap());
    println!("Latest seasonal: {:.3}", seasonal.last().unwrap());
}

// Measure seasonal and trend strength
let seasonal_strength = stl.seasonal_strength()?;
let trend_strength = stl.trend_strength()?;
println!("Seasonal strength: {:.3}", seasonal_strength);
println!("Trend strength: {:.3}", trend_strength);

// Forecast with decomposed components
let forecasts = stl.forecast(12)?; // Next 12 periods
```

#### **✨ NEW: Enhanced Regime-Switching Models**
```rust
use oxidiviner::models::regime_switching::{
    MultivariateMarkovSwitchingModel, HigherOrderMarkovModel, 
    DurationDependentMarkovModel, RegimeSwitchingARModel
};

// Multivariate regime detection for portfolio analysis
let mut mv_model = MultivariateMarkovSwitchingModel::portfolio_three_regime(
    vec!["stocks".to_string(), "bonds".to_string(), "commodities".to_string()],
    Some(100),
    Some(1e-4),
)?;

// Fit to multiple correlated time series
let mut data_map = HashMap::new();
data_map.insert("stocks".to_string(), stock_data);
data_map.insert("bonds".to_string(), bond_data);
data_map.insert("commodities".to_string(), commodity_data);

mv_model.fit_multiple(&data_map)?;

// Portfolio regime analysis with risk metrics
let weights = vec![0.6, 0.3, 0.1]; // 60% stocks, 30% bonds, 10% commodities
let analysis = mv_model.portfolio_regime_analysis(&weights)?;

println!("Current Regime: {} ({:.1}% confidence)", 
    analysis.current_regime, analysis.regime_probability * 100.0);
println!("Portfolio Volatility: {:.1}%", analysis.portfolio_volatility * 100.0);
println!("Diversification Ratio: {:.2}", analysis.diversification_ratio);

// Cross-asset correlation analysis
let correlations = mv_model.regime_correlation_analysis()?;
for regime_corr in &correlations {
    println!("Regime {} correlations:", regime_corr.regime);
    let avg_correlation = calculate_average_correlation(&regime_corr.correlations);
    println!("  Average correlation: {:.3}", avg_correlation);
}

// Higher-order Markov model for complex dependencies
let mut higher_order = HigherOrderMarkovModel::second_order(2, Some(100), Some(1e-4))?;
higher_order.fit(&data)?;

// Analyze regime persistence patterns
let persistence_stats = higher_order.analyze_regime_persistence()?;
for (regime, duration) in persistence_stats {
    println!("Regime {} average duration: {:.1} periods", regime, duration);
}

// Duration-dependent regime model
let duration_model = DurationDependentMarkovModel::new(2, 20, Some(50), Some(1e-3))?;
let expected_durations = duration_model.expected_durations()?;
for (regime, duration) in expected_durations.iter().enumerate() {
    println!("Regime {} expected duration: {:.1} periods", regime, duration);
}

// Regime-switching autoregressive model
let ar_orders = vec![2, 3]; // AR(2) in regime 0, AR(3) in regime 1
let ar_model = RegimeSwitchingARModel::new(2, ar_orders, Some(50), Some(1e-3))?;
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

#### **📊 Traditional Models**
| Model | Constructor | Key Parameters | Use Case |
|-------|-------------|----------------|-----------|
| **SimpleESModel** | `new(alpha)` | α ∈ (0,1) | Stable series, no trend |
| **HoltLinearModel** | `new(alpha, beta)` | α,β ∈ (0,1) | Linear trend |
| **HoltWintersModel** | `new(alpha, beta, gamma, period)` | α,β,γ ∈ (0,1), period > 1 | Seasonal patterns |
| **ARIMAModel** | `new(p, d, q, intercept)` | p,d,q ≥ 0 | Complex temporal dependencies |
| **MAModel** | `new(window)` | window > 0 | Smoothing, noise reduction |
| **GARCHModel** | `new(p, q, mean_model)` | p,q ≥ 1 | Volatility modeling |

#### **🚀 Advanced Models**
| Model | Constructor | Key Parameters | Use Case |
|-------|-------------|----------------|-----------|
| **KalmanFilter** | `local_level(proc_var, obs_var)` | variances > 0 | State-space filtering |
| **KalmanFilter** | `local_linear_trend(level_var, trend_var, obs_var)` | variances > 0 | Dynamic trend tracking |
| **KalmanFilter** | `seasonal_model(level_var, trend_var, seasonal_var, obs_var, period)` | variances > 0, period ≥ 2 | Seasonal state-space |
| **MarkovSwitchingModel** | `two_regime(max_iter, tolerance)` | max_iter > 0, tolerance > 0 | Bull/bear markets |
| **MarkovSwitchingModel** | `three_regime(max_iter, tolerance)` | max_iter > 0, tolerance > 0 | Bear/neutral/bull |
| **VECMModel** | `new(coint_relations, lag_order, constant, trend)` | relations ≥ 1, lag ≥ 1 | Cointegrated series |
| **TARModel** | `new(ar_orders, delay)` | orders ≥ 1, delay ≥ 1 | Non-linear dynamics |
| **STLModel** | `new(period, seasonal_smoother, trend_smoother)` | period ≥ 2, smoothers odd | Seasonal decomposition |

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

## 💼 **Financial Models**

OxiDiviner provides industry-standard financial models for quantitative finance applications:

### **Jump-Diffusion Models**

#### **Merton Jump-Diffusion Model**
The industry standard for modeling sudden market crashes and extraordinary events.

```rust
use oxidiviner::models::financial::MertonJumpDiffusionModel;

// Create model with typical equity parameters
let model = MertonJumpDiffusionModel::new(
    100.0,   // S₀: Initial stock price
    0.05,    // μ: Drift rate (5%)
    0.20,    // σ: Diffusion volatility (20%)
    0.10,    // λ: Jump intensity (10% annual probability)
    -0.05,   // μⱼ: Average jump size (-5% crash)
    0.15,    // σⱼ: Jump volatility (15%)
    1.0/252.0 // Δt: Daily time step
)?;

// Monte Carlo simulation for 1 year (252 trading days)
let paths = model.simulate_paths(252, 5000, Some(42))?;
println!("Simulated {} price paths", paths.len());

// Calculate Value-at-Risk with jump risk
let var_95 = model.calculate_var(1_000_000.0, 0.95, 1.0/252.0, 10000)?;
println!("Daily 95% VaR: ${:.0}", var_95);

// Price European options with jump risk
let call_price = model.option_price(100.0, 105.0, 0.25, 0.98, true)?;
println!("Call option price: ${:.2}", call_price);
```

#### **Kou Jump-Diffusion Model**
Advanced asymmetric jump model with realistic upward/downward jump distributions.

```rust
use oxidiviner::models::financial::KouJumpDiffusionModel;

// Create model with asymmetric jumps
let model = KouJumpDiffusionModel::new(
    100.0,   // S₀: Initial price
    0.08,    // μ: Drift rate
    0.25,    // σ: Diffusion volatility
    0.15,    // λ: Jump intensity
    0.6,     // p: Upward jump probability
    20.0,    // η₁: Upward jump decay (smaller = larger jumps)
    10.0,    // η₂: Downward jump decay
    1.0/252.0 // Δt: Time step
)?;

// Analyze asymmetric jump events
let simulation = model.simulate_with_jumps(252, Some(42))?;
println!("Detected {} jumps over 1 year", simulation.jump_events.len());

// Compare with symmetric Merton model
let kou_var = model.calculate_var(1_000_000.0, 0.99, 1.0/252.0, 10000)?;
println!("Kou 99% VaR: ${:.0} (captures tail asymmetry)", kou_var);
```

### **Stochastic Volatility Models**

#### **Heston Stochastic Volatility Model**
The gold standard for volatility modeling with mean reversion and correlation effects.

```rust
use oxidiviner::models::financial::HestonStochasticVolatilityModel;

// Create model with typical equity parameters
let model = HestonStochasticVolatilityModel::new(
    100.0,   // S₀: Initial stock price
    0.04,    // V₀: Initial variance (20% vol)
    0.05,    // μ: Risk-neutral drift
    2.0,     // κ: Mean reversion speed
    0.04,    // θ: Long-run variance (20% long-run vol)
    0.3,     // σᵥ: Volatility of volatility
    -0.7,    // ρ: Correlation (leverage effect)
    1.0/252.0 // Δt: Time step
)?;

// Check Feller condition for variance process stability
if model.feller_condition() {
    println!("✓ Feller condition satisfied: variance stays positive");
}

// Simulate correlated price and volatility paths
let paths = model.simulate_paths(252, 3000, Some(42))?;
let final_vols: Vec<f64> = paths.iter().map(|p| p.volatilities.last().unwrap().sqrt()).collect();
let avg_final_vol = final_vols.iter().sum::<f64>() / final_vols.len() as f64;
println!("Average 1Y volatility: {:.1}%", avg_final_vol * 100.0);

// Generate volatility surface for options trading
let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
let expiries = vec![0.25, 0.5, 1.0]; // 3M, 6M, 1Y
let surface = model.volatility_surface(100.0, &strikes, &expiries)?;

println!("Heston Volatility Surface:");
for point in surface.iter().take(5) {
    println!("K={:.0}, T={:.2}: {:.1}% vol", 
             point.strike, point.expiry, point.implied_volatility * 100.0);
}
```

#### **SABR Volatility Model**
Industry standard for FX and interest rate derivatives pricing.

```rust
use oxidiviner::models::financial::SABRVolatilityModel;

// Create FX model (EUR/USD)
let fx_model = SABRVolatilityModel::new(
    1.20,    // F₀: Forward rate (EUR/USD)
    0.10,    // σ₀: Initial volatility (10%)
    0.30,    // α: Vol of vol (30%)
    0.5,     // β: Backbone parameter (square-root model)
    -0.3,    // ρ: Correlation (-30%)
    1.0/252.0 // Δt: Time step
)?;

println!("Model type: {}", fx_model.get_model_type());

// SABR implied volatility using Hagan's approximation
let strikes = vec![1.15, 1.175, 1.20, 1.225, 1.25];
let expiry = 1.0; // 1 year

println!("SABR Implied Volatilities (1Y):");
for &strike in &strikes {
    let vol = fx_model.sabr_implied_volatility(1.20, strike, expiry)?;
    let moneyness = (strike / 1.20).ln() * 100.0;
    println!("K={:.3} (ln(K/F)={:+.1}%): {:.1}%", 
             strike, moneyness, vol * 100.0);
}

// European option pricing with SABR volatility
let call_price = fx_model.option_price(1.20, 1.25, 0.5, 0.97, true)?;
println!("6M EUR/USD call (K=1.25): {:.4}", call_price);

// Multi-market applications
let rates_model = SABRVolatilityModel::new(0.05, 0.30, 0.50, 0.2, -0.4, 1.0/252.0)?;
println!("Rates model type: {}", rates_model.get_model_type());
```

### **Risk Management Applications**

```rust
// Portfolio VaR with different models
let portfolio_value = 10_000_000.0; // $10M portfolio

// Compare VaR across models
let merton_var = merton_model.calculate_var(portfolio_value, 0.95, 1.0/252.0, 10000)?;
let kou_var = kou_model.calculate_var(portfolio_value, 0.95, 1.0/252.0, 10000)?;
let heston_var = heston_model.calculate_var(portfolio_value, 0.95, 1.0/252.0, 10000)?;

println!("Daily 95% VaR Comparison:");
println!("  Merton (symmetric jumps): ${:.0}", merton_var);
println!("  Kou (asymmetric jumps):   ${:.0}", kou_var);  
println!("  Heston (stoch volatility): ${:.0}", heston_var);

// Stress testing with multiple scenarios
let stress_scenarios = vec![1.0/252.0, 5.0/252.0, 21.0/252.0]; // 1D, 1W, 1M
for &horizon in &stress_scenarios {
    let stress_var = heston_model.calculate_var(portfolio_value, 0.99, horizon, 5000)?;
    let days = (horizon * 252.0).round() as i32;
    println!("{}D 99% VaR: ${:.0}", days, stress_var);
}
```

### **Financial Model Integration**

```rust
// Model comparison and selection
use oxidiviner::models::financial::*;

// Fit multiple models to the same dataset
let mut merton = MertonJumpDiffusionModel::new_equity_default()?;
let mut heston = HestonStochasticVolatilityModel::new_equity_default()?;

// Simplified fitting (in practice, use maximum likelihood estimation)
// let _ = merton.fit(&price_data)?;
// let _ = heston.fit(&price_data)?;

// Compare model characteristics
println!("Model Comparison:");
println!("  Merton: Captures sudden crashes with normal jumps");
println!("  Kou: Captures asymmetric jumps (realistic crash vs rally)");
println!("  Heston: Captures volatility clustering and leverage effects");
println!("  SABR: Industry standard for options smile modeling");
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
| **[Advanced Forecasting Models](oxidiviner/examples/advanced_forecasting_models.rs)** | State-space, regime-switching, cointegration, TAR, STL | `cargo run --example advanced_forecasting_models` |

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

#### **💼 Financial Models & Advanced Risk Management**
| Example | Description | Command |
|---------|-------------|---------|
| **[Heston Stochastic Volatility Demo](examples/heston_stochastic_volatility_demo.rs)** | Gold-standard volatility modeling with mean reversion | `cargo run --bin heston_stochastic_volatility_demo` |
| **[SABR Volatility Demo](examples/sabr_volatility_demo.rs)** | Industry-standard FX and rates volatility surface modeling | `cargo run --bin sabr_volatility_demo` |
| **[✨ Enhanced Regime-Switching Demo](examples/enhanced_regime_switching_demo.rs)** | **NEW**: Comprehensive multivariate and higher-order regime analysis | `cargo run --bin enhanced_regime_switching_demo` |
| **[Optimization Demo](examples/optimization_demo.rs)** | Advanced parameter optimization techniques | `cargo run --bin optimization_demo` |
| **[Advanced Diagnostics Demo](examples/advanced_diagnostics_demo.rs)** | Comprehensive model diagnostics and validation | `cargo run --bin advanced_diagnostics_demo` |

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
- **💼 Financial Models**: 5 advanced financial modeling examples (enhanced regime-switching + 2 working demos + 2 optimization/diagnostics)
- **🛠️ Utilities**: 3 development and utility examples

**Total: 35+ comprehensive examples covering every forecasting scenario including advanced regime-switching**

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

**Current Status**: 240+ tests passing with comprehensive coverage across all modules.

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
- **🐦 Follow on X**: [@CelsisDurham](https://x.com/CelsisDurham)

---

**Built with ❤️ in Rust** 🦀 | **Predicting the Future, One Timestamp at a Time** ⏰ 