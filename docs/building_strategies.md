# Building Strategies with OxiDiviner

OxiDiviner provides a comprehensive suite of time series forecasting models for building sophisticated trading and investment strategies. This guide outlines how to effectively use the various models and their combinations for different strategic objectives.

**Disclaimer**: The information provided here is for educational purposes only and should not be considered financial advice. All trading strategies involve risk, and past performance is not indicative of future results. Always conduct thorough backtesting and risk assessment before deploying any strategy with real capital.

## Core Strategy Components

### 1. Time Series Models
- **ARIMA Family**: For trend and mean-reversion strategies
  - `ARModel`: Basic autoregressive patterns
  - `ARMAModel`: Combined AR and MA effects
  - `ARIMAModel`: Non-stationary series
  - `SARIMAModel`: Seasonal patterns
  - `VARModel`: Multi-asset relationships

- **Exponential Smoothing**: For adaptive trend following
  - `SimpleESModel`: Basic trend following
  - `HoltLinearModel`: Enhanced trend capture
  - `HoltWintersModel`: Seasonal decomposition
  - `ETSModel`: State space formulation

- **GARCH Models**: For volatility-based strategies
  - `GARCHModel`: Basic volatility clustering
  - `EGARCHModel`: Asymmetric volatility
  - `GJRGARCHModel`: Leverage effects
  - `GARCHMModel`: Risk premium strategies

### 2. Advanced Financial Models
- **Jump Diffusion**: For tail risk strategies
  - `MertonJumpDiffusionModel`: Crash risk hedging
  - `KouJumpDiffusionModel`: Asymmetric jump trading

- **Stochastic Volatility**: For volatility trading
  - `HestonStochasticVolatilityModel`: Vol surface trading
  - `SABRVolatilityModel`: Rates and FX strategies

### 3. Regime Detection
- **Markov Switching**: For regime-based allocation
  - `MarkovSwitchingModel`: Basic regime trading
  - `MultivariateMarkovSwitchingModel`: Cross-asset regimes
  - `HigherOrderMarkovModel`: Complex regime patterns

### 4. Dependency Modeling
- **Copula Models**: For correlation trading
  - `GaussianCopulaModel`: Linear dependencies
  - `TCopulaModel`: Tail risk dependencies
  - `ArchimedeanCopulaModel`: Specialized structures

## Strategy Building Blocks

### 1. Signal Generation
- **Trend Signals**:
  ```rust
  // ARIMA trend detection
  let mut model = ARIMAModel::new(2, 1, 2, true)?;
  model.fit(&data)?;
  let forecast = model.forecast(10)?;
  let trend_signal = forecast.last() > data.last();
  ```

- **Volatility Signals**:
  ```rust
  // GARCH volatility threshold
  let mut model = GARCHModel::new(1, 1, None)?;
  model.fit(&returns)?;
  let vol_forecast = model.forecast(1)?;
  let high_vol = vol_forecast[0] > historical_vol * 1.5;
  ```

- **Regime Signals**:
  ```rust
  // Markov regime detection
  let mut model = MarkovSwitchingModel::new(2)?;
  model.fit(&data)?;
  let current_regime = model.most_likely_regime()?;
  ```

### 2. Portfolio Construction
- **Risk Parity**:
  ```rust
  // Using GARCH for dynamic allocation
  let mut allocations = vec![];
  for asset in assets {
      let vol = GARCHModel::new(1, 1, None)?.fit(&asset)?.forecast(1)?[0];
      allocations.push(1.0 / vol);
  }
  ```

- **Regime-Based Allocation**:
  ```rust
  // Multi-asset regime allocation
  let mut model = MultivariateMarkovSwitchingModel::new(2, asset_names)?;
  model.fit(&data)?;
  let regime = model.portfolio_regime_analysis()?;
  ```

### 3. Risk Management
- **Value at Risk**:
  ```rust
  // Jump diffusion VaR
  let mut model = MertonJumpDiffusionModel::new(0.05, 0.2, 2.0, -0.01, 0.05)?;
  model.fit(&data)?;
  let var_95 = model.calculate_var(portfolio_value, 0.95, 1.0/252.0)?;
  ```

- **Tail Risk Hedging**:
  ```rust
  // Copula-based tail risk
  let mut model = TCopulaModel::new(n_assets, 5.0)?;
  model.fit(&returns)?;
  let tail_dep = model.tail_dependence_coefficient(0, 1)?;
  ```

## Example Strategies

### 1. Adaptive Trend Following
```rust
use oxidiviner::prelude::*;

// Combine ETS and GARCH
let mut trend_model = ETSModel::new()?;
let mut vol_model = GARCHModel::new(1, 1, None)?;

// Fit models
trend_model.fit(&price_data)?;
vol_model.fit(&returns_data)?;

// Generate signals
let trend = trend_model.forecast(1)?[0];
let vol = vol_model.forecast(1)?[0];

// Position sizing
let position_size = if trend > current_price {
    1.0 / vol  // Long position, inverse vol sizing
} else {
    0.0  // Flat
};
```

### 2. Regime-Switching Portfolio
```rust
// Multi-asset regime model
let mut regime_model = MultivariateMarkovSwitchingModel::new(2, assets)?;
regime_model.fit(&returns)?;

// Get regime analysis
let analysis = regime_model.portfolio_regime_analysis()?;

// Regime-based allocation
let allocations = match analysis.current_regime {
    0 => vec![0.6, 0.3, 0.1],  // Risk-on regime
    1 => vec![0.2, 0.6, 0.2],  // Risk-off regime
    _ => vec![1.0/3.0; 3],     // Neutral
};
```

### 3. Options Volatility Trading
```rust
// SABR volatility surface
let mut model = SABRVolatilityModel::new(
    forward, 0.2, 0.4, 0.6, -0.3, 1.0/252.0
)?;

// Analyze vol surface
let surface_point = model.implied_volatility(strike, expiry)?;
let vol_signal = surface_point.sabr_implied_vol > surface_point.implied_volatility;
```

### 4. Cross-Asset Arbitrage
```rust
// Copula-based pairs trading
let mut copula = GaussianCopulaModel::new(2)?;
copula.fit(&pair_returns)?;

// Generate spread distribution
let scenarios = copula.simulate(1000)?;
let spread_zscore = (current_spread - mean) / std_dev;
```

## Implementation Best Practices

### 1. Model Selection
- Start with simpler models (AR, MA, ES) before complex ones
- Use information criteria (AIC, BIC) for model order selection
- Consider ensemble methods for robustness
- Validate using both in-sample and out-of-sample metrics

### 2. Risk Management
- Always implement position sizing
- Use stop-losses and take-profits
- Monitor portfolio-level risk
- Consider correlation risk
- Implement circuit breakers

### 3. Performance Monitoring
- Track Sharpe ratio, Sortino ratio, and maximum drawdown
- Monitor hit rate and profit factor
- Analyze regime-dependent performance
- Track transaction costs and slippage

### 4. Production Implementation
- Use proper error handling
- Implement logging and monitoring
- Plan for regular model retraining
- Set up alerts for abnormal conditions

## Strategy Validation Framework

### 1. Backtesting
```rust
use oxidiviner::core::validation::backtest;

let bt_results = backtest(&model, &data, 252, 10)?;
println!("Hit rate: {}%", bt_results.hit_rate * 100.0);
println!("Profit factor: {}", bt_results.profit_factor);
```

### 2. Cross-Validation
```rust
use oxidiviner::core::validation::cross_validate;

let cv_results = cross_validate(&model, &data, 5, 10)?;
println!("CV Sharpe: {}", cv_results.sharpe_ratio);
```

### 3. Stress Testing
```rust
// Regime-based stress testing
let stress_scenarios = vec![
    ("Bear Market", -0.3, 0.4),
    ("Bull Market", 0.2, 0.2),
    ("Crisis", -0.5, 0.6),
];

for (scenario, ret, vol) in stress_scenarios {
    let stress_pnl = strategy.simulate_scenario(ret, vol)?;
    println!("{}: ${:.2}", scenario, stress_pnl);
}
```

## Conclusion

Building successful strategies with OxiDiviner requires:
1. Understanding the strengths of each model type
2. Proper signal generation and validation
3. Robust risk management
4. Comprehensive testing and monitoring
5. Careful production implementation 