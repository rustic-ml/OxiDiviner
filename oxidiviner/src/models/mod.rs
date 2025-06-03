/*!
# OxiDiviner Models

This module contains all the time series forecasting models available in OxiDiviner.

## Available Model Categories

* **GARCH Models** - For volatility modeling and forecasting
* **Exponential Smoothing** - For trend and seasonal decomposition
* **Autoregressive Models** - For linear time series modeling
* **Moving Average Models** - For smoothing and trend analysis
* **State-Space Models** - For dynamic forecasting with hidden states
* **Regime-Switching Models** - For capturing different market states
* **Cointegration Models** - For long-run equilibrium relationships
* **Non-linear Models** - For threshold and regime-dependent behavior
* **Decomposition Models** - For seasonal-trend analysis and forecasting
* **Copula Models** - For dependency structure modeling
* **Financial Models** - For advanced quantitative finance applications

## Usage

```rust
use oxidiviner::models::garch::GARCHModel;
use oxidiviner::models::exponential_smoothing::SimpleExponentialSmoothing;
use oxidiviner::models::state_space::KalmanFilter;
use oxidiviner::models::regime_switching::MarkovSwitchingModel;
use oxidiviner::models::financial::MertonJumpDiffusionModel;

// Create and use models...
```
*/

// Re-export the unified error system
pub use crate::core::{OxiError, Result};

// Model modules
pub mod autoregressive;
pub mod exponential_smoothing;
pub mod garch;
pub mod moving_average;

// Advanced forecasting models
pub mod cointegration;
pub mod copula;
pub mod decomposition;
pub mod nonlinear;
pub mod regime_switching;
pub mod state_space;

// Advanced financial models
pub mod financial;
