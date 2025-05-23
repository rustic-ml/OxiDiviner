/*!
# OxiDiviner Models

This module contains all the time series forecasting models available in OxiDiviner.

## Available Model Categories

* **GARCH Models** - For volatility modeling and forecasting
* **Exponential Smoothing** - For trend and seasonal decomposition
* **Autoregressive Models** - For linear time series modeling
* **Moving Average Models** - For smoothing and trend analysis

## Usage

```rust
use oxidiviner::models::garch::GARCHModel;
use oxidiviner::models::exponential_smoothing::SimpleExponentialSmoothing;

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
