//! Financial time series analysis module
//!
//! This module provides specialized functionality for financial data analysis,
//! including price data handling, return calculations, and financial-specific
//! forecasting methods.

use crate::core::{OxiError, Result, TimeSeriesData};
use crate::models::autoregressive::ARIMAModel;
use crate::models::exponential_smoothing::SimpleESModel;
use crate::models::moving_average::MAModel;
use chrono::{DateTime, Utc};

/// Specialized wrapper for financial time series data
pub struct FinancialTimeSeries {
    /// The underlying time series data
    data: TimeSeriesData,
    /// Asset symbol or identifier
    symbol: String,
}

impl FinancialTimeSeries {
    /// Create a new financial time series from price data
    pub fn from_prices(
        timestamps: Vec<DateTime<Utc>>,
        prices: Vec<f64>,
        symbol: &str,
    ) -> Result<Self> {
        let data = TimeSeriesData::new(timestamps, prices, &format!("{}_prices", symbol))?;
        Ok(Self {
            data,
            symbol: symbol.to_string(),
        })
    }

    /// Get the symbol
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Get the underlying time series data
    pub fn data(&self) -> &TimeSeriesData {
        &self.data
    }

    /// Calculate simple returns (price[t] / price[t-1] - 1)
    pub fn simple_returns(&self) -> Result<Vec<f64>> {
        let prices = &self.data.values;
        if prices.len() < 2 {
            return Err(OxiError::ESInsufficientData {
                actual: prices.len(),
                expected: 2,
            });
        }

        let mut returns = Vec::with_capacity(prices.len() - 1);
        for i in 1..prices.len() {
            returns.push(prices[i] / prices[i - 1] - 1.0);
        }
        Ok(returns)
    }

    /// Calculate log returns (ln(price[t] / price[t-1]))
    pub fn log_returns(&self) -> Result<Vec<f64>> {
        let prices = &self.data.values;
        if prices.len() < 2 {
            return Err(OxiError::ESInsufficientData {
                actual: prices.len(),
                expected: 2,
            });
        }

        let mut returns = Vec::with_capacity(prices.len() - 1);
        for i in 1..prices.len() {
            returns.push((prices[i] / prices[i - 1]).ln());
        }
        Ok(returns)
    }

    /// Automatic forecasting with financial-appropriate defaults
    pub fn auto_forecast(&self, periods: usize) -> Result<(Vec<f64>, String)> {
        // Try models in order of preference for financial data
        let models = vec![
            ("ARIMA", self.try_arima_forecast(periods)),
            ("SimpleES", self.try_es_forecast(periods)),
            ("MA", self.try_ma_forecast(periods)),
        ];

        for (name, result) in models {
            if let Ok(forecast) = result {
                return Ok((forecast, name.to_string()));
            }
        }

        Err(OxiError::ModelError(
            "All financial models failed".to_string(),
        ))
    }

    /// Compare multiple models and return the best one
    pub fn compare_models(&self, periods: usize) -> Result<ModelComparison> {
        let mut results = Vec::new();

        // Try ARIMA
        if let Ok(forecast) = self.try_arima_forecast(periods) {
            results.push(ModelResult {
                name: "ARIMA".to_string(),
                forecast,
                error: 0.0, // Would need validation data to calculate properly
            });
        }

        // Try Exponential Smoothing
        if let Ok(forecast) = self.try_es_forecast(periods) {
            results.push(ModelResult {
                name: "SimpleES".to_string(),
                forecast,
                error: 0.0,
            });
        }

        // Try Moving Average
        if let Ok(forecast) = self.try_ma_forecast(periods) {
            results.push(ModelResult {
                name: "MA".to_string(),
                forecast,
                error: 0.0,
            });
        }

        if results.is_empty() {
            return Err(OxiError::ModelError("No models succeeded".to_string()));
        }

        Ok(ModelComparison { results })
    }

    // Private helper methods
    fn try_arima_forecast(&self, periods: usize) -> Result<Vec<f64>> {
        let mut model = ARIMAModel::new(1, 1, 1, true)?;
        model.fit(&self.data)?;
        model.forecast(periods)
    }

    fn try_es_forecast(&self, periods: usize) -> Result<Vec<f64>> {
        let mut model = SimpleESModel::new(0.3)?;
        model.fit(&self.data)?;
        model.forecast(periods)
    }

    fn try_ma_forecast(&self, periods: usize) -> Result<Vec<f64>> {
        let mut model = MAModel::new(5)?;
        model.fit(&self.data)?;
        model.forecast(periods)
    }
}

/// Result of a single model forecast
#[derive(Debug, Clone)]
pub struct ModelResult {
    /// Model name
    pub name: String,
    /// Forecast values
    pub forecast: Vec<f64>,
    /// Error metric (lower is better)
    pub error: f64,
}

/// Comparison of multiple models
#[derive(Debug, Clone)]
pub struct ModelComparison {
    /// Results from all models
    pub results: Vec<ModelResult>,
}

impl ModelComparison {
    /// Get the best model (lowest error)
    pub fn best(&self) -> Option<&ModelResult> {
        self.results.iter().min_by(|a, b| {
            a.error
                .partial_cmp(&b.error)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get all model names
    pub fn model_names(&self) -> Vec<&str> {
        self.results.iter().map(|r| r.name.as_str()).collect()
    }
}
