#![allow(clippy::needless_range_loop)]

use crate::models::ESError;
use crate::core::{Forecaster, ModelEvaluation, ModelOutput, OxiError, Result, TimeSeriesData};
use crate::math::metrics::{mae, mape, mse, rmse, smape};

/// Holt-Winters (Triple Exponential Smoothing) model for forecasting seasonal time series.
///
/// This model has level, trend, and seasonal components, making it suitable for forecasting
/// time series with both trend and seasonality. It's equivalent to ETS(A,A,A) in the ETS framework.
///
/// This implementation uses the additive form of the Holt-Winters model.
///
/// # Model Equations:
/// - Level: l_t = α * (y_t - s_{t-m}) + (1 - α) * (l_{t-1} + b_{t-1})
/// - Trend: b_t = β * (l_t - l_{t-1}) + (1 - β) * b_{t-1}
/// - Seasonal: s_t = γ * (y_t - l_t) + (1 - γ) * s_{t-m}
/// - Forecast: ŷ_{t+h|t} = l_t + h * b_t + s_{t-m+h mod m}
///
/// where:
/// - l_t is the level at time t
/// - b_t is the trend at time t
/// - s_t is the seasonal component at time t
/// - y_t is the observed value at time t
/// - m is the seasonal period
/// - α is the smoothing parameter for level (0 < α < 1)
/// - β is the smoothing parameter for trend (0 < β < 1)
/// - γ is the smoothing parameter for seasonal component (0 < γ < 1)
/// - ŷ_{t+h|t} is the h-step ahead forecast from time t
pub struct HoltWintersModel {
    /// Model name
    name: String,
    /// Smoothing parameter for level (0 < α < 1)
    alpha: f64,
    /// Smoothing parameter for trend (0 < β < 1)
    beta: f64,
    /// Smoothing parameter for seasonal component (0 < γ < 1)
    gamma: f64,
    /// Seasonal period (number of observations in one cycle)
    period: usize,
    /// Current level value (after fitting)
    level: Option<f64>,
    /// Current trend value (after fitting)
    trend: Option<f64>,
    /// Seasonal components (after fitting)
    seasonal: Option<Vec<f64>>,
    /// Fitted values over the training period
    fitted_values: Option<Vec<f64>>,
}

impl HoltWintersModel {
    /// Creates a new Holt-Winters (triple exponential smoothing) model.
    ///
    /// # Arguments
    /// * `alpha` - Smoothing parameter for level (0 < α < 1)
    /// * `beta` - Smoothing parameter for trend (0 < β < 1)
    /// * `gamma` - Smoothing parameter for seasonal component (0 < γ < 1)
    /// * `period` - Seasonal period (number of observations in one cycle)
    ///
    /// # Returns
    /// * `Result<Self>` - A new Holt-Winters model if parameters are valid
    pub fn new(
        alpha: f64,
        beta: f64,
        gamma: f64,
        period: usize,
    ) -> std::result::Result<Self, ESError> {
        // Validate parameters
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(OxiError::from(ESError::InvalidAlpha(alpha));
        }

        if beta <= 0.0 || beta >= 1.0 {
            return Err(OxiError::from(ESError::InvalidBeta(beta));
        }

        if gamma <= 0.0 || gamma >= 1.0 {
            return Err(OxiError::from(ESError::InvalidGamma(gamma));
        }

        if period < 2 {
            return Err(OxiError::from(ESError::InvalidPeriod(period));
        }

        let name = format!(
            "HoltWinters(α={:.3}, β={:.3}, γ={:.3}, m={})",
            alpha, beta, gamma, period
        );

        Ok(HoltWintersModel {
            name,
            alpha,
            beta,
            gamma,
            period,
            level: None,
            trend: None,
            seasonal: None,
            fitted_values: None,
        })
    }

    /// Fit the model to the provided time series data.
    /// This is a convenience method that calls the trait method directly.
    pub fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        <Self as Forecaster>::fit(self, data)
    }

    /// Forecast future values.
    /// This is a convenience method that calls the trait method directly.
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        <Self as Forecaster>::forecast(self, horizon)
    }

    /// Evaluate the model on test data.
    /// This is a convenience method that calls the trait method directly.
    pub fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        <Self as Forecaster>::evaluate(self, test_data)
    }

    /// Generate forecasts and evaluation in a standardized format.
    /// This is a convenience method that calls the trait method directly.
    pub fn predict(
        &self,
        horizon: usize,
        test_data: Option<&TimeSeriesData>,
    ) -> Result<ModelOutput> {
        <Self as Forecaster>::predict(self, horizon, test_data)
    }

    /// Get the fitted values if available.
    pub fn fitted_values(&self) -> Option<&Vec<f64>> {
        self.fitted_values.as_ref()
    }

    /// Get the seasonal components if available.
    pub fn seasonal_components(&self) -> Option<&Vec<f64>> {
        self.seasonal.as_ref()
    }
}

impl Forecaster for HoltWintersModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        if data.is_empty() {
            return Err(OxiError::from(ESError::EmptyData));
        }

        let n = data.values.len();

        // Need at least 2 full seasons of data
        if n < 2 * self.period {
            return Err(OxiError::from(ESError::InsufficientData {
                actual: n,
                expected: 2 * self.period,
            }));
        }

        // Initialize seasonal components
        // We use the ratio-to-moving-average method
        let mut seasonal = vec![0.0; self.period];

        // Calculate initial level using the first period
        let first_period_sum: f64 = data.values[0..self.period].iter().sum();
        let mut level = first_period_sum / self.period as f64;

        // Calculate initial trend using the first two periods
        let second_period_sum: f64 = data.values[self.period..2 * self.period].iter().sum();
        let mut trend = (second_period_sum - first_period_sum) / (self.period * self.period) as f64;

        // Initialize seasonal components
        for i in 0..self.period {
            seasonal[i] = data.values[i] - level;
        }

        // Prepare to store fitted values
        let mut fitted_values = Vec::with_capacity(n);

        // First period's fitted values are based on initial level and seasonal components
        for i in 0..self.period {
            fitted_values.push(level + seasonal[i]);
        }

        // Apply the Holt-Winters model recursively
        for i in self.period..n {
            let s_idx = i % self.period; // Index into seasonal array

            // Calculate forecast for this step
            let forecast = level + trend + seasonal[s_idx];

            // Store the forecast
            fitted_values.push(forecast);

            // Update the level, trend, and seasonal components based on the observed value
            let old_level = level;
            level = self.alpha * (data.values[i] - seasonal[s_idx])
                + (1.0 - self.alpha) * (level + trend);
            trend = self.beta * (level - old_level) + (1.0 - self.beta) * trend;
            seasonal[s_idx] =
                self.gamma * (data.values[i] - level) + (1.0 - self.gamma) * seasonal[s_idx];
        }

        // Store the final values
        self.level = Some(level);
        self.trend = Some(trend);
        self.seasonal = Some(seasonal);
        self.fitted_values = Some(fitted_values);

        Ok(())
    }

    fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        if let (Some(level), Some(trend)) = (self.level, self.trend) {
            if let Some(seasonal) = &self.seasonal {
                if horizon == 0 {
                    return Err(OxiError::from(ESError::InvalidHorizon(horizon));
                }

                // For Holt-Winters model, forecasts include trend and seasonal components
                let mut forecasts = Vec::with_capacity(horizon);
                for h in 1..=horizon {
                    let seasonal_idx = (h - 1) % self.period;
                    forecasts.push(level + h as f64 * trend + seasonal[seasonal_idx]);
                }

                Ok(forecasts)
            } else {
                Err(OxiError::from(ESError::NotFitted))
            }
        } else {
            Err(OxiError::from(ESError::NotFitted))
        }
    }

    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        if self.level.is_none() || self.trend.is_none() || self.seasonal.is_none() {
            return Err(OxiError::from(ESError::NotFitted));
        }

        let forecast = self.forecast(test_data.values.len())?;

        // Calculate error metrics
        let mae = mae(&test_data.values, &forecast);
        let mse = mse(&test_data.values, &forecast);
        let rmse = rmse(&test_data.values, &forecast);
        let mape = mape(&test_data.values, &forecast);
        let smape = smape(&test_data.values, &forecast);

        Ok(ModelEvaluation {
            model_name: self.name.clone(),
            mae,
            mse,
            rmse,
            mape,
            smape,
        })
    }

    // Using the default predict implementation from the trait
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, TimeZone, Utc};
    use std::f64::consts::PI;

    #[test]
    fn test_holt_winters_forecast() {
        // Create a time series with trend and seasonality
        let now = Utc::now();
        let period = 4; // Quarterly data
        let n = 20; // 5 years of quarterly data

        let timestamps: Vec<DateTime<Utc>> = (0..n)
            .map(|i| {
                Utc.timestamp_opt(now.timestamp() + i as i64 * 86400 * 90, 0)
                    .unwrap()
            })
            .collect();

        // Generate data with trend and seasonality
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            // Base level starts at 100
            let base = 100.0;
            // Linear trend adds 5 per quarter
            let trend = i as f64 * 5.0;
            // Seasonal pattern with amplitude of 20
            let seasonal = 20.0 * (2.0 * PI * (i % period) as f64 / period as f64).sin();
            // Small random noise
            let noise = (i as f64 * 0.1).cos() * 2.0;

            values.push(base + trend + seasonal + noise);
        }

        let time_series = TimeSeriesData::new(timestamps, values, "test_series").unwrap();

        // Create and fit the Holt-Winters model
        let mut model = HoltWintersModel::new(0.2, 0.1, 0.3, period).unwrap();
        model.fit(&time_series).unwrap();

        // Forecast 2 periods ahead
        let forecast_horizon = 2 * period; // 2 years of quarterly data
        let forecasts = model.forecast(forecast_horizon).unwrap();

        // Check that the number of forecasts matches the horizon
        assert_eq!(forecasts.len(), forecast_horizon);

        // The model should detect the seasonal pattern
        // Check that the seasonal components have a reasonable range
        if let Some(seasonal) = model.seasonal_components() {
            assert_eq!(seasonal.len(), period);

            // Calculate min and max of seasonal components
            let min_seasonal = seasonal.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_seasonal = seasonal.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            // Range should be approximately the amplitude of our seasonal pattern
            let seasonal_range = max_seasonal - min_seasonal;
            assert!(seasonal_range > 10.0); // Seasonal pattern had amplitude of 20
        } else {
            panic!("Seasonal components should be available");
        }

        // Test the standardized ModelOutput from predict()
        let output = model.predict(forecast_horizon, None).unwrap();

        // Check basic properties of the ModelOutput
        assert_eq!(output.model_name, model.name());
        assert_eq!(output.forecasts.len(), forecast_horizon);
        assert_eq!(output.forecasts, forecasts);
    }
}
