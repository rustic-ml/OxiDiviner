#![allow(clippy::needless_range_loop)]

use crate::core::{Forecaster, ModelEvaluation, ModelOutput, OxiError, Result, TimeSeriesData};
use crate::math::metrics::{mae, mape, mse, rmse, smape};
use crate::models::exponential_smoothing::ESError;

use std::fmt;

/// Error component type for ETS models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorType {
    /// Additive errors
    Additive,
    /// Multiplicative errors
    Multiplicative,
    /// No error component
    None,
}

impl fmt::Display for ErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorType::Additive => write!(f, "A"),
            ErrorType::Multiplicative => write!(f, "M"),
            ErrorType::None => write!(f, "N"),
        }
    }
}

/// Trend component type for ETS models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrendType {
    /// Additive trend
    Additive,
    /// Multiplicative trend
    Multiplicative,
    /// Damped additive trend
    DampedAdditive,
    /// Damped multiplicative trend
    DampedMultiplicative,
    /// No trend component
    None,
}

impl fmt::Display for TrendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrendType::Additive => write!(f, "A"),
            TrendType::Multiplicative => write!(f, "M"),
            TrendType::DampedAdditive => write!(f, "Ad"),
            TrendType::DampedMultiplicative => write!(f, "Md"),
            TrendType::None => write!(f, "N"),
        }
    }
}

/// Seasonal component type for ETS models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeasonalType {
    /// Additive seasonality
    Additive,
    /// Multiplicative seasonality
    Multiplicative,
    /// No seasonal component
    None,
}

impl fmt::Display for SeasonalType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeasonalType::Additive => write!(f, "A"),
            SeasonalType::Multiplicative => write!(f, "M"),
            SeasonalType::None => write!(f, "N"),
        }
    }
}

/// Error-Trend-Seasonal (ETS) model for forecasting.
///
/// This is a comprehensive implementation of the ETS framework, which includes many
/// exponential smoothing variants as special cases:
/// - ETS(A,N,N): Simple Exponential Smoothing with additive errors
/// - ETS(A,A,N): Holt's Linear Method with additive errors
/// - ETS(A,A,A): Additive Holt-Winters Method
/// - ETS(A,Ad,N): Damped Trend Method with additive errors
/// - ETS(M,N,N): Simple Exponential Smoothing with multiplicative errors
/// - and many others
///
/// The model components are specified using the following notation:
/// - Error: Additive (A), Multiplicative (M)
/// - Trend: None (N), Additive (A), Additive Damped (Ad), Multiplicative (M), Multiplicative Damped (Md)
/// - Seasonal: None (N), Additive (A), Multiplicative (M)
///
/// # References
/// Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia.
pub struct ETSModel {
    /// Model name
    name: String,
    /// Type of error
    error_type: ErrorType,
    /// Type of trend
    trend_type: TrendType,
    /// Type of seasonality
    seasonal_type: SeasonalType,
    /// Smoothing parameter for level (0 < α < 1)
    alpha: f64,
    /// Smoothing parameter for trend (0 < β < 1)
    beta: Option<f64>,
    /// Damping parameter for trend (0 < φ < 1)
    phi: Option<f64>,
    /// Smoothing parameter for seasonal component (0 < γ < 1)
    gamma: Option<f64>,
    /// Seasonal period (e.g., 4 for quarterly, 12 for monthly, 7 for daily with weekly seasonality)
    period: Option<usize>,
    /// Current level value (after fitting)
    level: Option<f64>,
    /// Current trend value (after fitting)
    trend: Option<f64>,
    /// Current seasonal components (after fitting)
    seasonal: Option<Vec<f64>>,
    /// Fitted values over the training period
    fitted_values: Option<Vec<f64>>,
}

impl ETSModel {
    /// Creates a new ETS model with specified components.
    ///
    /// # Arguments
    /// * `error_type` - Type of error component (A, M)
    /// * `trend_type` - Type of trend component (N, A, Ad, M, Md)
    /// * `seasonal_type` - Type of seasonal component (N, A, M)
    /// * `alpha` - Smoothing parameter for level (0 < α < 1)
    /// * `beta` - Smoothing parameter for trend (0 < β < 1), None if trend_type is None
    /// * `phi` - Damping parameter (0 < φ < 1), required if trend is damped
    /// * `gamma` - Smoothing parameter for seasonality (0 < γ < 1), None if seasonal_type is None
    /// * `period` - Seasonal period, None if seasonal_type is None
    ///
    /// # Returns
    /// * `Result<Self>` - A new ETS model if parameters are valid
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        error_type: ErrorType,
        trend_type: TrendType,
        seasonal_type: SeasonalType,
        alpha: f64,
        beta: Option<f64>,
        phi: Option<f64>,
        gamma: Option<f64>,
        period: Option<usize>,
    ) -> std::result::Result<Self, ESError> {
        // Validate parameters
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(ESError::InvalidAlpha(alpha));
        }

        // Validate beta if trend is present
        if trend_type != TrendType::None {
            if let Some(beta_val) = beta {
                if beta_val <= 0.0 || beta_val >= 1.0 {
                    return Err(ESError::InvalidBeta(beta_val));
                }
            } else {
                return Err(ESError::MissingParameter("beta".to_string()));
            }
        }

        // Validate phi if trend is damped
        if trend_type == TrendType::DampedAdditive || trend_type == TrendType::DampedMultiplicative
        {
            if let Some(phi_val) = phi {
                if phi_val <= 0.0 || phi_val >= 1.0 {
                    return Err(ESError::InvalidDampingFactor(phi_val));
                }
            } else {
                return Err(ESError::MissingParameter("phi".to_string()));
            }
        }

        // Validate gamma and period if seasonality is present
        if seasonal_type != SeasonalType::None {
            if let Some(gamma_val) = gamma {
                if gamma_val <= 0.0 || gamma_val >= 1.0 {
                    return Err(ESError::InvalidGamma(gamma_val));
                }
            } else {
                return Err(ESError::MissingParameter("gamma".to_string()));
            }

            if let Some(period_val) = period {
                if period_val < 2 {
                    return Err(ESError::InvalidPeriod(period_val));
                }
            } else {
                return Err(ESError::MissingParameter("period".to_string()));
            }
        }

        // Construct model name
        let mut name = format!("ETS({},{},{})", error_type, trend_type, seasonal_type);

        // Add parameters to name
        name.push_str(&format!(",α={:.3}", alpha));

        if let Some(beta_val) = beta {
            name.push_str(&format!(",β={:.3}", beta_val));
        }

        if let Some(phi_val) = phi {
            name.push_str(&format!(",φ={:.3}", phi_val));
        }

        if let Some(gamma_val) = gamma {
            name.push_str(&format!(",γ={:.3}", gamma_val));
        }

        if let Some(period_val) = period {
            name.push_str(&format!(",m={}", period_val));
        }

        Ok(ETSModel {
            name,
            error_type,
            trend_type,
            seasonal_type,
            alpha,
            beta,
            phi,
            gamma,
            period,
            level: None,
            trend: None,
            seasonal: None,
            fitted_values: None,
        })
    }

    /// Convenience method to create a Simple Exponential Smoothing model (ETS(A,N,N).
    pub fn simple(alpha: f64) -> std::result::Result<Self, ESError> {
        Self::new(
            ErrorType::Additive,
            TrendType::None,
            SeasonalType::None,
            alpha,
            None,
            None,
            None,
            None,
        )
    }

    /// Convenience method to create a Holt Linear model (ETS(A,A,N).
    pub fn holt(alpha: f64, beta: f64) -> std::result::Result<Self, ESError> {
        Self::new(
            ErrorType::Additive,
            TrendType::Additive,
            SeasonalType::None,
            alpha,
            Some(beta),
            None,
            None,
            None,
        )
    }

    /// Convenience method to create a Damped Trend model (ETS(A,Ad,N).
    pub fn damped_trend(alpha: f64, beta: f64, phi: f64) -> std::result::Result<Self, ESError> {
        Self::new(
            ErrorType::Additive,
            TrendType::DampedAdditive,
            SeasonalType::None,
            alpha,
            Some(beta),
            Some(phi),
            None,
            None,
        )
    }

    /// Convenience method to create an additive Holt-Winters model (ETS(A,A,A).
    pub fn holt_winters_additive(
        alpha: f64,
        beta: f64,
        gamma: f64,
        period: usize,
    ) -> std::result::Result<Self, ESError> {
        Self::new(
            ErrorType::Additive,
            TrendType::Additive,
            SeasonalType::Additive,
            alpha,
            Some(beta),
            None,
            Some(gamma),
            Some(period),
        )
    }

    /// Convenience method to create a multiplicative Holt-Winters model (ETS(A,A,M).
    pub fn holt_winters_multiplicative(
        alpha: f64,
        beta: f64,
        gamma: f64,
        period: usize,
    ) -> std::result::Result<Self, ESError> {
        Self::new(
            ErrorType::Additive,
            TrendType::Additive,
            SeasonalType::Multiplicative,
            alpha,
            Some(beta),
            None,
            Some(gamma),
            Some(period),
        )
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
}

impl Forecaster for ETSModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        if data.is_empty() {
            return Err(OxiError::ESEmptyData);
        }

        let n = data.values.len();
        let min_size = match self.seasonal_type {
            SeasonalType::None => 2,
            _ => 2 * self.period.unwrap(),
        };

        if n < min_size {
            return Err(OxiError::ESInsufficientData {
                actual: n,
                expected: min_size,
            });
        }

        // Setup initial values
        match (self.trend_type, self.seasonal_type) {
            // Simple Exponential Smoothing
            (TrendType::None, SeasonalType::None) => {
                self.level = Some(data.values[0]);
                self.trend = None;
                self.seasonal = None;
            }

            // Holt's Linear Method (with or without damping)
            (TrendType::Additive | TrendType::DampedAdditive, SeasonalType::None) => {
                self.level = Some(data.values[0]);
                self.trend = Some(data.values[1] - data.values[0]);
                self.seasonal = None;
            }

            // Holt-Winters Methods
            (
                TrendType::Additive | TrendType::DampedAdditive,
                SeasonalType::Additive | SeasonalType::Multiplicative,
            ) => {
                let period = self.period.unwrap();

                // Initialize level as average of first season
                self.level = Some(data.values[0..period].iter().sum::<f64>() / period as f64);

                // Initialize trend using first two seasons
                let first_season_avg = data.values[0..period].iter().sum::<f64>() / period as f64;
                let second_season_avg =
                    data.values[period..2 * period].iter().sum::<f64>() / period as f64;
                self.trend = Some((second_season_avg - first_season_avg) / period as f64);

                // Initialize seasonal components
                let mut seasonal = vec![0.0; period];
                if self.seasonal_type == SeasonalType::Additive {
                    // Additive seasonality
                    for i in 0..period {
                        seasonal[i] = data.values[i] - self.level.unwrap();
                    }
                } else {
                    // Multiplicative seasonality
                    for i in 0..period {
                        seasonal[i] = data.values[i] / self.level.unwrap();
                    }
                }
                self.seasonal = Some(seasonal);
            }

            // Unsupported model types
            _ => {
                return Err(OxiError::ESUnsupportedModelType(format!(
                    "ETS({},{},{})",
                    self.error_type, self.trend_type, self.seasonal_type
                )));
            }
        }

        // Prepare to store fitted values
        let mut fitted_values = Vec::with_capacity(n);

        // Initialize with the first-step forecast
        match (self.trend_type, self.seasonal_type) {
            (TrendType::None, SeasonalType::None) => {
                fitted_values.push(self.level.unwrap());
            }
            (TrendType::Additive | TrendType::DampedAdditive, SeasonalType::None) => {
                fitted_values.push(self.level.unwrap());
                fitted_values.push(self.level.unwrap() + self.trend.unwrap());
            }
            (TrendType::Additive | TrendType::DampedAdditive, SeasonalType::Additive) => {
                let period = self.period.unwrap();
                let level = self.level.unwrap();
                let _trend = self.trend.unwrap();
                let seasonal = self.seasonal.as_ref().unwrap();

                for i in 0..period {
                    fitted_values.push(level + seasonal[i]);
                }
            }
            (TrendType::Additive | TrendType::DampedAdditive, SeasonalType::Multiplicative) => {
                let period = self.period.unwrap();
                let level = self.level.unwrap();
                let seasonal = self.seasonal.as_ref().unwrap();

                for i in 0..period {
                    fitted_values.push(level * seasonal[i]);
                }
            }
            _ => {
                return Err(OxiError::ESUnsupportedModelType(format!(
                    "ETS({},{},{})",
                    self.error_type, self.trend_type, self.seasonal_type
                )));
            }
        }

        // Apply the model recursively to update parameters
        let start_idx = match self.seasonal_type {
            SeasonalType::None => 1,
            _ => self.period.unwrap(),
        };

        let mut level = self.level.unwrap();
        let mut trend = self.trend.unwrap_or(0.0);
        let mut seasonal = self.seasonal.clone().unwrap_or_default();

        for i in start_idx..n {
            let s_idx = if self.seasonal_type != SeasonalType::None {
                i % self.period.unwrap()
            } else {
                0
            };

            // Store the forecast for this step
            let forecast = match (self.trend_type, self.seasonal_type) {
                (TrendType::None, SeasonalType::None) => level,
                (TrendType::Additive, SeasonalType::None) => level + trend,
                (TrendType::DampedAdditive, SeasonalType::None) => {
                    level + self.phi.unwrap() * trend
                }
                (TrendType::Additive, SeasonalType::Additive) => level + trend + seasonal[s_idx],
                (TrendType::DampedAdditive, SeasonalType::Additive) => {
                    level + self.phi.unwrap() * trend + seasonal[s_idx]
                }
                (TrendType::Additive, SeasonalType::Multiplicative) => {
                    (level + trend) * seasonal[s_idx]
                }
                (TrendType::DampedAdditive, SeasonalType::Multiplicative) => {
                    (level + self.phi.unwrap() * trend) * seasonal[s_idx]
                }
                _ => {
                    return Err(OxiError::ESUnsupportedModelType(format!(
                        "ETS({},{},{})",
                        self.error_type, self.trend_type, self.seasonal_type
                    )));
                }
            };

            fitted_values.push(forecast);

            // Update parameters
            match (self.trend_type, self.seasonal_type) {
                // Simple Exponential Smoothing
                (TrendType::None, SeasonalType::None) => {
                    level = self.alpha * data.values[i] + (1.0 - self.alpha) * level;
                }

                // Holt's Linear Method
                (TrendType::Additive, SeasonalType::None) => {
                    let old_level = level;
                    level = self.alpha * data.values[i] + (1.0 - self.alpha) * (level + trend);
                    trend = self.beta.unwrap() * (level - old_level)
                        + (1.0 - self.beta.unwrap()) * trend;
                }

                // Damped Trend Method
                (TrendType::DampedAdditive, SeasonalType::None) => {
                    let old_level = level;
                    level = self.alpha * data.values[i]
                        + (1.0 - self.alpha) * (level + self.phi.unwrap() * trend);
                    trend = self.beta.unwrap() * (level - old_level)
                        + (1.0 - self.beta.unwrap()) * self.phi.unwrap() * trend;
                }

                // Additive Holt-Winters
                (TrendType::Additive, SeasonalType::Additive) => {
                    let old_level = level;
                    let period = self.period.unwrap();
                    let s_idx = i % period;
                    let prev_s_idx = (i + period - (period % n)) % period;

                    level = self.alpha * (data.values[i] - seasonal[prev_s_idx])
                        + (1.0 - self.alpha) * (level + trend);
                    trend = self.beta.unwrap() * (level - old_level)
                        + (1.0 - self.beta.unwrap()) * trend;
                    seasonal[s_idx] = self.gamma.unwrap() * (data.values[i] - level)
                        + (1.0 - self.gamma.unwrap()) * seasonal[prev_s_idx];
                }

                // Multiplicative Holt-Winters
                (TrendType::Additive, SeasonalType::Multiplicative) => {
                    let old_level = level;
                    let period = self.period.unwrap();
                    let s_idx = i % period;
                    let prev_s_idx = (i + period - (period % n)) % period;

                    level = self.alpha * (data.values[i] / seasonal[prev_s_idx])
                        + (1.0 - self.alpha) * (level + trend);
                    trend = self.beta.unwrap() * (level - old_level)
                        + (1.0 - self.beta.unwrap()) * trend;
                    seasonal[s_idx] = self.gamma.unwrap() * (data.values[i] / level)
                        + (1.0 - self.gamma.unwrap()) * seasonal[prev_s_idx];
                }

                // Other combinations are not yet implemented
                _ => {
                    return Err(OxiError::ESUnsupportedModelType(format!(
                        "ETS({},{},{})",
                        self.error_type, self.trend_type, self.seasonal_type
                    )));
                }
            }
        }

        // Store the final values
        self.level = Some(level);
        self.trend = if self.trend_type != TrendType::None {
            Some(trend)
        } else {
            None
        };
        self.seasonal = if self.seasonal_type != SeasonalType::None {
            Some(seasonal)
        } else {
            None
        };
        self.fitted_values = Some(fitted_values);

        Ok(())
    }

    fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        if self.level.is_none() {
            return Err(OxiError::ESNotFitted);
        }

        if horizon == 0 {
            return Err(OxiError::ESInvalidHorizon(horizon));
        }

        let level = self.level.unwrap();
        let _trend = self.trend.unwrap_or(0.0);

        let mut forecasts = Vec::with_capacity(horizon);

        match (self.trend_type, self.seasonal_type) {
            // Simple Exponential Smoothing
            (TrendType::None, SeasonalType::None) => {
                // In SES, all forecasts are the same
                forecasts = vec![level; horizon];
            }

            // Holt's Linear Method
            (TrendType::Additive, SeasonalType::None) => {
                // Linear increase
                for h in 1..=horizon {
                    forecasts.push(level + h as f64 * _trend);
                }
            }

            // Damped Trend Method
            (TrendType::DampedAdditive, SeasonalType::None) => {
                // Damped trend
                let phi = self.phi.unwrap();
                for h in 1..=horizon {
                    let mut damping_sum = 0.0;
                    let mut phi_power = phi;

                    for _ in 1..=h {
                        damping_sum += phi_power;
                        phi_power *= phi;
                    }

                    forecasts.push(level + damping_sum * _trend);
                }
            }

            // Additive Holt-Winters
            (TrendType::Additive, SeasonalType::Additive) => {
                if let Some(ref seasonal) = self.seasonal {
                    let period = self.period.unwrap();

                    for h in 1..=horizon {
                        let s_idx = (h - 1) % period;
                        forecasts.push(level + h as f64 * _trend + seasonal[s_idx]);
                    }
                } else {
                    return Err(OxiError::ESNotFitted);
                }
            }

            // Multiplicative Holt-Winters
            (TrendType::Additive, SeasonalType::Multiplicative) => {
                if let Some(ref seasonal) = self.seasonal {
                    let period = self.period.unwrap();

                    for h in 1..=horizon {
                        let s_idx = (h - 1) % period;
                        forecasts.push((level + h as f64 * _trend) * seasonal[s_idx]);
                    }
                } else {
                    return Err(OxiError::ESNotFitted);
                }
            }

            // Other combinations are not yet implemented
            _ => {
                return Err(OxiError::ESUnsupportedModelType(format!(
                    "ETS({},{},{})",
                    self.error_type, self.trend_type, self.seasonal_type
                )));
            }
        }

        Ok(forecasts)
    }

    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        if self.level.is_none() {
            return Err(OxiError::ESNotFitted);
        }

        let forecast = self.forecast(test_data.values.len())?;

        // Calculate error metrics
        let mae = mae(&test_data.values, &forecast);
        let mse = mse(&test_data.values, &forecast);
        let rmse = rmse(&test_data.values, &forecast);
        let mape = mape(&test_data.values, &forecast);
        let smape = smape(&test_data.values, &forecast);

        // Calculate R-squared
        let actual_mean = test_data.values.iter().sum::<f64>() / test_data.values.len() as f64;
        let tss = test_data
            .values
            .iter()
            .map(|x| (x - actual_mean).powi(2))
            .sum::<f64>();
        let r_squared = if tss > 0.0 {
            1.0 - (mse * test_data.values.len() as f64) / tss
        } else {
            0.0
        };

        Ok(ModelEvaluation {
            model_name: self.name.clone(),
            mae,
            mse,
            rmse,
            mape,
            smape,
            r_squared,
            aic: None,
            bic: None,
        })
    }

    // Using the default predict implementation from the trait
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, TimeZone, Utc};

    #[test]
    fn test_ets_simple() {
        // Test ETS(A,N,N) which is equivalent to Simple Exponential Smoothing
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();

        // Some data: 10, 11, 9, 10.5, 11.2, 10.8, 9.5, 10, 10.5, 11
        let values: Vec<f64> = vec![10.0, 11.0, 9.0, 10.5, 11.2, 10.8, 9.5, 10.0, 10.5, 11.0];

        let time_series = TimeSeriesData::new(timestamps, values, "test_series").unwrap();

        // Create and fit an ETS model with alpha = 0.3
        let mut model = ETSModel::simple(0.3).unwrap();
        model.fit(&time_series).unwrap();

        // Forecast 5 periods ahead
        let forecast_horizon = 5;
        let forecasts = model.forecast(forecast_horizon).unwrap();

        // Check that the number of forecasts matches the horizon
        assert_eq!(forecasts.len(), forecast_horizon);

        // For Simple ES, all forecasts should be the same
        let first_forecast = forecasts[0];
        for forecast in forecasts.iter() {
            assert_eq!(*forecast, first_forecast);
        }
    }

    #[test]
    fn test_ets_holt() {
        // Test ETS(A,A,N) which is equivalent to Holt Linear
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();

        // Linear trend data: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        let values: Vec<f64> = (1..=10).map(|i| i as f64).collect();

        let time_series = TimeSeriesData::new(timestamps, values, "test_series").unwrap();

        // Create and fit an ETS model with alpha = 0.8, beta = 0.2
        let mut model = ETSModel::holt(0.8, 0.2).unwrap();
        model.fit(&time_series).unwrap();

        // Forecast 5 periods ahead
        let forecast_horizon = 5;
        let forecasts = model.forecast(forecast_horizon).unwrap();

        // Check that the forecasts continue the linear trend
        for (i, forecast) in forecasts.iter().enumerate() {
            let expected = 11.0 + i as f64;
            assert!(
                (forecast - expected).abs() < 0.5,
                "Forecast {} expected to be close to {}, but was {}",
                i + 1,
                expected,
                forecast
            );
        }
    }
}
