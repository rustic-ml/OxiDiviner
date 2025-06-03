#![allow(clippy::needless_range_loop)]

use crate::core::{Forecaster, ModelEvaluation, ModelOutput, OxiError, Result, TimeSeriesData};
use crate::math::metrics::{mae, mape, mse, rmse, smape};
use crate::models::autoregressive::arima::ARIMAModel;
use crate::models::autoregressive::error::ARError;

/// Seasonal Autoregressive Integrated Moving Average (SARIMA) model for time series forecasting.
///
/// This model extends ARIMA to include seasonal components and is denoted as SARIMA(p,d,q)(P,D,Q)s
/// where:
/// - p, d, q: Non-seasonal AR order, differencing order, and MA order
/// - P, D, Q: Seasonal AR order, seasonal differencing order, and seasonal MA order
/// - s: Seasonal period (e.g., 12 for monthly data with yearly seasonality)
///
/// The model works by:
/// 1. Applying seasonal differencing to handle seasonal non-stationarity
/// 2. Applying regular differencing to handle trend non-stationarity
/// 3. Fitting an ARIMA model to the resulting series
/// 4. Integrating back to get forecasts in the original scale
///
/// SARIMA models are useful for:
/// - Data with both trend and seasonal components
/// - Economic indicators with seasonal patterns (retail sales, tourism)
/// - Climate data with yearly patterns
/// - Any time series with regular, periodic fluctuations
pub struct SARIMAModel {
    /// Model name
    name: String,
    /// Non-seasonal AR order (p)
    p: usize,
    /// Non-seasonal differencing order (d)
    d: usize,
    /// Non-seasonal MA order (q)
    q: usize,
    /// Seasonal AR order (P)
    seasonal_p: usize,
    /// Seasonal differencing order (D)
    seasonal_d: usize,
    /// Seasonal MA order (Q)
    seasonal_q: usize,
    /// Seasonal period (s)
    s: usize,
    /// Include intercept/constant term
    include_intercept: bool,
    /// Internal ARIMA model after seasonal adjustments
    arima_model: Option<ARIMAModel>,
    /// Last seasonal values for forecasting
    last_values: Option<Vec<f64>>,
}

impl SARIMAModel {
    /// Create a new SARIMA model.
    ///
    /// # Arguments
    /// * `p` - Non-seasonal AR order
    /// * `d` - Non-seasonal differencing order
    /// * `q` - Non-seasonal MA order
    /// * `P` - Seasonal AR order
    /// * `D` - Seasonal differencing order
    /// * `Q` - Seasonal MA order
    /// * `s` - Seasonal period (e.g., 12 for monthly data, 4 for quarterly data)
    /// * `include_intercept` - Whether to include an intercept term
    ///
    /// # Returns
    /// * `Result<Self>` - A new SARIMA model if parameters are valid
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        p: usize,
        d: usize,
        q: usize,
        seasonal_p: usize,
        seasonal_d: usize,
        seasonal_q: usize,
        s: usize,
        include_intercept: bool,
    ) -> Result<Self> {
        // Validate parameters
        if p == 0 && q == 0 && seasonal_p == 0 && seasonal_q == 0 {
            return Err(OxiError::from(ARError::InvalidLagOrder(0)));
        }

        if s < 2 {
            return Err(OxiError::from(ARError::InvalidParameter(format!(
                "Seasonal period must be at least 2, got {}",
                s
            ))));
        }

        let name = if include_intercept {
            format!(
                "SARIMA({},{},{})({},{},{}){}+intercept",
                p, d, q, seasonal_p, seasonal_d, seasonal_q, s
            )
        } else {
            format!(
                "SARIMA({},{},{})({},{},{}){}",
                p, d, q, seasonal_p, seasonal_d, seasonal_q, s
            )
        };

        Ok(SARIMAModel {
            name,
            p,
            d,
            q,
            seasonal_p,
            seasonal_d,
            seasonal_q,
            s,
            include_intercept,
            arima_model: None,
            last_values: None,
        })
    }

    /// Apply seasonal differencing to make the series more stationary
    fn seasonal_difference(&self, data: &[f64], d: usize, s: usize) -> Vec<f64> {
        if d == 0 || data.len() <= s * d {
            return data.to_vec();
        }

        let mut result = data.to_vec();

        // Apply d-order seasonal differencing
        for _ in 0..d {
            let mut temp = Vec::with_capacity(result.len() - s);
            for i in s..result.len() {
                temp.push(result[i] - result[i - s]);
            }
            result = temp;
        }

        result
    }

    /// Reverse seasonal differencing to get forecasts in original scale
    fn seasonal_integrate(&self, forecast: &[f64], last_values: &[f64]) -> Vec<f64> {
        if self.seasonal_d == 0 || last_values.len() < self.s * self.seasonal_d {
            return forecast.to_vec();
        }

        // Start with the differenced forecasts
        let mut result = forecast.to_vec();

        // For each level of seasonal differencing (working backwards)
        for diff_level in (1..=self.seasonal_d).rev() {
            let mut integrated = Vec::with_capacity(result.len());

            // For each forecast point
            for (i, &diff_val) in result.iter().enumerate() {
                // Find the appropriate seasonal lag to add
                let season_idx = last_values.len() - self.s * diff_level + i;
                if season_idx < last_values.len() {
                    // We have the actual value from history
                    integrated.push(diff_val + last_values[season_idx]);
                } else {
                    // We need to use a previously generated forecast point
                    let prev_season_idx = i - self.s;
                    integrated.push(diff_val + integrated[prev_season_idx]);
                }
            }

            result = integrated;
        }

        result
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

    /// Get the non-seasonal AR order (p)
    pub fn p(&self) -> usize {
        self.p
    }

    /// Get the non-seasonal differencing order (d)
    pub fn d(&self) -> usize {
        self.d
    }

    /// Get the non-seasonal MA order (q)
    pub fn q(&self) -> usize {
        self.q
    }

    /// Get the seasonal AR order (P)
    pub fn seasonal_p(&self) -> usize {
        self.seasonal_p
    }

    /// Get the seasonal differencing order (D)
    pub fn seasonal_d(&self) -> usize {
        self.seasonal_d
    }

    /// Get the seasonal MA order (Q)
    pub fn seasonal_q(&self) -> usize {
        self.seasonal_q
    }

    /// Get the seasonal period (s)
    pub fn s(&self) -> usize {
        self.s
    }

    /// Get whether the model includes an intercept
    pub fn include_intercept(&self) -> bool {
        self.include_intercept
    }

    /// Get the internal ARIMA model
    pub fn arima_model(&self) -> Option<&ARIMAModel> {
        self.arima_model.as_ref()
    }
}

impl Forecaster for SARIMAModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        if data.is_empty() {
            return Err(OxiError::from(ARError::EmptyData));
        }

        let n = data.values.len();

        // Need enough data for both seasonal and non-seasonal components
        let min_required = self.s * self.seasonal_d
            + self.d
            + self
                .p
                .max(self.seasonal_p * self.s)
                .max(self.q)
                .max(self.seasonal_q * self.s);

        if n <= min_required {
            return Err(OxiError::from(ARError::InsufficientData {
                actual: n,
                expected: min_required + 1,
            }));
        }

        // First apply seasonal differencing
        let seasonally_differenced =
            self.seasonal_difference(&data.values, self.seasonal_d, self.s);

        // Store original values for later integration during forecasting
        self.last_values = Some(data.values[(n - self.s * self.seasonal_d - 1)..].to_vec());

        // Create combined orders for the ARIMA model to handle both regular and seasonal components
        // In a full implementation, this would use polynomial multiplication of the AR and MA operators
        // Here we use a simplified approach where we combine the orders
        let combined_p = self.p + self.seasonal_p * self.s;
        let combined_q = self.q + self.seasonal_q * self.s;

        // Create timestamps for the seasonally differenced series
        let differenced_timestamps = data.timestamps[(self.s * self.seasonal_d)..].to_vec();
        let diff_series_name = format!("seasonally_diff_{}_of_{}", self.seasonal_d, data.name);

        let differenced_series = TimeSeriesData::new(
            differenced_timestamps,
            seasonally_differenced,
            &diff_series_name,
        )
        .map_err(|e| OxiError::ModelError(format!("Failed to create differenced series: {}", e)))?;

        // Create and fit an ARIMA model on the seasonally differenced data
        let mut arima_model =
            ARIMAModel::new(combined_p, self.d, combined_q, self.include_intercept)?;

        arima_model.fit(&differenced_series)?;

        // Store the fitted ARIMA model
        self.arima_model = Some(arima_model);

        Ok(())
    }

    fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        if horizon == 0 {
            return Err(OxiError::from(ARError::InvalidHorizon(horizon)));
        }

        if self.arima_model.is_none() || self.last_values.is_none() {
            return Err(OxiError::from(ARError::NotFitted));
        }

        // Forecast the differenced series using the ARIMA model
        let diff_forecasts = self.arima_model.as_ref().unwrap().forecast(horizon)?;

        // Reverse the seasonal differencing to get forecasts in the original scale
        let forecasts =
            self.seasonal_integrate(&diff_forecasts, self.last_values.as_ref().unwrap());

        Ok(forecasts)
    }

    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        if self.arima_model.is_none() {
            return Err(OxiError::from(ARError::NotFitted));
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
    use std::f64::consts::PI;

    #[test]
    fn test_seasonal_difference_and_integrate() {
        // Create a SARIMA model with seasonal period 4 (e.g., quarterly data)
        let model = SARIMAModel::new(1, 0, 0, 1, 1, 0, 4, false).unwrap();

        // Test data: values with a seasonal pattern (period 4)
        // Base: 10, 20, 15, 25, 12, 22, 17, 27, 14, 24, 19, 29
        let data: Vec<f64> = vec![
            10.0, 20.0, 15.0, 25.0, 12.0, 22.0, 17.0, 27.0, 14.0, 24.0, 19.0, 29.0,
        ];

        // Seasonal differencing (D=1, s=4) should give:
        // 12-10=2, 22-20=2, 17-15=2, 27-25=2, 14-12=2, 24-22=2, 19-17=2, 29-27=2
        let diff = model.seasonal_difference(&data, 1, 4);
        assert_eq!(diff, vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);

        // Test seasonal integration
        let forecast_diff = vec![2.0, 2.0, 2.0, 2.0]; // Continuation of the same seasonal pattern

        // For seasonal integration, we need the last original values (including seasonal lags)
        // Let's use the last 8 values: 14, 24, 19, 29, 16, 26, 21, 31
        let last_values = data.clone();

        // Integration should give: 14+2=16, 24+2=26, 19+2=21, 29+2=31
        let forecast = model.seasonal_integrate(&forecast_diff, &last_values);
        assert_eq!(forecast, vec![16.0, 26.0, 21.0, 31.0]);
    }

    #[test]
    fn test_sarima_seasonal_pattern() {
        // Create test data with a seasonal pattern (period 12, like monthly data)
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..48)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();

        // Create synthetic data with a seasonal pattern: base + seasonal component
        // Base value increases slightly each period
        // Seasonal component has 12-period cycle with sin function
        let mut values = Vec::with_capacity(48);
        for i in 0..48 {
            let base = 100.0 + i as f64 * 0.5; // Slight upward trend
            let seasonal = 20.0 * (2.0 * PI * (i % 12) as f64 / 12.0).sin(); // Seasonal component
            values.push(base + seasonal);
        }

        let time_series = TimeSeriesData::new(timestamps, values, "seasonal_series").unwrap();

        // Create and fit a SARIMA(1,0,0)(1,1,0)12 model
        // This handles a slight AR trend with seasonal differencing
        let mut model = SARIMAModel::new(1, 0, 0, 1, 1, 0, 12, true).unwrap();
        model.fit(&time_series).unwrap();

        // Test forecasting
        let forecast_horizon = 12; // Forecast one full seasonal cycle
        let forecasts = model.forecast(forecast_horizon).unwrap();

        // Check that we got the right number of forecasts
        assert_eq!(forecasts.len(), forecast_horizon);

        // For a proper test, we'd compare to known expected values
        // Here we just check that the forecasts follow a reasonable pattern
        for i in 0..12 {
            // The forecast should continue the seasonal pattern
            // Check that the sign of the monthly effect is consistent with past pattern
            let month_idx = i % 12;
            let past_value_for_month = time_series.values[36 + month_idx]; // Use 3rd year as reference
            let expected_trend = past_value_for_month + 0.5 * 12.0; // Adjust for trend increase

            // Allow reasonable deviation
            assert!(
                (forecasts[i] - expected_trend).abs() < 10.0,
                "Forecast should follow the seasonal pattern"
            );
        }
    }
}
