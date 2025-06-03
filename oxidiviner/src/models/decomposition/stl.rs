//! STL (Seasonal-Trend decomposition using Loess) for forecasting components
//!
//! STL decomposes a time series into seasonal, trend, and remainder components
//! using locally weighted regression (loess). Particularly useful for:
//! - Seasonal adjustment and forecasting
//! - Identifying structural breaks and outliers
//! - Understanding long-term trends vs seasonal patterns

use crate::core::{Forecaster, ModelEvaluation, OxiError, Result, TimeSeriesData};
use crate::math::metrics::{mae, mape, mse, rmse, smape};

/// STL Decomposition Model for seasonal time series
#[derive(Debug, Clone)]
pub struct STLModel {
    name: String,
    seasonal_period: usize,
    seasonal_smoother: usize,
    trend_smoother: usize,
    low_pass_smoother: usize,
    seasonal_jump: usize,
    trend_jump: usize,
    low_pass_jump: usize,
    max_iterations: usize,

    // Decomposition results
    trend: Option<Vec<f64>>,
    seasonal: Option<Vec<f64>>,
    remainder: Option<Vec<f64>>,
    fitted_values: Option<Vec<f64>>,
    residuals: Option<Vec<f64>>,

    training_data: Option<TimeSeriesData>,
}

impl STLModel {
    /// Create a new STL model
    ///
    /// # Arguments
    /// * `seasonal_period` - Length of seasonal cycle
    /// * `seasonal_smoother` - Seasonal smoother parameter (odd integer)
    /// * `trend_smoother` - Trend smoother parameter (odd integer, optional)
    pub fn new(
        seasonal_period: usize,
        seasonal_smoother: Option<usize>,
        trend_smoother: Option<usize>,
    ) -> Result<Self> {
        if seasonal_period < 2 {
            return Err(OxiError::InvalidParameter(
                "Seasonal period must be at least 2".to_string(),
            ));
        }

        let seasonal_smoother = seasonal_smoother.unwrap_or_else(|| {
            // Default: smallest odd integer >= 7
            let default = 7;
            if default % 2 == 0 {
                default + 1
            } else {
                default
            }
        });

        let trend_smoother = trend_smoother.unwrap_or_else(|| {
            // Default: smallest odd integer >= ceil(1.5 * period / (1 - 1.5/seasonal_smoother))
            let default = ((1.5 * seasonal_period as f64) / (1.0 - 1.5 / seasonal_smoother as f64))
                .ceil() as usize;
            if default % 2 == 0 {
                default + 1
            } else {
                default
            }
        });

        if seasonal_smoother % 2 == 0 || trend_smoother % 2 == 0 {
            return Err(OxiError::InvalidParameter(
                "Smoother parameters must be odd integers".to_string(),
            ));
        }

        Ok(Self {
            name: format!(
                "STL(period={}, seasonal={}, trend={})",
                seasonal_period, seasonal_smoother, trend_smoother
            ),
            seasonal_period,
            seasonal_smoother,
            trend_smoother,
            low_pass_smoother: seasonal_period, // Default
            seasonal_jump: (seasonal_smoother as f64 * 0.1).ceil() as usize,
            trend_jump: (trend_smoother as f64 * 0.1).ceil() as usize,
            low_pass_jump: (seasonal_period as f64 * 0.1).ceil() as usize,
            max_iterations: 2,

            trend: None,
            seasonal: None,
            remainder: None,
            fitted_values: None,
            residuals: None,
            training_data: None,
        })
    }

    /// Fit the STL model and decompose the time series
    pub fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        let n = data.values.len();
        if n < 2 * self.seasonal_period {
            return Err(OxiError::DataError(format!(
                "Insufficient data: need at least {} points, got {}",
                2 * self.seasonal_period,
                n
            )));
        }

        // Perform STL decomposition
        let (trend, seasonal, remainder) = self.stl_decomposition(&data.values)?;

        // Calculate fitted values and residuals
        let mut fitted_values = Vec::with_capacity(n);
        let mut residuals = Vec::with_capacity(n);

        for i in 0..n {
            let fitted = trend[i] + seasonal[i];
            fitted_values.push(fitted);
            residuals.push(data.values[i] - fitted);
        }

        // Store results
        self.trend = Some(trend);
        self.seasonal = Some(seasonal);
        self.remainder = Some(remainder);
        self.fitted_values = Some(fitted_values);
        self.residuals = Some(residuals);
        self.training_data = Some(data.clone());

        Ok(())
    }

    /// Forecast using decomposed components
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        let trend = self
            .trend
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("STL not fitted".to_string()))?;
        let _seasonal = self
            .seasonal
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("STL not fitted".to_string()))?;

        let _n = trend.len();
        let mut forecasts = Vec::with_capacity(horizon);

        // Forecast trend component (linear extrapolation)
        let trend_forecasts = self.forecast_trend_component(horizon)?;

        // Forecast seasonal component (repeat last complete cycle)
        let seasonal_forecasts = self.forecast_seasonal_component(horizon);

        // Combine components
        for h in 0..horizon {
            forecasts.push(trend_forecasts[h] + seasonal_forecasts[h]);
        }

        Ok(forecasts)
    }

    /// Get decomposed components
    pub fn get_components(&self) -> Option<(&Vec<f64>, &Vec<f64>, &Vec<f64>)> {
        if let (Some(trend), Some(seasonal), Some(remainder)) =
            (&self.trend, &self.seasonal, &self.remainder)
        {
            Some((trend, seasonal, remainder))
        } else {
            None
        }
    }

    /// Get seasonal strength (0 = no seasonality, 1 = pure seasonality)
    pub fn seasonal_strength(&self) -> Result<f64> {
        let remainder = self
            .remainder
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("STL not fitted".to_string()))?;
        let _seasonal = self
            .seasonal
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("STL not fitted".to_string()))?;

        let var_remainder = self.variance(remainder);
        let var_seasonal_remainder = self.variance(
            &_seasonal
                .iter()
                .zip(remainder.iter())
                .map(|(s, r)| s + r)
                .collect::<Vec<f64>>(),
        );

        if var_seasonal_remainder > 0.0 {
            Ok(1.0 - var_remainder / var_seasonal_remainder)
        } else {
            Ok(0.0)
        }
    }

    /// Get trend strength
    pub fn trend_strength(&self) -> Result<f64> {
        let remainder = self
            .remainder
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("STL not fitted".to_string()))?;
        let trend = self
            .trend
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("STL not fitted".to_string()))?;

        let var_remainder = self.variance(remainder);
        let var_trend_remainder = self.variance(
            &trend
                .iter()
                .zip(remainder.iter())
                .map(|(t, r)| t + r)
                .collect::<Vec<f64>>(),
        );

        if var_trend_remainder > 0.0 {
            Ok(1.0 - var_remainder / var_trend_remainder)
        } else {
            Ok(0.0)
        }
    }

    // Private implementation methods

    fn stl_decomposition(&self, data: &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let n = data.len();
        let mut trend = vec![0.0; n];
        let mut seasonal = vec![0.0; n];
        let mut detrended = data.to_vec();

        // Initialize trend using moving average
        self.moving_average(&data, &mut trend)?;

        for _iteration in 0..self.max_iterations {
            // Step 1: Remove trend to get detrended series
            for i in 0..n {
                detrended[i] = data[i] - trend[i];
            }

            // Step 2: Extract seasonal component
            self.extract_seasonal_component(&detrended, &mut seasonal)?;

            // Step 3: Remove seasonal component and smooth to get trend
            let mut seasonally_adjusted = vec![0.0; n];
            for i in 0..n {
                seasonally_adjusted[i] = data[i] - seasonal[i];
            }

            self.loess_smooth(&seasonally_adjusted, &mut trend, self.trend_smoother)?;
        }

        // Calculate remainder
        let mut remainder = vec![0.0; n];
        for i in 0..n {
            remainder[i] = data[i] - trend[i] - seasonal[i];
        }

        Ok((trend, seasonal, remainder))
    }

    fn extract_seasonal_component(&self, detrended: &[f64], seasonal: &mut Vec<f64>) -> Result<()> {
        let n = detrended.len();
        let period = self.seasonal_period;

        // Initialize seasonal component
        seasonal.fill(0.0);

        // Calculate seasonal averages for each position in the cycle
        let mut seasonal_cycle = vec![0.0; period];
        let mut counts = vec![0; period];

        for (i, &value) in detrended.iter().enumerate() {
            let pos = i % period;
            seasonal_cycle[pos] += value;
            counts[pos] += 1;
        }

        // Average and center the seasonal component
        let mut sum = 0.0;
        for i in 0..period {
            if counts[i] > 0 {
                seasonal_cycle[i] /= counts[i] as f64;
                sum += seasonal_cycle[i];
            }
        }

        let mean_seasonal = sum / period as f64;
        for i in 0..period {
            seasonal_cycle[i] -= mean_seasonal;
        }

        // Apply seasonal pattern to full series
        for i in 0..n {
            seasonal[i] = seasonal_cycle[i % period];
        }

        // Smooth seasonal component
        let seasonal_copy = seasonal.clone();
        self.loess_smooth(&seasonal_copy, seasonal, self.seasonal_smoother)?;

        Ok(())
    }

    fn moving_average(&self, data: &[f64], trend: &mut Vec<f64>) -> Result<()> {
        let n = data.len();
        let window = self.seasonal_period;

        for i in 0..n {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(n);

            let sum: f64 = data[start..end].iter().sum();
            trend[i] = sum / (end - start) as f64;
        }

        Ok(())
    }

    fn loess_smooth(&self, input: &[f64], output: &mut Vec<f64>, bandwidth: usize) -> Result<()> {
        let n = input.len();

        for i in 0..n {
            // Simple local regression (simplified LOESS)
            let half_bandwidth = bandwidth / 2;
            let start = i.saturating_sub(half_bandwidth);
            let end = (i + half_bandwidth + 1).min(n);

            // Weighted average with triangular weights
            let mut sum_weights = 0.0;
            let mut weighted_sum = 0.0;

            for j in start..end {
                let distance = (i as i32 - j as i32).abs() as f64;
                let weight = if distance <= half_bandwidth as f64 {
                    1.0 - distance / (half_bandwidth as f64 + 1.0)
                } else {
                    0.0
                };

                weighted_sum += weight * input[j];
                sum_weights += weight;
            }

            output[i] = if sum_weights > 0.0 {
                weighted_sum / sum_weights
            } else {
                input[i]
            };
        }

        Ok(())
    }

    fn forecast_trend_component(&self, horizon: usize) -> Result<Vec<f64>> {
        let trend = self.trend.as_ref().unwrap();
        let _n = trend.len();

        // Simple linear extrapolation using last few points
        let extrapolation_points = (10).min(_n / 2);
        let start_idx = _n - extrapolation_points;

        // Calculate slope using least squares
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for i in 0..extrapolation_points {
            let x = i as f64;
            let y = trend[start_idx + i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let n_points = extrapolation_points as f64;
        let slope = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_x2 - sum_x * sum_x);
        let _intercept = (sum_y - slope * sum_x) / n_points;

        // Generate forecasts
        let last_trend = trend[_n - 1];
        let mut forecasts = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let forecast = last_trend + slope * h as f64;
            forecasts.push(forecast);
        }

        Ok(forecasts)
    }

    fn forecast_seasonal_component(&self, horizon: usize) -> Vec<f64> {
        let seasonal = self.seasonal.as_ref().unwrap();
        let n = seasonal.len();
        let period = self.seasonal_period;

        // Extract the last complete seasonal cycle
        let start_idx = n - (n % period);
        let last_cycle = if start_idx >= period {
            &seasonal[start_idx - period..start_idx]
        } else {
            &seasonal[n - period.min(n)..]
        };

        // Repeat the cycle for forecasting
        let mut forecasts = Vec::with_capacity(horizon);
        for h in 0..horizon {
            forecasts.push(last_cycle[h % last_cycle.len()]);
        }

        forecasts
    }

    fn variance(&self, data: &[f64]) -> f64 {
        let n = data.len() as f64;
        if n <= 1.0 {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        variance
    }
}

impl Forecaster for STLModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        self.fit(data)
    }

    fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        self.forecast(horizon)
    }

    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        let forecasts = self.forecast(test_data.values.len())?;
        let actual = &test_data.values;
        let predicted = &forecasts[..test_data.values.len().min(forecasts.len())];

        Ok(ModelEvaluation {
            model_name: self.name.clone(),
            mae: mae(actual, predicted),
            mse: mse(actual, predicted),
            rmse: rmse(actual, predicted),
            mape: mape(actual, predicted),
            smape: smape(actual, predicted),
            r_squared: self.calculate_r_squared(actual, predicted)?,
            aic: None, // STL doesn't have likelihood-based parameters
            bic: None,
        })
    }
}

impl STLModel {
    fn calculate_r_squared(&self, actual: &[f64], predicted: &[f64]) -> Result<f64> {
        if actual.len() != predicted.len() {
            return Err(OxiError::ModelError(
                "Actual and predicted lengths don't match".to_string(),
            ));
        }

        let mean_actual = actual.iter().sum::<f64>() / actual.len() as f64;
        let ss_tot: f64 = actual.iter().map(|&x| (x - mean_actual).powi(2)).sum();
        let ss_res: f64 = actual
            .iter()
            .zip(predicted.iter())
            .map(|(&a, &p)| (a - p).powi(2))
            .sum();

        if ss_tot > 0.0 {
            Ok(1.0 - ss_res / ss_tot)
        } else {
            Ok(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    fn create_seasonal_data() -> TimeSeriesData {
        let start_time = Utc::now();
        let timestamps: Vec<chrono::DateTime<Utc>> =
            (0..100).map(|i| start_time + Duration::days(i)).collect();

        // Create data with trend and seasonal pattern
        let values: Vec<f64> = (0..100)
            .map(|i| {
                let trend = 100.0 + 0.5 * i as f64;
                let seasonal = 10.0 * ((i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin());
                let noise = rand::random::<f64>() * 2.0 - 1.0;
                trend + seasonal + noise
            })
            .collect();

        TimeSeriesData::new(timestamps, values, "seasonal_series").unwrap()
    }

    #[test]
    fn test_stl_model() {
        let mut model = STLModel::new(12, Some(7), Some(15)).unwrap();
        let data = create_seasonal_data();

        assert!(model.fit(&data).is_ok());
        assert!(model.forecast(10).is_ok());

        // Check that we have decomposed components
        assert!(model.get_components().is_some());
    }

    #[test]
    fn test_seasonal_strength() {
        let mut model = STLModel::new(12, Some(7), Some(15)).unwrap();
        let data = create_seasonal_data();
        model.fit(&data).unwrap();

        let seasonal_strength = model.seasonal_strength().unwrap();
        assert!(seasonal_strength >= 0.0);
        assert!(seasonal_strength <= 1.0);
    }

    #[test]
    fn test_trend_strength() {
        let mut model = STLModel::new(12, Some(7), Some(15)).unwrap();
        let data = create_seasonal_data();
        model.fit(&data).unwrap();

        let trend_strength = model.trend_strength().unwrap();
        assert!(trend_strength >= 0.0);
        assert!(trend_strength <= 1.0);
    }

    #[test]
    fn test_stl_components() {
        let mut model = STLModel::new(12, Some(7), Some(15)).unwrap();
        let data = create_seasonal_data();
        model.fit(&data).unwrap();

        let (trend, seasonal, remainder) = model.get_components().unwrap();
        assert_eq!(trend.len(), data.values.len());
        assert_eq!(seasonal.len(), data.values.len());
        assert_eq!(remainder.len(), data.values.len());

        // Check that components sum to original data (approximately)
        for i in 0..data.values.len() {
            let reconstructed = trend[i] + seasonal[i] + remainder[i];
            let diff = (reconstructed - data.values[i]).abs();
            assert!(diff < 1e-10, "Component reconstruction error too large");
        }
    }
}
