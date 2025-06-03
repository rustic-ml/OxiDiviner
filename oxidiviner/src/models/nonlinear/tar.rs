//! Threshold Autoregressive (TAR) Models for non-linear regime forecasting
//!
//! TAR models capture non-linear dynamics where the autoregressive behavior
//! changes based on a threshold variable. Useful for modeling:
//! - Financial time series with different behavior above/below certain levels
//! - Business cycle analysis with expansion/recession regimes
//! - Volatility clustering with high/low volatility regimes

use crate::core::{Forecaster, ModelEvaluation, OxiError, Result, TimeSeriesData};
use crate::math::metrics::{mae, mape, mse, rmse, smape};

/// Threshold Autoregressive Model
#[derive(Debug, Clone)]
pub struct TARModel {
    name: String,
    threshold: Option<f64>,
    delay: usize,
    ar_orders: Vec<usize>,
    ar_coefficients: Option<Vec<Vec<f64>>>,
    constants: Option<Vec<f64>>,
    residual_variances: Option<Vec<f64>>,
    fitted_values: Option<Vec<f64>>,
    residuals: Option<Vec<f64>>,
    regime_sequence: Option<Vec<usize>>,
    log_likelihood: Option<f64>,
    training_data: Option<TimeSeriesData>,
}

impl TARModel {
    /// Create a new TAR model
    pub fn new(ar_orders: Vec<usize>, delay: usize) -> Result<Self> {
        if ar_orders.is_empty() {
            return Err(OxiError::InvalidParameter(
                "Must specify at least one AR order".to_string(),
            ));
        }

        Ok(Self {
            name: format!("TAR({:?}, {})", ar_orders, delay),
            threshold: None,
            delay,
            ar_orders,
            ar_coefficients: None,
            constants: None,
            residual_variances: None,
            fitted_values: None,
            residuals: None,
            regime_sequence: None,
            log_likelihood: None,
            training_data: None,
        })
    }

    /// Fit the TAR model using grid search for threshold
    pub fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        let n = data.values.len();
        let max_ar_order = *self.ar_orders.iter().max().unwrap();

        if n < max_ar_order + self.delay + 10 {
            return Err(OxiError::DataError(format!(
                "Insufficient data: need at least {} points, got {}",
                max_ar_order + self.delay + 10,
                n
            )));
        }

        // Grid search for optimal threshold
        let (threshold, best_ll) = self.find_optimal_threshold(&data.values)?;
        self.threshold = Some(threshold);

        // Estimate parameters with optimal threshold
        self.estimate_parameters(&data.values, threshold)?;

        // Calculate fitted values and residuals
        self.calculate_fitted_values(&data.values)?;

        self.log_likelihood = Some(best_ll);
        self.training_data = Some(data.clone());

        Ok(())
    }

    /// Forecast future values
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("TAR model not fitted".to_string()))?;

        let threshold = self
            .threshold
            .ok_or_else(|| OxiError::ModelError("TAR model not fitted".to_string()))?;

        let n = training_data.values.len();
        let _max_ar_order = *self.ar_orders.iter().max().unwrap();

        // Initialize with last observations
        let mut extended_series = training_data.values.clone();
        extended_series.reserve(horizon);

        for _ in 0..horizon {
            let current_n = extended_series.len();

            // Determine regime based on threshold variable
            let threshold_var = if current_n > self.delay {
                extended_series[current_n - 1 - self.delay]
            } else {
                extended_series[0]
            };

            let regime = if threshold_var <= threshold { 0 } else { 1 };

            // Forecast next value
            let mut forecast = self.constants.as_ref().unwrap()[regime];

            let ar_order = self.ar_orders[regime];
            for lag in 1..=ar_order {
                if current_n >= lag {
                    forecast += self.ar_coefficients.as_ref().unwrap()[regime][lag - 1]
                        * extended_series[current_n - lag];
                }
            }

            extended_series.push(forecast);
        }

        Ok(extended_series[n..].to_vec())
    }

    /// Get the estimated threshold
    pub fn get_threshold(&self) -> Option<f64> {
        self.threshold
    }

    /// Get regime sequence
    pub fn get_regime_sequence(&self) -> Option<&Vec<usize>> {
        self.regime_sequence.as_ref()
    }

    // Private implementation methods

    fn find_optimal_threshold(&self, data: &[f64]) -> Result<(f64, f64)> {
        let n = data.len();

        // Get potential threshold values (exclude extreme quantiles)
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let start_idx = (n as f64 * 0.15) as usize;
        let end_idx = (n as f64 * 0.85) as usize;

        let mut best_threshold = sorted_data[start_idx];
        let mut best_log_likelihood = f64::NEG_INFINITY;

        // Grid search over potential thresholds
        for i in start_idx..end_idx {
            let threshold = sorted_data[i];

            if let Ok(ll) = self.calculate_log_likelihood_for_threshold(data, threshold) {
                if ll > best_log_likelihood {
                    best_log_likelihood = ll;
                    best_threshold = threshold;
                }
            }
        }

        Ok((best_threshold, best_log_likelihood))
    }

    fn calculate_log_likelihood_for_threshold(&self, data: &[f64], threshold: f64) -> Result<f64> {
        let n = data.len();
        let max_ar_order = *self.ar_orders.iter().max().unwrap();
        let start_idx = max_ar_order + self.delay;

        // Separate data into regimes
        let mut regime1_data = Vec::new();
        let mut regime2_data = Vec::new();

        for t in start_idx..n {
            let threshold_var = data[t - 1 - self.delay];
            if threshold_var <= threshold {
                regime1_data.push((t, 0));
            } else {
                regime2_data.push((t, 1));
            }
        }

        if regime1_data.len() < self.ar_orders[0] + 2 || regime2_data.len() < self.ar_orders[1] + 2
        {
            return Ok(f64::NEG_INFINITY); // Not enough data in one regime
        }

        // Estimate AR parameters for each regime
        let mut log_likelihood = 0.0;

        for (regime_data, regime) in [(regime1_data, 0), (regime2_data, 1)] {
            let ar_order = self.ar_orders[regime];
            let m = regime_data.len();

            // Set up regression matrices
            let mut y = vec![0.0; m];
            let mut x = vec![vec![0.0; ar_order + 1]; m];

            for (i, &(t, _)) in regime_data.iter().enumerate() {
                y[i] = data[t];
                x[i][0] = 1.0; // Constant term

                for lag in 1..=ar_order {
                    if t >= lag {
                        x[i][lag] = data[t - lag];
                    }
                }
            }

            // Simple OLS estimation (in practice, use more robust methods)
            if let Ok((coeffs, residual_var)) = self.ols_regression(&y, &x) {
                // Add to log-likelihood
                log_likelihood -= 0.5 * m as f64 * (2.0 * std::f64::consts::PI * residual_var).ln();

                let mut ss_res = 0.0;
                for (_i, &(t, _)) in regime_data.iter().enumerate() {
                    let mut fitted = coeffs[0];
                    for lag in 1..=ar_order {
                        if t >= lag {
                            fitted += coeffs[lag] * data[t - lag];
                        }
                    }
                    let residual = data[t] - fitted;
                    ss_res += residual * residual;
                }
                log_likelihood -= 0.5 * ss_res / residual_var;
            } else {
                return Ok(f64::NEG_INFINITY);
            }
        }

        Ok(log_likelihood)
    }

    fn estimate_parameters(&mut self, data: &[f64], threshold: f64) -> Result<()> {
        let n = data.len();
        let max_ar_order = *self.ar_orders.iter().max().unwrap();
        let start_idx = max_ar_order + self.delay;

        // Separate data into regimes
        let mut regime_data = vec![Vec::new(); 2];

        for t in start_idx..n {
            let threshold_var = data[t - 1 - self.delay];
            let regime = if threshold_var <= threshold { 0 } else { 1 };
            regime_data[regime].push(t);
        }

        let mut ar_coefficients = vec![Vec::new(); 2];
        let mut constants = vec![0.0; 2];
        let mut residual_variances = vec![0.0; 2];

        // Estimate parameters for each regime
        for regime in 0..2 {
            let ar_order = self.ar_orders[regime];
            let regime_indices = &regime_data[regime];
            let m = regime_indices.len();

            if m < ar_order + 2 {
                return Err(OxiError::DataError(format!(
                    "Insufficient data for regime {}: need at least {} points, got {}",
                    regime,
                    ar_order + 2,
                    m
                )));
            }

            // Set up regression
            let mut y = vec![0.0; m];
            let mut x = vec![vec![0.0; ar_order + 1]; m];

            for (_i, &t) in regime_indices.iter().enumerate() {
                y[_i] = data[t];
                x[_i][0] = 1.0; // Constant term

                for lag in 1..=ar_order {
                    if t >= lag {
                        x[_i][lag] = data[t - lag];
                    }
                }
            }

            // OLS estimation
            let (coeffs, residual_var) = self.ols_regression(&y, &x)?;

            constants[regime] = coeffs[0];
            ar_coefficients[regime] = coeffs[1..].to_vec();
            residual_variances[regime] = residual_var;
        }

        self.ar_coefficients = Some(ar_coefficients);
        self.constants = Some(constants);
        self.residual_variances = Some(residual_variances);

        Ok(())
    }

    fn calculate_fitted_values(&mut self, data: &[f64]) -> Result<()> {
        let n = data.len();
        let max_ar_order = *self.ar_orders.iter().max().unwrap();
        let start_idx = max_ar_order + self.delay;
        let threshold = self.threshold.unwrap();

        let mut fitted_values = vec![0.0; n - start_idx];
        let mut residuals = vec![0.0; n - start_idx];
        let mut regime_sequence = vec![0; n - start_idx];

        for t in start_idx..n {
            let threshold_var = data[t - 1 - self.delay];
            let regime = if threshold_var <= threshold { 0 } else { 1 };
            regime_sequence[t - start_idx] = regime;

            let mut fitted = self.constants.as_ref().unwrap()[regime];
            let ar_order = self.ar_orders[regime];

            for lag in 1..=ar_order {
                if t >= lag {
                    fitted +=
                        self.ar_coefficients.as_ref().unwrap()[regime][lag - 1] * data[t - lag];
                }
            }

            fitted_values[t - start_idx] = fitted;
            residuals[t - start_idx] = data[t] - fitted;
        }

        self.fitted_values = Some(fitted_values);
        self.residuals = Some(residuals);
        self.regime_sequence = Some(regime_sequence);

        Ok(())
    }

    fn ols_regression(&self, y: &[f64], x: &[Vec<f64>]) -> Result<(Vec<f64>, f64)> {
        let n = y.len();
        let k = x[0].len();

        if n < k {
            return Err(OxiError::DataError(format!(
                "Insufficient data: need at least {} observations, got {}",
                k, n
            )));
        }

        // Simple OLS: β = (X'X)^(-1)X'y
        // This is a simplified implementation

        let mut xtx = vec![vec![0.0; k]; k];
        let mut xty = vec![0.0; k];

        // Calculate X'X and X'y
        for i in 0..n {
            for j in 0..k {
                xty[j] += x[i][j] * y[i];
                for l in 0..k {
                    xtx[j][l] += x[i][j] * x[i][l];
                }
            }
        }

        // Solve using Gaussian elimination (simplified)
        let mut coeffs = vec![0.0; k];

        // Forward elimination
        for i in 0..k {
            // Find pivot
            let mut max_row = i;
            for j in i + 1..k {
                if xtx[j][i].abs() > xtx[max_row][i].abs() {
                    max_row = j;
                }
            }

            // Swap rows
            if max_row != i {
                xtx.swap(i, max_row);
                xty.swap(i, max_row);
            }

            // Make diagonal element 1
            let pivot = xtx[i][i];
            if pivot.abs() < 1e-10 {
                return Err(OxiError::ModelError("Matrix is singular".to_string()));
            }

            for j in i..k {
                xtx[i][j] /= pivot;
            }
            xty[i] /= pivot;

            // Eliminate column
            for j in i + 1..k {
                let factor = xtx[j][i];
                for l in i..k {
                    xtx[j][l] -= factor * xtx[i][l];
                }
                xty[j] -= factor * xty[i];
            }
        }

        // Back substitution
        for i in (0..k).rev() {
            coeffs[i] = xty[i];
            for j in i + 1..k {
                coeffs[i] -= xtx[i][j] * coeffs[j];
            }
        }

        // Calculate residual variance
        let mut ss_res = 0.0;
        for i in 0..n {
            let mut fitted = 0.0;
            for j in 0..k {
                fitted += x[i][j] * coeffs[j];
            }
            let residual = y[i] - fitted;
            ss_res += residual * residual;
        }

        let residual_variance = ss_res / (n - k) as f64;

        Ok((coeffs, residual_variance))
    }
}

impl Forecaster for TARModel {
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
            aic: None, // Could be calculated from log_likelihood if available
            bic: None,
        })
    }
}

impl TARModel {
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
