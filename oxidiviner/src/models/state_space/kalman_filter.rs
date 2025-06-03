//! Kalman Filter - State-space models for dynamic forecasting
//!
//! The Kalman filter is a state-space model that provides optimal estimates of unknown
//! variables in the presence of uncertain or noisy measurements. It's particularly useful
//! for financial time series forecasting where:
//! - The underlying process has hidden states (trend, seasonality, cycles)
//! - Observations are noisy or incomplete
//! - Real-time updating is required as new data arrives
//!
//! ## State-Space Representation
//!
//! State equation: x[t] = F * x[t-1] + B * u[t] + w[t]
//! Observation equation: y[t] = H * x[t] + v[t]
//!
//! Where:
//! - x[t] is the state vector at time t
//! - y[t] is the observation vector at time t
//! - F is the state transition matrix
//! - H is the observation matrix
//! - B is the control matrix (optional)
//! - u[t] is the control vector (optional)
//! - w[t] ~ N(0, Q) is process noise
//! - v[t] ~ N(0, R) is observation noise

use crate::core::{Forecaster, ModelEvaluation, OxiError, Result, TimeSeriesData};
use crate::math::metrics::{mae, mape, mse, rmse, smape};
use chrono::{DateTime, Utc};
use std::f64::consts::PI;

/// Kalman Filter for dynamic time series forecasting
///
/// This implementation supports multiple variants:
/// - Local Level Model (random walk with noise)
/// - Local Linear Trend Model (trend + noise)
/// - Seasonal Model (trend + seasonal + noise)
/// - Custom state-space models
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    /// Model name for identification
    name: String,
    /// State dimension
    state_dim: usize,
    /// Observation dimension
    obs_dim: usize,
    /// State transition matrix F (state_dim x state_dim)
    state_transition: Vec<Vec<f64>>,
    /// Observation matrix H (obs_dim x state_dim)
    observation: Vec<Vec<f64>>,
    /// Process noise covariance Q (state_dim x state_dim)
    process_noise: Vec<Vec<f64>>,
    /// Observation noise covariance R (obs_dim x obs_dim)
    observation_noise: Vec<Vec<f64>>,
    /// Current state estimate
    state_estimate: Option<Vec<f64>>,
    /// Current state covariance
    state_covariance: Option<Vec<Vec<f64>>>,
    /// Fitted values
    fitted_values: Option<Vec<f64>>,
    /// Residuals
    residuals: Option<Vec<f64>>,
    /// Log-likelihood
    log_likelihood: Option<f64>,
    /// State history for analysis
    state_history: Option<Vec<Vec<f64>>>,
    /// Innovation sequence
    innovations: Option<Vec<f64>>,
    /// Training data for forecasting
    training_data: Option<TimeSeriesData>,
}

impl KalmanFilter {
    /// Create a Local Level Model (random walk with noise)
    ///
    /// State: [level]
    /// Observation: level + noise
    ///
    /// # Arguments
    /// * `process_variance` - Variance of level innovations
    /// * `observation_variance` - Variance of observation noise
    pub fn local_level(process_variance: f64, observation_variance: f64) -> Result<Self> {
        if process_variance <= 0.0 || observation_variance <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Variances must be positive".to_string(),
            ));
        }

        Ok(Self {
            name: "Local Level Kalman Filter".to_string(),
            state_dim: 1,
            obs_dim: 1,
            state_transition: vec![vec![1.0]], // Random walk
            observation: vec![vec![1.0]],      // Direct observation
            process_noise: vec![vec![process_variance]],
            observation_noise: vec![vec![observation_variance]],
            state_estimate: None,
            state_covariance: None,
            fitted_values: None,
            residuals: None,
            log_likelihood: None,
            state_history: None,
            innovations: None,
            training_data: None,
        })
    }

    /// Create a Local Linear Trend Model
    ///
    /// State: [level, trend]
    /// Observation: level + noise
    ///
    /// # Arguments
    /// * `level_variance` - Variance of level innovations
    /// * `trend_variance` - Variance of trend innovations
    /// * `observation_variance` - Variance of observation noise
    pub fn local_linear_trend(
        level_variance: f64,
        trend_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        if level_variance <= 0.0 || trend_variance <= 0.0 || observation_variance <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Variances must be positive".to_string(),
            ));
        }

        Ok(Self {
            name: "Local Linear Trend Kalman Filter".to_string(),
            state_dim: 2,
            obs_dim: 1,
            state_transition: vec![
                vec![1.0, 1.0], // level[t] = level[t-1] + trend[t-1]
                vec![0.0, 1.0], // trend[t] = trend[t-1]
            ],
            observation: vec![vec![1.0, 0.0]], // Observe level only
            process_noise: vec![vec![level_variance, 0.0], vec![0.0, trend_variance]],
            observation_noise: vec![vec![observation_variance]],
            state_estimate: None,
            state_covariance: None,
            fitted_values: None,
            residuals: None,
            log_likelihood: None,
            state_history: None,
            innovations: None,
            training_data: None,
        })
    }

    /// Create a Seasonal Model with trend
    ///
    /// State: [level, trend, seasonal_1, ..., seasonal_s-1]
    /// where s is the seasonal period
    ///
    /// # Arguments
    /// * `level_variance` - Variance of level innovations
    /// * `trend_variance` - Variance of trend innovations
    /// * `seasonal_variance` - Variance of seasonal innovations
    /// * `observation_variance` - Variance of observation noise
    /// * `seasonal_period` - Length of seasonal cycle
    pub fn seasonal_model(
        level_variance: f64,
        trend_variance: f64,
        seasonal_variance: f64,
        observation_variance: f64,
        seasonal_period: usize,
    ) -> Result<Self> {
        if level_variance <= 0.0
            || trend_variance <= 0.0
            || seasonal_variance <= 0.0
            || observation_variance <= 0.0
        {
            return Err(OxiError::InvalidParameter(
                "Variances must be positive".to_string(),
            ));
        }

        if seasonal_period < 2 {
            return Err(OxiError::InvalidParameter(
                "Seasonal period must be at least 2".to_string(),
            ));
        }

        let state_dim = 2 + seasonal_period - 1; // level + trend + (s-1) seasonal states

        // Build state transition matrix
        let mut state_transition = vec![vec![0.0; state_dim]; state_dim];

        // Level and trend transitions
        state_transition[0][0] = 1.0; // level[t] = level[t-1] + trend[t-1] + seasonal[t-s]
        state_transition[0][1] = 1.0;
        state_transition[0][2] = 1.0; // Add first seasonal component
        state_transition[1][1] = 1.0; // trend[t] = trend[t-1]

        // Seasonal transitions (sum-to-zero constraint)
        for i in 2..state_dim - 1 {
            state_transition[i + 1][i] = 1.0; // Shift seasonal states
        }
        // Sum-to-zero constraint for seasonals
        for i in 2..state_dim {
            state_transition[2][i] = -1.0;
        }

        // Observation matrix [1, 0, 1, 0, ..., 0]
        let mut observation = vec![vec![0.0; state_dim]];
        observation[0][0] = 1.0; // Observe level
        observation[0][2] = 1.0; // Plus current seasonal

        // Process noise covariance
        let mut process_noise = vec![vec![0.0; state_dim]; state_dim];
        process_noise[0][0] = level_variance;
        process_noise[1][1] = trend_variance;
        process_noise[2][2] = seasonal_variance;

        Ok(Self {
            name: format!("Seasonal Kalman Filter (period={})", seasonal_period),
            state_dim,
            obs_dim: 1,
            state_transition,
            observation,
            process_noise,
            observation_noise: vec![vec![observation_variance]],
            state_estimate: None,
            state_covariance: None,
            fitted_values: None,
            residuals: None,
            log_likelihood: None,
            state_history: None,
            innovations: None,
            training_data: None,
        })
    }

    /// Fit the Kalman filter to time series data
    pub fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        let n = data.values.len();
        if n < self.state_dim + 1 {
            return Err(OxiError::DataError(format!(
                "Insufficient data: need at least {} points, got {}",
                self.state_dim + 1,
                n
            )));
        }

        // Initialize state and covariance
        let mut state = self.initialize_state(&data.values)?;
        let mut covariance = self.initialize_covariance()?;

        let mut fitted_values = Vec::with_capacity(n);
        let mut residuals = Vec::with_capacity(n);
        let mut state_history = Vec::with_capacity(n);
        let mut innovations = Vec::with_capacity(n);
        let mut log_likelihood = 0.0;

        for (t, &observation) in data.values.iter().enumerate() {
            // Prediction step
            if t > 0 {
                state = self.predict_state(&state)?;
                covariance = self.predict_covariance(&covariance)?;
            }

            // Update step
            let (updated_state, updated_covariance, innovation, innovation_var, fitted) =
                self.update_step(&state, &covariance, observation)?;

            state = updated_state;
            covariance = updated_covariance;

            fitted_values.push(fitted);
            residuals.push(observation - fitted);
            innovations.push(innovation);
            state_history.push(state.clone());

            // Update log-likelihood
            if innovation_var > 0.0 {
                log_likelihood -= 0.5
                    * ((2.0 * PI).ln() + innovation_var.ln() + innovation.powi(2) / innovation_var);
            }
        }

        // Store results
        self.state_estimate = Some(state);
        self.state_covariance = Some(covariance);
        self.fitted_values = Some(fitted_values);
        self.residuals = Some(residuals);
        self.log_likelihood = Some(log_likelihood);
        self.state_history = Some(state_history);
        self.innovations = Some(innovations);
        self.training_data = Some(data.clone());

        Ok(())
    }

    /// Forecast future values
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        let state = self
            .state_estimate
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("Kalman Filter not fitted".to_string()))?;
        let covariance = self
            .state_covariance
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("Kalman Filter not fitted".to_string()))?;

        let mut forecasts = Vec::with_capacity(horizon);
        let mut current_state = state.clone();
        let mut current_covariance = covariance.clone();

        for _ in 0..horizon {
            // Predict next state
            current_state = self.predict_state(&current_state)?;
            current_covariance = self.predict_covariance(&current_covariance)?;

            // Forecast observation
            let forecast = self.matrix_vector_multiply(&self.observation, &current_state)?[0];
            forecasts.push(forecast);
        }

        Ok(forecasts)
    }

    /// Forecast with confidence intervals
    pub fn forecast_with_intervals(
        &self,
        horizon: usize,
        confidence: f64,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(OxiError::InvalidParameter(
                "Confidence must be between 0 and 1".to_string(),
            ));
        }

        let state = self
            .state_estimate
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("Kalman Filter not fitted".to_string()))?;
        let covariance = self
            .state_covariance
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("Kalman Filter not fitted".to_string()))?;

        let mut forecasts = Vec::with_capacity(horizon);
        let mut lower_bounds = Vec::with_capacity(horizon);
        let mut upper_bounds = Vec::with_capacity(horizon);

        let mut current_state = state.clone();
        let mut current_covariance = covariance.clone();

        // Critical value for confidence interval
        let z = Self::normal_quantile((1.0 + confidence) / 2.0);

        for _ in 0..horizon {
            // Predict next state
            current_state = self.predict_state(&current_state)?;
            current_covariance = self.predict_covariance(&current_covariance)?;

            // Forecast observation
            let forecast = self.matrix_vector_multiply(&self.observation, &current_state)?[0];

            // Calculate forecast variance
            let h_p_ht = self.matrix_multiply_transpose(
                &self.observation,
                &current_covariance,
                &self.observation,
            )?;
            let forecast_var = h_p_ht[0][0] + self.observation_noise[0][0];
            let forecast_std = forecast_var.sqrt();

            forecasts.push(forecast);
            lower_bounds.push(forecast - z * forecast_std);
            upper_bounds.push(forecast + z * forecast_std);
        }

        Ok((forecasts, lower_bounds, upper_bounds))
    }

    /// Get the current state estimate
    pub fn get_state(&self) -> Option<&Vec<f64>> {
        self.state_estimate.as_ref()
    }

    /// Get the state history
    pub fn get_state_history(&self) -> Option<&Vec<Vec<f64>>> {
        self.state_history.as_ref()
    }

    /// Get innovation sequence
    pub fn get_innovations(&self) -> Option<&Vec<f64>> {
        self.innovations.as_ref()
    }

    /// Perform Ljung-Box test on innovations for model validation
    pub fn ljung_box_test(&self, lags: usize) -> Result<(f64, f64)> {
        let innovations = self
            .innovations
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("Kalman Filter not fitted".to_string()))?;

        if innovations.len() < lags + 1 {
            return Err(OxiError::DataError(format!(
                "Insufficient data for Ljung-Box test: need at least {} points, got {}",
                lags + 1,
                innovations.len()
            )));
        }

        let n = innovations.len() as f64;
        let mut statistic = 0.0;

        for k in 1..=lags {
            let autocorr = self.calculate_autocorrelation(innovations, k)?;
            statistic += autocorr.powi(2) / (n - k as f64);
        }

        statistic *= n * (n + 2.0);

        // Chi-square critical value (approximate)
        let critical_value = Self::chi_square_quantile(0.95, lags as f64);

        Ok((statistic, critical_value))
    }

    // Private helper methods

    fn initialize_state(&self, data: &[f64]) -> Result<Vec<f64>> {
        let mut state = vec![0.0; self.state_dim];

        match self.state_dim {
            1 => {
                // Local level: initialize with first observation
                state[0] = data[0];
            }
            2 => {
                // Local linear trend: initialize level and trend
                state[0] = data[0];
                if data.len() > 1 {
                    state[1] = data[1] - data[0]; // Initial trend
                }
            }
            _ => {
                // Seasonal model: initialize with decomposition
                state[0] = data.iter().sum::<f64>() / data.len() as f64; // Mean level
                if self.state_dim > 1 {
                    state[1] = 0.0; // Initial trend
                }
                // Initialize seasonal components (simplified)
                for i in 2..self.state_dim {
                    state[i] = 0.0;
                }
            }
        }

        Ok(state)
    }

    fn initialize_covariance(&self) -> Result<Vec<Vec<f64>>> {
        let mut covariance = vec![vec![0.0; self.state_dim]; self.state_dim];

        // Initialize with large diagonal values for diffuse prior
        for i in 0..self.state_dim {
            covariance[i][i] = 1000.0;
        }

        Ok(covariance)
    }

    fn predict_state(&self, state: &[f64]) -> Result<Vec<f64>> {
        self.matrix_vector_multiply(&self.state_transition, state)
    }

    fn predict_covariance(&self, covariance: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        // P = F * P * F' + Q
        let fp = self.matrix_multiply(&self.state_transition, covariance)?;
        let fpf =
            self.matrix_multiply_transpose(&fp, &self.state_transition, &self.state_transition)?;
        self.matrix_add(&fpf, &self.process_noise)
    }

    fn update_step(
        &self,
        state: &[f64],
        covariance: &[Vec<f64>],
        observation: f64,
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>, f64, f64, f64)> {
        // Innovation and its covariance
        let h_state = self.matrix_vector_multiply(&self.observation, state)?;
        let innovation = observation - h_state[0];

        let h_p = self.matrix_multiply(&self.observation, covariance)?;
        let h_p_ht = self.matrix_multiply_transpose(&h_p, &self.observation, &self.observation)?;
        let innovation_cov = h_p_ht[0][0] + self.observation_noise[0][0];

        if innovation_cov <= 0.0 {
            return Err(OxiError::ModelError(
                "Innovation covariance is not positive".to_string(),
            ));
        }

        // Kalman gain
        let p_ht =
            self.matrix_multiply_transpose(covariance, &self.observation, &self.observation)?;
        let mut kalman_gain = vec![0.0; self.state_dim];
        for i in 0..self.state_dim {
            kalman_gain[i] = p_ht[i][0] / innovation_cov;
        }

        // Updated state
        let mut updated_state = state.to_vec();
        for i in 0..self.state_dim {
            updated_state[i] += kalman_gain[i] * innovation;
        }

        // Updated covariance: P = (I - K*H) * P
        let mut kh = vec![vec![0.0; self.state_dim]; self.state_dim];
        for i in 0..self.state_dim {
            for j in 0..self.state_dim {
                kh[i][j] = kalman_gain[i] * self.observation[0][j];
            }
        }

        let mut identity = vec![vec![0.0; self.state_dim]; self.state_dim];
        for i in 0..self.state_dim {
            identity[i][i] = 1.0;
        }

        let ikh = self.matrix_subtract(&identity, &kh)?;
        let updated_covariance = self.matrix_multiply(&ikh, covariance)?;

        let fitted = h_state[0];

        Ok((
            updated_state,
            updated_covariance,
            innovation,
            innovation_cov,
            fitted,
        ))
    }

    // Matrix operations

    fn matrix_vector_multiply(&self, matrix: &[Vec<f64>], vector: &[f64]) -> Result<Vec<f64>> {
        if matrix[0].len() != vector.len() {
            return Err(OxiError::ModelError(
                "Matrix-vector dimension mismatch".to_string(),
            ));
        }

        let mut result = vec![0.0; matrix.len()];
        for i in 0..matrix.len() {
            for j in 0..vector.len() {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        Ok(result)
    }

    fn matrix_multiply(&self, a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if a[0].len() != b.len() {
            return Err(OxiError::ModelError(
                "Matrix multiplication dimension mismatch".to_string(),
            ));
        }

        let mut result = vec![vec![0.0; b[0].len()]; a.len()];
        for i in 0..a.len() {
            for j in 0..b[0].len() {
                for k in 0..b.len() {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        Ok(result)
    }

    fn matrix_multiply_transpose(
        &self,
        a: &[Vec<f64>],
        b: &[Vec<f64>],
        c: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>> {
        // Calculate A * B * C^T
        let ab = self.matrix_multiply(a, b)?;
        let mut result = vec![vec![0.0; c.len()]; ab.len()];

        for i in 0..ab.len() {
            for j in 0..c.len() {
                for k in 0..c[0].len() {
                    result[i][j] += ab[i][k] * c[j][k]; // Note: c[j][k] instead of c[k][j] for transpose
                }
            }
        }
        Ok(result)
    }

    fn matrix_add(&self, a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if a.len() != b.len() || a[0].len() != b[0].len() {
            return Err(OxiError::ModelError(
                "Matrix addition dimension mismatch".to_string(),
            ));
        }

        let mut result = vec![vec![0.0; a[0].len()]; a.len()];
        for i in 0..a.len() {
            for j in 0..a[0].len() {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        Ok(result)
    }

    fn matrix_subtract(&self, a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if a.len() != b.len() || a[0].len() != b[0].len() {
            return Err(OxiError::ModelError(
                "Matrix subtraction dimension mismatch".to_string(),
            ));
        }

        let mut result = vec![vec![0.0; a[0].len()]; a.len()];
        for i in 0..a.len() {
            for j in 0..a[0].len() {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        Ok(result)
    }

    fn calculate_autocorrelation(&self, data: &[f64], lag: usize) -> Result<f64> {
        if data.len() <= lag {
            return Ok(0.0);
        }

        let n = data.len();
        let mean = data.iter().sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            denominator += (data[i] - mean).powi(2);
        }

        for i in lag..n {
            numerator += (data[i] - mean) * (data[i - lag] - mean);
        }

        if denominator > 0.0 {
            Ok(numerator / denominator)
        } else {
            Ok(0.0)
        }
    }

    fn normal_quantile(p: f64) -> f64 {
        // Approximate inverse normal CDF using Beasley-Springer-Moro algorithm
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        if p == 0.5 {
            return 0.0;
        }

        let a0 = 2.515517;
        let a1 = 0.802853;
        let a2 = 0.010328;
        let b1 = 1.432788;
        let b2 = 0.189269;
        let b3 = 0.001308;

        let sign = if p > 0.5 { 1.0 } else { -1.0 };
        let x = if p > 0.5 { 1.0 - p } else { p };

        let t = (-2.0 * x.ln()).sqrt();
        let numerator = a0 + a1 * t + a2 * t * t;
        let denominator = 1.0 + b1 * t + b2 * t * t + b3 * t * t * t;

        sign * (t - numerator / denominator)
    }

    fn chi_square_quantile(p: f64, df: f64) -> f64 {
        // Simplified approximation for chi-square quantile
        // For more accuracy, would need proper implementation
        if df <= 0.0 {
            return 0.0;
        }

        // Wilson-Hilferty approximation
        let h = 2.0 / (9.0 * df);
        let z = Self::normal_quantile(p);

        df * (1.0 - h + z * (h.sqrt())).powi(3)
    }
}

impl Forecaster for KalmanFilter {
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
            aic: self
                .log_likelihood
                .map(|ll| -2.0 * ll + 2.0 * (self.state_dim * 2) as f64),
            bic: self.log_likelihood.map(|ll| {
                let n = self.training_data.as_ref().map_or(0, |td| td.values.len());
                -2.0 * ll + (self.state_dim * 2) as f64 * (n as f64).ln()
            }),
        })
    }
}

impl KalmanFilter {
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

    fn create_test_data() -> TimeSeriesData {
        let start_time = Utc::now();
        let timestamps: Vec<DateTime<Utc>> =
            (0..100).map(|i| start_time + Duration::days(i)).collect();

        // Create deterministic data with trend and noise
        let values: Vec<f64> = (0..100)
            .map(|i| {
                let trend = 100.0 + 0.5 * i as f64;
                let seasonal = (i as f64 * 0.1).sin() * 2.0;
                let noise = (i as f64 * 0.137).sin() * 0.5; // Deterministic noise
                trend + seasonal + noise
            })
            .collect();

        TimeSeriesData::new(timestamps, values, "test_series").unwrap()
    }

    #[test]
    fn test_local_level_model() {
        let mut model = KalmanFilter::local_level(1.0, 0.5).unwrap();
        let data = create_test_data();

        assert!(model.fit(&data).is_ok());
        assert!(model.forecast(10).is_ok());
    }

    #[test]
    fn test_local_linear_trend_model() {
        let model = KalmanFilter::local_linear_trend(1.0, 0.1, 0.5).unwrap();

        // Test that the model was created with correct dimensions
        assert_eq!(model.state_dim, 2);
        assert_eq!(model.name, "Local Linear Trend Kalman Filter");

        // Test basic properties
        assert_eq!(model.state_transition.len(), 2);
        assert_eq!(model.observation.len(), 1);
        assert_eq!(model.process_noise.len(), 2);
        assert_eq!(model.observation_noise.len(), 1);
    }

    #[test]
    fn test_seasonal_model() {
        let model = KalmanFilter::seasonal_model(1.0, 0.1, 0.5, 0.2, 7).unwrap();

        // Test that the model was created with correct dimensions
        assert_eq!(model.state_dim, 8); // 2 + 7 - 1 = 8 (level + trend + 6 seasonal states)
        assert!(model.name.contains("Seasonal Kalman Filter"));

        // Test basic properties
        assert_eq!(model.state_transition.len(), 8);
        assert_eq!(model.observation.len(), 1);
        assert_eq!(model.process_noise.len(), 8);
        assert_eq!(model.observation_noise.len(), 1);
    }

    #[test]
    fn test_forecast_with_intervals() {
        let mut model = KalmanFilter::local_level(1.0, 0.5).unwrap();
        let data = create_test_data();
        model.fit(&data).unwrap();

        let result = model.forecast_with_intervals(5, 0.95);
        assert!(result.is_ok());

        let (forecasts, lower, upper) = result.unwrap();
        assert_eq!(forecasts.len(), 5);
        assert_eq!(lower.len(), 5);
        assert_eq!(upper.len(), 5);

        // Check that intervals make sense
        for i in 0..5 {
            assert!(lower[i] <= forecasts[i]);
            assert!(forecasts[i] <= upper[i]);
        }
    }

    #[test]
    fn test_ljung_box_test() {
        let mut model = KalmanFilter::local_level(1.0, 0.5).unwrap();
        let data = create_test_data();
        model.fit(&data).unwrap();

        let result = model.ljung_box_test(10);
        assert!(result.is_ok());

        let (statistic, critical_value) = result.unwrap();
        assert!(statistic >= 0.0);
        assert!(critical_value > 0.0);
    }
}
