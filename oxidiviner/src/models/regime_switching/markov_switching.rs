//! Markov Regime-Switching Models for different market states
//!
//! Regime-switching models capture the idea that financial time series exhibit
//! different behaviors in different time periods (regimes). These models are
//! particularly useful for modeling:
//! - Bull vs Bear markets
//! - High vs Low volatility periods
//! - Normal vs Crisis periods
//! - Different monetary policy regimes
//!
//! ## Model Specification
//!
//! The basic two-regime model:
//! y[t] = μ[S[t]] + σ[S[t]] * ε[t]
//!
//! Where:
//! - S[t] is the regime at time t (hidden Markov chain)
//! - μ[S[t]] is the regime-dependent mean
//! - σ[S[t]] is the regime-dependent volatility
//! - ε[t] ~ N(0,1) is white noise
//!
//! The regime S[t] follows a Markov chain with transition probabilities:
//! P(S[t] = j | S[t-1] = i) = p[i,j]

use crate::core::{Forecaster, ModelEvaluation, OxiError, Result, TimeSeriesData};
use crate::math::metrics::{mae, mape, mse, rmse, smape};
use std::f64::consts::{E, PI};

// Type aliases for complex types from e_step and m_step
pub type XiMatrix = Vec<Vec<Vec<f64>>>; // P(S_{t-1}=i, S_t=j | Y_1, ..., Y_T)
pub type GammaMatrix = Vec<Vec<f64>>; // P(S_t=j | Y_1, ..., Y_T)

/// Markov Regime-Switching Model
///
/// This implementation supports:
/// - Two-regime models (most common)
/// - Three-regime models (bull/neutral/bear markets)
/// - Custom number of regimes
/// - Regime-dependent means and variances
/// - Time-varying transition probabilities (optional)
#[derive(Debug, Clone)]
pub struct MarkovSwitchingModel {
    /// Model name
    name: String,
    /// Number of regimes
    num_regimes: usize,
    /// Maximum number of EM iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Regime-dependent means
    means: Option<Vec<f64>>,
    /// Regime-dependent standard deviations
    std_devs: Option<Vec<f64>>,
    /// Transition probability matrix (regime i to regime j)
    transition_matrix: Option<Vec<Vec<f64>>>,
    /// Initial regime probabilities
    initial_probs: Option<Vec<f64>>,
    /// Fitted values
    fitted_values: Option<Vec<f64>>,
    /// Residuals
    residuals: Option<Vec<f64>>,
    /// Regime probabilities for each time point
    regime_probabilities: Option<Vec<Vec<f64>>>,
    /// Most likely regime sequence (Viterbi path)
    regime_sequence: Option<Vec<usize>>,
    /// Log-likelihood
    log_likelihood: Option<f64>,
    /// AIC and BIC
    information_criteria: Option<(f64, f64)>,
    /// Training data
    training_data: Option<TimeSeriesData>,
    /// Converged flag
    converged: bool,
}

impl MarkovSwitchingModel {
    /// Create a new two-regime switching model
    ///
    /// # Arguments
    /// * `max_iterations` - Maximum EM iterations (default: 1000)
    /// * `tolerance` - Convergence tolerance (default: 1e-6)
    pub fn two_regime(max_iterations: Option<usize>, tolerance: Option<f64>) -> Self {
        Self {
            name: "Two-Regime Markov Switching Model".to_string(),
            num_regimes: 2,
            max_iterations: max_iterations.unwrap_or(1000),
            tolerance: tolerance.unwrap_or(1e-6),
            means: None,
            std_devs: None,
            transition_matrix: None,
            initial_probs: None,
            fitted_values: None,
            residuals: None,
            regime_probabilities: None,
            regime_sequence: None,
            log_likelihood: None,
            information_criteria: None,
            training_data: None,
            converged: false,
        }
    }

    /// Create a three-regime switching model (e.g., bear/neutral/bull)
    pub fn three_regime(max_iterations: Option<usize>, tolerance: Option<f64>) -> Self {
        Self {
            name: "Three-Regime Markov Switching Model".to_string(),
            num_regimes: 3,
            max_iterations: max_iterations.unwrap_or(1000),
            tolerance: tolerance.unwrap_or(1e-6),
            means: None,
            std_devs: None,
            transition_matrix: None,
            initial_probs: None,
            fitted_values: None,
            residuals: None,
            regime_probabilities: None,
            regime_sequence: None,
            log_likelihood: None,
            information_criteria: None,
            training_data: None,
            converged: false,
        }
    }

    /// Create a custom n-regime switching model
    pub fn n_regime(
        n: usize,
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
    ) -> Result<Self> {
        if n < 2 {
            return Err(OxiError::InvalidParameter(
                "Number of regimes must be at least 2".to_string(),
            ));
        }

        Ok(Self {
            name: format!("{}-Regime Markov Switching Model", n),
            num_regimes: n,
            max_iterations: max_iterations.unwrap_or(1000),
            tolerance: tolerance.unwrap_or(1e-6),
            means: None,
            std_devs: None,
            transition_matrix: None,
            initial_probs: None,
            fitted_values: None,
            residuals: None,
            regime_probabilities: None,
            regime_sequence: None,
            log_likelihood: None,
            information_criteria: None,
            training_data: None,
            converged: false,
        })
    }

    /// Fit the model using the Expectation-Maximization algorithm
    pub fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        let n = data.values.len();
        if n < self.num_regimes * 3 {
            return Err(OxiError::DataError(format!(
                "Insufficient data: need at least {} points, got {}",
                self.num_regimes * 3,
                n
            )));
        }

        // Initialize parameters
        self.initialize_parameters(&data.values)?;

        let mut log_likelihood = f64::NEG_INFINITY;

        for iteration in 0..self.max_iterations {
            // E-step: Calculate regime probabilities
            let (xi, gamma) = self.e_step(&data.values)?;

            // M-step: Update parameters
            self.m_step(&data.values, &xi, &gamma)?;

            // Calculate new log-likelihood
            let new_log_likelihood = self.calculate_log_likelihood(&data.values)?;

            // Check convergence
            if (new_log_likelihood - log_likelihood).abs() < self.tolerance {
                self.converged = true;
                break;
            }

            log_likelihood = new_log_likelihood;

            if iteration == self.max_iterations - 1 {
                eprintln!(
                    "Warning: EM algorithm did not converge after {} iterations",
                    self.max_iterations
                );
            }
        }

        // Calculate fitted values and residuals
        let regime_probs = self.forward_backward(&data.values)?;
        let mut fitted_values = Vec::with_capacity(n);
        let mut residuals = Vec::with_capacity(n);
        let means_vec = self.means.as_ref().unwrap();

        for (t, current_regime_prob_row) in regime_probs.iter().enumerate().take(n) {
            let fitted: f64 = current_regime_prob_row
                .iter()
                .zip(means_vec.iter())
                .map(|(&prob, &mean_val)| prob * mean_val)
                .sum();
            fitted_values.push(fitted);
            residuals.push(data.values[t] - fitted);
        }

        // Calculate most likely regime sequence using Viterbi algorithm
        let regime_sequence = self.viterbi(&data.values)?;

        // Calculate information criteria
        let num_params = self.num_regimes * 2 + self.num_regimes * (self.num_regimes - 1);
        let aic = -2.0 * log_likelihood + 2.0 * num_params as f64;
        let bic = -2.0 * log_likelihood + (num_params as f64) * (n as f64).ln();

        // Store results
        self.fitted_values = Some(fitted_values);
        self.residuals = Some(residuals);
        self.regime_probabilities = Some(regime_probs);
        self.regime_sequence = Some(regime_sequence);
        self.log_likelihood = Some(log_likelihood);
        self.information_criteria = Some((aic, bic));
        self.training_data = Some(data.clone());

        Ok(())
    }

    /// Forecast future values with regime uncertainty
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        let means = self
            .means
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("Markov Switching Model not fitted".to_string()))?;
        let transition_matrix = self
            .transition_matrix
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("Markov Switching Model not fitted".to_string()))?;

        // Get the final regime probabilities
        let final_probs = self
            .regime_probabilities
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("Markov Switching Model not fitted".to_string()))?
            .last()
            .unwrap();

        let mut forecasts = Vec::with_capacity(horizon);
        let mut current_probs = final_probs.clone();

        for _ in 0..horizon {
            // Forecast as weighted average of regime means
            let forecast = (0..self.num_regimes)
                .map(|i| current_probs[i] * means[i])
                .sum();
            forecasts.push(forecast);

            // Update regime probabilities using transition matrix
            let mut new_probs_vec = vec![0.0; self.num_regimes];
            for (j_idx, new_prob_val_ref) in
                new_probs_vec.iter_mut().enumerate().take(self.num_regimes)
            {
                for i_idx in 0..self.num_regimes {
                    *new_prob_val_ref += current_probs[i_idx] * transition_matrix[i_idx][j_idx];
                }
            }
            current_probs = new_probs_vec;
        }

        Ok(forecasts)
    }

    /// Forecast with regime-specific predictions
    pub fn forecast_by_regime(&self, horizon: usize) -> Result<Vec<Vec<f64>>> {
        let means = self
            .means
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("Markov Switching Model not fitted".to_string()))?;

        let mut regime_forecasts = vec![vec![0.0; horizon]; self.num_regimes];

        for regime in 0..self.num_regimes {
            for h in 0..horizon {
                regime_forecasts[regime][h] = means[regime];
            }
        }

        Ok(regime_forecasts)
    }

    /// Get regime probabilities for each time point
    pub fn get_regime_probabilities(&self) -> Option<&Vec<Vec<f64>>> {
        self.regime_probabilities.as_ref()
    }

    /// Get the most likely regime sequence
    pub fn get_regime_sequence(&self) -> Option<&Vec<usize>> {
        self.regime_sequence.as_ref()
    }

    /// Get regime parameters
    pub fn get_regime_parameters(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        if let (Some(means), Some(std_devs)) = (&self.means, &self.std_devs) {
            Some((means.clone(), std_devs.clone()))
        } else {
            None
        }
    }

    /// Get transition matrix
    pub fn get_transition_matrix(&self) -> Option<&Vec<Vec<f64>>> {
        self.transition_matrix.as_ref()
    }

    /// Calculate regime duration statistics
    pub fn regime_duration_stats(&self) -> Result<Vec<f64>> {
        let transition_matrix = self
            .transition_matrix
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("Markov Switching Model not fitted".to_string()))?;

        let mut expected_durations = Vec::with_capacity(self.num_regimes);

        for (regime_idx, transition_matrix_row) in
            transition_matrix.iter().enumerate().take(self.num_regimes)
        {
            let persistence = transition_matrix_row[regime_idx];
            if persistence < 1.0 {
                expected_durations.push(1.0 / (1.0 - persistence));
            } else {
                expected_durations.push(f64::INFINITY);
            }
        }

        Ok(expected_durations)
    }

    /// Classify current market regime based on recent data
    pub fn classify_current_regime(&self) -> Result<(usize, f64)> {
        let regime_probs = self
            .regime_probabilities
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("Markov Switching Model not fitted".to_string()))?;

        let final_probs = regime_probs.last().unwrap();
        let (most_likely_regime, max_prob) = final_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        Ok((most_likely_regime, *max_prob))
    }

    // Private implementation methods

    fn initialize_parameters(&mut self, data: &[f64]) -> Result<()> {
        let n = data.len();

        // Initialize means using quantiles
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut means = Vec::with_capacity(self.num_regimes);
        for i in 0..self.num_regimes {
            let quantile_pos = (i + 1) * n / (self.num_regimes + 1);
            means.push(sorted_data[quantile_pos.min(n - 1)]);
        }

        // Initialize standard deviations
        let overall_std = self.calculate_std_dev(data);
        let std_devs = vec![overall_std; self.num_regimes];

        // Initialize transition matrix with high persistence
        let mut transition_matrix = vec![vec![0.0; self.num_regimes]; self.num_regimes];
        let persistence = 0.9;
        let off_diagonal = (1.0 - persistence) / (self.num_regimes - 1) as f64;

        for (i_idx, row) in transition_matrix.iter_mut().enumerate() {
            for (j_idx, val_in_row_ref) in row.iter_mut().enumerate() {
                if i_idx == j_idx {
                    *val_in_row_ref = persistence;
                } else {
                    *val_in_row_ref = off_diagonal;
                }
            }
        }

        // Initialize equal initial probabilities
        let initial_probs = vec![1.0 / self.num_regimes as f64; self.num_regimes];

        self.means = Some(means);
        self.std_devs = Some(std_devs);
        self.transition_matrix = Some(transition_matrix);
        self.initial_probs = Some(initial_probs);

        Ok(())
    }

    fn e_step(&self, data: &[f64]) -> Result<(XiMatrix, GammaMatrix)> {
        let n = data.len();

        // Forward-backward algorithm to get regime probabilities
        let gamma = self.forward_backward(data)?;

        // Calculate xi (joint probabilities of consecutive regimes)
        let mut xi = vec![vec![vec![0.0; self.num_regimes]; self.num_regimes]; n - 1];
        let alpha = self.forward(data)?;
        let beta = self.backward(data)?;

        let means = self.means.as_ref().unwrap();
        let std_devs = self.std_devs.as_ref().unwrap();
        let transition_matrix = self.transition_matrix.as_ref().unwrap();

        for t in 0..n - 1 {
            let mut normalizer = 0.0;

            for (i_idx, transition_matrix_row_i) in
                transition_matrix.iter().enumerate().take(self.num_regimes)
            {
                for j_idx in 0..self.num_regimes {
                    let prob = alpha[t][i_idx]
                        * transition_matrix_row_i[j_idx]
                        * self.normal_density(data[t + 1], means[j_idx], std_devs[j_idx])
                        * beta[t + 1][j_idx];
                    xi[t][i_idx][j_idx] = prob;
                    normalizer += prob;
                }
            }

            // Normalize
            if normalizer > 0.0 {
                for i_idx_norm in 0..self.num_regimes {
                    // Renamed i to i_idx_norm for clarity
                    for j_idx_norm in 0..self.num_regimes {
                        // Renamed j to j_idx_norm for clarity
                        xi[t][i_idx_norm][j_idx_norm] /= normalizer;
                    }
                }
            }
        }

        Ok((xi, gamma))
    }

    fn m_step(&mut self, data: &[f64], xi: &[Vec<Vec<f64>>], gamma: &[Vec<f64>]) -> Result<()> {
        let n = data.len();
        let mut new_means = vec![0.0; self.num_regimes];
        let mut new_std_devs = vec![0.0; self.num_regimes];
        let mut new_transition_matrix = vec![vec![0.0; self.num_regimes]; self.num_regimes];
        let mut new_initial_probs = vec![0.0; self.num_regimes];

        // Update initial probabilities
        new_initial_probs.copy_from_slice(&gamma[0]);

        // Update means
        let current_means = self.means.as_ref().unwrap();
        for (regime_idx, new_mean_val_ref) in new_means.iter_mut().enumerate() {
            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for t in 0..n {
                numerator += gamma[t][regime_idx] * data[t];
                denominator += gamma[t][regime_idx];
            }

            if denominator > 0.0 {
                *new_mean_val_ref = numerator / denominator;
            } else {
                *new_mean_val_ref = current_means[regime_idx];
            }
        }

        // Update standard deviations
        let current_std_devs = self.std_devs.as_ref().unwrap();
        for (regime_idx, new_std_dev_val_ref) in new_std_devs.iter_mut().enumerate() {
            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for t in 0..n {
                let diff = data[t] - new_means[regime_idx]; // Use new_means here
                numerator += gamma[t][regime_idx] * diff * diff;
                denominator += gamma[t][regime_idx];
            }

            if denominator > 0.0 {
                *new_std_dev_val_ref = (numerator / denominator).sqrt();
                *new_std_dev_val_ref = new_std_dev_val_ref.max(1e-6);
            } else {
                *new_std_dev_val_ref = current_std_devs[regime_idx];
            }
        }

        // Update transition matrix
        let current_transition_matrix = self.transition_matrix.as_ref().unwrap();
        for (i_idx, row_ref) in new_transition_matrix.iter_mut().enumerate() {
            let mut row_sum = 0.0;
            for (j_idx, val_ref) in row_ref.iter_mut().enumerate() {
                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for t_idx in 0..n - 1 {
                    numerator += xi[t_idx][i_idx][j_idx];
                    denominator += gamma[t_idx][i_idx];
                }

                if denominator > 0.0 {
                    *val_ref = numerator / denominator;
                } else {
                    *val_ref = current_transition_matrix[i_idx][j_idx];
                }
                row_sum += *val_ref;
            }

            // Normalize transition probabilities
            if row_sum > 0.0 {
                for val_in_row in row_ref.iter_mut() {
                    // .take(self.num_regimes) is redundant
                    *val_in_row /= row_sum;
                }
            }
        }

        // Update parameters
        self.means = Some(new_means);
        self.std_devs = Some(new_std_devs);
        self.transition_matrix = Some(new_transition_matrix);
        self.initial_probs = Some(new_initial_probs);

        Ok(())
    }

    fn forward_backward(&self, data: &[f64]) -> Result<Vec<Vec<f64>>> {
        let alpha = self.forward(data)?;
        let beta = self.backward(data)?;
        let n = data.len();

        let mut gamma = vec![vec![0.0; self.num_regimes]; n];

        for t in 0..n {
            let mut normalizer = 0.0;
            for regime in 0..self.num_regimes {
                gamma[t][regime] = alpha[t][regime] * beta[t][regime];
                normalizer += gamma[t][regime];
            }

            // Normalize
            if normalizer > 0.0 {
                for regime in 0..self.num_regimes {
                    gamma[t][regime] /= normalizer;
                }
            }
        }

        Ok(gamma)
    }

    fn forward(&self, data: &[f64]) -> Result<Vec<Vec<f64>>> {
        let n = data.len();
        let mut alpha = vec![vec![0.0; self.num_regimes]; n];

        let means = self.means.as_ref().unwrap();
        let std_devs = self.std_devs.as_ref().unwrap();
        let transition_matrix = self.transition_matrix.as_ref().unwrap();
        let initial_probs = self.initial_probs.as_ref().unwrap();

        // Initialize
        for regime in 0..self.num_regimes {
            alpha[0][regime] = initial_probs[regime]
                * self.normal_density(data[0], means[regime], std_devs[regime]);
        }

        // Forward pass
        for t in 1..n {
            for j_idx in 0..self.num_regimes {
                let mut sum = 0.0;
                for (i_idx, transition_matrix_row_i) in
                    transition_matrix.iter().enumerate().take(self.num_regimes)
                {
                    sum += alpha[t - 1][i_idx] * transition_matrix_row_i[j_idx];
                }
                alpha[t][j_idx] = sum * self.normal_density(data[t], means[j_idx], std_devs[j_idx]);
            }
        }

        Ok(alpha)
    }

    fn backward(&self, data: &[f64]) -> Result<Vec<Vec<f64>>> {
        let n = data.len();
        let mut beta = vec![vec![0.0; self.num_regimes]; n];

        let means_vec = self.means.as_ref().unwrap();
        let std_devs_vec = self.std_devs.as_ref().unwrap();
        let transition_matrix_ref = self.transition_matrix.as_ref().unwrap();

        if n == 0 {
            return Ok(beta); // Return empty beta if data is empty
        }

        // Initialize
        for regime_probs_at_n_minus_1 in beta[n - 1].iter_mut() {
            *regime_probs_at_n_minus_1 = 1.0;
        }

        // Backward pass
        for t in (0..n - 1).rev() {
            let mut next_beta_t_row = vec![0.0; self.num_regimes];
            for i_idx in 0..self.num_regimes {
                let sum_val: f64 = (0..self.num_regimes)
                    .map(|j_idx| {
                        transition_matrix_ref[i_idx][j_idx]
                            * self.normal_density(
                                data[t + 1],
                                means_vec[j_idx],
                                std_devs_vec[j_idx],
                            )
                            * beta[t + 1][j_idx] // Immutable borrow of beta[t+1]
                    })
                    .sum();
                next_beta_t_row[i_idx] = sum_val;
            }
            beta[t] = next_beta_t_row;
        }

        Ok(beta)
    }

    fn viterbi(&self, data: &[f64]) -> Result<Vec<usize>> {
        let n = data.len();
        let mut delta = vec![vec![0.0; self.num_regimes]; n];
        let mut psi = vec![vec![0; self.num_regimes]; n];

        let means = self.means.as_ref().unwrap();
        let std_devs = self.std_devs.as_ref().unwrap();
        let transition_matrix = self.transition_matrix.as_ref().unwrap();
        let initial_probs = self.initial_probs.as_ref().unwrap();

        // Initialize
        for regime in 0..self.num_regimes {
            delta[0][regime] = initial_probs[regime].ln()
                + self
                    .normal_density(data[0], means[regime], std_devs[regime])
                    .ln();
        }

        // Forward pass
        for t in 1..n {
            for j_idx in 0..self.num_regimes {
                let mut max_val = f64::NEG_INFINITY;
                let mut max_arg = 0;

                for (i_idx, transition_matrix_row_i) in
                    transition_matrix.iter().enumerate().take(self.num_regimes)
                {
                    let val = delta[t - 1][i_idx] + transition_matrix_row_i[j_idx].ln();
                    if val > max_val {
                        max_val = val;
                        max_arg = i_idx;
                    }
                }

                delta[t][j_idx] = max_val
                    + self
                        .normal_density(data[t], means[j_idx], std_devs[j_idx])
                        .ln();
                psi[t][j_idx] = max_arg;
            }
        }

        // Backward pass
        let mut path = vec![0; n];

        // Find the maximum final state
        let mut max_val = f64::NEG_INFINITY;
        for regime in 0..self.num_regimes {
            if delta[n - 1][regime] > max_val {
                max_val = delta[n - 1][regime];
                path[n - 1] = regime;
            }
        }

        // Backtrack
        for t in (0..n - 1).rev() {
            path[t] = psi[t + 1][path[t + 1]];
        }

        Ok(path)
    }

    fn calculate_log_likelihood(&self, data: &[f64]) -> Result<f64> {
        let alpha = self.forward(data)?;
        let n = data.len();

        let final_prob: f64 = alpha[n - 1].iter().sum();
        if final_prob > 0.0 {
            Ok(final_prob.ln())
        } else {
            Ok(f64::NEG_INFINITY)
        }
    }

    fn normal_density(&self, x: f64, mean: f64, std_dev: f64) -> f64 {
        let variance = std_dev * std_dev;
        let coefficient = 1.0 / (std_dev * (2.0 * PI).sqrt());
        let exponent = -0.5 * (x - mean).powi(2) / variance;
        coefficient * E.powf(exponent)
    }

    fn calculate_std_dev(&self, data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        variance.sqrt()
    }
}

impl Forecaster for MarkovSwitchingModel {
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
            aic: self.information_criteria.map(|(aic, _)| aic),
            bic: self.information_criteria.map(|(_, bic)| bic),
        })
    }
}

impl MarkovSwitchingModel {
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
    use chrono::{DateTime, Duration, Utc};

    fn create_regime_switching_data() -> TimeSeriesData {
        let start_time = Utc::now();
        let timestamps: Vec<DateTime<Utc>> =
            (0..200).map(|i| start_time + Duration::days(i)).collect();

        // Create data with two regimes: low volatility (first 100) and high volatility (last 100)
        let mut values = Vec::with_capacity(200);

        // Regime 1: Low volatility, mean around 0
        for i in 0..100 {
            values.push(0.5 * ((i as f64 * 0.1).sin() + rand::random::<f64>() * 0.2 - 0.1));
        }

        // Regime 2: High volatility, mean around 2
        for i in 100..200 {
            values.push(2.0 + (i as f64 * 0.1).sin() + rand::random::<f64>() * 1.0 - 0.5);
        }

        TimeSeriesData::new(timestamps, values, "regime_switching_series").unwrap()
    }

    #[test]
    fn test_two_regime_model() {
        let mut model = MarkovSwitchingModel::two_regime(Some(100), Some(1e-4));
        let data = create_regime_switching_data();

        assert!(model.fit(&data).is_ok());
        assert!(model.forecast(10).is_ok());

        // Check that we have regime parameters
        assert!(model.get_regime_parameters().is_some());
        assert!(model.get_transition_matrix().is_some());
    }

    #[test]
    fn test_three_regime_model() {
        let mut model = MarkovSwitchingModel::three_regime(Some(50), Some(1e-3));
        let data = create_regime_switching_data();

        assert!(model.fit(&data).is_ok());
        assert!(model.forecast(5).is_ok());
    }

    #[test]
    fn test_regime_classification() {
        let mut model = MarkovSwitchingModel::two_regime(Some(50), Some(1e-3));
        let data = create_regime_switching_data();
        model.fit(&data).unwrap();

        let (regime, prob) = model.classify_current_regime().unwrap();
        assert!(regime < 2);
        assert!((0.0..=1.0).contains(&prob));
    }

    #[test]
    fn test_regime_duration_stats() {
        let mut model = MarkovSwitchingModel::two_regime(Some(50), Some(1e-3));
        let data = create_regime_switching_data();
        model.fit(&data).unwrap();

        let durations = model.regime_duration_stats().unwrap();
        assert_eq!(durations.len(), 2);
        assert!(durations.iter().all(|&d| d > 0.0));
    }

    #[test]
    fn test_forecast_by_regime() {
        let mut model = MarkovSwitchingModel::two_regime(Some(50), Some(1e-3));
        let data = create_regime_switching_data();
        model.fit(&data).unwrap();

        let regime_forecasts = model.forecast_by_regime(5).unwrap();
        assert_eq!(regime_forecasts.len(), 2);
        assert_eq!(regime_forecasts[0].len(), 5);
        assert_eq!(regime_forecasts[1].len(), 5);
    }
}
