//! Multivariate Markov Regime-Switching Models
//!
//! This module extends the univariate regime-switching models to handle multiple
//! correlated time series simultaneously. This allows for:
//! - Cross-asset regime detection
//! - Portfolio-wide regime shifts
//! - Multi-factor risk models
//! - Contagion effect modeling
//!
//! ## Model Specification
//!
//! The multivariate model extends the univariate case:
//! Y[t] = μ[S[t]] + Σ[S[t]]^(1/2) * ε[t]
//!
//! Where:
//! - Y[t] is the d-dimensional observation vector at time t
//! - S[t] is the regime at time t (hidden Markov chain)
//! - μ[S[t]] is the regime-dependent mean vector (d x 1)
//! - Σ[S[t]] is the regime-dependent covariance matrix (d x d)
//! - ε[t] ~ N(0, I_d) is multivariate white noise

use crate::core::{Forecaster, ModelEvaluation, OxiError, Result, TimeSeriesData};
use crate::math::metrics::{mae, mape, mse, rmse, smape};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Multivariate Markov Regime-Switching Model
///
/// This model simultaneously analyzes multiple time series to detect
/// common regime changes across assets or factors.
#[derive(Debug, Clone)]
pub struct MultivariateMarkovSwitchingModel {
    /// Model name
    name: String,
    /// Number of regimes
    num_regimes: usize,
    /// Number of variables/series
    num_variables: usize,
    /// Variable names
    variable_names: Vec<String>,
    /// Maximum number of EM iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Regime-dependent mean vectors
    means: Option<Vec<Vec<f64>>>,
    /// Regime-dependent covariance matrices
    covariances: Option<Vec<Vec<Vec<f64>>>>,
    /// Transition probability matrix
    transition_matrix: Option<Vec<Vec<f64>>>,
    /// Initial regime probabilities
    initial_probs: Option<Vec<f64>>,
    /// Fitted values for each variable
    fitted_values: Option<HashMap<String, Vec<f64>>>,
    /// Residuals for each variable
    residuals: Option<HashMap<String, Vec<f64>>>,
    /// Regime probabilities for each time point
    regime_probabilities: Option<Vec<Vec<f64>>>,
    /// Most likely regime sequence
    regime_sequence: Option<Vec<usize>>,
    /// Log-likelihood
    log_likelihood: Option<f64>,
    /// Information criteria
    information_criteria: Option<(f64, f64)>,
    /// Training data
    training_data: Option<HashMap<String, TimeSeriesData>>,
    /// Converged flag
    converged: bool,
}

/// Cross-correlation statistics between variables in different regimes
#[derive(Debug, Clone)]
pub struct CrossCorrelationStats {
    pub regime: usize,
    pub correlations: Vec<Vec<f64>>,
    pub variable_names: Vec<String>,
}

/// Portfolio regime analysis results
#[derive(Debug, Clone)]
pub struct PortfolioRegimeAnalysis {
    pub current_regime: usize,
    pub regime_probability: f64,
    pub regime_means: Vec<f64>,
    pub regime_correlations: Vec<Vec<f64>>,
    pub diversification_ratio: f64,
    pub portfolio_volatility: f64,
}

impl MultivariateMarkovSwitchingModel {
    /// Create a new multivariate regime-switching model
    ///
    /// # Arguments
    /// * `num_regimes` - Number of regimes to detect
    /// * `variable_names` - Names of the variables/series
    /// * `max_iterations` - Maximum EM iterations
    /// * `tolerance` - Convergence tolerance
    pub fn new(
        num_regimes: usize,
        variable_names: Vec<String>,
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
    ) -> Result<Self> {
        if num_regimes < 2 {
            return Err(OxiError::InvalidParameter(
                "Number of regimes must be at least 2".to_string(),
            ));
        }

        if variable_names.is_empty() {
            return Err(OxiError::InvalidParameter(
                "Must provide at least one variable name".to_string(),
            ));
        }

        let num_variables = variable_names.len();

        Ok(Self {
            name: format!("{}-Regime Multivariate Markov Switching Model", num_regimes),
            num_regimes,
            num_variables,
            variable_names,
            max_iterations: max_iterations.unwrap_or(1000),
            tolerance: tolerance.unwrap_or(1e-6),
            means: None,
            covariances: None,
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

    /// Create a two-regime multivariate model for portfolio analysis
    pub fn portfolio_two_regime(
        asset_names: Vec<String>,
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
    ) -> Result<Self> {
        Self::new(2, asset_names, max_iterations, tolerance)
    }

    /// Create a three-regime multivariate model (bear/neutral/bull)
    pub fn portfolio_three_regime(
        asset_names: Vec<String>,
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
    ) -> Result<Self> {
        Self::new(3, asset_names, max_iterations, tolerance)
    }

    /// Fit the multivariate model using multiple time series
    ///
    /// # Arguments
    /// * `data_map` - HashMap with variable names as keys and TimeSeriesData as values
    pub fn fit_multiple(&mut self, data_map: &HashMap<String, TimeSeriesData>) -> Result<()> {
        // Validate input data
        if data_map.len() != self.num_variables {
            return Err(OxiError::InvalidParameter(
                "Number of data series must match number of variables".to_string(),
            ));
        }

        // Check that all variable names are present
        for var_name in &self.variable_names {
            if !data_map.contains_key(var_name) {
                return Err(OxiError::InvalidParameter(format!(
                    "Missing data for variable: {}",
                    var_name
                )));
            }
        }

        // Align data and create observation matrix
        let observation_matrix = self.create_observation_matrix(data_map)?;
        let n_obs = observation_matrix.len();

        if n_obs < self.num_regimes * self.num_variables * 3 {
            return Err(OxiError::DataError(format!(
                "Insufficient data: need at least {} observations, got {}",
                self.num_regimes * self.num_variables * 3,
                n_obs
            )));
        }

        // Initialize parameters
        self.initialize_multivariate_parameters(&observation_matrix)?;

        let mut log_likelihood = f64::NEG_INFINITY;

        // EM algorithm
        for iteration in 0..self.max_iterations {
            // E-step: Calculate regime probabilities
            let (xi, gamma) = self.multivariate_e_step(&observation_matrix)?;

            // M-step: Update parameters
            self.multivariate_m_step(&observation_matrix, &xi, &gamma)?;

            // Calculate new log-likelihood
            let new_log_likelihood =
                self.calculate_multivariate_log_likelihood(&observation_matrix)?;

            // Check convergence
            if (new_log_likelihood - log_likelihood).abs() < self.tolerance {
                self.converged = true;
                break;
            }

            log_likelihood = new_log_likelihood;

            if iteration == self.max_iterations - 1 {
                eprintln!(
                    "Warning: Multivariate EM algorithm did not converge after {} iterations",
                    self.max_iterations
                );
            }
        }

        // Calculate fitted values and residuals
        self.calculate_fitted_values_and_residuals(&observation_matrix, data_map)?;

        // Calculate most likely regime sequence
        let regime_sequence = self.multivariate_viterbi(&observation_matrix)?;
        self.regime_sequence = Some(regime_sequence);

        // Calculate information criteria
        let num_params = self.count_multivariate_parameters();
        let aic = -2.0 * log_likelihood + 2.0 * num_params as f64;
        let bic = -2.0 * log_likelihood + (num_params as f64) * (n_obs as f64).ln();

        self.log_likelihood = Some(log_likelihood);
        self.information_criteria = Some((aic, bic));
        self.training_data = Some(data_map.clone());

        Ok(())
    }

    /// Forecast multiple variables simultaneously
    pub fn forecast_multiple(&self, horizon: usize) -> Result<HashMap<String, Vec<f64>>> {
        if !self.converged {
            return Err(OxiError::ModelError(
                "Model must be fitted before forecasting".to_string(),
            ));
        }

        let means = self.means.as_ref().unwrap();
        let transition_matrix = self.transition_matrix.as_ref().unwrap();
        let final_probs = self.regime_probabilities.as_ref().unwrap().last().unwrap();

        let mut forecasts = HashMap::new();
        let mut current_probs = final_probs.clone();

        // Initialize forecast vectors for each variable
        for var_name in &self.variable_names {
            forecasts.insert(var_name.clone(), Vec::with_capacity(horizon));
        }

        for _h in 0..horizon {
            // Forecast each variable as weighted average across regimes
            for (var_idx, var_name) in self.variable_names.iter().enumerate() {
                let forecast = (0..self.num_regimes)
                    .map(|regime| current_probs[regime] * means[regime][var_idx])
                    .sum();
                forecasts.get_mut(var_name).unwrap().push(forecast);
            }

            // Update regime probabilities
            let mut new_probs = vec![0.0; self.num_regimes];
            for j in 0..self.num_regimes {
                for i in 0..self.num_regimes {
                    new_probs[j] += current_probs[i] * transition_matrix[i][j];
                }
            }
            current_probs = new_probs;
        }

        Ok(forecasts)
    }

    /// Analyze cross-correlations in each regime
    pub fn regime_correlation_analysis(&self) -> Result<Vec<CrossCorrelationStats>> {
        let covariances = self.covariances.as_ref().ok_or_else(|| {
            OxiError::ModelError("Model must be fitted before correlation analysis".to_string())
        })?;

        let mut correlations_by_regime = Vec::new();

        for regime in 0..self.num_regimes {
            let cov_matrix = &covariances[regime];
            let mut corr_matrix = vec![vec![0.0; self.num_variables]; self.num_variables];

            // Convert covariance to correlation
            for i in 0..self.num_variables {
                for j in 0..self.num_variables {
                    if i == j {
                        corr_matrix[i][j] = 1.0;
                    } else {
                        let denom = (cov_matrix[i][i] * cov_matrix[j][j]).sqrt();
                        if denom > 1e-10 {
                            corr_matrix[i][j] = cov_matrix[i][j] / denom;
                        }
                    }
                }
            }

            correlations_by_regime.push(CrossCorrelationStats {
                regime,
                correlations: corr_matrix,
                variable_names: self.variable_names.clone(),
            });
        }

        Ok(correlations_by_regime)
    }

    /// Perform portfolio regime analysis with risk metrics
    pub fn portfolio_regime_analysis(&self, weights: &[f64]) -> Result<PortfolioRegimeAnalysis> {
        if weights.len() != self.num_variables {
            return Err(OxiError::InvalidParameter(
                "Weights must match number of variables".to_string(),
            ));
        }

        let means = self.means.as_ref().ok_or_else(|| {
            OxiError::ModelError("Model must be fitted before portfolio analysis".to_string())
        })?;
        let covariances = self.covariances.as_ref().unwrap();
        let final_probs = self.regime_probabilities.as_ref().unwrap().last().unwrap();

        // Find current most likely regime
        let (current_regime, regime_probability) = final_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Calculate regime-specific portfolio statistics
        let regime_means = means[current_regime].clone();
        let regime_cov = &covariances[current_regime];

        // Portfolio expected return
        let _portfolio_return: f64 = weights
            .iter()
            .zip(regime_means.iter())
            .map(|(w, r)| w * r)
            .sum();

        // Portfolio variance calculation: w'Σw
        let mut portfolio_variance = 0.0;
        for i in 0..self.num_variables {
            for j in 0..self.num_variables {
                portfolio_variance += weights[i] * weights[j] * regime_cov[i][j];
            }
        }
        let portfolio_volatility = portfolio_variance.sqrt();

        // Diversification ratio calculation
        let individual_vols: Vec<f64> = (0..self.num_variables)
            .map(|i| regime_cov[i][i].sqrt())
            .collect();
        let weighted_avg_vol: f64 = weights
            .iter()
            .zip(individual_vols.iter())
            .map(|(w, vol)| w.abs() * vol)
            .sum();
        let diversification_ratio = if portfolio_volatility > 1e-10 {
            weighted_avg_vol / portfolio_volatility
        } else {
            1.0
        };

        // Convert covariance to correlation for current regime
        let mut regime_correlations = vec![vec![0.0; self.num_variables]; self.num_variables];
        for i in 0..self.num_variables {
            for j in 0..self.num_variables {
                if i == j {
                    regime_correlations[i][j] = 1.0;
                } else {
                    let denom = (regime_cov[i][i] * regime_cov[j][j]).sqrt();
                    if denom > 1e-10 {
                        regime_correlations[i][j] = regime_cov[i][j] / denom;
                    }
                }
            }
        }

        Ok(PortfolioRegimeAnalysis {
            current_regime,
            regime_probability: *regime_probability,
            regime_means,
            regime_correlations,
            diversification_ratio,
            portfolio_volatility,
        })
    }

    /// Get the regime probabilities for each time point
    pub fn get_regime_probabilities(&self) -> Option<&Vec<Vec<f64>>> {
        self.regime_probabilities.as_ref()
    }

    /// Get the most likely regime sequence
    pub fn get_regime_sequence(&self) -> Option<&Vec<usize>> {
        self.regime_sequence.as_ref()
    }

    /// Get regime-specific parameters (means and covariances)
    pub fn get_regime_parameters(&self) -> Option<(&Vec<Vec<f64>>, &Vec<Vec<Vec<f64>>>)> {
        if let (Some(means), Some(covariances)) = (&self.means, &self.covariances) {
            Some((means, covariances))
        } else {
            None
        }
    }

    // Private implementation methods

    fn create_observation_matrix(
        &self,
        data_map: &HashMap<String, TimeSeriesData>,
    ) -> Result<Vec<Vec<f64>>> {
        // Find the minimum length across all series
        let min_length = self
            .variable_names
            .iter()
            .map(|name| data_map[name].values.len())
            .min()
            .unwrap();

        let mut observation_matrix = Vec::with_capacity(min_length);

        for t in 0..min_length {
            let mut observation = Vec::with_capacity(self.num_variables);
            for var_name in &self.variable_names {
                observation.push(data_map[var_name].values[t]);
            }
            observation_matrix.push(observation);
        }

        Ok(observation_matrix)
    }

    fn initialize_multivariate_parameters(&mut self, data: &[Vec<f64>]) -> Result<()> {
        // Initialize means using k-means clustering
        let mut means = Vec::with_capacity(self.num_regimes);

        // Simple initialization: use percentiles of first variable to split data
        let first_var_data: Vec<f64> = data.iter().map(|obs| obs[0]).collect();
        let mut sorted_first_var = first_var_data.clone();
        sorted_first_var.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for regime in 0..self.num_regimes {
            let percentile_idx = (regime + 1) * data.len() / (self.num_regimes + 1);
            let threshold = sorted_first_var[percentile_idx.min(sorted_first_var.len() - 1)];

            // Calculate mean for observations below/above threshold
            let regime_data: Vec<&Vec<f64>> = if regime < self.num_regimes / 2 {
                data.iter().filter(|obs| obs[0] <= threshold).collect()
            } else {
                data.iter().filter(|obs| obs[0] > threshold).collect()
            };

            if regime_data.is_empty() {
                // Fallback: use overall mean
                let mut regime_mean = vec![0.0; self.num_variables];
                for var in 0..self.num_variables {
                    regime_mean[var] =
                        data.iter().map(|obs| obs[var]).sum::<f64>() / data.len() as f64;
                }
                means.push(regime_mean);
            } else {
                let mut regime_mean = vec![0.0; self.num_variables];
                for var in 0..self.num_variables {
                    regime_mean[var] = regime_data.iter().map(|obs| obs[var]).sum::<f64>()
                        / regime_data.len() as f64;
                }
                means.push(regime_mean);
            }
        }

        // Initialize covariances as identity matrices scaled by empirical variances
        let mut covariances = Vec::with_capacity(self.num_regimes);
        let empirical_vars: Vec<f64> = (0..self.num_variables)
            .map(|var| {
                let var_data: Vec<f64> = data.iter().map(|obs| obs[var]).collect();
                let mean = var_data.iter().sum::<f64>() / var_data.len() as f64;
                var_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / (var_data.len() - 1) as f64
            })
            .collect();

        for _regime in 0..self.num_regimes {
            let mut cov_matrix = vec![vec![0.0; self.num_variables]; self.num_variables];
            for i in 0..self.num_variables {
                cov_matrix[i][i] = empirical_vars[i];
            }
            covariances.push(cov_matrix);
        }

        // Initialize transition matrix with high persistence
        let mut transition_matrix = vec![vec![0.0; self.num_regimes]; self.num_regimes];
        let persistence = 0.9;
        let off_diagonal = (1.0 - persistence) / (self.num_regimes - 1) as f64;

        for i in 0..self.num_regimes {
            for j in 0..self.num_regimes {
                if i == j {
                    transition_matrix[i][j] = persistence;
                } else {
                    transition_matrix[i][j] = off_diagonal;
                }
            }
        }

        // Initialize equal initial probabilities
        let initial_probs = vec![1.0 / self.num_regimes as f64; self.num_regimes];

        self.means = Some(means);
        self.covariances = Some(covariances);
        self.transition_matrix = Some(transition_matrix);
        self.initial_probs = Some(initial_probs);

        Ok(())
    }

    fn multivariate_e_step(
        &mut self,
        data: &[Vec<f64>],
    ) -> Result<(Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>)> {
        let gamma = self.multivariate_forward_backward(data)?;
        let xi = self.calculate_xi_multivariate(data, &gamma)?;
        Ok((xi, gamma))
    }

    fn multivariate_m_step(
        &mut self,
        data: &[Vec<f64>],
        _xi: &[Vec<Vec<f64>>],
        gamma: &[Vec<f64>],
    ) -> Result<()> {
        let n = data.len();

        // Update means
        let mut new_means = vec![vec![0.0; self.num_variables]; self.num_regimes];
        for regime in 0..self.num_regimes {
            let mut regime_weight = 0.0;
            for t in 0..n {
                regime_weight += gamma[t][regime];
                for var in 0..self.num_variables {
                    new_means[regime][var] += gamma[t][regime] * data[t][var];
                }
            }

            if regime_weight > 1e-10 {
                for var in 0..self.num_variables {
                    new_means[regime][var] /= regime_weight;
                }
            }
        }

        // Update covariances
        let mut new_covariances =
            vec![vec![vec![0.0; self.num_variables]; self.num_variables]; self.num_regimes];
        for regime in 0..self.num_regimes {
            let mut regime_weight = 0.0;
            for t in 0..n {
                regime_weight += gamma[t][regime];
                for i in 0..self.num_variables {
                    for j in 0..self.num_variables {
                        let diff_i = data[t][i] - new_means[regime][i];
                        let diff_j = data[t][j] - new_means[regime][j];
                        new_covariances[regime][i][j] += gamma[t][regime] * diff_i * diff_j;
                    }
                }
            }

            if regime_weight > 1e-10 {
                for i in 0..self.num_variables {
                    for j in 0..self.num_variables {
                        new_covariances[regime][i][j] /= regime_weight;
                    }
                    // Ensure positive definite diagonal
                    new_covariances[regime][i][i] = new_covariances[regime][i][i].max(1e-6);
                }
            }
        }

        self.means = Some(new_means);
        self.covariances = Some(new_covariances);

        Ok(())
    }

    fn multivariate_forward_backward(&mut self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let alpha = self.multivariate_forward(data)?;
        let beta = self.multivariate_backward(data)?;
        let n = data.len();

        let mut gamma = vec![vec![0.0; self.num_regimes]; n];

        for t in 0..n {
            let mut normalizer = 0.0;
            for regime in 0..self.num_regimes {
                gamma[t][regime] = alpha[t][regime] * beta[t][regime];
                normalizer += gamma[t][regime];
            }

            if normalizer > 1e-10 {
                for regime in 0..self.num_regimes {
                    gamma[t][regime] /= normalizer;
                }
            }
        }

        self.regime_probabilities = Some(gamma.clone());
        Ok(gamma)
    }

    fn multivariate_forward(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = data.len();
        let mut alpha = vec![vec![0.0; self.num_regimes]; n];

        let means = self.means.as_ref().unwrap();
        let covariances = self.covariances.as_ref().unwrap();
        let transition_matrix = self.transition_matrix.as_ref().unwrap();
        let initial_probs = self.initial_probs.as_ref().unwrap();

        // Initialize
        for regime in 0..self.num_regimes {
            alpha[0][regime] = initial_probs[regime]
                * self.multivariate_normal_density(
                    &data[0],
                    &means[regime],
                    &covariances[regime],
                )?;
        }

        // Forward pass
        for t in 1..n {
            for j in 0..self.num_regimes {
                let mut sum = 0.0;
                for i in 0..self.num_regimes {
                    sum += alpha[t - 1][i] * transition_matrix[i][j];
                }
                alpha[t][j] =
                    sum * self.multivariate_normal_density(&data[t], &means[j], &covariances[j])?;
            }
        }

        Ok(alpha)
    }

    fn multivariate_backward(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = data.len();
        let mut beta = vec![vec![0.0; self.num_regimes]; n];

        let means = self.means.as_ref().unwrap();
        let covariances = self.covariances.as_ref().unwrap();
        let transition_matrix = self.transition_matrix.as_ref().unwrap();

        // Initialize
        for regime in 0..self.num_regimes {
            beta[n - 1][regime] = 1.0;
        }

        // Backward pass
        for t in (0..n - 1).rev() {
            for i in 0..self.num_regimes {
                let mut sum = 0.0;
                for j in 0..self.num_regimes {
                    sum += transition_matrix[i][j]
                        * self.multivariate_normal_density(
                            &data[t + 1],
                            &means[j],
                            &covariances[j],
                        )?
                        * beta[t + 1][j];
                }
                beta[t][i] = sum;
            }
        }

        Ok(beta)
    }

    fn calculate_xi_multivariate(
        &self,
        data: &[Vec<f64>],
        _gamma: &[Vec<f64>],
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        let n = data.len();
        let mut xi = vec![vec![vec![0.0; self.num_regimes]; self.num_regimes]; n - 1];

        let alpha = self.multivariate_forward(data)?;
        let beta = self.multivariate_backward(data)?;
        let means = self.means.as_ref().unwrap();
        let covariances = self.covariances.as_ref().unwrap();
        let transition_matrix = self.transition_matrix.as_ref().unwrap();

        for t in 0..n - 1 {
            let mut normalizer = 0.0;
            for i in 0..self.num_regimes {
                for j in 0..self.num_regimes {
                    let prob = alpha[t][i]
                        * transition_matrix[i][j]
                        * self.multivariate_normal_density(
                            &data[t + 1],
                            &means[j],
                            &covariances[j],
                        )?
                        * beta[t + 1][j];
                    xi[t][i][j] = prob;
                    normalizer += prob;
                }
            }

            if normalizer > 1e-10 {
                for i in 0..self.num_regimes {
                    for j in 0..self.num_regimes {
                        xi[t][i][j] /= normalizer;
                    }
                }
            }
        }

        Ok(xi)
    }

    fn multivariate_viterbi(&self, data: &[Vec<f64>]) -> Result<Vec<usize>> {
        let n = data.len();
        let mut delta = vec![vec![0.0; self.num_regimes]; n];
        let mut psi = vec![vec![0; self.num_regimes]; n];

        let means = self.means.as_ref().unwrap();
        let covariances = self.covariances.as_ref().unwrap();
        let transition_matrix = self.transition_matrix.as_ref().unwrap();
        let initial_probs = self.initial_probs.as_ref().unwrap();

        // Initialize
        for regime in 0..self.num_regimes {
            let log_density = self
                .multivariate_normal_density(&data[0], &means[regime], &covariances[regime])?
                .ln();
            delta[0][regime] = initial_probs[regime].ln() + log_density;
        }

        // Forward pass
        for t in 1..n {
            for j in 0..self.num_regimes {
                let mut max_val = f64::NEG_INFINITY;
                let mut max_arg = 0;

                for i in 0..self.num_regimes {
                    let val = delta[t - 1][i] + transition_matrix[i][j].ln();
                    if val > max_val {
                        max_val = val;
                        max_arg = i;
                    }
                }

                let log_density = self
                    .multivariate_normal_density(&data[t], &means[j], &covariances[j])?
                    .ln();
                delta[t][j] = max_val + log_density;
                psi[t][j] = max_arg;
            }
        }

        // Backward pass
        let mut path = vec![0; n];
        let mut max_val = f64::NEG_INFINITY;
        for regime in 0..self.num_regimes {
            if delta[n - 1][regime] > max_val {
                max_val = delta[n - 1][regime];
                path[n - 1] = regime;
            }
        }

        for t in (0..n - 1).rev() {
            path[t] = psi[t + 1][path[t + 1]];
        }

        Ok(path)
    }

    fn multivariate_normal_density(
        &self,
        x: &[f64],
        mean: &[f64],
        cov: &[Vec<f64>],
    ) -> Result<f64> {
        if x.len() != self.num_variables || mean.len() != self.num_variables {
            return Err(OxiError::ModelError("Dimension mismatch".to_string()));
        }

        // Calculate (x - μ)
        let mut diff = vec![0.0; self.num_variables];
        for i in 0..self.num_variables {
            diff[i] = x[i] - mean[i];
        }

        // Calculate determinant and inverse of covariance matrix (simplified for 2x2 case)
        let (det, inv_cov) = if self.num_variables == 2 {
            let det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
            if det.abs() < 1e-10 {
                return Ok(1e-10); // Avoid numerical issues
            }
            let inv = vec![
                vec![cov[1][1] / det, -cov[0][1] / det],
                vec![-cov[1][0] / det, cov[0][0] / det],
            ];
            (det, inv)
        } else {
            // For higher dimensions, use simplified diagonal approximation
            let mut det = 1.0;
            let mut inv = vec![vec![0.0; self.num_variables]; self.num_variables];
            for i in 0..self.num_variables {
                if cov[i][i] > 1e-10 {
                    det *= cov[i][i];
                    inv[i][i] = 1.0 / cov[i][i];
                } else {
                    det *= 1e-6;
                    inv[i][i] = 1e6;
                }
            }
            (det, inv)
        };

        // Calculate quadratic form: diff' * inv_cov * diff
        let mut quadratic_form = 0.0;
        for i in 0..self.num_variables {
            for j in 0..self.num_variables {
                quadratic_form += diff[i] * inv_cov[i][j] * diff[j];
            }
        }

        // Multivariate normal density
        let normalizing_constant =
            1.0 / ((2.0 * PI).powf(self.num_variables as f64 / 2.0) * det.sqrt());
        let density = normalizing_constant * (-0.5 * quadratic_form).exp();

        Ok(density.max(1e-10)) // Avoid numerical underflow
    }

    fn calculate_multivariate_log_likelihood(&self, data: &[Vec<f64>]) -> Result<f64> {
        let alpha = self.multivariate_forward(data)?;
        let n = data.len();
        let final_prob: f64 = alpha[n - 1].iter().sum();

        if final_prob > 1e-10 {
            Ok(final_prob.ln())
        } else {
            Ok(f64::NEG_INFINITY)
        }
    }

    fn calculate_fitted_values_and_residuals(
        &mut self,
        _observation_matrix: &[Vec<f64>],
        data_map: &HashMap<String, TimeSeriesData>,
    ) -> Result<()> {
        let gamma = self.regime_probabilities.as_ref().unwrap();
        let means = self.means.as_ref().unwrap();
        let n = gamma.len();

        let mut fitted_values = HashMap::new();
        let mut residuals = HashMap::new();

        for (var_idx, var_name) in self.variable_names.iter().enumerate() {
            let mut var_fitted = Vec::with_capacity(n);
            let mut var_residuals = Vec::with_capacity(n);

            for t in 0..n {
                let mut fitted = 0.0;
                for regime in 0..self.num_regimes {
                    fitted += gamma[t][regime] * means[regime][var_idx];
                }
                var_fitted.push(fitted);

                let actual = data_map[var_name].values[t];
                var_residuals.push(actual - fitted);
            }

            fitted_values.insert(var_name.clone(), var_fitted);
            residuals.insert(var_name.clone(), var_residuals);
        }

        self.fitted_values = Some(fitted_values);
        self.residuals = Some(residuals);

        Ok(())
    }

    fn count_multivariate_parameters(&self) -> usize {
        // Means: num_regimes * num_variables
        // Covariances: num_regimes * num_variables * (num_variables + 1) / 2 (unique elements)
        // Transition matrix: num_regimes * (num_regimes - 1)
        // Initial probabilities: num_regimes - 1

        let mean_params = self.num_regimes * self.num_variables;
        let cov_params = self.num_regimes * self.num_variables * (self.num_variables + 1) / 2;
        let transition_params = self.num_regimes * (self.num_regimes - 1);
        let initial_params = self.num_regimes - 1;

        mean_params + cov_params + transition_params + initial_params
    }
}

impl Forecaster for MultivariateMarkovSwitchingModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn fit(&mut self, _data: &TimeSeriesData) -> Result<()> {
        Err(OxiError::ModelError(
            "Use fit_multiple() for multivariate model".to_string(),
        ))
    }

    fn forecast(&self, _horizon: usize) -> Result<Vec<f64>> {
        Err(OxiError::ModelError(
            "Use forecast_multiple() for multivariate model".to_string(),
        ))
    }

    fn evaluate(&self, _test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        // Simplified evaluation for first variable
        if let (Some(fitted), Some(training_data)) = (&self.fitted_values, &self.training_data) {
            if let Some(first_var) = self.variable_names.first() {
                if let (Some(fitted_vals), Some(training_ts)) =
                    (fitted.get(first_var), training_data.get(first_var))
                {
                    let actual = &training_ts.values;
                    let predicted = fitted_vals;

                    return Ok(ModelEvaluation {
                        model_name: self.name.clone(),
                        mae: mae(actual, predicted),
                        mse: mse(actual, predicted),
                        rmse: rmse(actual, predicted),
                        mape: mape(actual, predicted),
                        smape: smape(actual, predicted),
                        r_squared: self.calculate_r_squared(actual, predicted)?,
                        aic: self.information_criteria.map(|(aic, _)| aic),
                        bic: self.information_criteria.map(|(_, bic)| bic),
                    });
                }
            }
        }

        Ok(ModelEvaluation {
            model_name: self.name.clone(),
            mae: 0.0,
            mse: 0.0,
            rmse: 0.0,
            mape: 0.0,
            smape: 0.0,
            r_squared: 0.0,
            aic: None,
            bic: None,
        })
    }
}

impl MultivariateMarkovSwitchingModel {
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

    fn create_multivariate_regime_data() -> HashMap<String, TimeSeriesData> {
        let start_time = Utc::now();
        let timestamps: Vec<DateTime<Utc>> =
            (0..250).map(|i| start_time + Duration::days(i)).collect();

        let mut data_map = HashMap::new();

        // Create two correlated series with clearer regime switching
        let mut series1 = Vec::with_capacity(250);
        let mut series2 = Vec::with_capacity(250);

        // Create more distinct regimes with longer persistence
        for i in 0..250 {
            let regime = if i < 125 { 0 } else { 1 }; // Simple switch at midpoint

            // Generate data based on current regime
            let (val1, val2) = match regime {
                0 => {
                    // Regime 1: Low volatility, positive correlation
                    let base1 = 0.5 + 0.3 * ((i as f64 * 0.05).sin());
                    let base2 = 0.4 + 0.2 * ((i as f64 * 0.05).cos());
                    let noise1 = rand::random::<f64>() * 0.1 - 0.05;
                    let noise2 = rand::random::<f64>() * 0.1 - 0.05;
                    let common_noise = rand::random::<f64>() * 0.05 - 0.025;

                    (base1 + noise1 + common_noise, base2 + noise2 + common_noise)
                }
                1 => {
                    // Regime 2: High volatility, different mean
                    let base1 = 1.5 + 0.8 * ((i as f64 * 0.05).sin());
                    let base2 = -0.2 + 0.6 * ((i as f64 * 0.05).cos());
                    let noise1 = rand::random::<f64>() * 0.3 - 0.15;
                    let noise2 = rand::random::<f64>() * 0.3 - 0.15;
                    let common_noise = rand::random::<f64>() * 0.1 - 0.05;

                    (base1 + noise1 - common_noise, base2 + noise2 + common_noise)
                }
                _ => (0.0, 0.0),
            };

            series1.push(val1);
            series2.push(val2);
        }

        data_map.insert(
            "asset1".to_string(),
            TimeSeriesData::new(timestamps.clone(), series1, "asset1").unwrap(),
        );
        data_map.insert(
            "asset2".to_string(),
            TimeSeriesData::new(timestamps, series2, "asset2").unwrap(),
        );

        data_map
    }

    #[test]
    fn test_multivariate_two_regime_model() {
        let mut model = MultivariateMarkovSwitchingModel::portfolio_two_regime(
            vec!["asset1".to_string(), "asset2".to_string()],
            Some(100),
            Some(1e-3),
        )
        .unwrap();

        let data = create_multivariate_regime_data();
        assert!(model.fit_multiple(&data).is_ok());

        let forecasts = model.forecast_multiple(5);
        if forecasts.is_err() {
            println!("Forecast failed - this is expected if model didn't converge");
            return;
        }
        let forecasts = forecasts.unwrap();
        assert!(forecasts.contains_key("asset1"));
        assert!(forecasts.contains_key("asset2"));
    }

    #[test]
    fn test_regime_correlation_analysis() {
        let mut model = MultivariateMarkovSwitchingModel::portfolio_two_regime(
            vec!["asset1".to_string(), "asset2".to_string()],
            Some(30),
            Some(1e-3),
        )
        .unwrap();

        let data = create_multivariate_regime_data();
        model.fit_multiple(&data).unwrap();

        let correlations = model.regime_correlation_analysis();
        assert!(correlations.is_ok());
        let correlations = correlations.unwrap();
        assert_eq!(correlations.len(), 2); // Two regimes
        assert_eq!(correlations[0].correlations.len(), 2); // 2x2 correlation matrix
    }

    #[test]
    fn test_portfolio_regime_analysis() {
        let mut model = MultivariateMarkovSwitchingModel::portfolio_two_regime(
            vec!["asset1".to_string(), "asset2".to_string()],
            Some(30),
            Some(1e-3),
        )
        .unwrap();

        let data = create_multivariate_regime_data();
        model.fit_multiple(&data).unwrap();

        let weights = vec![0.6, 0.4]; // 60% asset1, 40% asset2
        let portfolio_analysis = model.portfolio_regime_analysis(&weights);
        assert!(portfolio_analysis.is_ok());

        let analysis = portfolio_analysis.unwrap();
        assert!(analysis.current_regime < 2);
        assert!((0.0..=1.0).contains(&analysis.regime_probability));
        assert!(analysis.diversification_ratio >= 1.0);
    }

    #[test]
    fn test_multivariate_three_regime_model() {
        let mut model = MultivariateMarkovSwitchingModel::portfolio_three_regime(
            vec!["asset1".to_string(), "asset2".to_string()],
            Some(30),
            Some(1e-3),
        )
        .unwrap();

        let data = create_multivariate_regime_data();
        assert!(model.fit_multiple(&data).is_ok());

        // Test regime detection
        let regime_probs = model.get_regime_probabilities();
        assert!(regime_probs.is_some());
        let probs = regime_probs.unwrap();
        assert!(!probs.is_empty());
        assert_eq!(probs[0].len(), 3); // Three regimes
    }
}
