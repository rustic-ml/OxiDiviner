//! Student's t-Copula Model with GARCH Marginals
//!
//! Implementation of multivariate time series forecasting using Student's t-copulas
//! with GARCH(1,1) marginal distributions. The t-copula extends the Gaussian copula
//! by introducing tail dependence through the degrees of freedom parameter.

use crate::core::{Forecaster, ModelEvaluation, OxiError, Result, TimeSeriesData};
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Student's t-Copula Model with GARCH(1,1) marginals
///
/// The t-copula is particularly useful for financial time series as it captures:
/// 1. Tail dependence between series
/// 2. Fat-tailed behavior through degrees of freedom parameter
/// 3. Asymmetric correlations during market stress
#[derive(Debug, Clone)]
pub struct TCopulaModel {
    /// Number of time series
    n_series: usize,
    /// Degrees of freedom parameter (controls tail thickness)
    degrees_of_freedom: f64,
    /// GARCH parameters for each marginal series [omega, alpha, beta]
    garch_params: Vec<[f64; 3]>,
    /// Conditional variances for each series
    conditional_vars: Vec<Vec<f64>>,
    /// Correlation matrix of the t-copula
    correlation_matrix: DMatrix<f64>,
    /// Fitted marginal residuals (standardized)
    marginal_residuals: Vec<Vec<f64>>,
    /// Uniform transforms of residuals
    uniform_residuals: Vec<Vec<f64>>,
    /// t-distributed transforms for copula fitting
    t_residuals: Vec<Vec<f64>>,
    /// Whether the model has been fitted
    is_fitted: bool,
    /// Sample size
    n_obs: usize,
}

impl TCopulaModel {
    /// Create a new t-Copula model
    ///
    /// # Arguments
    /// * `n_series` - Number of time series to model jointly
    /// * `degrees_of_freedom` - Degrees of freedom parameter (>2, typically 3-10)
    ///
    /// # Example
    /// ```rust
    /// use oxidiviner::models::copula::TCopulaModel;
    ///
    /// let model = TCopulaModel::new(3, 5.0).unwrap();
    /// ```
    pub fn new(n_series: usize, degrees_of_freedom: f64) -> Result<Self> {
        if n_series < 2 {
            return Err(OxiError::ModelError(
                "Copula model requires at least 2 time series".to_string(),
            ));
        }

        if degrees_of_freedom <= 2.0 {
            return Err(OxiError::ModelError(
                "Degrees of freedom must be > 2".to_string(),
            ));
        }

        Ok(Self {
            n_series,
            degrees_of_freedom,
            garch_params: vec![[0.01, 0.05, 0.9]; n_series],
            conditional_vars: vec![Vec::new(); n_series],
            correlation_matrix: DMatrix::identity(n_series, n_series),
            marginal_residuals: vec![Vec::new(); n_series],
            uniform_residuals: vec![Vec::new(); n_series],
            t_residuals: vec![Vec::new(); n_series],
            is_fitted: false,
            n_obs: 0,
        })
    }

    /// Fit GARCH(1,1) model to a single time series
    fn fit_garch_marginal(&mut self, series_idx: usize, data: &[f64]) -> Result<()> {
        let n = data.len();
        if n < 10 {
            return Err(OxiError::DataError(
                "Need at least 10 observations for GARCH fitting".to_string(),
            ));
        }

        // Calculate returns
        let mut returns = Vec::with_capacity(n - 1);
        for i in 1..n {
            returns.push((data[i] / data[i - 1] - 1.0) * 100.0);
        }

        // Initialize conditional variances
        let mean_return_sq = returns.iter().map(|x| x * x).sum::<f64>() / returns.len() as f64;
        let mut h = vec![mean_return_sq; returns.len()];

        // Simple parameter estimation (in practice, would use MLE)
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let squared_residuals: Vec<f64> =
            returns.iter().map(|r| (r - mean_return).powi(2)).collect();

        let unconditional_var =
            squared_residuals.iter().sum::<f64>() / squared_residuals.len() as f64;

        let omega = 0.01 * unconditional_var;
        let alpha = 0.05;
        let beta = 0.90;

        self.garch_params[series_idx] = [omega, alpha, beta];

        // Update conditional variances
        h[0] = unconditional_var;
        for t in 1..returns.len() {
            let prev_residual_sq = squared_residuals[t - 1];
            h[t] = omega + alpha * prev_residual_sq + beta * h[t - 1];
        }

        self.conditional_vars[series_idx] = h.clone();

        // Calculate standardized residuals
        let mut std_residuals = Vec::with_capacity(returns.len());
        for t in 0..returns.len() {
            let residual = returns[t] - mean_return;
            std_residuals.push(residual / h[t].sqrt());
        }

        self.marginal_residuals[series_idx] = std_residuals;
        Ok(())
    }

    /// Transform standardized residuals to uniform using empirical CDF
    fn transform_to_uniform(&mut self, series_idx: usize) -> Result<()> {
        let residuals = &self.marginal_residuals[series_idx];
        let n = residuals.len();

        let mut indexed_residuals: Vec<(f64, usize)> = residuals
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .filter(|(val, _)| val.is_finite()) // Filter out NaN and infinite values
            .collect();

        if indexed_residuals.is_empty() {
            return Err(OxiError::ModelError(
                "No valid residuals after filtering".to_string(),
            ));
        }

        indexed_residuals
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut uniform_vals = vec![0.5; n]; // Default value for invalid residuals
        for (rank, &(_, original_idx)) in indexed_residuals.iter().enumerate() {
            uniform_vals[original_idx] = (rank + 1) as f64 / (indexed_residuals.len() + 1) as f64;
        }

        self.uniform_residuals[series_idx] = uniform_vals;
        Ok(())
    }

    /// Transform uniform residuals to t-distribution for copula fitting
    fn transform_to_t_distribution(&mut self, series_idx: usize) -> Result<()> {
        let uniform_vals = &self.uniform_residuals[series_idx];
        let t_dist = StudentsT::new(0.0, 1.0, self.degrees_of_freedom)
            .map_err(|_| OxiError::ModelError("Failed to create t-distribution".to_string()))?;

        let t_vals: Vec<f64> = uniform_vals
            .iter()
            .map(|&u| {
                let clamped_u = u.clamp(1e-10, 1.0 - 1e-10);
                t_dist.inverse_cdf(clamped_u)
            })
            .collect();

        self.t_residuals[series_idx] = t_vals;
        Ok(())
    }

    /// Estimate correlation matrix from t-distributed residuals
    fn estimate_correlation_matrix(&mut self) -> Result<()> {
        let n_obs = self.t_residuals[0].len();
        let mut corr_matrix = DMatrix::zeros(self.n_series, self.n_series);

        // Calculate sample correlation matrix
        for i in 0..self.n_series {
            for j in 0..self.n_series {
                if i == j {
                    corr_matrix[(i, j)] = 1.0;
                } else {
                    let mut sum_xy = 0.0;
                    let mut sum_x_sq = 0.0;
                    let mut sum_y_sq = 0.0;

                    for t in 0..n_obs {
                        let x = self.t_residuals[i][t];
                        let y = self.t_residuals[j][t];
                        sum_xy += x * y;
                        sum_x_sq += x * x;
                        sum_y_sq += y * y;
                    }

                    let correlation = sum_xy / (sum_x_sq.sqrt() * sum_y_sq.sqrt());
                    corr_matrix[(i, j)] = correlation;
                }
            }
        }

        self.correlation_matrix = corr_matrix;
        Ok(())
    }

    /// Fit the t-copula model to multivariate data
    pub fn fit_multivariate(&mut self, data: &[Vec<f64>]) -> Result<()> {
        if data.len() != self.n_series {
            return Err(OxiError::DataError(format!(
                "Expected {} series, got {}",
                self.n_series,
                data.len()
            )));
        }

        // Check all series have same length
        let n_obs = data[0].len();
        for series in data.iter() {
            if series.len() != n_obs {
                return Err(OxiError::DataError(
                    "All time series must have the same length".to_string(),
                ));
            }
        }

        if n_obs < 20 {
            return Err(OxiError::DataError(
                "Need at least 20 observations for t-copula fitting".to_string(),
            ));
        }

        self.n_obs = n_obs;

        // Fit GARCH marginals for each series
        for (i, series_data) in data.iter().enumerate() {
            self.fit_garch_marginal(i, series_data)?;
        }

        // Transform to uniform and then to t-distribution
        for i in 0..self.n_series {
            self.transform_to_uniform(i)?;
            self.transform_to_t_distribution(i)?;
        }

        // Estimate correlation matrix
        self.estimate_correlation_matrix()?;

        self.is_fitted = true;
        Ok(())
    }

    /// Calculate log-likelihood of the t-copula
    pub fn log_likelihood(&self) -> Result<f64> {
        if !self.is_fitted {
            return Err(OxiError::ModelError("Model not fitted".to_string()));
        }

        let nu = self.degrees_of_freedom;
        let k = self.n_series as f64;

        // Log-likelihood calculation for multivariate t-distribution
        let det_r = self.correlation_matrix.determinant();
        if det_r <= 0.0 {
            return Err(OxiError::ModelError(
                "Correlation matrix is singular".to_string(),
            ));
        }

        let log_det_r = det_r.ln();

        // Gamma function approximation using Sterling's approximation
        let gamma_half_nu_k = log_gamma((nu + k) / 2.0);
        let gamma_half_nu = log_gamma(nu / 2.0);

        let mut log_likelihood = 0.0;

        for t in 0..self.n_obs {
            // Get t-residuals for time t
            let mut z_t = DVector::zeros(self.n_series);
            for i in 0..self.n_series {
                z_t[i] = self.t_residuals[i][t];
            }

            // Calculate quadratic form z'R^(-1)z
            let r_inv = self
                .correlation_matrix
                .clone()
                .try_inverse()
                .ok_or_else(|| {
                    OxiError::ModelError("Cannot invert correlation matrix".to_string())
                })?;
            let quad_form = z_t.transpose() * r_inv * z_t;

            // Add to log-likelihood
            log_likelihood += gamma_half_nu_k
                - gamma_half_nu
                - 0.5 * log_det_r
                - 0.5 * k * (nu * std::f64::consts::PI).ln()
                - 0.5 * (nu + k) * (1.0 + quad_form[0] / nu).ln();
        }

        Ok(log_likelihood)
    }

    /// Generate multivariate forecasts
    pub fn forecast_multivariate(&self, horizon: usize) -> Result<Vec<Vec<f64>>> {
        if !self.is_fitted {
            return Err(OxiError::ModelError("Model not fitted".to_string()));
        }

        let mut forecasts = vec![Vec::new(); self.n_series];

        // For simplicity, generate point forecasts using expected values
        for (series_idx, current_forecast_series) in forecasts.iter_mut().enumerate().take(self.n_series) {
            let [omega, alpha, beta] = self.garch_params[series_idx];
            let last_var = if !self.conditional_vars[series_idx].is_empty() {
                *self.conditional_vars[series_idx].last().unwrap()
            } else {
                omega
            };

            let mut h_forecast_variance = last_var; // Renamed for clarity, as it's variance

            for _ in 0..horizon {
                h_forecast_variance = omega + (alpha + beta) * h_forecast_variance;
                let point_forecast = 0.0; // Expected return for t-distribution
                current_forecast_series.push(point_forecast);
            }
        }

        Ok(forecasts)
    }

    /// Get the correlation matrix
    pub fn correlation_matrix(&self) -> &DMatrix<f64> {
        &self.correlation_matrix
    }

    /// Get degrees of freedom parameter
    pub fn degrees_of_freedom(&self) -> f64 {
        self.degrees_of_freedom
    }

    /// Get information criteria
    pub fn information_criteria(&self) -> Result<(f64, f64)> {
        let log_lik = self.log_likelihood()?;
        let n_params = self.n_series * 3 + (self.n_series * (self.n_series - 1)) / 2 + 1; // GARCH + correlation + df
        let n_residuals = self.t_residuals[0].len() as f64;

        let aic = -2.0 * log_lik + 2.0 * n_params as f64;
        let bic = -2.0 * log_lik + (n_params as f64) * n_residuals.ln();

        Ok((aic, bic))
    }
}

impl Forecaster for TCopulaModel {
    fn name(&self) -> &str {
        "t-Copula with GARCH Marginals"
    }

    fn fit(&mut self, _data: &TimeSeriesData) -> Result<()> {
        Err(OxiError::ModelError(
            "Use fit_multivariate for copula models".to_string(),
        ))
    }

    fn forecast(&self, _horizon: usize) -> Result<Vec<f64>> {
        Err(OxiError::ModelError(
            "Use forecast_multivariate for copula models".to_string(),
        ))
    }

    fn evaluate(&self, _test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        Err(OxiError::ModelError(
            "Use evaluate_multivariate for copula models".to_string(),
        ))
    }
}

/// Log gamma function approximation using Stirling's approximation
fn log_gamma(x: f64) -> f64 {
    if x < 1.0 {
        // Use recursion formula: Γ(x) = Γ(x+1)/x
        log_gamma(x + 1.0) - x.ln()
    } else {
        // Stirling's approximation: ln(Γ(x)) ≈ x*ln(x) - x + 0.5*ln(2π/x)
        x * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI / x).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data() -> Vec<Vec<f64>> {
        // Generate correlated financial returns
        let n = 100;
        let mut data1 = vec![100.0];
        let mut data2 = vec![50.0];

        for i in 1..n {
            let shock1 = 0.02 * (i as f64 * 0.1).sin() + 0.01;
            let shock2 = 0.015 * (i as f64 * 0.15).cos() + 0.008;

            data1.push(data1[i - 1] * (1.0 + shock1));
            data2.push(data2[i - 1] * (1.0 + shock2));
        }

        vec![data1, data2]
    }

    #[test]
    fn test_t_copula_creation() {
        let model = TCopulaModel::new(2, 5.0);
        assert!(model.is_ok());

        let model = TCopulaModel::new(1, 5.0);
        assert!(model.is_err());

        let model = TCopulaModel::new(2, 1.0);
        assert!(model.is_err());
    }

    #[test]
    fn test_t_copula_fitting() {
        let data = generate_test_data();
        let mut model = TCopulaModel::new(2, 5.0).unwrap();

        let result = model.fit_multivariate(&data);
        assert!(result.is_ok());
        assert!(model.is_fitted);
    }

    #[test]
    fn test_degrees_of_freedom() {
        let model = TCopulaModel::new(2, 7.5).unwrap();
        assert_eq!(model.degrees_of_freedom(), 7.5);
    }

    #[test]
    fn test_t_copula_forecast() {
        let data = generate_test_data();
        let mut model = TCopulaModel::new(2, 5.0).unwrap();

        model.fit_multivariate(&data).unwrap();
        let forecasts = model.forecast_multivariate(5).unwrap();

        assert_eq!(forecasts.len(), 2);
        assert_eq!(forecasts[0].len(), 5);
        assert_eq!(forecasts[1].len(), 5);
    }
}
