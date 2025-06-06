//! Gaussian Copula Model with GARCH Marginals
//!
//! Implementation of multivariate time series forecasting using Gaussian copulas
//! with GARCH(1,1) marginal distributions. This approach models the dependency
//! structure separately from the marginal distributions, allowing for flexible
//! modeling of complex multivariate relationships.

use crate::core::{Forecaster, ModelEvaluation, OxiError, Result, TimeSeriesData};
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, Normal};

/// Gaussian Copula Model with GARCH(1,1) marginals
///
/// This model separates the modeling of:
/// 1. Marginal distributions (using GARCH models)
/// 2. Dependency structure (using Gaussian copula)
///
/// The Gaussian copula assumes that after transforming marginals to uniform [0,1],
/// the dependency structure follows a multivariate normal distribution.
#[derive(Debug, Clone)]
pub struct GaussianCopulaModel {
    /// Number of time series
    n_series: usize,
    /// GARCH parameters for each marginal series [omega, alpha, beta]
    garch_params: Vec<[f64; 3]>,
    /// Conditional variances for each series
    conditional_vars: Vec<Vec<f64>>,
    /// Correlation matrix of the Gaussian copula
    correlation_matrix: DMatrix<f64>,
    /// Fitted marginal residuals (standardized)
    marginal_residuals: Vec<Vec<f64>>,
    /// Uniform transforms of residuals
    uniform_residuals: Vec<Vec<f64>>,
    /// Normal transforms for copula fitting
    normal_residuals: Vec<Vec<f64>>,
    /// Whether the model has been fitted
    is_fitted: bool,
    /// Sample size
    n_obs: usize,
}

impl GaussianCopulaModel {
    /// Create a new Gaussian Copula model
    ///
    /// # Arguments
    /// * `n_series` - Number of time series to model jointly
    ///
    /// # Example
    /// ```rust
    /// use oxidiviner::models::copula::GaussianCopulaModel;
    ///
    /// let model = GaussianCopulaModel::new(3).unwrap();
    /// ```
    pub fn new(n_series: usize) -> Result<Self> {
        if n_series < 2 {
            return Err(OxiError::ModelError(
                "Copula model requires at least 2 time series".to_string(),
            ));
        }

        Ok(Self {
            n_series,
            garch_params: vec![[0.01, 0.05, 0.9]; n_series], // Default GARCH params
            conditional_vars: vec![Vec::new(); n_series],
            correlation_matrix: DMatrix::identity(n_series, n_series),
            marginal_residuals: vec![Vec::new(); n_series],
            uniform_residuals: vec![Vec::new(); n_series],
            normal_residuals: vec![Vec::new(); n_series],
            is_fitted: false,
            n_obs: 0,
        })
    }

    /// Fit GARCH(1,1) model to a single time series
    fn fit_garch_marginal(&mut self, series_idx: usize, data: &[f64]) -> Result<()> {
        let n = data.len();
        if n < 10 {
            // Needs at least 2 data points for one return, plus more for GARCH
            return Err(OxiError::DataError(
                "Need at least 10 observations for GARCH fitting".to_string(),
            ));
        }

        // Calculate returns
        if n < 2 {
            // Not enough data to calculate returns, handle appropriately or return error earlier.
            // For now, an empty returns vector will likely cause issues later.
            // This case should ideally be caught by the n < 10 check if GARCH needs several returns.
            self.marginal_residuals[series_idx] = Vec::new();
            self.conditional_vars[series_idx] = Vec::new();
            return Ok(()); // Or an error
        }
        let returns: Vec<f64> = data
            .windows(2)
            .map(|window| (window[1] / window[0] - 1.0) * 100.0) // Convert to percentage returns
            .collect();

        if returns.is_empty() {
            // This can happen if n=1, though caught by n<2. Defensive check.
            return Err(OxiError::DataError(
                "Not enough data to calculate returns for GARCH fitting".to_string(),
            ));
        }

        // Initialize conditional variances
        let mean_return_sq = returns.iter().map(|x| x * x).sum::<f64>() / returns.len() as f64;
        let mut h = vec![mean_return_sq; returns.len()];

        // Simple method of moments estimation for GARCH parameters
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let squared_residuals: Vec<f64> =
            returns.iter().map(|r| (r - mean_return).powi(2)).collect();

        // Estimate GARCH parameters using method of moments
        let unconditional_var =
            squared_residuals.iter().sum::<f64>() / squared_residuals.len() as f64;

        // Simple parameter estimation (in practice, would use MLE)
        let omega = 0.01 * unconditional_var;
        let alpha = 0.05;
        let beta = 0.90;

        self.garch_params[series_idx] = [omega, alpha, beta];

        // Update conditional variances with estimated parameters
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

        // Create sorted residuals with original indices
        let mut indexed_residuals: Vec<(f64, usize)> = residuals
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();
        indexed_residuals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Calculate empirical CDF values
        let mut uniform_vals = vec![0.0; n];
        for (rank, &(_, original_idx)) in indexed_residuals.iter().enumerate() {
            // Use (rank + 1) / (n + 1) to avoid 0 and 1
            uniform_vals[original_idx] = (rank + 1) as f64 / (n + 1) as f64;
        }

        self.uniform_residuals[series_idx] = uniform_vals;
        Ok(())
    }

    /// Transform uniform residuals to standard normal for copula fitting
    fn transform_to_normal(&mut self, series_idx: usize) -> Result<()> {
        let uniform_vals = &self.uniform_residuals[series_idx];
        let normal = Normal::new(0.0, 1.0).unwrap();

        let normal_vals: Vec<f64> = uniform_vals
            .iter()
            .map(|&u| {
                // Ensure u is in (0, 1)
                let clamped_u = u.clamp(1e-10, 1.0 - 1e-10);
                normal.inverse_cdf(clamped_u)
            })
            .collect();

        self.normal_residuals[series_idx] = normal_vals;
        Ok(())
    }

    /// Estimate correlation matrix from normal-transformed residuals
    fn estimate_correlation_matrix(&mut self) -> Result<()> {
        let n_obs = self.normal_residuals[0].len();
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
                        let x = self.normal_residuals[i][t];
                        let y = self.normal_residuals[j][t];
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

    /// Calculate log-likelihood for the Gaussian copula
    pub fn log_likelihood(&self) -> Result<f64> {
        if !self.is_fitted {
            return Err(OxiError::ModelError("Model not fitted".to_string()));
        }

        // Use the actual number of residuals (which is n_obs - 1 due to return calculation)
        let n_residuals = self.normal_residuals[0].len();
        let mut log_lik = 0.0;

        // Calculate copula density for each observation
        for t in 0..n_residuals {
            let mut z = DVector::zeros(self.n_series);
            for i in 0..self.n_series {
                z[i] = self.normal_residuals[i][t];
            }

            // Multivariate normal density calculation
            let inv_corr = self
                .correlation_matrix
                .clone()
                .try_inverse()
                .ok_or_else(|| OxiError::ModelError("Singular correlation matrix".to_string()))?;

            let det = self.correlation_matrix.determinant();
            if det <= 0.0 {
                return Err(OxiError::ModelError(
                    "Non-positive definite correlation matrix".to_string(),
                ));
            }

            // Copula density = exp(-0.5 * z' * (R^-1 - I) * z) / sqrt(det(R))
            let quad_form = z.transpose() * &inv_corr * &z;
            let z_norm_sq = z.norm_squared();

            log_lik += -0.5 * (quad_form[0] - z_norm_sq) - 0.5 * det.ln();
        }

        Ok(log_lik)
    }

    /// Generate forecasts using the fitted copula model
    pub fn forecast_multivariate(
        &self,
        horizon: usize,
        _n_simulations: usize,
    ) -> Result<Vec<Vec<f64>>> {
        if !self.is_fitted {
            return Err(OxiError::ModelError("Model not fitted".to_string()));
        }

        let mut forecasts = vec![Vec::new(); self.n_series];

        // For simplicity, we'll generate point forecasts using the expected values
        // In practice, you'd want to simulate from the copula

        for (series_idx, current_forecast_series) in
            forecasts.iter_mut().enumerate().take(self.n_series)
        {
            let [omega, alpha, beta] = self.garch_params[series_idx];
            let last_var = self.conditional_vars[series_idx].last().unwrap_or(&omega);

            // Forecast conditional variances
            let mut h_forecast_variance = *last_var;
            // let _long_run_var = omega / (1.0 - alpha - beta); // _long_run_var is unused

            for _ in 0..horizon {
                // GARCH variance forecast converges to long-run variance
                h_forecast_variance = omega + (alpha + beta) * h_forecast_variance;

                // For point forecast, use zero mean (could be improved with ARMA mean)
                let point_forecast = 0.0; // Expected return
                current_forecast_series.push(point_forecast);
            }
        }

        Ok(forecasts)
    }

    /// Get model information criteria
    pub fn information_criteria(&self) -> Result<(f64, f64)> {
        let log_lik = self.log_likelihood()?;
        let n_params = self.n_series * 3 + (self.n_series * (self.n_series - 1)) / 2; // GARCH params + correlation params
        let n_residuals = self.normal_residuals[0].len() as f64; // Use actual number of residuals

        let aic = -2.0 * log_lik + 2.0 * n_params as f64;
        let bic = -2.0 * log_lik + (n_params as f64) * n_residuals.ln();

        Ok((aic, bic))
    }

    /// Get the fitted correlation matrix
    pub fn correlation_matrix(&self) -> &DMatrix<f64> {
        &self.correlation_matrix
    }

    /// Get GARCH parameters for a specific series
    pub fn garch_parameters(&self, series_idx: usize) -> Result<[f64; 3]> {
        if series_idx >= self.n_series {
            return Err(OxiError::DataError(
                "Series index out of bounds".to_string(),
            ));
        }
        Ok(self.garch_params[series_idx])
    }
}

impl Forecaster for GaussianCopulaModel {
    fn name(&self) -> &str {
        "Gaussian Copula with GARCH Marginals"
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

impl GaussianCopulaModel {
    /// Fit the copula model to multivariate time series data
    ///
    /// # Arguments
    /// * `data` - Matrix where each row is an observation and each column is a time series
    ///
    /// # Example
    /// ```rust
    /// use oxidiviner::models::copula::GaussianCopulaModel;
    ///
    /// let mut model = GaussianCopulaModel::new(2).unwrap();
    /// let data = vec![
    ///     vec![100.0, 200.0],
    ///     vec![101.0, 201.0],
    ///     vec![102.0, 199.0],
    ///     // ... more observations
    /// ];
    /// model.fit_multivariate(&data).unwrap();
    /// ```
    pub fn fit_multivariate(&mut self, data: &[Vec<f64>]) -> Result<()> {
        if data.is_empty() {
            return Err(OxiError::DataError("No data provided".to_string()));
        }

        let n_obs = data.len();
        let n_series = data[0].len();

        if n_series != self.n_series {
            return Err(OxiError::DataError(format!(
                "Expected {} series, got {}",
                self.n_series, n_series
            )));
        }

        if n_obs < 50 {
            return Err(OxiError::DataError(
                "Need at least 50 observations for reliable copula estimation".to_string(),
            ));
        }

        self.n_obs = n_obs;

        // Transpose data for easier processing
        let mut series_data = vec![Vec::with_capacity(n_obs); n_series];
        for obs in data {
            for (i, &value) in obs.iter().enumerate() {
                series_data[i].push(value);
            }
        }

        // Fit GARCH marginals for each series
        for (i, current_series_data) in series_data.iter().enumerate().take(self.n_series) {
            self.fit_garch_marginal(i, current_series_data)?;
            self.transform_to_uniform(i)?;
            self.transform_to_normal(i)?;
        }

        // Estimate copula correlation matrix
        self.estimate_correlation_matrix()?;

        self.is_fitted = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data() -> Vec<Vec<f64>> {
        // Generate simple deterministic correlated time series
        let n_obs = 100;
        let n_series = 2;
        let mut data = vec![vec![100.0; n_series]; n_obs];

        // Simple deterministic pattern with some correlation
        for (t_idx, row_data) in data.iter_mut().enumerate().skip(1) {
            let t_f = (t_idx + 1) as f64; // t_idx is 0-based from enumerate().skip(1)
                                          // Series 1: trending upward with sine wave
            row_data[0] = 100.0 + t_f * 0.1 + (t_f * 0.2).sin() * 2.0;
            // Series 2: correlated with series 1 but with different volatility
            row_data[1] = 200.0 + t_f * 0.05 + (t_f * 0.15).cos() * 3.0 + row_data[0] * 0.1;
        }

        data
    }

    #[test]
    fn test_copula_creation() {
        let model = GaussianCopulaModel::new(2);
        assert!(model.is_ok());

        let model = GaussianCopulaModel::new(1);
        assert!(model.is_err());
    }

    #[test]
    fn test_copula_fitting() {
        let mut model = GaussianCopulaModel::new(2).unwrap();
        let data = generate_test_data();

        let result = model.fit_multivariate(&data);
        assert!(result.is_ok());
        assert!(model.is_fitted);
    }

    #[test]
    fn test_correlation_matrix() {
        let mut model = GaussianCopulaModel::new(2).unwrap();
        let data = generate_test_data();

        model.fit_multivariate(&data).unwrap();
        let corr_matrix = model.correlation_matrix();

        // Check that diagonal elements are 1
        assert!((corr_matrix[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((corr_matrix[(1, 1)] - 1.0).abs() < 1e-10);

        // Check symmetry
        assert!((corr_matrix[(0, 1)] - corr_matrix[(1, 0)]).abs() < 1e-10);

        // Check that off-diagonal correlation is reasonable
        assert!(corr_matrix[(0, 1)].abs() <= 1.0);
    }

    #[test]
    fn test_information_criteria() {
        let mut model = GaussianCopulaModel::new(2).unwrap();
        let data = generate_test_data();

        model.fit_multivariate(&data).unwrap();
        let (aic, bic) = model.information_criteria().unwrap();

        assert!(aic.is_finite());
        assert!(bic.is_finite());
        assert!(bic > aic); // BIC typically larger than AIC for reasonable sample sizes
    }

    #[test]
    fn test_garch_parameters() {
        let mut model = GaussianCopulaModel::new(2).unwrap();
        let data = generate_test_data();

        model.fit_multivariate(&data).unwrap();

        for i in 0..2 {
            let params = model.garch_parameters(i).unwrap();
            let [omega, alpha, beta] = params;

            // Check GARCH parameter constraints
            assert!(omega > 0.0);
            assert!(alpha >= 0.0);
            assert!(beta >= 0.0);
            assert!(alpha + beta < 1.0); // Stationarity condition
        }
    }

    #[test]
    fn test_multivariate_forecast() {
        let mut model = GaussianCopulaModel::new(2).unwrap();
        let data = generate_test_data();

        model.fit_multivariate(&data).unwrap();
        let forecasts = model.forecast_multivariate(5, 1000).unwrap();

        assert_eq!(forecasts.len(), 2); // Two series
        assert_eq!(forecasts[0].len(), 5); // Five periods ahead
        assert_eq!(forecasts[1].len(), 5);
    }
}
