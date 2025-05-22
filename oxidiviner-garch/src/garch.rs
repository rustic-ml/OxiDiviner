use crate::error::{GARCHError, Result};
use chrono::{DateTime, Utc};
use std::fmt;

/// GARCH(p,q) Model - Generalized Autoregressive Conditional Heteroskedasticity
///
/// The GARCH(p,q) model is defined as:
/// y_t = μ + ε_t
/// ε_t = σ_t * z_t, where z_t ~ N(0,1)
/// σ²_t = ω + Σ(i=1 to p) α_i * ε²_{t-i} + Σ(j=1 to q) β_j * σ²_{t-j}
///
/// Where:
/// - p is the order of the ARCH terms (ε²)
/// - q is the order of the GARCH terms (σ²)
/// - ω is the constant term (omega)
/// - α_i are the ARCH parameters
/// - β_j are the GARCH parameters
#[derive(Debug, Clone)]
pub struct GARCHModel {
    /// Mean of the process
    pub mean: f64,
    /// Constant term in the variance equation
    pub omega: f64,
    /// ARCH coefficients (alpha)
    pub alpha: Vec<f64>,
    /// GARCH coefficients (beta)
    pub beta: Vec<f64>,
    /// Fitted variance series
    pub fitted_variance: Option<Vec<f64>>,
    /// Residuals
    pub residuals: Option<Vec<f64>>,
    /// Log-likelihood of the fitted model
    pub log_likelihood: Option<f64>,
    /// Information criteria (AIC, BIC)
    pub info_criteria: Option<(f64, f64)>,
    /// Parameter standard errors
    pub std_errors: Option<Vec<f64>>,
    /// Training data timestamps
    pub timestamps: Option<Vec<DateTime<Utc>>>,
}

impl GARCHModel {
    /// Create a new GARCH(p,q) model with specified parameters
    ///
    /// # Arguments
    ///
    /// * `p` - Order of the ARCH terms
    /// * `q` - Order of the GARCH terms
    /// * `params` - Optional parameters [mean, omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]
    ///
    /// # Returns
    ///
    /// A Result containing the GARCHModel or an error
    pub fn new(p: usize, q: usize, params: Option<Vec<f64>>) -> Result<Self> {
        // Validate p and q
        if p == 0 && q == 0 {
            return Err(GARCHError::InvalidParameters(
                "Both p and q cannot be zero".to_string(),
            ));
        }

        let alpha = vec![0.0; p];
        let beta = vec![0.0; q];
        let mean = 0.0;
        let omega = 0.0001; // Small positive value

        let model = if let Some(params) = params {
            // Validate parameters length
            if params.len() != 1 + 1 + p + q {
                return Err(GARCHError::InvalidParameters(format!(
                    "Expected {} parameters, got {}",
                    1 + 1 + p + q,
                    params.len()
                )));
            }

            let mean = params[0];
            let omega = params[1];

            // Extract alpha and beta parameters
            let mut alpha = Vec::with_capacity(p);
            let mut beta = Vec::with_capacity(q);

            for i in 0..p {
                alpha.push(params[2 + i]);
            }

            for i in 0..q {
                beta.push(params[2 + p + i]);
            }

            // Validate parameters for stationarity and positivity
            Self::validate_parameters(&omega, &alpha, &beta)?;

            GARCHModel {
                mean,
                omega,
                alpha,
                beta,
                fitted_variance: None,
                residuals: None,
                log_likelihood: None,
                info_criteria: None,
                std_errors: None,
                timestamps: None,
            }
        } else {
            // Default initialization with zero parameters
            GARCHModel {
                mean,
                omega,
                alpha,
                beta,
                fitted_variance: None,
                residuals: None,
                log_likelihood: None,
                info_criteria: None,
                std_errors: None,
                timestamps: None,
            }
        };

        Ok(model)
    }

    /// Validate GARCH parameters for stationarity and positivity constraints
    fn validate_parameters(omega: &f64, alpha: &[f64], beta: &[f64]) -> Result<()> {
        // Check positivity constraints
        if *omega <= 0.0 {
            return Err(GARCHError::InvalidParameters(
                "Omega must be positive".to_string(),
            ));
        }

        for &a in alpha {
            if a < 0.0 {
                return Err(GARCHError::InvalidParameters(
                    "Alpha parameters must be non-negative".to_string(),
                ));
            }
        }

        for &b in beta {
            if b < 0.0 {
                return Err(GARCHError::InvalidParameters(
                    "Beta parameters must be non-negative".to_string(),
                ));
            }
        }

        // Check stationarity condition: sum of alpha and beta < 1
        let sum: f64 = alpha.iter().sum::<f64>() + beta.iter().sum::<f64>();
        if sum >= 1.0 {
            return Err(GARCHError::InvalidParameters(
                "Sum of alpha and beta must be less than 1 for stationarity".to_string(),
            ));
        }

        Ok(())
    }

    /// Fit the GARCH model to the data
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data
    /// * `timestamps` - Optional timestamps for the data
    ///
    /// # Returns
    ///
    /// A Result indicating success or failure
    pub fn fit(&mut self, data: &[f64], timestamps: Option<&[DateTime<Utc>]>) -> Result<()> {
        if data.len() < 2 {
            return Err(GARCHError::InvalidData(
                "Data must have at least 2 points".to_string(),
            ));
        }

        // Calculate mean
        self.mean = data.iter().sum::<f64>() / data.len() as f64;

        // Calculate residuals
        let residuals: Vec<f64> = data.iter().map(|&x| x - self.mean).collect();

        // Initial parameter guess
        let p = self.alpha.len();
        let q = self.beta.len();

        // Default initial parameters if not already set
        if self.omega <= 0.0 {
            self.omega =
                residuals.iter().map(|&r| r * r).sum::<f64>() / residuals.len() as f64 * 0.1;
        }

        if self.alpha.iter().all(|&a| a == 0.0) {
            for i in 0..p {
                self.alpha[i] = 0.05 + (0.1 / (i + 1) as f64);
            }
        }

        if self.beta.iter().all(|&b| b == 0.0) {
            for i in 0..q {
                self.beta[i] = 0.1 + (0.5 / (i + 1) as f64);
            }
        }

        // Calculate the fitted variance
        let fitted_variance = self.calculate_variance(&residuals)?;

        // Store results
        self.residuals = Some(residuals);
        self.fitted_variance = Some(fitted_variance);

        // Store timestamps if provided
        if let Some(ts) = timestamps {
            self.timestamps = Some(ts.to_vec());
        }

        // Calculate log-likelihood and information criteria
        self.calculate_statistics()?;

        Ok(())
    }

    /// Calculate the conditional variance based on the model parameters
    fn calculate_variance(&self, residuals: &[f64]) -> Result<Vec<f64>> {
        let n = residuals.len();
        let p = self.alpha.len();
        let q = self.beta.len();
        let max_lag = p.max(q);

        if n <= max_lag {
            return Err(GARCHError::InvalidData(
                "Not enough data points for the specified model".to_string(),
            ));
        }

        let mut variance = vec![0.0; n];

        // Initialize with unconditional variance
        let unconditional_variance = residuals.iter().map(|&r| r * r).sum::<f64>() / n as f64;

        // Initialize the first elements with historical values
        let max_lag = p.max(q);
        
        // Instead of a range loop, use an iterator on the slice
        for variance_item in variance.iter_mut().take(max_lag) {
            *variance_item = unconditional_variance;
        }

        // Calculate variance for the rest of the series
        for t in max_lag..n {
            let mut var_t = self.omega;

            // Add ARCH components
            for i in 0..p {
                var_t += self.alpha[i] * residuals[t - i - 1].powi(2);
            }

            // Add GARCH components
            for j in 0..q {
                var_t += self.beta[j] * variance[t - j - 1];
            }

            variance[t] = var_t;
        }

        Ok(variance)
    }

    /// Calculate log-likelihood and information criteria
    fn calculate_statistics(&mut self) -> Result<()> {
        let residuals = match &self.residuals {
            Some(r) => r,
            None => return Err(GARCHError::EstimationError("Model not fitted".to_string())),
        };

        let variance = match &self.fitted_variance {
            Some(v) => v,
            None => return Err(GARCHError::EstimationError("Model not fitted".to_string())),
        };

        let n = residuals.len();
        let p = self.alpha.len();
        let q = self.beta.len();
        let num_params = 2 + p + q; // mean, omega, alphas, betas

        // Calculate log-likelihood
        let mut log_likelihood = 0.0;
        for t in 0..n {
            if variance[t] <= 0.0 {
                return Err(GARCHError::NumericalError(
                    "Negative or zero variance encountered".to_string(),
                ));
            }

            log_likelihood += -0.5
                * (std::f64::consts::LN_2
                    + std::f64::consts::PI.ln()
                    + (variance[t]).ln()
                    + residuals[t].powi(2) / variance[t]);
        }

        self.log_likelihood = Some(log_likelihood);

        // Calculate information criteria
        let aic = -2.0 * log_likelihood + 2.0 * num_params as f64;
        let bic = -2.0 * log_likelihood + (num_params as f64) * (n as f64).ln();

        self.info_criteria = Some((aic, bic));

        Ok(())
    }

    /// Forecast future volatility
    ///
    /// # Arguments
    ///
    /// * `horizon` - Number of steps to forecast
    ///
    /// # Returns
    ///
    /// A Result containing the forecasted variance
    pub fn forecast_variance(&self, horizon: usize) -> Result<Vec<f64>> {
        if horizon == 0 {
            return Ok(vec![]);
        }

        let residuals = match &self.residuals {
            Some(r) => r,
            None => return Err(GARCHError::ForecastError("Model not fitted".to_string())),
        };

        let variance = match &self.fitted_variance {
            Some(v) => v,
            None => return Err(GARCHError::ForecastError("Model not fitted".to_string())),
        };

        let n = residuals.len();
        let p = self.alpha.len();
        let q = self.beta.len();

        let mut forecast = Vec::with_capacity(horizon);

        // For each forecast step
        for h in 1..=horizon {
            let mut var_h = self.omega;

            // Add ARCH components
            for i in 0..p {
                if h <= i {
                    // We have actual residuals available
                    // Safe indexing: Only access if within bounds
                    let idx = n.checked_sub(h).and_then(|v| v.checked_add(i));
                    if let Some(idx) = idx {
                        if idx < residuals.len() {
                            var_h += self.alpha[i] * residuals[idx].powi(2);
                        }
                    }
                } else if h > i + 1 && (h - i - 2) < forecast.len() {
                    // Expected value of future squared residuals is the variance
                    // Use forecasted variance safely
                    var_h += self.alpha[i] * forecast[h - i - 2];
                }
            }

            // Add GARCH components
            for j in 0..q {
                if h <= j + 1 {
                    // We have actual variances available
                    // Safe indexing: Only access if within bounds
                    let idx = n.checked_sub(h).and_then(|v| v.checked_add(j + 1));
                    if let Some(idx) = idx {
                        if idx < variance.len() {
                            var_h += self.beta[j] * variance[idx];
                        }
                    }
                } else if (h - j - 2) < forecast.len() {
                    // Use previously forecasted variances safely
                    var_h += self.beta[j] * forecast[h - j - 2];
                }
            }

            forecast.push(var_h);
        }

        Ok(forecast)
    }

    /// Get the order of the GARCH model as (p,q)
    pub fn order(&self) -> (usize, usize) {
        (self.alpha.len(), self.beta.len())
    }
}

impl fmt::Display for GARCHModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (p, q) = self.order();
        writeln!(f, "GARCH({}, {}) Model", p, q)?;
        writeln!(f, "Mean: {:.6}", self.mean)?;
        writeln!(f, "Omega: {:.6}", self.omega)?;

        for (i, &alpha) in self.alpha.iter().enumerate() {
            writeln!(f, "Alpha[{}]: {:.6}", i + 1, alpha)?;
        }

        for (i, &beta) in self.beta.iter().enumerate() {
            writeln!(f, "Beta[{}]: {:.6}", i + 1, beta)?;
        }

        if let Some(ll) = self.log_likelihood {
            writeln!(f, "Log-Likelihood: {:.6}", ll)?;
        }

        if let Some((aic, bic)) = self.info_criteria {
            writeln!(f, "AIC: {:.6}, BIC: {:.6}", aic, bic)?;
        }

        Ok(())
    }
}
