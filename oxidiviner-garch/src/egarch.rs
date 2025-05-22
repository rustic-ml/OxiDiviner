use crate::error::{GARCHError, Result};
use chrono::{DateTime, Utc};
use std::fmt;

/// EGARCH(p,q) Model - Exponential Generalized Autoregressive Conditional Heteroskedasticity
///
/// The EGARCH(p,q) model is defined as:
/// y_t = μ + ε_t
/// ε_t = σ_t * z_t, where z_t ~ N(0,1)
/// ln(σ²_t) = ω + Σ(i=1 to p) [α_i * |z_{t-i}| + γ_i * z_{t-i}] + Σ(j=1 to q) β_j * ln(σ²_{t-j})
///
/// Where:
/// - p is the order of the ARCH terms
/// - q is the order of the GARCH terms
/// - ω is the constant term (omega)
/// - α_i are the ARCH parameters
/// - γ_i are the leverage parameters (asymmetric effects)
/// - β_j are the GARCH parameters
/// - z_t = ε_t / σ_t are standardized residuals
///
/// The model can capture asymmetric effects in volatility (leverage effect):
/// - When γ < 0, negative shocks have a larger impact on volatility than positive shocks
/// - When γ > 0, positive shocks have a larger impact on volatility than negative shocks
/// - When γ = 0, the model is symmetric
#[derive(Debug, Clone)]
pub struct EGARCHModel {
    /// Mean of the process
    pub mean: f64,
    /// Constant term in the variance equation
    pub omega: f64,
    /// ARCH coefficients (alpha)
    pub alpha: Vec<f64>,
    /// Leverage parameters (gamma) for asymmetric effects
    pub gamma: Vec<f64>,
    /// GARCH coefficients (beta)
    pub beta: Vec<f64>,
    /// Fitted variance series
    pub fitted_variance: Option<Vec<f64>>,
    /// Standardized residuals (z_t = ε_t / σ_t)
    pub std_residuals: Option<Vec<f64>>,
    /// Residuals (ε_t)
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

impl EGARCHModel {
    /// Create a new EGARCH(p,q) model with specified parameters
    ///
    /// # Arguments
    ///
    /// * `p` - Order of the ARCH terms
    /// * `q` - Order of the GARCH terms
    /// * `params` - Optional parameters [mean, omega, alpha_1, ..., alpha_p, gamma_1, ..., gamma_p, beta_1, ..., beta_q]
    ///
    /// # Returns
    ///
    /// A Result containing the EGARCHModel or an error
    pub fn new(p: usize, q: usize, params: Option<Vec<f64>>) -> Result<Self> {
        // Validate p and q
        if p == 0 && q == 0 {
            return Err(GARCHError::InvalidParameters(
                "Both p and q cannot be zero".to_string(),
            ));
        }

        let alpha = vec![0.0; p];
        let gamma = vec![0.0; p];
        let beta = vec![0.0; q];
        let mean = 0.0;
        let omega = 0.0;

        let model = if let Some(params) = params {
            // Validate parameters length (mean, omega, alphas, gammas, betas)
            if params.len() != 1 + 1 + p + p + q {
                return Err(GARCHError::InvalidParameters(format!(
                    "Expected {} parameters, got {}",
                    1 + 1 + p + p + q,
                    params.len()
                )));
            }

            let mean = params[0];
            let omega = params[1];

            // Extract alpha, gamma, and beta parameters
            let mut alpha = Vec::with_capacity(p);
            let mut gamma = Vec::with_capacity(p);
            let mut beta = Vec::with_capacity(q);

            for i in 0..p {
                alpha.push(params[2 + i]);
            }

            for i in 0..p {
                gamma.push(params[2 + p + i]);
            }

            for i in 0..q {
                beta.push(params[2 + 2 * p + i]);
            }

            // Validate parameters for stationarity
            Self::validate_parameters(&alpha, &beta)?;

            EGARCHModel {
                mean,
                omega,
                alpha,
                gamma,
                beta,
                fitted_variance: None,
                std_residuals: None,
                residuals: None,
                log_likelihood: None,
                info_criteria: None,
                std_errors: None,
                timestamps: None,
            }
        } else {
            // Default initialization
            EGARCHModel {
                mean,
                omega,
                alpha,
                gamma,
                beta,
                fitted_variance: None,
                std_residuals: None,
                residuals: None,
                log_likelihood: None,
                info_criteria: None,
                std_errors: None,
                timestamps: None,
            }
        };

        Ok(model)
    }

    /// Validate EGARCH parameters for stationarity
    fn validate_parameters(_alpha: &[f64], beta: &[f64]) -> Result<()> {
        // Check stationarity condition: sum of beta < 1
        let sum_beta: f64 = beta.iter().sum();
        if sum_beta >= 1.0 {
            return Err(GARCHError::InvalidParameters(
                "Sum of beta must be less than 1 for stationarity".to_string(),
            ));
        }

        Ok(())
    }

    /// Fit the EGARCH model to the data
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

        // Initial parameter guess if not already set
        let p = self.alpha.len();
        let q = self.beta.len();

        // Default initial omega
        if self.omega == 0.0 {
            // Use log of sample variance as initial omega
            let sample_var = residuals.iter().map(|&r| r * r).sum::<f64>() / residuals.len() as f64;
            self.omega = sample_var.ln() * 0.1;
        }

        // Default initial alpha values
        if self.alpha.iter().all(|&a| a == 0.0) {
            for i in 0..p {
                self.alpha[i] = 0.05 + (0.05 / (i + 1) as f64);
            }
        }

        // Default initial gamma values (slight negative bias for leverage effect)
        if self.gamma.iter().all(|&g| g == 0.0) {
            for i in 0..p {
                self.gamma[i] = -0.05;
            }
        }

        // Default initial beta values
        if self.beta.iter().all(|&b| b == 0.0) {
            for i in 0..q {
                self.beta[i] = 0.8 / q as f64;
            }
        }

        // Calculate the fitted variance
        let (fitted_variance, std_residuals) = self.calculate_variance(&residuals)?;

        // Store results
        self.residuals = Some(residuals);
        self.fitted_variance = Some(fitted_variance);
        self.std_residuals = Some(std_residuals);

        // Store timestamps if provided
        if let Some(ts) = timestamps {
            self.timestamps = Some(ts.to_vec());
        }

        // Calculate log-likelihood and information criteria
        self.calculate_statistics()?;

        Ok(())
    }

    /// Calculate the conditional variance based on the model parameters
    fn calculate_variance(&self, residuals: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = residuals.len();
        let p = self.alpha.len();
        let q = self.beta.len();
        let max_lag = p.max(q);

        if n <= max_lag {
            return Err(GARCHError::InvalidData(
                "Not enough data points for the specified model".to_string(),
            ));
        }

        // Calculate unconditional variance for initialization
        let unconditional_variance = residuals.iter().map(|&r| r * r).sum::<f64>() / n as f64;
        let log_unconditional_var = unconditional_variance.ln();

        let mut log_variance = vec![log_unconditional_var; n];
        let mut variance = vec![unconditional_variance; n];
        let mut std_residuals = vec![0.0; n];

        // Initialize standardized residuals for the first max_lag observations
        for i in 0..max_lag {
            std_residuals[i] = residuals[i] / unconditional_variance.sqrt();
        }

        // Calculate variance for the rest of the series
        for t in max_lag..n {
            let mut log_var_t = self.omega;

            // Add ARCH and leverage components
            for i in 0..p {
                let z_t_i = std_residuals[t - i - 1];
                log_var_t += self.alpha[i] * z_t_i.abs() + self.gamma[i] * z_t_i;
            }

            // Add GARCH components
            for j in 0..q {
                log_var_t += self.beta[j] * log_variance[t - j - 1];
            }

            log_variance[t] = log_var_t;
            variance[t] = log_var_t.exp();

            // Calculate standardized residual
            std_residuals[t] = residuals[t] / variance[t].sqrt();
        }

        Ok((variance, std_residuals))
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
        let num_params = 2 + 2 * p + q; // mean, omega, alphas, gammas, betas

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

        let std_residuals = match &self.std_residuals {
            Some(z) => z,
            None => return Err(GARCHError::ForecastError("Model not fitted".to_string())),
        };

        let variance = match &self.fitted_variance {
            Some(v) => v,
            None => return Err(GARCHError::ForecastError("Model not fitted".to_string())),
        };

        let n = std_residuals.len();
        let p = self.alpha.len();
        let q = self.beta.len();

        // Create vectors to store forecasted log-variances and variances
        let mut forecast_log_var = Vec::with_capacity(horizon);
        let mut forecast_var = Vec::with_capacity(horizon);

        // Last log-variance from fitted data
        let mut last_log_variances = Vec::with_capacity(q);
        for j in 0..q {
            if j < n {
                last_log_variances.push(variance[n - j - 1].ln());
            }
        }

        // Last standardized residuals
        let mut last_std_residuals = Vec::with_capacity(p);
        for i in 0..p {
            if i < n {
                last_std_residuals.push(std_residuals[n - i - 1]);
            }
        }

        // For each forecast step
        for h in 1..=horizon {
            let mut log_var_h = self.omega;

            // Add ARCH and leverage components
            for i in 0..p {
                if i < last_std_residuals.len() {
                    let z = last_std_residuals[i];
                    // For h=1, we use actual standardized residuals
                    // For h>1, the expectation of |z| is sqrt(2/π) and E[z] is 0
                    if h == 1 {
                        log_var_h += self.alpha[i] * z.abs() + self.gamma[i] * z;
                    } else {
                        log_var_h += self.alpha[i] * (2.0 / std::f64::consts::PI).sqrt();
                        // E[z] = 0, so gamma term disappears
                    }
                }
            }

            // Add GARCH components
            for j in 0..q {
                if j < last_log_variances.len() {
                    log_var_h += self.beta[j] * last_log_variances[j];
                } else if j < forecast_log_var.len() {
                    log_var_h += self.beta[j] * forecast_log_var[j];
                }
            }

            // Add to forecast and update last values
            forecast_log_var.push(log_var_h);
            forecast_var.push(log_var_h.exp());

            // Shift last values for next iteration
            if h < horizon {
                // For standardized residuals beyond h=1, we use expected value
                // which is 0 under normality assumption
                last_std_residuals.insert(0, 0.0);
                if last_std_residuals.len() > p {
                    last_std_residuals.pop();
                }

                last_log_variances.insert(0, log_var_h);
                if last_log_variances.len() > q {
                    last_log_variances.pop();
                }
            }
        }

        Ok(forecast_var)
    }

    /// Get the order of the EGARCH model as (p,q)
    pub fn order(&self) -> (usize, usize) {
        (self.alpha.len(), self.beta.len())
    }

    /// Calculate the news impact curve (NIC)
    ///
    /// The news impact curve shows the relationship between past shocks (ε_{t-1})
    /// and current conditional variance (σ²_t), holding everything else constant.
    ///
    /// # Arguments
    ///
    /// * `points` - Number of points to evaluate the curve
    /// * `range` - Range of standardized residuals to evaluate (-range to +range)
    ///
    /// # Returns
    ///
    /// A tuple of (shock_values, variance_values) representing the news impact curve
    pub fn news_impact_curve(&self, points: usize, range: f64) -> (Vec<f64>, Vec<f64>) {
        if points == 0 || range <= 0.0 {
            return (vec![], vec![]);
        }

        let step = 2.0 * range / (points as f64 - 1.0);
        let mut shock_values = Vec::with_capacity(points);
        let mut variance_values = Vec::with_capacity(points);

        // Assume we only care about the first lag effect (most recent shock)
        // and we set all other lags to their unconditional expectations
        for i in 0..points {
            let z = -range + (i as f64) * step;
            shock_values.push(z);

            // Calculate log variance based on this shock
            let mut log_var = self.omega;

            // First lag uses the shock value we're evaluating
            if !self.alpha.is_empty() {
                log_var += self.alpha[0] * z.abs() + self.gamma[0] * z;
            }

            // Other ARCH lags use expected value of |z| which is sqrt(2/π)
            for i in 1..self.alpha.len() {
                log_var += self.alpha[i] * (2.0 / std::f64::consts::PI).sqrt();
                // E[z] = 0 so gamma terms for lags > 1 disappear
            }

            // GARCH lags use long-run average log variance
            // In long run, E[ln(σ²)] = ω / (1 - sum(β))
            let sum_beta: f64 = self.beta.iter().sum();
            if sum_beta < 1.0 {
                let long_run_log_var = self.omega / (1.0 - sum_beta);
                for &beta in &self.beta {
                    log_var += beta * long_run_log_var;
                }
            }

            // Convert log variance to variance
            variance_values.push(log_var.exp());
        }

        (shock_values, variance_values)
    }
}

impl fmt::Display for EGARCHModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (p, q) = self.order();
        writeln!(f, "EGARCH({}, {}) Model", p, q)?;
        writeln!(f, "Mean: {:.6}", self.mean)?;
        writeln!(f, "Omega: {:.6}", self.omega)?;

        for (i, &alpha) in self.alpha.iter().enumerate() {
            writeln!(f, "Alpha[{}]: {:.6}", i + 1, alpha)?;
        }

        for (i, &gamma) in self.gamma.iter().enumerate() {
            writeln!(f, "Gamma[{}]: {:.6}", i + 1, gamma)?;
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
