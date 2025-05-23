use crate::models::{GARCHError, Result};
use chrono::{DateTime, Utc};
use std::fmt;

/// GJR-GARCH(p,q) Model - Glosten-Jagannathan-Runkle GARCH
///
/// The GJR-GARCH(p,q) model is defined as:
/// y_t = μ + ε_t
/// ε_t = σ_t * z_t, where z_t ~ N(0,1)
/// σ²_t = ω + Σ(i=1 to p) [α_i * ε²_{t-i} + γ_i * ε²_{t-i} * I_{t-i}] + Σ(j=1 to q) β_j * σ²_{t-j}
///
/// Where:
/// - p is the order of the ARCH terms
/// - q is the order of the GARCH terms
/// - ω is the constant term (omega)
/// - α_i are the ARCH parameters
/// - γ_i are the asymmetry parameters
/// - β_j are the GARCH parameters
/// - I_{t-i} is an indicator function: I_{t-i} = 1 if ε_{t-i} < 0 (negative shock), otherwise 0
///
/// The model can capture asymmetric effects in volatility (leverage effect):
/// - When γ > 0, negative shocks have a larger impact (α + γ) on volatility than positive shocks (α)
/// - This is consistent with the leverage effect observed in financial markets
#[derive(Debug, Clone)]
pub struct GJRGARCHModel {
    /// Mean of the process
    pub mean: f64,
    /// Constant term in the variance equation
    pub omega: f64,
    /// ARCH coefficients (alpha)
    pub alpha: Vec<f64>,
    /// Asymmetry parameters (gamma) for negative shocks
    pub gamma: Vec<f64>,
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

impl GJRGARCHModel {
    /// Create a new GJR-GARCH(p,q) model with specified parameters
    ///
    /// # Arguments
    ///
    /// * `p` - Order of the ARCH terms
    /// * `q` - Order of the GARCH terms
    /// * `params` - Optional parameters [mean, omega, alpha_1, ..., alpha_p, gamma_1, ..., gamma_p, beta_1, ..., beta_q]
    ///
    /// # Returns
    ///
    /// A Result containing the GJRGARCHModel or an error
    pub fn new(p: usize, q: usize, params: Option<Vec<f64>>) -> Result<Self> {
        // Validate p and q
        if p == 0 && q == 0 {
            return Err(OxiError::from(GARCHError::InvalidParameters(
                "Both p and q cannot be zero".to_string(),
            ));
        }

        let alpha = vec![0.0; p];
        let gamma = vec![0.0; p];
        let beta = vec![0.0; q];
        let mean = 0.0;
        let omega = 0.0001; // Small positive value

        let model = if let Some(params) = params {
            // Validate parameters length
            if params.len() != 1 + 1 + p + p + q {
                return Err(OxiError::from(GARCHError::InvalidParameters(format!(
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

            // Validate parameters for stationarity and positivity
            Self::validate_parameters(&omega, &alpha, &gamma, &beta)?;

            GJRGARCHModel {
                mean,
                omega,
                alpha,
                gamma,
                beta,
                fitted_variance: None,
                residuals: None,
                log_likelihood: None,
                info_criteria: None,
                std_errors: None,
                timestamps: None,
            }
        } else {
            // Default initialization
            GJRGARCHModel {
                mean,
                omega,
                alpha,
                gamma,
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

    /// Validate GJR-GARCH parameters for stationarity and positivity constraints
    fn validate_parameters(omega: &f64, alpha: &[f64], gamma: &[f64], beta: &[f64]) -> Result<()> {
        // Check positivity constraints
        if *omega <= 0.0 {
            return Err(OxiError::from(GARCHError::InvalidParameters(
                "Omega must be positive".to_string(),
            ));
        }

        for &a in alpha {
            if a < 0.0 {
                return Err(OxiError::from(GARCHError::InvalidParameters(
                    "Alpha parameters must be non-negative".to_string(),
                ));
            }
        }

        // Gamma can be negative as long as alpha + gamma/2 > 0
        // This ensures positive variance when considering the expected contribution
        for i in 0..alpha.len() {
            let a = alpha[i];
            let g = gamma[i];
            if a + g / 2.0 < 0.0 {
                return Err(OxiError::from(GARCHError::InvalidParameters(format!(
                    "Alpha[{}] + Gamma[{}]/2 must be non-negative for positive variance",
                    i + 1,
                    i + 1
                )));
            }
        }

        for &b in beta {
            if b < 0.0 {
                return Err(OxiError::from(GARCHError::InvalidParameters(
                    "Beta parameters must be non-negative".to_string(),
                ));
            }
        }

        // Check stationarity condition: sum of alpha, gamma/2, and beta < 1
        // (gamma/2 because negative shocks occur 50% of the time under normality)
        let sum: f64 =
            alpha.iter().sum::<f64>() + gamma.iter().sum::<f64>() / 2.0 + beta.iter().sum::<f64>();

        if sum >= 1.0 {
            return Err(OxiError::from(GARCHError::InvalidParameters(
                "Sum of alpha + gamma/2 + beta must be less than 1 for stationarity".to_string(),
            ));
        }

        Ok(())
    }

    /// Fit the GJR-GARCH model to the data
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
            return Err(OxiError::from(GARCHError::InvalidData(
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
                self.alpha[i] = 0.03 + (0.02 / (i + 1) as f64);
            }
        }

        // Default initial gamma values (positive for leverage effect)
        if self.gamma.iter().all(|&g| g == 0.0) {
            for i in 0..p {
                self.gamma[i] = 0.05 + (0.02 / (i + 1) as f64);
            }
        }

        if self.beta.iter().all(|&b| b == 0.0) {
            for i in 0..q {
                self.beta[i] = 0.7 / q as f64;
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
            return Err(OxiError::from(GARCHError::InvalidData(
                "Not enough data points for the specified model".to_string(),
            ));
        }

        let mut variance = vec![0.0; n];

        // Initialize the first elements with historical values
        let max_lag = p.max(q);

        // Initialize with unconditional variance
        let unconditional_variance = residuals.iter().map(|&r| r * r).sum::<f64>() / n as f64;

        // Instead of a range loop, use an iterator on the slice
        for variance_item in variance.iter_mut().take(max_lag) {
            *variance_item = unconditional_variance;
        }

        // Calculate variance for the rest of the series
        for t in max_lag..n {
            let mut var_t = self.omega;

            // Add ARCH and asymmetry components
            for i in 0..p {
                let eps_squared = residuals[t - i - 1].powi(2);
                let indicator = if residuals[t - i - 1] < 0.0 { 1.0 } else { 0.0 };

                var_t += self.alpha[i] * eps_squared + self.gamma[i] * eps_squared * indicator;
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
            None => return Err(OxiError::from(GARCHError::EstimationError("Model not fitted".to_string())),
        };

        let variance = match &self.fitted_variance {
            Some(v) => v,
            None => return Err(OxiError::from(GARCHError::EstimationError("Model not fitted".to_string())),
        };

        let n = residuals.len();
        let p = self.alpha.len();
        let q = self.beta.len();
        let num_params = 2 + 2 * p + q; // mean, omega, alphas, gammas, betas

        // Calculate log-likelihood
        let mut log_likelihood = 0.0;
        for t in 0..n {
            if variance[t] <= 0.0 {
                return Err(OxiError::from(GARCHError::NumericalError(
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
            None => return Err(OxiError::from(GARCHError::ForecastError("Model not fitted".to_string())),
        };

        let variance = match &self.fitted_variance {
            Some(v) => v,
            None => return Err(OxiError::from(GARCHError::ForecastError("Model not fitted".to_string())),
        };

        let n = residuals.len();
        let p = self.alpha.len();
        let q = self.beta.len();

        let mut forecast = Vec::with_capacity(horizon);

        // For each forecast step
        for h in 1..=horizon {
            let mut var_h = self.omega;

            // Add ARCH and asymmetry components
            for i in 0..p {
                if h <= i {
                    // We have actual residuals available
                    let eps_squared = residuals[n - h + i].powi(2);
                    let indicator = if residuals[n - h + i] < 0.0 { 1.0 } else { 0.0 };
                    var_h += self.alpha[i] * eps_squared + self.gamma[i] * eps_squared * indicator;
                } else {
                    // Expected value of future squared residuals is the variance
                    // Expected value of indicator is 0.5 under zero mean normality
                    if h - i - 2 < forecast.len() {
                        var_h += (self.alpha[i] + self.gamma[i] * 0.5) * forecast[h - i - 2];
                    }
                }
            }

            // Add GARCH components
            for j in 0..q {
                if h <= j + 1 {
                    // We have actual variances available
                    var_h += self.beta[j] * variance[n - h + j + 1];
                } else if h - j - 2 < forecast.len() {
                    // Use previously forecasted variances
                    var_h += self.beta[j] * forecast[h - j - 2];
                }
            }

            forecast.push(var_h);
        }

        Ok(forecast)
    }

    /// Get the order of the GJR-GARCH model as (p,q)
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
    /// * `range` - Range of residuals to evaluate (-range to +range)
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

        // Calculate unconditional variance for GARCH terms
        let sum_alpha = self.alpha.iter().sum::<f64>();
        let sum_gamma_half = self.gamma.iter().sum::<f64>() / 2.0; // 50% of shocks are negative
        let sum_beta = self.beta.iter().sum::<f64>();

        let unconditional_variance = if sum_alpha + sum_gamma_half + sum_beta < 1.0 {
            self.omega / (1.0 - sum_alpha - sum_gamma_half - sum_beta)
        } else {
            1.0 // Default if model is not stationary
        };

        for i in 0..points {
            let shock = -range + (i as f64) * step;
            shock_values.push(shock);

            // Calculate variance based on this shock
            let mut var = self.omega;

            // First lag uses the shock value we're evaluating
            if !self.alpha.is_empty() {
                let indicator = if shock < 0.0 { 1.0 } else { 0.0 };
                var += self.alpha[0] * shock.powi(2) + self.gamma[0] * shock.powi(2) * indicator;
            }

            // For higher order models, other ARCH lags use unconditional variance
            for i in 1..self.alpha.len() {
                var += (self.alpha[i] + self.gamma[i] * 0.5) * unconditional_variance;
            }

            // GARCH lags use unconditional variance
            for &beta in &self.beta {
                var += beta * unconditional_variance;
            }

            variance_values.push(var);
        }

        (shock_values, variance_values)
    }
}

impl fmt::Display for GJRGARCHModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (p, q) = self.order();
        writeln!(f, "GJR-GARCH({}, {}) Model", p, q)?;
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
