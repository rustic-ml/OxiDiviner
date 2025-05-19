use crate::error::{GARCHError, Result};
use chrono::{DateTime, Utc};
use statrs::statistics::Statistics;
use std::fmt;

/// GARCH-M(p,q) Model - GARCH-in-Mean
///
/// The GARCH-M(p,q) model is defined as:
/// y_t = μ + λ * f(σ²_t) + ε_t
/// ε_t = σ_t * z_t, where z_t ~ N(0,1)
/// σ²_t = ω + Σ(i=1 to p) α_i * ε²_{t-i} + Σ(j=1 to q) β_j * σ²_{t-j}
///
/// Where:
/// - p is the order of the ARCH terms
/// - q is the order of the GARCH terms
/// - μ is the constant term in the mean equation
/// - λ is the risk premium parameter
/// - f(σ²_t) is the risk premium function (variance, standard deviation, or log-variance)
/// - ω is the constant term in the variance equation
/// - α_i are the ARCH parameters
/// - β_j are the GARCH parameters
///
/// The model allows volatility to directly affect returns through the risk premium:
/// - Positive λ: Higher volatility leads to higher expected returns (risk-return tradeoff)
/// - Negative λ: Higher volatility leads to lower expected returns
/// - Zero λ: Standard GARCH model with no risk premium
#[derive(Debug, Clone)]
pub struct GARCHMModel {
    /// Constant term in the mean equation
    pub mean: f64,
    /// Risk premium parameter (lambda)
    pub lambda: f64,
    /// Type of risk premium function
    pub risk_type: RiskPremiumType,
    /// Constant term in the variance equation
    pub omega: f64,
    /// ARCH coefficients (alpha)
    pub alpha: Vec<f64>,
    /// GARCH coefficients (beta)
    pub beta: Vec<f64>,
    /// Fitted variance series
    pub fitted_variance: Option<Vec<f64>>,
    /// Fitted mean series
    pub fitted_mean: Option<Vec<f64>>,
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

/// Type of risk premium function used in the GARCH-M model
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RiskPremiumType {
    /// Use variance (σ²_t) in the mean equation
    Variance,
    /// Use standard deviation (σ_t) in the mean equation
    StdDev,
    /// Use log-variance (ln(σ²_t)) in the mean equation
    LogVariance,
}

impl GARCHMModel {
    /// Create a new GARCH-M(p,q) model with specified parameters
    ///
    /// # Arguments
    ///
    /// * `p` - Order of the ARCH terms
    /// * `q` - Order of the GARCH terms
    /// * `risk_type` - Type of risk premium function to use
    /// * `params` - Optional parameters [mean, lambda, omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]
    ///
    /// # Returns
    ///
    /// A Result containing the GARCHMModel or an error
    pub fn new(
        p: usize,
        q: usize,
        risk_type: RiskPremiumType,
        params: Option<Vec<f64>>,
    ) -> Result<Self> {
        // Validate p and q
        if p == 0 && q == 0 {
            return Err(GARCHError::InvalidParameters(
                "Both p and q cannot be zero".to_string(),
            ));
        }

        let alpha = vec![0.0; p];
        let beta = vec![0.0; q];
        let mean = 0.0;
        let lambda = 0.0;
        let omega = 0.0001; // Small positive value

        let model = if let Some(params) = params {
            // Validate parameters length
            if params.len() != 2 + 1 + p + q {
                return Err(GARCHError::InvalidParameters(format!(
                    "Expected {} parameters, got {}",
                    2 + 1 + p + q,
                    params.len()
                )));
            }

            let mean = params[0];
            let lambda = params[1];
            let omega = params[2];
            
            // Extract alpha and beta parameters
            let mut alpha = Vec::with_capacity(p);
            let mut beta = Vec::with_capacity(q);
            
            for i in 0..p {
                alpha.push(params[3 + i]);
            }
            
            for i in 0..q {
                beta.push(params[3 + p + i]);
            }
            
            // Validate parameters for stationarity and positivity
            Self::validate_parameters(&omega, &alpha, &beta)?;
            
            GARCHMModel {
                mean,
                lambda,
                risk_type,
                omega,
                alpha,
                beta,
                fitted_variance: None,
                fitted_mean: None,
                residuals: None,
                log_likelihood: None,
                info_criteria: None,
                std_errors: None,
                timestamps: None,
            }
        } else {
            // Default initialization
            GARCHMModel {
                mean,
                lambda,
                risk_type,
                omega,
                alpha,
                beta,
                fitted_variance: None,
                fitted_mean: None,
                residuals: None,
                log_likelihood: None,
                info_criteria: None,
                std_errors: None,
                timestamps: None,
            }
        };

        Ok(model)
    }

    /// Validate GARCH-M parameters for stationarity and positivity constraints
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

    /// Apply the risk premium function to the variance
    fn apply_risk_premium(&self, variance: f64) -> f64 {
        match self.risk_type {
            RiskPremiumType::Variance => variance,
            RiskPremiumType::StdDev => variance.sqrt(),
            RiskPremiumType::LogVariance => variance.ln(),
        }
    }

    /// Fit the GARCH-M model to the data
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

        // Initial parameter guess if not already set
        let p = self.alpha.len();
        let q = self.beta.len();
        
        // Default initial parameters if not already set
        if self.omega <= 0.0 {
            // Use sample variance as initial omega
            let sample_mean = data.iter().sum::<f64>() / data.len() as f64;
            let sample_var = data.iter().map(|&x| (x - sample_mean).powi(2)).sum::<f64>() / data.len() as f64;
            self.omega = sample_var * 0.1;
        }
        
        if self.lambda == 0.0 {
            // Small positive lambda as initial value
            self.lambda = 0.05;
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
        
        // Iterative process to fit the model
        // 1. Calculate variance using previous residuals
        // 2. Calculate mean using the variance (risk premium)
        // 3. Calculate new residuals
        // 4. Repeat until convergence or max iterations
        
        // Start with initial residuals (data - mean)
        let initial_residuals: Vec<f64> = data.iter().map(|&x| x - self.mean).collect();
        
        // Calculate the fitted variance
        let mut fitted_variance = self.calculate_variance(&initial_residuals)?;
        
        // Calculate the fitted mean using the risk premium
        let mut fitted_mean = Vec::with_capacity(data.len());
        for &var in &fitted_variance {
            let risk_premium = self.apply_risk_premium(var);
            fitted_mean.push(self.mean + self.lambda * risk_premium);
        }
        
        // Calculate the residuals
        let mut residuals = Vec::with_capacity(data.len());
        for i in 0..data.len() {
            residuals.push(data[i] - fitted_mean[i]);
        }
        
        // Update the variance using the new residuals
        fitted_variance = self.calculate_variance(&residuals)?;
        
        // Recalculate fitted mean with updated variance
        for i in 0..fitted_mean.len() {
            let risk_premium = self.apply_risk_premium(fitted_variance[i]);
            fitted_mean[i] = self.mean + self.lambda * risk_premium;
        }
        
        // Update residuals with final fitted mean
        for i in 0..residuals.len() {
            residuals[i] = data[i] - fitted_mean[i];
        }
        
        // Store results
        self.fitted_variance = Some(fitted_variance);
        self.fitted_mean = Some(fitted_mean);
        self.residuals = Some(residuals);
        
        // Store timestamps if provided
        if let Some(ts) = timestamps {
            self.timestamps = Some(ts.to_vec());
        }
        
        // Calculate log-likelihood and information criteria
        self.calculate_statistics(data)?;
        
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
        for i in 0..max_lag {
            variance[i] = unconditional_variance;
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
    fn calculate_statistics(&mut self, data: &[f64]) -> Result<()> {
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
        let num_params = 3 + p + q; // mean, lambda, omega, alphas, betas
        
        // Calculate log-likelihood
        let mut log_likelihood = 0.0;
        for t in 0..n {
            if variance[t] <= 0.0 {
                return Err(GARCHError::NumericalError(
                    "Negative or zero variance encountered".to_string(),
                ));
            }
            
            log_likelihood += -0.5 * (std::f64::consts::LN_2 + std::f64::consts::PI.ln() + (variance[t]).ln() + residuals[t].powi(2) / variance[t]);
        }
        
        self.log_likelihood = Some(log_likelihood);
        
        // Calculate information criteria
        let aic = -2.0 * log_likelihood + 2.0 * num_params as f64;
        let bic = -2.0 * log_likelihood + (num_params as f64) * (n as f64).ln();
        
        self.info_criteria = Some((aic, bic));
        
        Ok(())
    }
    
    /// Forecast future values and volatility
    ///
    /// # Arguments
    ///
    /// * `horizon` - Number of steps to forecast
    ///
    /// # Returns
    ///
    /// A Result containing (forecasted_means, forecasted_variance)
    pub fn forecast(&self, horizon: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        if horizon == 0 {
            return Ok((vec![], vec![]));
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
        
        let mut forecast_var = Vec::with_capacity(horizon);
        let mut forecast_mean = Vec::with_capacity(horizon);
        
        // For each forecast step
        for h in 1..=horizon {
            let mut var_h = self.omega;
            
            // Add ARCH components
            for i in 0..p {
                if h <= i {
                    // We have actual residuals available
                    var_h += self.alpha[i] * residuals[n - h + i].powi(2);
                } else {
                    // Expected value of future squared residuals is the variance
                    if h - i - 2 < forecast_var.len() {
                        var_h += self.alpha[i] * forecast_var[h - i - 2];
                    }
                }
            }
            
            // Add GARCH components
            for j in 0..q {
                if h <= j + 1 {
                    // We have actual variances available
                    var_h += self.beta[j] * variance[n - h + j + 1];
                } else if h - j - 2 < forecast_var.len() {
                    // Use previously forecasted variances
                    var_h += self.beta[j] * forecast_var[h - j - 2];
                }
            }
            
            forecast_var.push(var_h);
            
            // Calculate mean forecast using the risk premium
            let risk_premium = self.apply_risk_premium(var_h);
            forecast_mean.push(self.mean + self.lambda * risk_premium);
        }
        
        Ok((forecast_mean, forecast_var))
    }
    
    /// Get the order of the GARCH-M model as (p,q)
    pub fn order(&self) -> (usize, usize) {
        (self.alpha.len(), self.beta.len())
    }
}

impl fmt::Display for GARCHMModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (p, q) = self.order();
        let risk_type_str = match self.risk_type {
            RiskPremiumType::Variance => "Variance",
            RiskPremiumType::StdDev => "Standard Deviation",
            RiskPremiumType::LogVariance => "Log-Variance",
        };
        
        writeln!(f, "GARCH-M({}, {}) Model", p, q)?;
        writeln!(f, "Risk Premium Type: {}", risk_type_str)?;
        writeln!(f, "Mean: {:.6}", self.mean)?;
        writeln!(f, "Lambda: {:.6}", self.lambda)?;
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