//! Archimedean Copula Models
//!
//! Implementation of Archimedean copulas including Clayton, Gumbel, and Frank copulas.
//! These copulas are particularly useful for modeling different types of dependence:
//! - Clayton: Lower tail dependence
//! - Gumbel: Upper tail dependence  
//! - Frank: Symmetric dependence

use crate::core::{Forecaster, ModelEvaluation, OxiError, Result, TimeSeriesData};

/// Type of Archimedean copula
#[derive(Debug, Clone, Copy)]
pub enum ArchimedeanType {
    /// Clayton copula - exhibits lower tail dependence
    Clayton,
    /// Gumbel copula - exhibits upper tail dependence
    Gumbel,
    /// Frank copula - symmetric dependence, no tail dependence
    Frank,
}

impl std::fmt::Display for ArchimedeanType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchimedeanType::Clayton => write!(f, "Clayton"),
            ArchimedeanType::Gumbel => write!(f, "Gumbel"),
            ArchimedeanType::Frank => write!(f, "Frank"),
        }
    }
}

/// Archimedean Copula Model
///
/// Models dependency using generator functions specific to each copula family.
/// Each copula family has different tail dependence properties suitable for
/// different applications.
#[derive(Debug, Clone)]
pub struct ArchimedeanCopulaModel {
    /// Type of Archimedean copula
    copula_type: ArchimedeanType,
    /// Copula parameter (θ)
    theta: f64,
    /// Marginal data (after transformation to uniform)
    uniform_data: Vec<Vec<f64>>,
    /// Whether the model has been fitted
    is_fitted: bool,
    /// Sample size
    n_obs: usize,
}

impl ArchimedeanCopulaModel {
    /// Create a new Archimedean copula model
    ///
    /// # Arguments
    /// * `copula_type` - Type of Archimedean copula
    ///
    /// # Example
    /// ```rust
    /// use oxidiviner::models::copula::{ArchimedeanCopulaModel, ArchimedeanType};
    ///
    /// let model = ArchimedeanCopulaModel::new(ArchimedeanType::Clayton).unwrap();
    /// ```
    pub fn new(copula_type: ArchimedeanType) -> Result<Self> {
        // Currently only supports bivariate case, n_variables was removed.
        Ok(Self {
            copula_type,
            theta: 1.0,                        // Default parameter
            uniform_data: vec![Vec::new(); 2], // Hardcoded to 2 for bivariate
            is_fitted: false,
            n_obs: 0,
        })
    }

    /// Transform data to uniform margins using empirical CDF
    fn transform_to_uniform(&mut self, data: &[Vec<f64>]) -> Result<()> {
        for (var_idx, series) in data.iter().enumerate() {
            let n = series.len();

            let mut indexed_data: Vec<(f64, usize)> = series
                .iter()
                .enumerate()
                .map(|(i, &val)| (val, i))
                .collect();
            indexed_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let mut uniform_vals = vec![0.0; n];
            for (rank, &(_, original_idx)) in indexed_data.iter().enumerate() {
                // Use (rank + 1) / (n + 1) to avoid 0 and 1
                uniform_vals[original_idx] = (rank + 1) as f64 / (n + 1) as f64;
            }

            self.uniform_data[var_idx] = uniform_vals;
        }
        Ok(())
    }

    /// Log-likelihood function for parameter estimation
    fn log_likelihood(&self, theta: f64) -> f64 {
        if self.uniform_data.is_empty() || self.uniform_data[0].is_empty() {
            return f64::NEG_INFINITY;
        }

        let _old_theta = self.theta;
        // Temporarily set theta for calculation
        let mut temp_model = self.clone();
        temp_model.theta = theta;

        let mut log_lik = 0.0;
        let n = self.uniform_data[0].len();

        for i in 0..n {
            let u = self.uniform_data[0][i];
            let v = self.uniform_data[1][i];

            // Copula density calculation depends on the type
            let density = match self.copula_type {
                ArchimedeanType::Clayton => {
                    if theta <= 0.0 {
                        return f64::NEG_INFINITY;
                    }
                    let term1 = 1.0 + theta;
                    let term2 = u.powf(-theta - 1.0) + v.powf(-theta - 1.0) - 1.0;
                    let term3 = term2.powf(-1.0 / theta - 2.0);
                    term1 * term3
                }
                ArchimedeanType::Gumbel => {
                    if theta < 1.0 {
                        return f64::NEG_INFINITY;
                    }
                    let ln_u = u.ln();
                    let ln_v = v.ln();
                    let term1 = (-ln_u).powf(theta) + (-ln_v).powf(theta);
                    let term2 = term1.powf(1.0 / theta);
                    let term3 = (-term2).exp();
                    let term4 = term1.powf(1.0 / theta - 2.0);
                    let _term5 = (theta - 1.0) * (ln_u * ln_v);
                    let density_val = term3 * term4 * (term1 + (theta - 1.0)) / (u * v);
                    density_val.abs() // Ensure positive
                }
                ArchimedeanType::Frank => {
                    if theta == 0.0 {
                        return 0.0; // Independence case
                    }
                    let exp_theta = (-theta).exp();
                    let exp_theta_u = (-theta * u).exp();
                    let exp_theta_v = (-theta * v).exp();
                    let numerator =
                        theta * (1.0 - exp_theta) * (exp_theta_u + exp_theta_v - exp_theta);
                    let denominator = (1.0 - exp_theta_u) * (1.0 - exp_theta_v) * (1.0 - exp_theta);
                    if denominator != 0.0 {
                        numerator / denominator
                    } else {
                        1e-10
                    }
                }
            };

            if density > 0.0 && density.is_finite() {
                log_lik += density.ln();
            } else {
                return f64::NEG_INFINITY;
            }
        }

        log_lik
    }

    /// Estimate copula parameter using maximum likelihood
    fn estimate_parameter(&mut self) -> Result<()> {
        if self.uniform_data.is_empty() || self.uniform_data[0].is_empty() {
            return Err(OxiError::ModelError(
                "No data available for parameter estimation".to_string(),
            ));
        }

        // Parameter bounds for different copula types
        let (theta_min, theta_max, theta_init) = match self.copula_type {
            ArchimedeanType::Clayton => (0.1, 20.0, 1.0),
            ArchimedeanType::Gumbel => (1.001, 20.0, 2.0),
            ArchimedeanType::Frank => (-20.0, 20.0, 1.0),
        };

        // Simple grid search for parameter estimation
        let mut best_theta = theta_init;
        let mut best_log_lik = self.log_likelihood(theta_init);

        let n_grid = 50;
        for i in 0..n_grid {
            let theta = theta_min + (theta_max - theta_min) * i as f64 / (n_grid - 1) as f64;
            let log_lik = self.log_likelihood(theta);

            if log_lik > best_log_lik {
                best_log_lik = log_lik;
                best_theta = theta;
            }
        }

        self.theta = best_theta;
        Ok(())
    }

    /// Fit the Archimedean copula to bivariate data
    pub fn fit_bivariate(&mut self, data: &[Vec<f64>]) -> Result<()> {
        if data.len() != 2 {
            return Err(OxiError::DataError(
                "Archimedean copula requires exactly 2 variables".to_string(),
            ));
        }

        let n_obs = data[0].len();
        if data[1].len() != n_obs {
            return Err(OxiError::DataError(
                "All variables must have the same number of observations".to_string(),
            ));
        }

        if n_obs < 10 {
            return Err(OxiError::DataError(
                "Need at least 10 observations for copula fitting".to_string(),
            ));
        }

        self.n_obs = n_obs;

        // Transform to uniform margins
        self.transform_to_uniform(data)?;

        // Estimate copula parameter
        self.estimate_parameter()?;

        self.is_fitted = true;
        Ok(())
    }

    /// Generate bivariate samples from the fitted copula
    pub fn simulate(&self, n_samples: usize) -> Result<Vec<(f64, f64)>> {
        if !self.is_fitted {
            return Err(OxiError::ModelError("Model not fitted".to_string()));
        }

        // For simplicity, return empirical samples
        // In practice, would implement proper simulation algorithms
        let mut samples = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let idx = i % self.n_obs;
            let u = self.uniform_data[0][idx];
            let v = self.uniform_data[1][idx];
            samples.push((u, v));
        }

        Ok(samples)
    }

    /// Get the estimated parameter
    pub fn parameter(&self) -> f64 {
        self.theta
    }

    /// Get the copula type
    pub fn copula_type(&self) -> ArchimedeanType {
        self.copula_type
    }

    /// Calculate Kendall's tau (rank correlation)
    pub fn kendall_tau(&self) -> Result<f64> {
        if !self.is_fitted {
            return Err(OxiError::ModelError("Model not fitted".to_string()));
        }

        // Theoretical Kendall's tau for different copula families
        let tau = match self.copula_type {
            ArchimedeanType::Clayton => self.theta / (self.theta + 2.0),
            ArchimedeanType::Gumbel => (self.theta - 1.0) / self.theta,
            ArchimedeanType::Frank => {
                // Approximation for Frank copula
                if self.theta.abs() < 1e-6 {
                    0.0
                } else {
                    1.0 - 4.0 / self.theta + 4.0 * self.debye_function(self.theta) / self.theta
                }
            }
        };

        assert!((0.0..=1.0).contains(&tau));

        Ok(tau)
    }

    /// Debye function for Frank copula Kendall's tau calculation
    fn debye_function(&self, x: f64) -> f64 {
        // First-order Debye function: D_1(x) = (1/x) * integral_0^x (t/(e^t - 1)) dt
        // Approximation for practical use
        if x.abs() < 1e-6 {
            1.0 - x / 4.0
        } else {
            x / (x.exp() - 1.0)
        }
    }

    /// Get information criteria
    pub fn information_criteria(&self) -> Result<(f64, f64)> {
        if !self.is_fitted {
            return Err(OxiError::ModelError("Model not fitted".to_string()));
        }

        let log_lik = self.log_likelihood(self.theta);
        let n_params = 1.0; // Only theta parameter
        let n_obs = self.n_obs as f64;

        let aic = -2.0 * log_lik + 2.0 * n_params;
        let bic = -2.0 * log_lik + n_params * n_obs.ln();

        Ok((aic, bic))
    }
}

impl Forecaster for ArchimedeanCopulaModel {
    fn name(&self) -> &str {
        match self.copula_type {
            ArchimedeanType::Clayton => "Clayton Copula",
            ArchimedeanType::Gumbel => "Gumbel Copula",
            ArchimedeanType::Frank => "Frank Copula",
        }
    }

    fn fit(&mut self, _data: &TimeSeriesData) -> Result<()> {
        Err(OxiError::ModelError(
            "Use fit_bivariate for Archimedean copula models".to_string(),
        ))
    }

    fn forecast(&self, _horizon: usize) -> Result<Vec<f64>> {
        Err(OxiError::ModelError(
            "Archimedean copulas are dependency models, not forecasting models".to_string(),
        ))
    }

    fn evaluate(&self, _test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        if !self.is_fitted {
            return Err(OxiError::ModelError("Model not fitted".to_string()));
        }

        // For copulas, evaluation is based on goodness-of-fit rather than forecasting
        let (aic, bic) = self.information_criteria()?;
        let tau = self.kendall_tau()?;

        Ok(ModelEvaluation {
            model_name: format!("{} Copula", self.copula_type),
            mae: f64::NAN, // Not applicable for copulas
            mse: f64::NAN,
            rmse: f64::NAN,
            mape: f64::NAN,
            smape: f64::NAN,
            r_squared: tau.powi(2), // Use tau^2 as a measure of dependence strength
            aic: Some(aic),
            bic: Some(bic),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data() -> Vec<Vec<f64>> {
        // Generate correlated data with known dependence structure
        let n = 100;
        let mut data1 = Vec::with_capacity(n);
        let mut data2 = Vec::with_capacity(n);

        for i in 0..n {
            let x = i as f64 / n as f64;
            data1.push(x + 0.1 * (x * 10.0).sin());
            data2.push(x.powf(1.5) + 0.05 * (x * 15.0).cos());
        }

        vec![data1, data2]
    }

    #[test]
    fn test_clayton_copula() {
        let mut model = ArchimedeanCopulaModel::new(ArchimedeanType::Clayton).unwrap();
        let data = generate_test_data();

        let result = model.fit_bivariate(&data);
        assert!(result.is_ok());
        assert!(model.is_fitted);
        assert!(model.parameter() > 0.0);
    }

    #[test]
    fn test_gumbel_copula() {
        let mut model = ArchimedeanCopulaModel::new(ArchimedeanType::Gumbel).unwrap();
        let data = generate_test_data();

        let result = model.fit_bivariate(&data);
        assert!(result.is_ok());
        assert!(model.parameter() >= 1.0);
    }

    #[test]
    fn test_frank_copula() {
        let mut model = ArchimedeanCopulaModel::new(ArchimedeanType::Frank).unwrap();
        let data = generate_test_data();

        let result = model.fit_bivariate(&data);
        assert!(result.is_ok());
        assert!(model.is_fitted);
    }

    #[test]
    fn test_kendall_tau() {
        let mut model = ArchimedeanCopulaModel::new(ArchimedeanType::Clayton).unwrap();
        let data = generate_test_data();

        model.fit_bivariate(&data).unwrap();
        let tau = model.kendall_tau().unwrap();
        assert!((0.0..=1.0).contains(&tau));
    }

    #[test]
    fn test_simulation() {
        let mut model = ArchimedeanCopulaModel::new(ArchimedeanType::Clayton).unwrap();
        let data = generate_test_data();

        model.fit_bivariate(&data).unwrap();
        let samples = model.simulate(50).unwrap();
        assert_eq!(samples.len(), 50);

        for (u, v) in samples {
            assert!((0.0..=1.0).contains(&u));
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_information_criteria() {
        let mut model = ArchimedeanCopulaModel::new(ArchimedeanType::Frank).unwrap();
        let data = generate_test_data();

        model.fit_bivariate(&data).unwrap();
        let (aic, bic) = model.information_criteria().unwrap();
        assert!(aic.is_finite());
        assert!(bic.is_finite());
    }
}
