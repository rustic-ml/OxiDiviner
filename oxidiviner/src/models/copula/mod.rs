//! Copula-based models for dependency structure forecasting
//!
//! Copulas model the dependency structure between variables separately from their
//! marginal distributions. This is particularly useful for:
//! - Multi-asset portfolio risk modeling
//! - Joint forecasting of correlated time series
//! - Tail dependency analysis
//! - Risk management and value-at-risk calculations
//!
//! ## Available Copula Models
//!
//! ### Gaussian Copula
//! - Models linear dependence with normal marginals
//! - No tail dependence
//! - Good for symmetric relationships
//!
//! ### Student's t-Copula
//! - Models dependence with heavy tails
//! - Symmetric tail dependence
//! - Controlled by degrees of freedom parameter
//!
//! ### Archimedean Copulas
//! - **Clayton**: Lower tail dependence, good for modeling crash dependencies
//! - **Gumbel**: Upper tail dependence, good for modeling boom dependencies
//! - **Frank**: Symmetric dependence without tail dependence

use crate::core::{OxiError, Result};

pub mod archimedean_copula;
pub mod gaussian_copula;
pub mod t_copula;

pub use archimedean_copula::{ArchimedeanCopulaModel, ArchimedeanType};
pub use gaussian_copula::GaussianCopulaModel;
pub use t_copula::TCopulaModel;

/// Copula model factory for creating different copula types
pub struct CopulaFactory;

impl CopulaFactory {
    /// Create a Gaussian copula model
    ///
    /// # Arguments
    /// * `n_series` - Number of time series to model jointly
    ///
    /// # Example
    /// ```rust
    /// use oxidiviner::models::copula::CopulaFactory;
    ///
    /// let gaussian_model = CopulaFactory::gaussian_copula(3).unwrap();
    /// ```
    pub fn gaussian_copula(n_series: usize) -> Result<GaussianCopulaModel> {
        GaussianCopulaModel::new(n_series)
    }

    /// Create a Student's t-copula model
    ///
    /// # Arguments
    /// * `n_series` - Number of time series to model jointly
    /// * `degrees_of_freedom` - Degrees of freedom parameter (>2)
    ///
    /// # Example
    /// ```rust
    /// use oxidiviner::models::copula::CopulaFactory;
    ///
    /// let t_model = CopulaFactory::t_copula(3, 5.0).unwrap();
    /// ```
    pub fn t_copula(n_series: usize, degrees_of_freedom: f64) -> Result<TCopulaModel> {
        TCopulaModel::new(n_series, degrees_of_freedom)
    }

    /// Create a Clayton copula model
    ///
    /// # Example
    /// ```rust
    /// use oxidiviner::models::copula::CopulaFactory;
    ///
    /// let clayton_model = CopulaFactory::clayton_copula().unwrap();
    /// ```
    pub fn clayton_copula() -> Result<ArchimedeanCopulaModel> {
        ArchimedeanCopulaModel::new(ArchimedeanType::Clayton, 2)
    }

    /// Create a Gumbel copula model
    ///
    /// # Example
    /// ```rust
    /// use oxidiviner::models::copula::CopulaFactory;
    ///
    /// let gumbel_model = CopulaFactory::gumbel_copula().unwrap();
    /// ```
    pub fn gumbel_copula() -> Result<ArchimedeanCopulaModel> {
        ArchimedeanCopulaModel::new(ArchimedeanType::Gumbel, 2)
    }

    /// Create a Frank copula model
    ///
    /// # Example
    /// ```rust
    /// use oxidiviner::models::copula::CopulaFactory;
    ///
    /// let frank_model = CopulaFactory::frank_copula().unwrap();
    /// ```
    pub fn frank_copula() -> Result<ArchimedeanCopulaModel> {
        ArchimedeanCopulaModel::new(ArchimedeanType::Frank, 2)
    }

    /// Create an Archimedean copula model of specified type
    ///
    /// # Arguments
    /// * `copula_type` - Type of Archimedean copula
    ///
    /// # Example
    /// ```rust
    /// use oxidiviner::models::copula::{CopulaFactory, ArchimedeanType};
    ///
    /// let copula_model = CopulaFactory::archimedean_copula(ArchimedeanType::Clayton).unwrap();
    /// ```
    pub fn archimedean_copula(copula_type: ArchimedeanType) -> Result<ArchimedeanCopulaModel> {
        ArchimedeanCopulaModel::new(copula_type, 2)
    }
}

/// Utility functions for copula analysis
pub mod utils {
    use super::*;

    /// Calculate empirical Kendall's tau between two series
    pub fn empirical_kendall_tau(x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(OxiError::DataError(
                "Series must have the same length".to_string(),
            ));
        }

        let n = x.len();
        if n < 2 {
            return Err(OxiError::DataError(
                "Need at least 2 observations".to_string(),
            ));
        }

        let mut concordant = 0;
        let mut discordant = 0;

        for i in 0..n {
            for j in i + 1..n {
                let sign_x = (x[i] - x[j]).signum();
                let sign_y = (y[i] - y[j]).signum();

                if sign_x * sign_y > 0.0 {
                    concordant += 1;
                } else if sign_x * sign_y < 0.0 {
                    discordant += 1;
                }
            }
        }

        let total_pairs = n * (n - 1) / 2;
        let tau = (concordant as f64 - discordant as f64) / total_pairs as f64;
        Ok(tau)
    }

    /// Calculate empirical Spearman's rho between two series
    pub fn empirical_spearman_rho(x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(OxiError::DataError(
                "Series must have the same length".to_string(),
            ));
        }

        let n = x.len();
        if n < 2 {
            return Err(OxiError::DataError(
                "Need at least 2 observations".to_string(),
            ));
        }

        // Convert to ranks
        let ranks_x = to_ranks(x);
        let ranks_y = to_ranks(y);

        // Calculate Pearson correlation of ranks
        let mean_x = ranks_x.iter().sum::<f64>() / n as f64;
        let mean_y = ranks_y.iter().sum::<f64>() / n as f64;

        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;

        for i in 0..n {
            let dx = ranks_x[i] - mean_x;
            let dy = ranks_y[i] - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }

        if sum_x2 == 0.0 || sum_y2 == 0.0 {
            return Ok(0.0);
        }

        let rho = sum_xy / (sum_x2.sqrt() * sum_y2.sqrt());
        Ok(rho)
    }

    /// Convert values to ranks
    fn to_ranks(values: &[f64]) -> Vec<f64> {
        let n = values.len();
        let mut indexed_values: Vec<(f64, usize)> = values
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();

        indexed_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut ranks = vec![0.0; n];
        for (rank, &(_, original_idx)) in indexed_values.iter().enumerate() {
            ranks[original_idx] = (rank + 1) as f64;
        }

        ranks
    }

    /// Select best copula model based on AIC
    pub fn select_best_copula(data: &[Vec<f64>]) -> Result<String> {
        if data.len() != 2 {
            return Err(OxiError::DataError(
                "Copula selection currently only supports bivariate data".to_string(),
            ));
        }

        let mut best_copula = String::new();
        let mut best_aic = f64::INFINITY;

        // Test Gaussian copula
        let mut gaussian_model = CopulaFactory::gaussian_copula(2)?;
        if gaussian_model.fit_multivariate(data).is_ok() {
            if let Ok((aic, _)) = gaussian_model.information_criteria() {
                if aic < best_aic {
                    best_aic = aic;
                    best_copula = "Gaussian".to_string();
                }
            }
        }

        // Test t-copula
        let mut t_model = CopulaFactory::t_copula(2, 5.0)?;
        if t_model.fit_multivariate(data).is_ok() {
            if let Ok((aic, _)) = t_model.information_criteria() {
                if aic < best_aic {
                    best_aic = aic;
                    best_copula = "t-Copula".to_string();
                }
            }
        }

        // Test Archimedean copulas
        for copula_type in [
            ArchimedeanType::Clayton,
            ArchimedeanType::Gumbel,
            ArchimedeanType::Frank,
        ] {
            let mut arch_model = CopulaFactory::archimedean_copula(copula_type)?;
            if arch_model.fit_bivariate(data).is_ok() {
                if let Ok((aic, _)) = arch_model.information_criteria() {
                    if aic < best_aic {
                        best_aic = aic;
                        best_copula = format!("{}", copula_type);
                    }
                }
            }
        }

        if best_copula.is_empty() {
            return Err(OxiError::ModelError(
                "No suitable copula model found".to_string(),
            ));
        }

        Ok(best_copula)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data() -> Vec<Vec<f64>> {
        // Generate correlated data
        let n = 100;
        let mut data1 = Vec::with_capacity(n);
        let mut data2 = Vec::with_capacity(n);

        for i in 0..n {
            let x = i as f64 / n as f64;
            data1.push(x + 0.1 * (x * 10.0).sin());
            data2.push(0.8 * x + 0.2 * (x * 5.0).cos() + 0.1);
        }

        vec![data1, data2]
    }

    #[test]
    fn test_copula_factory() {
        let gaussian = CopulaFactory::gaussian_copula(2);
        assert!(gaussian.is_ok());

        let t_copula = CopulaFactory::t_copula(2, 5.0);
        assert!(t_copula.is_ok());

        let clayton = CopulaFactory::clayton_copula();
        assert!(clayton.is_ok());

        let gumbel = CopulaFactory::gumbel_copula();
        assert!(gumbel.is_ok());

        let frank = CopulaFactory::frank_copula();
        assert!(frank.is_ok());
    }

    #[test]
    fn test_empirical_kendall_tau() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let tau = utils::empirical_kendall_tau(&x, &y).unwrap();
        assert!((tau - 1.0).abs() < 1e-10); // Perfect positive correlation
    }

    #[test]
    fn test_empirical_spearman_rho() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let rho = utils::empirical_spearman_rho(&x, &y).unwrap();
        assert!((rho - 1.0).abs() < 1e-10); // Perfect positive correlation
    }

    #[test]
    fn test_copula_selection() {
        let data = generate_test_data();
        let best_copula = utils::select_best_copula(&data).unwrap();
        assert!(!best_copula.is_empty());
    }
}
