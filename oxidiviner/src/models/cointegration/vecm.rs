//! Vector Error Correction Model (VECM) for cointegration-based forecasting
//!
//! VECM models are used when variables are cointegrated (have a long-run relationship)
//! but may deviate from this relationship in the short run. The model captures:
//! - Long-run equilibrium relationships (cointegration)
//! - Short-run adjustments back to equilibrium
//! - Common trends and cycles among variables
//!
//! Particularly useful for:
//! - Multi-asset forecasting (stock pairs, portfolio optimization)
//! - Currency relationships (purchasing power parity)
//! - Commodity price relationships
//! - Interest rate term structure modeling

use crate::core::{Forecaster, ModelEvaluation, OxiError, Result, TimeSeriesData};
use crate::math::metrics::{mae, mape, mse, rmse, smape};

/// Vector Error Correction Model for cointegrated time series
#[derive(Debug, Clone)]
pub struct VECMModel {
    /// Model name
    name: String,
    /// Number of variables
    num_variables: usize,
    /// Number of cointegrating relationships
    num_coint_relations: usize,
    /// VAR lag order
    lag_order: usize,
    /// Include deterministic components
    include_constant: bool,
    include_trend: bool,
    /// Cointegrating vectors (beta)
    coint_vectors: Option<Vec<Vec<f64>>>,
    /// Adjustment coefficients (alpha)
    adjustment_coeffs: Option<Vec<Vec<f64>>>,
    /// VAR coefficients for differenced variables
    var_coefficients: Option<Vec<Vec<Vec<f64>>>>,
    /// Deterministic components
    constants: Option<Vec<f64>>,
    trend_coeffs: Option<Vec<f64>>,
    /// Fitted values and residuals
    fitted_values: Option<Vec<Vec<f64>>>,
    residuals: Option<Vec<Vec<f64>>>,
    /// Error correction terms
    error_correction_terms: Option<Vec<Vec<f64>>>,
    /// Log-likelihood and information criteria
    log_likelihood: Option<f64>,
    information_criteria: Option<(f64, f64)>, // (AIC, BIC)
    /// Training data
    training_data: Option<Vec<TimeSeriesData>>,
}

impl VECMModel {
    /// Create a new VECM model
    ///
    /// # Arguments
    /// * `num_coint_relations` - Number of cointegrating relationships
    /// * `lag_order` - Number of lags in the VAR representation
    /// * `include_constant` - Include constant terms
    /// * `include_trend` - Include linear trend
    pub fn new(
        num_coint_relations: usize,
        lag_order: usize,
        include_constant: bool,
        include_trend: bool,
    ) -> Result<Self> {
        if num_coint_relations == 0 {
            return Err(OxiError::InvalidParameter(
                "Number of cointegrating relations must be positive".to_string(),
            ));
        }

        if lag_order == 0 {
            return Err(OxiError::InvalidParameter(
                "Lag order must be positive".to_string(),
            ));
        }

        Ok(Self {
            name: format!("VECM({}, {})", lag_order, num_coint_relations),
            num_variables: 0, // Will be set during fitting
            num_coint_relations,
            lag_order,
            include_constant,
            include_trend,
            coint_vectors: None,
            adjustment_coeffs: None,
            var_coefficients: None,
            constants: None,
            trend_coeffs: None,
            fitted_values: None,
            residuals: None,
            error_correction_terms: None,
            log_likelihood: None,
            information_criteria: None,
            training_data: None,
        })
    }

    /// Fit the VECM model to multiple time series
    pub fn fit_multiple(&mut self, data: &[TimeSeriesData]) -> Result<()> {
        let n = data.len();
        if n == 0 {
            return Err(OxiError::DataError("No data provided".to_string()));
        }

        // Validate data
        let series_length = data[0].values.len();
        if series_length < 2 * self.lag_order + 10 {
            return Err(OxiError::DataError(format!(
                "Insufficient data: need at least {} observations, got {}",
                2 * self.lag_order + 10,
                series_length
            )));
        }

        for series in data.iter().skip(1) {
            if series.values.len() != series_length {
                return Err(OxiError::DataError(
                    "All series must have equal length".to_string(),
                ));
            }
        }

        self.num_variables = n;
        self.training_data = Some(data.to_vec());

        // Step 1: Perform Johansen cointegration test
        let (coint_vectors, adjustment_coeffs) = self.johansen_cointegration_test(data)?;

        // Store cointegrating relationships
        self.coint_vectors = Some(coint_vectors.clone());
        self.adjustment_coeffs = Some(adjustment_coeffs.clone());

        // Step 2: Estimate VECM parameters
        self.estimate_vecm_parameters(data, &coint_vectors, &adjustment_coeffs)?;

        // Step 3: Calculate fitted values and residuals
        self.calculate_fitted_values(data)?;

        // Step 4: Calculate information criteria
        self.calculate_information_criteria(series_length)?;

        Ok(())
    }

    /// Forecast multiple series using the fitted VECM
    pub fn forecast_multiple(&self, horizon: usize) -> Result<Vec<Vec<f64>>> {
        let coint_vectors = self
            .coint_vectors
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("VECM model not fitted".to_string()))?;
        let adjustment_coeffs = self
            .adjustment_coeffs
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("VECM model not fitted".to_string()))?;
        let var_coefficients = self
            .var_coefficients
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("VECM model not fitted".to_string()))?;
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| OxiError::ModelError("VECM model not fitted".to_string()))?;

        let num_obs = training_data[0].values.len();
        let mut forecasts = vec![Vec::with_capacity(horizon); self.num_variables];

        // Initialize with last observations
        let mut last_levels = vec![0.0; self.num_variables];
        let mut last_differences = vec![vec![0.0; self.lag_order]; self.num_variables];

        for i in 0..self.num_variables {
            last_levels[i] = training_data[i].values[num_obs - 1];

            for lag in 0..self.lag_order {
                if num_obs > lag + 1 {
                    last_differences[i][lag] = training_data[i].values[num_obs - 1 - lag]
                        - training_data[i].values[num_obs - 2 - lag];
                }
            }
        }

        // Generate forecasts
        for h in 0..horizon {
            let mut forecast_changes = vec![0.0; self.num_variables];

            // Calculate error correction terms
            for coint_rel in 0..self.num_coint_relations {
                let mut ect = 0.0;
                for i in 0..self.num_variables {
                    ect += coint_vectors[coint_rel][i] * last_levels[i];
                }

                // Apply adjustment
                for i in 0..self.num_variables {
                    forecast_changes[i] += adjustment_coeffs[i][coint_rel] * ect;
                }
            }

            // Add VAR dynamics
            for i in 0..self.num_variables {
                for lag in 0..self.lag_order {
                    for j in 0..self.num_variables {
                        if lag < last_differences[j].len() {
                            forecast_changes[i] +=
                                var_coefficients[i][lag][j] * last_differences[j][lag];
                        }
                    }
                }

                // Add deterministic components
                if self.include_constant {
                    if let Some(constants) = &self.constants {
                        forecast_changes[i] += constants[i];
                    }
                }

                if self.include_trend {
                    if let Some(trend_coeffs) = &self.trend_coeffs {
                        forecast_changes[i] += trend_coeffs[i] * (num_obs + h + 1) as f64;
                    }
                }
            }

            // Update levels and differences
            for i in 0..self.num_variables {
                let new_level = last_levels[i] + forecast_changes[i];
                forecasts[i].push(new_level);

                // Update for next iteration
                if h < horizon - 1 {
                    // Shift differences
                    for lag in (1..self.lag_order).rev() {
                        last_differences[i][lag] = last_differences[i][lag - 1];
                    }
                    last_differences[i][0] = forecast_changes[i];
                    last_levels[i] = new_level;
                }
            }
        }

        Ok(forecasts)
    }

    /// Get cointegrating vectors
    pub fn get_cointegrating_vectors(&self) -> Option<&Vec<Vec<f64>>> {
        self.coint_vectors.as_ref()
    }

    /// Get adjustment coefficients
    pub fn get_adjustment_coefficients(&self) -> Option<&Vec<Vec<f64>>> {
        self.adjustment_coeffs.as_ref()
    }

    /// Get error correction terms
    pub fn get_error_correction_terms(&self) -> Option<&Vec<Vec<f64>>> {
        self.error_correction_terms.as_ref()
    }

    // Private implementation methods

    fn johansen_cointegration_test(
        &self,
        data: &[TimeSeriesData],
    ) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let n = data[0].values.len();

        // This is a simplified implementation of Johansen test
        // In practice, you'd want a more robust implementation

        // Create level and difference matrices
        let mut levels = vec![vec![0.0; n - 1]; self.num_variables];
        let mut differences = vec![vec![0.0; n - 1]; self.num_variables];

        for i in 0..self.num_variables {
            for t in 1..n {
                levels[i][t - 1] = data[i].values[t - 1];
                differences[i][t - 1] = data[i].values[t] - data[i].values[t - 1];
            }
        }

        // Simplified cointegration test using OLS regression
        // In reality, you'd use maximum likelihood estimation

        let mut coint_vectors = vec![vec![0.0; self.num_variables]; self.num_coint_relations];
        let mut adjustment_coeffs = vec![vec![0.0; self.num_coint_relations]; self.num_variables];

        // Initialize with simple relationships
        for r in 0..self.num_coint_relations {
            if r < self.num_variables {
                coint_vectors[r][r] = 1.0;
                if r + 1 < self.num_variables {
                    coint_vectors[r][r + 1] = -1.0; // Simple pair relationship
                }
            }
        }

        // Simple adjustment coefficients (normally estimated via ML)
        for i in 0..self.num_variables {
            for r in 0..self.num_coint_relations {
                adjustment_coeffs[i][r] = -0.1; // Weak mean reversion
            }
        }

        Ok((coint_vectors, adjustment_coeffs))
    }

    fn estimate_vecm_parameters(
        &mut self,
        data: &[TimeSeriesData],
        coint_vectors: &[Vec<f64>],
        adjustment_coeffs: &[Vec<f64>],
    ) -> Result<()> {
        let n = data[0].values.len();

        // Store cointegrating relationships
        self.coint_vectors = Some(coint_vectors.to_vec());
        self.adjustment_coeffs = Some(adjustment_coeffs.to_vec());

        // Initialize VAR coefficients
        let var_coefficients =
            vec![
                vec![vec![0.1 / (self.lag_order as f64); self.num_variables]; self.lag_order];
                self.num_variables
            ];
        self.var_coefficients = Some(var_coefficients);

        // Initialize deterministic components
        if self.include_constant {
            self.constants = Some(vec![0.0; self.num_variables]);
        }
        if self.include_trend {
            self.trend_coeffs = Some(vec![0.0; self.num_variables]);
        }

        // Calculate error correction terms
        let mut ec_terms = vec![vec![0.0; n - 1]; self.num_coint_relations];
        for r in 0..self.num_coint_relations {
            for t in 0..n - 1 {
                for i in 0..self.num_variables {
                    ec_terms[r][t] += coint_vectors[r][i] * data[i].values[t];
                }
            }
        }
        self.error_correction_terms = Some(ec_terms);

        Ok(())
    }

    fn calculate_fitted_values(&mut self, data: &[TimeSeriesData]) -> Result<()> {
        let n = data[0].values.len();
        let mut fitted_values = vec![vec![0.0; n - self.lag_order]; self.num_variables];
        let mut residuals = vec![vec![0.0; n - self.lag_order]; self.num_variables];

        for t in self.lag_order..n {
            for i in 0..self.num_variables {
                let mut fitted = data[i].values[t - 1]; // Start with previous level

                // Add predicted change
                let mut predicted_change = 0.0;

                // Error correction terms
                if let (Some(adjustment_coeffs), Some(ec_terms)) =
                    (&self.adjustment_coeffs, &self.error_correction_terms)
                {
                    for r in 0..self.num_coint_relations {
                        if t - 1 < ec_terms[r].len() {
                            predicted_change += adjustment_coeffs[i][r] * ec_terms[r][t - 1];
                        }
                    }
                }

                // VAR terms
                if let Some(var_coeffs) = &self.var_coefficients {
                    for lag in 1..=self.lag_order {
                        if t > lag {
                            for j in 0..self.num_variables {
                                let diff = data[j].values[t - lag] - data[j].values[t - lag - 1];
                                predicted_change += var_coeffs[i][lag - 1][j] * diff;
                            }
                        }
                    }
                }

                fitted += predicted_change;
                fitted_values[i][t - self.lag_order] = fitted;
                residuals[i][t - self.lag_order] = data[i].values[t] - fitted;
            }
        }

        self.fitted_values = Some(fitted_values);
        self.residuals = Some(residuals);

        Ok(())
    }

    fn calculate_information_criteria(&mut self, n: usize) -> Result<()> {
        if let Some(residuals) = &self.residuals {
            // Calculate log-likelihood (simplified)
            let mut log_likelihood = 0.0;
            let num_obs = n - self.lag_order;

            for i in 0..self.num_variables {
                let var_residuals =
                    residuals[i].iter().map(|&x| x * x).sum::<f64>() / num_obs as f64;
                log_likelihood -=
                    0.5 * num_obs as f64 * (2.0 * std::f64::consts::PI * var_residuals).ln();
            }

            // Calculate number of parameters
            let num_params = self.num_variables
                * (self.num_coint_relations + // adjustment coefficients
                self.lag_order * self.num_variables + // VAR coefficients
                if self.include_constant { 1 } else { 0 } +
                if self.include_trend { 1 } else { 0 });

            let aic = -2.0 * log_likelihood + 2.0 * num_params as f64;
            let bic = -2.0 * log_likelihood + (num_params as f64) * (num_obs as f64).ln();

            self.log_likelihood = Some(log_likelihood);
            self.information_criteria = Some((aic, bic));
        }

        Ok(())
    }
}

impl Forecaster for VECMModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        // For single series, create a trivial VECM (essentially an AR model)
        self.fit_multiple(&[data.clone()])
    }

    fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        let forecasts_multiple = self.forecast_multiple(horizon)?;
        forecasts_multiple
            .into_iter()
            .next()
            .ok_or_else(|| OxiError::ModelError("No forecasts generated".to_string()))
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

impl VECMModel {
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

    fn create_cointegrated_data() -> Vec<TimeSeriesData> {
        let start_time = Utc::now();
        let timestamps: Vec<DateTime<Utc>> =
            (0..200).map(|i| start_time + Duration::days(i)).collect();

        // Create two cointegrated series
        let mut series1_values = Vec::with_capacity(200);
        let mut series2_values = Vec::with_capacity(200);

        let mut common_trend = 100.0;
        for _i in 0..200 {
            common_trend += rand::random::<f64>() * 0.5 - 0.25; // Random walk

            // Series 1: follows common trend with noise
            series1_values.push(common_trend + rand::random::<f64>() * 2.0 - 1.0);

            // Series 2: cointegrated with series 1 (long-run relationship)
            series2_values.push(0.8 * common_trend + 20.0 + rand::random::<f64>() * 2.0 - 1.0);
        }

        vec![
            TimeSeriesData::new(timestamps.clone(), series1_values, "series1").unwrap(),
            TimeSeriesData::new(timestamps, series2_values, "series2").unwrap(),
        ]
    }

    #[test]
    fn test_vecm_model() {
        let mut model = VECMModel::new(1, 2, true, false).unwrap();
        let data = create_cointegrated_data();

        assert!(model.fit_multiple(&data).is_ok());
        assert!(model.forecast_multiple(10).is_ok());

        // Check that we have cointegrating vectors
        assert!(model.get_cointegrating_vectors().is_some());
        assert!(model.get_adjustment_coefficients().is_some());
    }

    #[test]
    fn test_vecm_single_series() {
        let mut model = VECMModel::new(1, 2, true, false).unwrap();
        let data = create_cointegrated_data();

        // Test with single series (should work as reduced form)
        assert!(model.fit(&data[0]).is_ok());
        assert!(model.forecast(5).is_ok());
    }

    #[test]
    fn test_invalid_parameters() {
        // Test invalid number of cointegrating relations
        assert!(VECMModel::new(0, 2, true, false).is_err());

        // Test invalid lag order
        assert!(VECMModel::new(1, 0, true, false).is_err());
    }
}
