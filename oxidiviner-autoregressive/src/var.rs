use crate::error::{ARError, Result as ARResult};
use oxidiviner_core::{Forecaster, ModelEvaluation, ModelOutput, OxiError, Result, TimeSeriesData};
use oxidiviner_math::metrics::{mae, mape, mse, rmse, smape};
use std::collections::HashMap;

/// Vector Autoregression (VAR) model for multivariate time series forecasting.
///
/// VAR extends the univariate autoregressive model to capture linear dependencies among
/// multiple time series. In a VAR model, each variable is a linear function of past values
/// of itself and past values of other variables in the system.
///
/// The general form of a VAR(p) model with k variables is:
/// Y_t = c + A_1*Y_{t-1} + A_2*Y_{t-2} + ... + A_p*Y_{t-p} + ε_t
///
/// where:
/// - Y_t is a k×1 vector of values at time t for all variables
/// - c is a k×1 vector of constants
/// - A_i are k×k matrices of coefficients
/// - p is the lag order
/// - ε_t is a k×1 vector of error terms
///
/// VAR models are useful for:
/// - Modeling relationships between multiple related time series
/// - Understanding how shocks to one variable affect other variables
/// - Forecasting multiple interrelated variables simultaneously
/// - Economic and financial applications (e.g., GDP, inflation, interest rates)
pub struct VARModel {
    /// Model name
    name: String,
    /// Number of variables (time series) in the system
    k: usize,
    /// Lag order (number of past values to consider)
    p: usize,
    /// Include intercept/constant term
    include_intercept: bool,
    /// Variable names
    variable_names: Vec<String>,
    /// Coefficient matrices A_1, A_2, ..., A_p (stored as Vec<Vec<Vec<f64>>>)
    /// The outer Vec has p elements (one per lag)
    /// Each inner Vec<Vec<f64>> is a k×k matrix of coefficients
    coefficient_matrices: Option<Vec<Vec<Vec<f64>>>>,
    /// Constant/intercept vector (k×1)
    intercept: Option<Vec<f64>>,
    /// Last p values for each variable (needed for forecasting)
    last_values: Option<Vec<Vec<f64>>>,
    /// Fitted values for each variable over the training period
    fitted_values: Option<Vec<Vec<f64>>>,
    /// Residuals for each variable over the training period
    residuals: Option<Vec<Vec<f64>>>,
}

impl VARModel {
    /// Creates a new Vector Autoregression model.
    ///
    /// # Arguments
    /// * `p` - Lag order (number of past values to use)
    /// * `variable_names` - Names of the variables in the system
    /// * `include_intercept` - Whether to include an intercept/constant term
    ///
    /// # Returns
    /// * `Result<Self>` - A new VAR model if parameters are valid
    pub fn new(p: usize, variable_names: Vec<String>, include_intercept: bool) -> ARResult<Self> {
        // Validate parameters
        if p == 0 {
            return Err(ARError::InvalidLagOrder(p));
        }

        if variable_names.is_empty() {
            return Err(ARError::InvalidParameter(
                "At least one variable required for VAR model".to_string(),
            ));
        }

        let k = variable_names.len();
        let name = if include_intercept {
            format!("VAR({})_{}vars+intercept", p, k)
        } else {
            format!("VAR({})_{}vars", p, k)
        };

        Ok(VARModel {
            name,
            k,
            p,
            include_intercept,
            variable_names,
            coefficient_matrices: None,
            intercept: None,
            last_values: None,
            fitted_values: None,
            residuals: None,
        })
    }

    /// Fit the VAR model to multiple time series data.
    ///
    /// # Arguments
    /// * `data_map` - A map from variable names to their TimeSeriesData
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn fit_multiple(&mut self, data_map: &HashMap<String, TimeSeriesData>) -> Result<()> {
        // Check if we have data for all variables
        for var_name in &self.variable_names {
            if !data_map.contains_key(var_name) {
                return Err(OxiError::DataError(format!(
                    "Missing data for variable '{}'",
                    var_name
                )));
            }
        }

        // Check if all time series have the same timestamps
        let first_var = &data_map[&self.variable_names[0]];
        let timestamps = &first_var.timestamps;
        let n = timestamps.len();

        for var_name in &self.variable_names {
            let var_data = &data_map[var_name];
            if var_data.timestamps.len() != n {
                return Err(OxiError::DataError(format!(
                    "Inconsistent time series lengths. Expected {} timestamps, but got {} for '{}'",
                    n,
                    var_data.timestamps.len(),
                    var_name
                )));
            }

            for i in 0..n {
                if var_data.timestamps[i] != timestamps[i] {
                    return Err(OxiError::DataError(format!(
                        "Inconsistent timestamps at position {} for variable '{}'",
                        i, var_name
                    )));
                }
            }
        }

        // Need more observations than the VAR order
        if n <= self.p {
            return Err(OxiError::from(ARError::InsufficientData {
                actual: n,
                expected: self.p + 1,
            }));
        }

        // Extract all time series values into a k×n matrix
        let mut all_values = Vec::with_capacity(self.k);
        for var_name in &self.variable_names {
            all_values.push(data_map[var_name].values.clone());
        }

        // Fit the VAR model
        self.fit_var_model(&all_values)
            .map_err(OxiError::from)?;

        Ok(())
    }

    /// Fit the model to the provided single time series data.
    ///
    /// This is a convenience method that assumes a univariate model (k=1).
    /// For multivariate models, use fit_multiple instead.
    ///
    /// # Arguments
    /// * `data` - A single time series
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        if self.k != 1 {
            return Err(OxiError::ModelError(format!(
                "Cannot use fit() for VAR model with {} variables. Use fit_multiple() instead.",
                self.k
            )));
        }

        let mut data_map = HashMap::new();
        data_map.insert(self.variable_names[0].clone(), data.clone());

        self.fit_multiple(&data_map)
    }

    /// Forecast future values for all variables.
    ///
    /// # Arguments
    /// * `horizon` - Number of periods to forecast
    ///
    /// # Returns
    /// * `Result<HashMap<String, Vec<f64>>>` - Forecasts for each variable
    pub fn forecast_multiple(&self, horizon: usize) -> Result<HashMap<String, Vec<f64>>> {
        if horizon == 0 {
            return Err(OxiError::from(ARError::InvalidHorizon(horizon)));
        }

        if self.coefficient_matrices.is_none() || self.last_values.is_none() {
            return Err(OxiError::from(ARError::NotFitted));
        }

        // Get coefficient matrices and last values
        let coef_matrices = self.coefficient_matrices.as_ref().unwrap();
        let last_values = self.last_values.as_ref().unwrap();

        // Initialize forecasts with empty vectors for each variable
        let mut forecasts: HashMap<String, Vec<f64>> = HashMap::new();
        for var_name in &self.variable_names {
            forecasts.insert(var_name.clone(), Vec::with_capacity(horizon));
        }

        // Create a matrix of historical + forecasted values for each variable
        // Start with the last p observed values (in reverse order, newest first)
        let mut extended_values = last_values.clone();

        // Create a default intercept vector if none is provided
        let default_intercept = vec![0.0; self.k];
        let intercept = self.intercept.as_ref().unwrap_or(&default_intercept);

        // Generate forecasts one step at a time
        for _ in 0..horizon {
            // Current forecast is a k×1 vector
            let mut current_forecast = vec![0.0; self.k];

            // Add intercept term if included
            if self.include_intercept {
                for i in 0..self.k {
                    current_forecast[i] = intercept[i];
                }
            }

            // Apply coefficient matrices to lagged values
            for lag in 0..self.p {
                let coef_matrix = &coef_matrices[lag];
                let lagged_values = &extended_values[lag];

                for i in 0..self.k {
                    // For each output variable
                    for j in 0..self.k {
                        // For each input variable
                        current_forecast[i] += coef_matrix[i][j] * lagged_values[j];
                    }
                }
            }

            // Add the current forecast to the historical values
            extended_values.insert(0, current_forecast.clone());
            // Remove the oldest value to maintain p lags
            extended_values.pop();

            // Add the forecast to each variable's forecast vector
            for (i, var_name) in self.variable_names.iter().enumerate() {
                forecasts
                    .get_mut(var_name)
                    .unwrap()
                    .push(current_forecast[i]);
            }
        }

        Ok(forecasts)
    }

    /// Forecast future values for a single variable.
    ///
    /// This is a convenience method that returns forecasts for the first variable only.
    /// For full multivariate forecasts, use forecast_multiple.
    ///
    /// # Arguments
    /// * `horizon` - Number of periods to forecast
    ///
    /// # Returns
    /// * `Result<Vec<f64>>` - Forecasts for the first variable
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        let forecasts_map = self.forecast_multiple(horizon)?;
        let var_name = &self.variable_names[0];

        Ok(forecasts_map[var_name].clone())
    }

    /// Evaluate the model on test data for multiple variables.
    ///
    /// # Arguments
    /// * `test_data_map` - A map from variable names to their test TimeSeriesData
    ///
    /// # Returns
    /// * `Result<HashMap<String, ModelEvaluation>>` - Evaluation for each variable
    pub fn evaluate_multiple(
        &self,
        test_data_map: &HashMap<String, TimeSeriesData>,
    ) -> Result<HashMap<String, ModelEvaluation>> {
        if self.coefficient_matrices.is_none() {
            return Err(OxiError::from(ARError::NotFitted));
        }

        // Check if we have test data for all variables
        for var_name in &self.variable_names {
            if !test_data_map.contains_key(var_name) {
                return Err(OxiError::DataError(format!(
                    "Missing test data for variable '{}'",
                    var_name
                )));
            }
        }

        // Check if all test time series have the same horizon
        let first_var = &test_data_map[&self.variable_names[0]];
        let horizon = first_var.values.len();

        for var_name in &self.variable_names {
            let var_data = &test_data_map[var_name];
            if var_data.values.len() != horizon {
                return Err(OxiError::DataError(format!(
                    "Inconsistent test series lengths. Expected {} values, but got {} for '{}'",
                    horizon,
                    var_data.values.len(),
                    var_name
                )));
            }
        }

        // Generate forecasts for all variables
        let forecasts_map = self.forecast_multiple(horizon)?;

        // Calculate evaluation metrics for each variable
        let mut evaluations = HashMap::new();

        for var_name in &self.variable_names {
            let actual = &test_data_map[var_name].values;
            let forecast = &forecasts_map[var_name];

            let mae_value = mae(actual, forecast);
            let mse_value = mse(actual, forecast);
            let rmse_value = rmse(actual, forecast);
            let mape_value = mape(actual, forecast);
            let smape_value = smape(actual, forecast);

            let model_name = format!("{}-{}", self.name, var_name);
            let eval = ModelEvaluation {
                model_name,
                mae: mae_value,
                mse: mse_value,
                rmse: rmse_value,
                mape: mape_value,
                smape: smape_value,
            };

            evaluations.insert(var_name.clone(), eval);
        }

        Ok(evaluations)
    }

    /// Evaluate the model on a single test series.
    ///
    /// This is a convenience method for univariate VAR models.
    /// For full multivariate evaluation, use evaluate_multiple.
    ///
    /// # Arguments
    /// * `test_data` - Test data for the first variable
    ///
    /// # Returns
    /// * `Result<ModelEvaluation>` - Evaluation metrics
    pub fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        if self.k != 1 {
            return Err(OxiError::ModelError(
                format!("Cannot use evaluate() for VAR model with {} variables. Use evaluate_multiple() instead.", self.k)
            ));
        }

        let mut test_data_map = HashMap::new();
        test_data_map.insert(self.variable_names[0].clone(), test_data.clone());

        let evaluations = self.evaluate_multiple(&test_data_map)?;

        Ok(evaluations[&self.variable_names[0]].clone())
    }

    /// Generate forecasts and evaluation for multiple variables.
    ///
    /// # Arguments
    /// * `horizon` - Number of periods to forecast
    /// * `test_data_map` - Optional map from variable names to test data
    ///
    /// # Returns
    /// * `Result<HashMap<String, ModelOutput>>` - Outputs for each variable
    pub fn predict_multiple(
        &self,
        horizon: usize,
        test_data_map: Option<&HashMap<String, TimeSeriesData>>,
    ) -> Result<HashMap<String, ModelOutput>> {
        if horizon == 0 {
            return Err(OxiError::from(ARError::InvalidHorizon(horizon)));
        }

        if self.coefficient_matrices.is_none() {
            return Err(OxiError::from(ARError::NotFitted));
        }

        // Generate forecasts for all variables
        let forecasts_map = self.forecast_multiple(horizon)?;

        // Initialize outputs with forecasts
        let mut outputs = HashMap::new();

        for var_name in &self.variable_names {
            let forecast = forecasts_map[var_name].clone();

            let output = ModelOutput {
                model_name: format!("{}-{}", self.name, var_name),
                forecasts: forecast,
                evaluation: None,
            };

            outputs.insert(var_name.clone(), output);
        }

        // Add evaluations if test data is provided
        if let Some(test_map) = test_data_map {
            let evaluations = self.evaluate_multiple(test_map)?;

            for var_name in &self.variable_names {
                if let Some(eval) = evaluations.get(var_name) {
                    if let Some(output) = outputs.get_mut(var_name) {
                        output.evaluation = Some(eval.clone());
                    }
                }
            }
        }

        Ok(outputs)
    }

    /// Generate forecasts and evaluation for a single variable.
    ///
    /// This is a convenience method for univariate VAR models.
    /// For full multivariate prediction, use predict_multiple.
    ///
    /// # Arguments
    /// * `horizon` - Number of periods to forecast
    /// * `test_data` - Optional test data for the first variable
    ///
    /// # Returns
    /// * `Result<ModelOutput>` - Output for the first variable
    pub fn predict(
        &self,
        horizon: usize,
        test_data: Option<&TimeSeriesData>,
    ) -> Result<ModelOutput> {
        if self.k != 1 {
            return Err(OxiError::ModelError(
                format!("Cannot use predict() for VAR model with {} variables. Use predict_multiple() instead.", self.k)
            ));
        }

        let mut test_data_map = None;

        if let Some(data) = test_data {
            let mut map = HashMap::new();
            map.insert(self.variable_names[0].clone(), data.clone());
            test_data_map = Some(map);
        }

        let test_map_ref = test_data_map.as_ref();
        let outputs = self.predict_multiple(horizon, test_map_ref)?;

        Ok(outputs[&self.variable_names[0]].clone())
    }

    /// Get the estimated coefficient matrices.
    pub fn coefficient_matrices(&self) -> Option<&Vec<Vec<Vec<f64>>>> {
        self.coefficient_matrices.as_ref()
    }

    /// Get the estimated intercept vector.
    pub fn intercept(&self) -> Option<&Vec<f64>> {
        self.intercept.as_ref()
    }

    /// Get the fitted values for each variable.
    pub fn fitted_values(&self) -> Option<&Vec<Vec<f64>>> {
        self.fitted_values.as_ref()
    }

    /// Get the residuals for each variable.
    pub fn residuals(&self) -> Option<&Vec<Vec<f64>>> {
        self.residuals.as_ref()
    }

    /// Fit the VAR model using OLS estimation.
    ///
    /// This method implements OLS estimation for each equation separately.
    /// For a VAR(p) model with k variables, we have k equations to estimate.
    ///
    /// # Arguments
    /// * `data` - k×n matrix where each row is a time series
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    fn fit_var_model(&mut self, data: &[Vec<f64>]) -> ARResult<()> {
        let k = data.len();
        let n = data[0].len();

        // Sanity check: all time series should have the same length
        for i in 1..k {
            if data[i].len() != n {
                return Err(ARError::InvalidParameter(
                    format!("Inconsistent time series lengths. Expected {} values, but got {} for series {}",
                           n, data[i].len(), i)
                ));
            }
        }

        // Effective sample size after losing p observations
        let t = n - self.p;

        if t <= k * self.p + self.include_intercept as usize {
            return Err(ARError::InsufficientData {
                actual: n,
                expected: k * self.p + self.include_intercept as usize + self.p + 1,
            });
        }

        // Prepare dependent and independent variables
        // Y is a T×k matrix of dependent variables
        // X is a T×(kp+1) matrix of regressors (if include_intercept is true, +1 for the constant term)

        // Y matrix (will be T×k)
        let mut y = vec![vec![0.0; k]; t];
        for t_idx in 0..t {
            // Fill Y matrix with current values
            for i in 0..k {
                y[t_idx][i] = data[i][t_idx + self.p];
            }
        }

        // X matrix (will be T×(kp+intercept))
        let num_regressors = k * self.p + self.include_intercept as usize;
        let mut x = vec![vec![0.0; num_regressors]; t];

        for t_idx in 0..t {
            // If intercept is included, set first column to 1
            if self.include_intercept {
                x[t_idx][0] = 1.0;
            }

            // Set other columns to lagged values
            let intercept_offset = self.include_intercept as usize;
            for lag in 0..self.p {
                for i in 0..k {
                    let col_idx = intercept_offset + lag * k + i;
                    x[t_idx][col_idx] = data[i][t_idx + self.p - lag - 1];
                }
            }
        }

        // Fit each equation using OLS: B = (X'X)^(-1)X'Y
        // Where B is a (kp+1)×k matrix of coefficients

        // First, compute X'X (num_regressors × num_regressors)
        let mut xtx = vec![vec![0.0; num_regressors]; num_regressors];
        for i in 0..num_regressors {
            for j in 0..num_regressors {
                for t_idx in 0..t {
                    xtx[i][j] += x[t_idx][i] * x[t_idx][j];
                }
            }
        }

        // Compute the inverse of X'X
        let xtx_inv = self.invert_matrix(&xtx)?;

        // Compute X'Y (num_regressors × k)
        let mut xty = vec![vec![0.0; k]; num_regressors];
        for i in 0..num_regressors {
            for j in 0..k {
                for t_idx in 0..t {
                    xty[i][j] += x[t_idx][i] * y[t_idx][j];
                }
            }
        }

        // Compute B = (X'X)^(-1)X'Y
        let mut b = vec![vec![0.0; k]; num_regressors];
        for i in 0..num_regressors {
            for j in 0..k {
                for r in 0..num_regressors {
                    b[i][j] += xtx_inv[i][r] * xty[r][j];
                }
            }
        }

        // Extract coefficient matrices and intercept from B
        let mut coefficient_matrices = Vec::with_capacity(self.p);
        let mut intercept = vec![0.0; k];

        // Set intercept if included
        if self.include_intercept {
            for i in 0..k {
                intercept[i] = b[0][i];
            }
        }

        // Extract coefficient matrices
        for lag in 0..self.p {
            let mut coef_matrix = vec![vec![0.0; k]; k];
            let intercept_offset = self.include_intercept as usize;

            for i in 0..k {
                for j in 0..k {
                    let row_idx = intercept_offset + lag * k + j;
                    coef_matrix[i][j] = b[row_idx][i];
                }
            }

            coefficient_matrices.push(coef_matrix);
        }

        // Calculate fitted values and residuals
        let mut fitted_values = vec![vec![0.0; n]; k];
        let mut residuals = vec![vec![0.0; n]; k];

        // First p values cannot be predicted
        for i in 0..k {
            for t_idx in 0..self.p {
                fitted_values[i][t_idx] = f64::NAN;
                residuals[i][t_idx] = f64::NAN;
            }
        }

        // Calculate fitted values for the rest
        for t_idx in self.p..n {
            for i in 0..k {
                let mut fitted = if self.include_intercept {
                    intercept[i]
                } else {
                    0.0
                };

                for lag in 0..self.p {
                    for j in 0..k {
                        fitted += coefficient_matrices[lag][i][j] * data[j][t_idx - lag - 1];
                    }
                }

                fitted_values[i][t_idx] = fitted;
                residuals[i][t_idx] = data[i][t_idx] - fitted;
            }
        }

        // Store last p values for forecasting
        let mut last_values = Vec::with_capacity(self.p);
        for lag in 0..self.p {
            let t_idx = n - lag - 1;
            let mut current_values = Vec::with_capacity(k);

            for i in 0..k {
                current_values.push(data[i][t_idx]);
            }

            last_values.push(current_values);
        }

        // Store results
        self.coefficient_matrices = Some(coefficient_matrices);
        self.intercept = if self.include_intercept {
            Some(intercept)
        } else {
            None
        };
        self.fitted_values = Some(fitted_values);
        self.residuals = Some(residuals);
        self.last_values = Some(last_values);

        Ok(())
    }

    /// Invert a matrix using Gaussian elimination.
    ///
    /// # Arguments
    /// * `a` - Square matrix to invert
    ///
    /// # Returns
    /// * `Result<Vec<Vec<f64>>>` - Inverted matrix
    fn invert_matrix(&self, a: &[Vec<f64>]) -> ARResult<Vec<Vec<f64>>> {
        let n = a.len();
        if n == 0 || a[0].len() != n {
            return Err(ARError::LinearSolveError(
                "Invalid matrix dimensions for inversion".to_string(),
            ));
        }

        // Create augmented matrix [A|I]
        let mut aug = vec![vec![0.0; 2 * n]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = a[i][j];
            }
            aug[i][n + i] = 1.0; // Identity matrix on the right
        }

        // Gaussian elimination
        for i in 0..n {
            // Find pivot
            let mut max_idx = i;
            let mut max_val = aug[i][i].abs();

            for j in (i + 1)..n {
                if aug[j][i].abs() > max_val {
                    max_idx = j;
                    max_val = aug[j][i].abs();
                }
            }

            // Check if matrix is singular
            if max_val < 1e-10 {
                return Err(ARError::LinearSolveError(
                    "Singular matrix detected".to_string(),
                ));
            }

            // Swap rows if needed
            if max_idx != i {
                aug.swap(i, max_idx);
            }

            // Scale pivot row
            let pivot = aug[i][i];
            for j in 0..(2 * n) {
                aug[i][j] /= pivot;
            }

            // Eliminate other rows
            for j in 0..n {
                if j != i {
                    let factor = aug[j][i];
                    for k in 0..(2 * n) {
                        aug[j][k] -= factor * aug[i][k];
                    }
                }
            }
        }

        // Extract inverse matrix
        let mut inverse = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                inverse[i][j] = aug[i][n + j];

                // Check for invalid values
                if inverse[i][j].is_nan() || inverse[i][j].is_infinite() {
                    return Err(ARError::InvalidCoefficient);
                }
            }
        }

        Ok(inverse)
    }
}

// Implement Forecaster trait for a univariate version of the VAR model
impl Forecaster for VARModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        if self.k != 1 {
            return Err(OxiError::ModelError(
                format!("Cannot use Forecaster::fit() for VAR model with {} variables. Use fit_multiple() instead.", self.k)
            ));
        }

        let mut data_map = HashMap::new();
        data_map.insert(self.variable_names[0].clone(), data.clone());

        self.fit_multiple(&data_map)
    }

    fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        if self.k != 1 {
            return Err(OxiError::ModelError(
                format!("Cannot use Forecaster::forecast() for VAR model with {} variables. Use forecast_multiple() instead.", self.k)
            ));
        }

        let forecasts_map = self.forecast_multiple(horizon)?;

        Ok(forecasts_map[&self.variable_names[0]].clone())
    }

    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        if self.k != 1 {
            return Err(OxiError::ModelError(
                format!("Cannot use Forecaster::evaluate() for VAR model with {} variables. Use evaluate_multiple() instead.", self.k)
            ));
        }

        let mut test_data_map = HashMap::new();
        test_data_map.insert(self.variable_names[0].clone(), test_data.clone());

        let evaluations = self.evaluate_multiple(&test_data_map)?;

        Ok(evaluations[&self.variable_names[0]].clone())
    }

    fn predict(&self, horizon: usize, test_data: Option<&TimeSeriesData>) -> Result<ModelOutput> {
        if self.k != 1 {
            return Err(OxiError::ModelError(
                format!("Cannot use Forecaster::predict() for VAR model with {} variables. Use predict_multiple() instead.", self.k)
            ));
        }

        let mut test_data_map = None;

        if let Some(data) = test_data {
            let mut map = HashMap::new();
            map.insert(self.variable_names[0].clone(), data.clone());
            test_data_map = Some(map);
        }

        let test_map_ref = test_data_map.as_ref();
        let outputs = self.predict_multiple(horizon, test_map_ref)?;

        Ok(outputs[&self.variable_names[0]].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, TimeZone, Utc};

    #[test]
    fn test_var_univariate() {
        // This test verifies that a VAR(1) model with a single variable
        // behaves similarly to an AR(1) model

        // Create linear trend data: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..12)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();

        let values: Vec<f64> = (1..=12).map(|i| i as f64).collect();

        let time_series = TimeSeriesData::new(timestamps, values, "trend_series").unwrap();

        // Create and fit a VAR(1) model with one variable
        let variable_names = vec!["y".to_string()];
        let mut model = VARModel::new(1, variable_names, true).unwrap();
        model.fit(&time_series).unwrap();

        // Forecast the next 5 values
        let forecast_horizon = 5;
        let forecasts = model.forecast(forecast_horizon).unwrap();

        // Check that we got the expected number of forecasts
        assert_eq!(forecasts.len(), forecast_horizon);

        // For a linear trend, the model should continue the trend (approximately)
        for (i, forecast) in forecasts.iter().enumerate() {
            let expected = 13.0 + i as f64;
            // Allow for some deviation in the forecast
            assert!(
                (forecast - expected).abs() < 1.0,
                "Forecast {} should be close to {} for trend data",
                forecast,
                expected
            );
        }
    }

    #[test]
    fn test_var_bivariate() {
        // Create two related time series with small variations to avoid singularity
        // y1: linear trend with small noise
        // y2: approximately 2*y1 with small noise
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..15)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();

        // Add small variations to avoid singularity issues
        let mut values1 = Vec::with_capacity(15);
        let mut values2 = Vec::with_capacity(15);
        
        for i in 0..15 {
            let base = (i + 1) as f64;
            // Small sinusoidal variations
            let noise1 = 0.05 * (i as f64 * 0.7).sin();
            let noise2 = 0.05 * (i as f64 * 0.5).cos();
            
            let y1 = base + noise1;
            let y2 = 2.0 * base + noise2; // Approximately 2*y1 with different noise
            
            values1.push(y1);
            values2.push(y2);
        }

        let ts1 = TimeSeriesData::new(timestamps.clone(), values1, "y1").unwrap();
        let ts2 = TimeSeriesData::new(timestamps, values2, "y2").unwrap();

        let mut data_map = HashMap::new();
        data_map.insert("y1".to_string(), ts1);
        data_map.insert("y2".to_string(), ts2);

        // Create and fit a VAR(1) model
        let variable_names = vec!["y1".to_string(), "y2".to_string()];
        let mut model = VARModel::new(1, variable_names, true).unwrap();
        model.fit_multiple(&data_map).unwrap();

        // Forecast the next 5 values
        let forecast_horizon = 5;
        let forecasts = model.forecast_multiple(forecast_horizon).unwrap();

        // Check that the relationship between y1 and y2 is approximately preserved in forecasts
        for i in 0..forecast_horizon {
            let y1_forecast = forecasts["y1"][i];
            let y2_forecast = forecasts["y2"][i];

            // The relationship should be approximately preserved
            let ratio = y2_forecast / y1_forecast;
            assert!(
                (ratio - 2.0).abs() < 0.3,
                "Forecast relationship should preserve y2 ≈ 2*y1, but got ratio {} at horizon {}",
                ratio,
                i
            );
        }
    }
}
