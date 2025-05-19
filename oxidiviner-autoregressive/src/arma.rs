use crate::error::{ARError, Result as ARResult};
use oxidiviner_core::{Forecaster, ModelEvaluation, ModelOutput, OxiError, Result, TimeSeriesData};
use oxidiviner_math::metrics::{mae, mape, mse, rmse, smape};

/// Autoregressive Moving Average (ARMA) model for time series forecasting.
///
/// This model combines autoregressive (AR) and moving average (MA) components.
/// The general form of an ARMA(p,q) model is:
/// y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p} +
///       θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θₑε_{t-q} + ε_t
///
/// where:
/// - y_t is the value at time t
/// - c is a constant (intercept)
/// - φ₁, φ₂, ..., φₚ are the autoregressive coefficients
/// - θ₁, θ₂, ..., θₑ are the moving average coefficients
/// - p is the AR lag order (number of past values)
/// - q is the MA lag order (number of past errors)
/// - ε_t is white noise
///
/// ARMA models are useful for:
/// - Capturing both AR and MA patterns in data
/// - Modeling more complex time series behavior
/// - Providing more flexible fit than pure AR or MA models
pub struct ARMAModel {
    /// Model name
    name: String,
    /// AR lag order (p)
    p: usize,
    /// MA lag order (q)
    q: usize,
    /// Include intercept/constant term
    include_intercept: bool,
    /// Estimated intercept/constant term
    intercept: Option<f64>,
    /// Estimated AR coefficients [φ₁, φ₂, ..., φₚ]
    ar_coefficients: Option<Vec<f64>>,
    /// Estimated MA coefficients [θ₁, θ₂, ..., θₑ]
    ma_coefficients: Option<Vec<f64>>,
    /// Fitted values over the training period
    fitted_values: Option<Vec<f64>>,
    /// Residuals (errors) from the fitting process
    residuals: Option<Vec<f64>>,
    /// Last p values from the training data (needed for forecasting)
    last_values: Option<Vec<f64>>,
    /// Last q residuals from the training data (needed for forecasting)
    last_residuals: Option<Vec<f64>>,
    /// Mean of the training data (for mean centering)
    mean: Option<f64>,
}

impl ARMAModel {
    /// Creates a new ARMA model.
    ///
    /// # Arguments
    /// * `p` - AR lag order (number of past values to use)
    /// * `q` - MA lag order (number of past errors to use)
    /// * `include_intercept` - Whether to include an intercept/constant term
    ///
    /// # Returns
    /// * `Result<Self>` - A new ARMA model if parameters are valid
    pub fn new(p: usize, q: usize, include_intercept: bool) -> ARResult<Self> {
        // Validate parameters
        if p == 0 && q == 0 {
            return Err(ARError::InvalidLagOrder(0));
        }

        let name = if include_intercept {
            format!("ARMA({},{})+intercept", p, q)
        } else {
            format!("ARMA({},{})", p, q)
        };

        Ok(ARMAModel {
            name,
            p,
            q,
            include_intercept,
            intercept: None,
            ar_coefficients: None,
            ma_coefficients: None,
            fitted_values: None,
            residuals: None,
            last_values: None,
            last_residuals: None,
            mean: None,
        })
    }

    /// Fit the model to the provided time series data.
    /// This is a convenience method that calls the trait method directly.
    pub fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        <Self as Forecaster>::fit(self, data)
    }

    /// Forecast future values.
    /// This is a convenience method that calls the trait method directly.
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        <Self as Forecaster>::forecast(self, horizon)
    }

    /// Evaluate the model on test data.
    /// This is a convenience method that calls the trait method directly.
    pub fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        <Self as Forecaster>::evaluate(self, test_data)
    }

    /// Generate forecasts and evaluation in a standardized format.
    /// This is a convenience method that calls the trait method directly.
    pub fn predict(
        &self,
        horizon: usize,
        test_data: Option<&TimeSeriesData>,
    ) -> Result<ModelOutput> {
        <Self as Forecaster>::predict(self, horizon, test_data)
    }

    /// Get the fitted values if available.
    pub fn fitted_values(&self) -> Option<&Vec<f64>> {
        self.fitted_values.as_ref()
    }

    /// Get the residuals from model fitting.
    pub fn residuals(&self) -> Option<&Vec<f64>> {
        self.residuals.as_ref()
    }

    /// Get the estimated AR coefficients.
    pub fn ar_coefficients(&self) -> Option<&Vec<f64>> {
        self.ar_coefficients.as_ref()
    }

    /// Get the estimated MA coefficients.
    pub fn ma_coefficients(&self) -> Option<&Vec<f64>> {
        self.ma_coefficients.as_ref()
    }

    /// Get the estimated intercept/constant term.
    pub fn intercept(&self) -> Option<f64> {
        self.intercept
    }

    /// Get the mean of the training data.
    pub fn mean(&self) -> Option<f64> {
        self.mean
    }

    /// Estimate ARMA parameters using an iterative method
    /// Note: For a production-ready implementation, you might want to use
    /// more sophisticated estimation methods like maximum likelihood
    fn fit_arma_parameters(&mut self, data: &[f64]) -> ARResult<()> {
        let n = data.len();
        let mean = data.iter().sum::<f64>() / n as f64;

        // Initialize coefficients
        let mut ar_coeffs = vec![0.0; self.p];
        let mut ma_coeffs = vec![0.0; self.q];

        // For simplicity, we'll first fit an AR model to get initial estimates
        if self.p > 0 {
            self.fit_ar_component(data, &mut ar_coeffs)?;
        }

        // Calculate residuals using AR model
        let mut residuals = vec![0.0; n];
        let max_lag = self.p.max(self.q);

        // First max_lag residuals are set to zero (can't estimate)
        for i in 0..max_lag {
            residuals[i] = 0.0;
        }

        // Calculate initial residuals using AR model
        for t in max_lag..n {
            let mut y_hat = self.include_intercept.then(|| mean).unwrap_or(0.0);

            // Add AR component
            for i in 0..self.p {
                if i < t {
                    y_hat += ar_coeffs[i] * data[t - i - 1];
                }
            }

            residuals[t] = data[t] - y_hat;
        }

        // If we have MA components, estimate them
        if self.q > 0 {
            self.fit_ma_component(&residuals, &mut ma_coeffs)?;
        }

        // Now iterate a few times to refine the estimates
        // In a production system, you would use proper convergence criteria
        for _ in 0..5 {
            // Recalculate residuals using both AR and MA components
            for t in max_lag..n {
                let mut y_hat = self.include_intercept.then(|| mean).unwrap_or(0.0);

                // Add AR component
                for i in 0..self.p {
                    if i < t {
                        y_hat += ar_coeffs[i] * data[t - i - 1];
                    }
                }

                // Add MA component
                for i in 0..self.q {
                    if i < t {
                        y_hat += ma_coeffs[i] * residuals[t - i - 1];
                    }
                }

                residuals[t] = data[t] - y_hat;
            }

            // Refit AR component
            if self.p > 0 {
                self.fit_ar_component(data, &mut ar_coeffs)?;
            }

            // Refit MA component
            if self.q > 0 {
                self.fit_ma_component(&residuals, &mut ma_coeffs)?;
            }
        }

        // Calculate final fitted values
        let mut fitted_values = vec![f64::NAN; max_lag]; // First values cannot be predicted

        for t in max_lag..n {
            let mut prediction = self.include_intercept.then(|| mean).unwrap_or(0.0);

            // Add AR component
            for i in 0..self.p {
                prediction += ar_coeffs[i] * data[t - i - 1];
            }

            // Add MA component
            for i in 0..self.q {
                prediction += ma_coeffs[i] * residuals[t - i - 1];
            }

            fitted_values.push(prediction);
        }

        // Calculate intercept
        let intercept = if self.include_intercept {
            let ar_sum: f64 = ar_coeffs.iter().sum();
            mean * (1.0 - ar_sum)
        } else {
            0.0
        };

        // Store the results
        self.ar_coefficients = Some(ar_coeffs);
        self.ma_coefficients = Some(ma_coeffs);
        self.intercept = Some(intercept);
        self.mean = Some(mean);
        self.fitted_values = Some(fitted_values);
        self.residuals = Some(residuals.clone());
        self.last_values = Some(data[(n - self.p.max(1))..].to_vec());
        self.last_residuals = Some(residuals[(n - self.q.max(1))..].to_vec());

        Ok(())
    }

    /// Fit AR component using Yule-Walker equations
    fn fit_ar_component(&self, data: &[f64], ar_coeffs: &mut Vec<f64>) -> ARResult<()> {
        if self.p == 0 {
            return Ok(());
        }

        let n = data.len();
        let mean = data.iter().sum::<f64>() / n as f64;

        // Calculate autocorrelations up to lag p
        let mut autocorr = Vec::with_capacity(self.p + 1);

        // Autocorrelation at lag 0 (variance)
        let mut sum_sq = 0.0;
        for &value in data {
            sum_sq += (value - mean).powi(2);
        }
        let _var = sum_sq / n as f64;
        autocorr.push(1.0); // r(0) = 1

        // Autocorrelation at lags 1 to p
        for lag in 1..=self.p {
            let mut sum_cross = 0.0;
            for i in lag..n {
                sum_cross += (data[i] - mean) * (data[i - lag] - mean);
            }
            autocorr.push(sum_cross / sum_sq);
        }

        // Create the Toeplitz matrix
        let mut matrix = vec![vec![0.0; self.p]; self.p];
        for i in 0..self.p {
            for j in 0..self.p {
                matrix[i][j] = autocorr[(i as isize - j as isize).abs() as usize];
            }
        }

        // Create the right-hand side
        let rhs: Vec<f64> = autocorr[1..=self.p].to_vec();

        // Solve the system using Gaussian elimination
        let phi = self.solve_linear_system(&matrix, &rhs)?;

        // Update the AR coefficients
        for i in 0..self.p {
            ar_coeffs[i] = phi[i];
        }

        Ok(())
    }

    /// Fit MA component using a simplified approach
    /// Note: In a production system, you'd use a more sophisticated method
    fn fit_ma_component(&self, residuals: &[f64], ma_coeffs: &mut Vec<f64>) -> ARResult<()> {
        if self.q == 0 {
            return Ok(());
        }

        let n = residuals.len();

        // Simplistic approach: use autocorrelation of residuals
        // In practice, you'd use maximum likelihood estimation
        for q in 0..self.q {
            let lag = q + 1;
            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for t in lag..n {
                numerator += residuals[t] * residuals[t - lag];
                denominator += residuals[t].powi(2);
            }

            ma_coeffs[q] = numerator / denominator;

            // Check for invalid coefficients
            if ma_coeffs[q].is_nan() || ma_coeffs[q].is_infinite() {
                return Err(ARError::InvalidCoefficient);
            }

            // MA coefficients should typically be between -1 and 1
            ma_coeffs[q] = ma_coeffs[q].clamp(-0.99, 0.99);
        }

        Ok(())
    }

    /// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
    fn solve_linear_system(&self, a: &[Vec<f64>], b: &[f64]) -> ARResult<Vec<f64>> {
        let n = a.len();
        if n == 0 || a[0].len() != n || b.len() != n {
            return Err(ARError::LinearSolveError(
                "Invalid matrix dimensions".to_string(),
            ));
        }

        // Create augmented matrix [A|b]
        let mut aug = vec![vec![0.0; n + 1]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = a[i][j];
            }
            aug[i][n] = b[i];
        }

        // Gaussian elimination with partial pivoting
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
            if max_val.abs() < 1e-10 {
                return Err(ARError::LinearSolveError(
                    "Singular matrix detected".to_string(),
                ));
            }

            // Swap rows if needed
            if max_idx != i {
                aug.swap(i, max_idx);
            }

            // Eliminate below
            for j in (i + 1)..n {
                let factor = aug[j][i] / aug[i][i];
                aug[j][i] = 0.0; // This is already zero in theory

                for k in (i + 1)..=n {
                    aug[j][k] -= factor * aug[i][k];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += aug[i][j] * x[j];
            }
            x[i] = (aug[i][n] - sum) / aug[i][i];

            // Check for invalid coefficients
            if x[i].is_nan() || x[i].is_infinite() {
                return Err(ARError::InvalidCoefficient);
            }
        }

        Ok(x)
    }
}

impl Forecaster for ARMAModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        if data.is_empty() {
            return Err(OxiError::from(ARError::EmptyData));
        }

        let n = data.values.len();
        let max_lag = self.p.max(self.q);

        // Need more observations than the max lag order
        if n <= max_lag {
            return Err(OxiError::from(ARError::InsufficientData {
                actual: n,
                expected: max_lag + 1,
            }));
        }

        // Fit the ARMA model parameters
        self.fit_arma_parameters(&data.values)
            .map_err(|e| OxiError::from(e))?;

        Ok(())
    }

    fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        if horizon == 0 {
            return Err(OxiError::from(ARError::InvalidHorizon(horizon)));
        }

        if self.ar_coefficients.is_none()
            || self.ma_coefficients.is_none()
            || self.last_values.is_none()
            || self.last_residuals.is_none()
        {
            return Err(OxiError::from(ARError::NotFitted));
        }

        let ar_coeffs = self.ar_coefficients.as_ref().unwrap();
        let ma_coeffs = self.ma_coefficients.as_ref().unwrap();
        let intercept = self.intercept.unwrap_or(0.0);

        // Initialize with the last observed values and residuals
        let mut extended_values = self.last_values.as_ref().unwrap().clone();
        let mut extended_residuals = self.last_residuals.as_ref().unwrap().clone();
        let mut forecasts = Vec::with_capacity(horizon);

        // Generate forecasts one step at a time
        for _ in 0..horizon {
            let mut prediction = intercept;

            // Apply AR coefficients to the most recent p values
            for i in 0..self.p {
                if i < extended_values.len() {
                    prediction += ar_coeffs[i] * extended_values[extended_values.len() - i - 1];
                }
            }

            // Apply MA coefficients to the most recent q residuals
            for i in 0..self.q {
                if i < extended_residuals.len() {
                    prediction +=
                        ma_coeffs[i] * extended_residuals[extended_residuals.len() - i - 1];
                }
            }

            // Add prediction to our forecasts
            forecasts.push(prediction);

            // Update the extended series
            extended_values.push(prediction);
            extended_residuals.push(0.0); // Future residuals are expected to be zero
        }

        Ok(forecasts)
    }

    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        if self.ar_coefficients.is_none() || self.ma_coefficients.is_none() {
            return Err(OxiError::from(ARError::NotFitted));
        }

        let forecast = self.forecast(test_data.values.len())?;

        // Calculate error metrics
        let mae = mae(&test_data.values, &forecast);
        let mse = mse(&test_data.values, &forecast);
        let rmse = rmse(&test_data.values, &forecast);
        let mape = mape(&test_data.values, &forecast);
        let smape = smape(&test_data.values, &forecast);

        Ok(ModelEvaluation {
            model_name: self.name.clone(),
            mae,
            mse,
            rmse,
            mape,
            smape,
        })
    }

    // Using the default predict implementation from the trait
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, TimeZone, Utc};

    #[test]
    fn test_arma_model_constant_data() {
        // Create constant time series: 10, 10, 10, 10, 10, 10, 10, 10, 10, 10
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();

        let values = vec![10.0; 10];

        let time_series = TimeSeriesData::new(timestamps, values, "constant_series").unwrap();

        // Create and fit an ARMA(1,1) model
        let mut model = ARMAModel::new(1, 1, true).unwrap();
        model.fit(&time_series).unwrap();

        // For constant data, forecasts should be the same constant
        let forecast_horizon = 5;
        let forecasts = model.forecast(forecast_horizon).unwrap();

        // Check that the forecasts are constant
        for forecast in forecasts {
            assert!(
                (forecast - 10.0).abs() < 1e-5,
                "Forecast {} should be close to 10.0 for constant data",
                forecast
            );
        }
    }

    #[test]
    fn test_arma_model_trending_data() {
        // Create linear trend data: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..12)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();

        let values: Vec<f64> = (1..=12).map(|i| i as f64).collect();

        let time_series = TimeSeriesData::new(timestamps, values, "trend_series").unwrap();

        // Create and fit an ARMA(2,1) model with intercept
        let mut model = ARMAModel::new(2, 1, true).unwrap();
        model.fit(&time_series).unwrap();

        // For a linear trend, an ARMA model should be able to capture it
        let forecast_horizon = 5;
        let forecasts = model.forecast(forecast_horizon).unwrap();

        // Check that forecasts continue the trend (approximately)
        for (i, forecast) in forecasts.iter().enumerate() {
            let expected = 13.0 + i as f64;
            // Allow for some deviation in the forecast
            assert!(
                (forecast - expected).abs() < 2.0,
                "Forecast {} should be reasonably close to {} for trend data",
                forecast,
                expected
            );
        }
    }

    #[test]
    fn test_arma_model_coefficient_access() {
        // Create a simple time series
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..20)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();

        let values: Vec<f64> = (1..=20).map(|i| i as f64).collect();

        let time_series = TimeSeriesData::new(timestamps, values, "test_series").unwrap();

        // Create and fit ARMA(2,1) model
        let mut model = ARMAModel::new(2, 1, true).unwrap();
        model.fit(&time_series).unwrap();

        // Should be able to access AR coefficients
        let ar_coefficients = model.ar_coefficients().unwrap();
        assert_eq!(
            ar_coefficients.len(),
            2,
            "ARMA(2,1) model should have 2 AR coefficients"
        );

        // Should be able to access MA coefficients
        let ma_coefficients = model.ma_coefficients().unwrap();
        assert_eq!(
            ma_coefficients.len(),
            1,
            "ARMA(2,1) model should have 1 MA coefficient"
        );

        // Should be able to access intercept
        let intercept = model.intercept().unwrap();
        assert!(intercept.is_finite(), "Intercept should be a finite number");
    }
}
