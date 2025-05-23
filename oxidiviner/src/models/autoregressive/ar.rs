#![allow(clippy::needless_range_loop)]

use crate::error::ARError;
use oxidiviner_core::{Forecaster, ModelEvaluation, ModelOutput, OxiError, Result, TimeSeriesData};
use oxidiviner_math::metrics::{mae, mape, mse, rmse, smape};

/// Autoregressive (AR) model for time series forecasting.
///
/// This model predicts future values of a time series as a linear combination of its past values.
/// The general form of an AR(p) model is:
/// y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p} + ε_t
///
/// where:
/// - y_t is the value at time t
/// - c is a constant (intercept)
/// - φ₁, φ₂, ..., φₚ are the autoregressive coefficients
/// - p is the lag order (number of past values to consider)
/// - ε_t is white noise
///
/// AR models are particularly useful for:
/// - Capturing linear relationships in a time series
/// - Handling autocorrelation in data
/// - Modeling trending data or mean-reverting processes
/// - Serving as a component in more complex models (ARMA, ARIMA)
pub struct ARModel {
    /// Model name
    name: String,
    /// Lag order (number of past values to consider)
    p: usize,
    /// Include intercept/constant term
    include_intercept: bool,
    /// Estimated intercept/constant term
    intercept: Option<f64>,
    /// Estimated AR coefficients [φ₁, φ₂, ..., φₚ]
    coefficients: Option<Vec<f64>>,
    /// Fitted values over the training period
    fitted_values: Option<Vec<f64>>,
    /// Last p values from the training data (needed for forecasting)
    last_values: Option<Vec<f64>>,
    /// Mean of the training data (for mean centering)
    mean: Option<f64>,
}

impl ARModel {
    /// Creates a new Autoregressive model.
    ///
    /// # Arguments
    /// * `p` - Lag order (number of past values to use)
    /// * `include_intercept` - Whether to include an intercept/constant term
    ///
    /// # Returns
    /// * `Result<Self>` - A new AR model if parameters are valid
    pub fn new(p: usize, include_intercept: bool) -> std::result::Result<Self, ARError> {
        // Validate parameters
        if p == 0 {
            return Err(ARError::InvalidLagOrder(p));
        }

        let name = if include_intercept {
            format!("AR({})+intercept", p)
        } else {
            format!("AR({})", p)
        };

        Ok(ARModel {
            name,
            p,
            include_intercept,
            intercept: None,
            coefficients: None,
            fitted_values: None,
            last_values: None,
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

    /// Get the estimated AR coefficients.
    pub fn coefficients(&self) -> Option<&Vec<f64>> {
        self.coefficients.as_ref()
    }

    /// Get the estimated intercept/constant term.
    pub fn intercept(&self) -> Option<f64> {
        self.intercept
    }

    /// Get the mean of the training data.
    pub fn mean(&self) -> Option<f64> {
        self.mean
    }

    /// Solve the Yule-Walker equations to find AR coefficients.
    ///
    /// This implements the method of moments estimator for AR models.
    fn fit_yule_walker(&mut self, data: &[f64]) -> std::result::Result<(), ARError> {
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
                matrix[i][j] = autocorr[(i as isize - j as isize).unsigned_abs()];
            }
        }

        // Create the right-hand side
        let rhs: Vec<f64> = autocorr[1..=self.p].to_vec();

        // Solve the system using Gaussian elimination
        let phi = self.solve_linear_system(&matrix, &rhs)?;

        // Calculate the intercept if needed
        let intercept = if self.include_intercept {
            let phi_sum: f64 = phi.iter().sum();
            mean * (1.0 - phi_sum)
        } else {
            0.0
        };

        // Store the results
        self.coefficients = Some(phi);
        self.intercept = Some(intercept);
        self.mean = Some(mean);

        Ok(())
    }

    /// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
    fn solve_linear_system(
        &self,
        a: &[Vec<f64>],
        b: &[f64],
    ) -> std::result::Result<Vec<f64>, ARError> {
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

impl Forecaster for ARModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        if data.is_empty() {
            return Err(OxiError::from(ARError::EmptyData));
        }

        let n = data.values.len();

        // Need more observations than the AR order
        if n <= self.p {
            return Err(OxiError::from(ARError::InsufficientData {
                actual: n,
                expected: self.p + 1,
            }));
        }

        // Fit the model using Yule-Walker equations
        self.fit_yule_walker(&data.values).map_err(OxiError::from)?;

        // Calculate fitted values
        let mut fitted_values = vec![f64::NAN; self.p]; // First p values cannot be predicted

        // Predict the rest of the series
        for t in self.p..n {
            let mut prediction = self.intercept.unwrap_or(0.0);

            for lag in 1..=self.p {
                prediction += self.coefficients.as_ref().unwrap()[lag - 1] * data.values[t - lag];
            }

            fitted_values.push(prediction);
        }

        // Store the last p values for forecasting
        self.last_values = Some(data.values[(n - self.p)..].to_vec());
        self.fitted_values = Some(fitted_values);

        Ok(())
    }

    fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        if horizon == 0 {
            return Err(OxiError::from(ARError::InvalidHorizon(horizon)));
        }

        if self.coefficients.is_none() || self.last_values.is_none() {
            return Err(OxiError::from(ARError::NotFitted));
        }

        let coefficients = self.coefficients.as_ref().unwrap();
        let intercept = self.intercept.unwrap_or(0.0);

        // Initialize forecasts with the last p observed values
        let mut extended_series = self.last_values.as_ref().unwrap().clone();
        let mut forecasts = Vec::with_capacity(horizon);

        // Generate forecasts one step at a time
        for _ in 0..horizon {
            let mut prediction = intercept;

            // Apply AR coefficients to the most recent p values
            let n = extended_series.len();
            for lag in 1..=self.p {
                prediction += coefficients[lag - 1] * extended_series[n - lag];
            }

            // Add prediction to our forecasts and to the extended series
            forecasts.push(prediction);
            extended_series.push(prediction);
        }

        Ok(forecasts)
    }

    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        if self.coefficients.is_none() {
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
    use chrono::Utc;

    #[test]
    fn test_ar_model_constant_data() {
        // Test with constant data
        let mut values = vec![10.0; 20];
        // Add tiny variation to prevent singular matrix
        values[5] = 10.01;
        values[10] = 9.99;
        values[15] = 10.02;

        let timestamps: Vec<chrono::DateTime<Utc>> = (0..values.len())
            .map(|i| Utc::now() + chrono::Duration::days(i as i64))
            .collect();

        let ts_data = TimeSeriesData::new(timestamps, values, "constant").unwrap();

        let mut model = ARModel::new(1, true).unwrap();
        let result = model.fit(&ts_data);

        // Should fit without errors
        assert!(result.is_ok());

        // The coefficient should be small for near-constant data
        let coeffs = model.coefficients().unwrap();
        assert!(coeffs[0].abs() < 0.5);

        // Forecast should predict values close to 10.0
        let forecast = model.forecast(5).unwrap();
        assert_eq!(forecast.len(), 5);
        for f in forecast {
            assert!((f - 10.0).abs() < 0.5);
        }
    }

    #[test]
    fn test_ar_model_trending_data() {
        // Create trending data with small variations to avoid numerical issues
        let mut values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        // Add tiny random variations
        values[5] += 0.01;
        values[10] -= 0.02;
        values[15] += 0.03;

        let timestamps: Vec<chrono::DateTime<Utc>> = (0..values.len())
            .map(|i| Utc::now() + chrono::Duration::days(i as i64))
            .collect();

        let ts_data = TimeSeriesData::new(timestamps, values, "trend").unwrap();

        let mut model = ARModel::new(2, true).unwrap();
        let result = model.fit(&ts_data);
        assert!(result.is_ok());

        // Forecast with trending data
        let forecast = model.forecast(5).unwrap();

        // Check that we can make forecasts for trending data
        assert_eq!(forecast.len(), 5);

        // Check that all forecasts are finite numbers
        for f in forecast {
            assert!(f.is_finite(), "Forecast should be a finite number");
        }
    }

    #[test]
    fn test_ar_model_coefficient_access() {
        // Test accessing model parameters
        let mut model = ARModel::new(2, true).unwrap();

        // Before fitting
        assert!(model.coefficients().is_none());
        assert!(model.intercept().is_none());
        assert!(model.mean().is_none());
        assert!(model.fitted_values().is_none());

        // Create some data with known pattern
        let values = [
            1.0, 1.2, 1.1, 1.15, 1.12, 1.14, 1.13, 1.15, 1.14, 1.16, 1.15, 1.17, 1.16, 1.18, 1.17,
        ];
        let timestamps = (0..values.len())
            .map(|i| Utc::now() + chrono::Duration::days(i as i64))
            .collect();

        let ts_data = TimeSeriesData::new(timestamps, values.to_vec(), "pattern").unwrap();

        // Fit the model
        let result = model.fit(&ts_data);
        assert!(result.is_ok());

        // After fitting, these should be available
        assert!(model.coefficients().is_some());
        assert!(model.intercept().is_some());
        assert!(model.mean().is_some());
        assert!(model.fitted_values().is_some());

        // The name should include the order and intercept info
        assert_eq!(model.name(), "AR(2)+intercept");
    }

    #[test]
    fn test_ar_model_invalid_parameters() {
        // Test with invalid lag order p=0
        let result = ARModel::new(0, true);
        assert!(result.is_err());

        if let Err(ARError::InvalidLagOrder(p)) = result {
            assert_eq!(p, 0);
        } else {
            panic!("Expected InvalidLagOrder error");
        }

        // Test with insufficient data
        let mut model = ARModel::new(5, true).unwrap();

        // Create data with fewer points than needed for AR(5)
        let values = vec![1.0, 2.0, 3.0, 4.0]; // Only 4 points, need at least 5
        let timestamps = (0..values.len())
            .map(|i| Utc::now() + chrono::Duration::days(i as i64))
            .collect();

        let ts_data = TimeSeriesData::new(timestamps, values, "short").unwrap();

        // Fitting should fail
        let result = model.fit(&ts_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_ar_model_evaluation() {
        // Create data with a clear AR(1) pattern
        let values = [
            1.0, 1.2, 1.1, 1.15, 1.12, 1.14, 1.13, 1.15, 1.14, 1.16, 1.15, 1.17, 1.16, 1.18, 1.17,
        ];
        let timestamps: Vec<chrono::DateTime<Utc>> = (0..values.len())
            .map(|i| Utc::now() + chrono::Duration::days(i as i64))
            .collect();

        // We don't need the full time series, just train and test splits
        let train_len = 10;
        let train_timestamps = timestamps[0..train_len].to_vec();
        let train_values = values[0..train_len].to_vec();
        let test_timestamps = timestamps[train_len..].to_vec();
        let test_values = values[train_len..].to_vec();

        let train_data = TimeSeriesData::new(train_timestamps, train_values, "train").unwrap();
        let test_data = TimeSeriesData::new(test_timestamps, test_values, "test").unwrap();

        // Fit an AR(1) model
        let mut model = ARModel::new(1, true).unwrap();
        model.fit(&train_data).unwrap();

        // Evaluate on test data
        let eval = model.evaluate(&test_data).unwrap();

        // Check evaluation metrics
        assert!(eval.mae > 0.0);
        assert!(eval.mse > 0.0);
        assert!(eval.rmse > 0.0);
        assert!(eval.mape > 0.0);

        // Try the predict method too
        let output = model.predict(5, Some(&test_data)).unwrap();
        assert_eq!(output.forecasts.len(), 5);
        assert!(output.evaluation.is_some());

        // Predict without test data
        let output_no_test = model.predict(5, None).unwrap();
        assert_eq!(output_no_test.forecasts.len(), 5);
        assert!(output_no_test.evaluation.is_none());
    }
}
