use oxidiviner_core::{Forecaster, ModelEvaluation, ModelOutput, OxiError, Result, TimeSeriesData};
use oxidiviner_math::metrics::{mae, mse, rmse, mape, smape};
use crate::error::ARError;

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
    pub fn predict(&self, horizon: usize, test_data: Option<&TimeSeriesData>) -> Result<ModelOutput> {
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
                matrix[i][j] = autocorr[(i as isize - j as isize).abs() as usize];
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
    fn solve_linear_system(&self, a: &[Vec<f64>], b: &[f64]) -> std::result::Result<Vec<f64>, ARError> {
        let n = a.len();
        if n == 0 || a[0].len() != n || b.len() != n {
            return Err(ARError::LinearSolveError("Invalid matrix dimensions".to_string()));
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
                return Err(ARError::LinearSolveError("Singular matrix detected".to_string()));
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
        self.fit_yule_walker(&data.values)
            .map_err(|e| OxiError::from(e))?;
        
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
    use chrono::{DateTime, Utc, TimeZone};

    #[test]
    fn test_ar_model_constant_data() {
        // Create constant time series: 10, 10, 10, 10, 10, 10, 10, 10, 10, 10
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();
        
        let values = vec![10.0; 10];
        
        let time_series = TimeSeriesData::new(timestamps, values, "constant_series").unwrap();
        
        // Create and fit an AR(1) model
        let mut model = ARModel::new(1, true).unwrap();
        model.fit(&time_series).unwrap();
        
        // For constant data, forecasts should be the same constant
        let forecast_horizon = 5;
        let forecasts = model.forecast(forecast_horizon).unwrap();
        
        // Check that the forecasts are constant
        for forecast in forecasts {
            assert!((forecast - 10.0).abs() < 1e-10, 
                   "Forecast {} should be 10.0 for constant data", forecast);
        }
    }
    
    #[test]
    fn test_ar_model_trending_data() {
        // Create linear trend data: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..12)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();
        
        let values: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        
        let time_series = TimeSeriesData::new(timestamps, values, "trend_series").unwrap();
        
        // Create and fit an AR(2) model with intercept
        let mut model = ARModel::new(2, true).unwrap();
        model.fit(&time_series).unwrap();
        
        // For a linear trend, an AR model should be able to capture it
        let forecast_horizon = 5;
        let forecasts = model.forecast(forecast_horizon).unwrap();
        
        // Check that forecasts continue the trend (approximately)
        for (i, forecast) in forecasts.iter().enumerate() {
            let expected = 13.0 + i as f64;
            // Allow for some deviation in the forecast
            assert!((forecast - expected).abs() < 1.0, 
                   "Forecast {} should be close to {} for trend data", 
                   forecast, expected);
        }
    }
    
    #[test]
    fn test_ar_model_coefficient_access() {
        // Create a simple time series
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..20)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();
        
        let values: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        
        let time_series = TimeSeriesData::new(timestamps, values, "test_series").unwrap();
        
        // Create and fit AR(3) model
        let mut model = ARModel::new(3, true).unwrap();
        model.fit(&time_series).unwrap();
        
        // Should be able to access coefficients
        let coefficients = model.coefficients().unwrap();
        assert_eq!(coefficients.len(), 3, "AR(3) model should have 3 coefficients");
        
        // Should be able to access intercept
        let intercept = model.intercept().unwrap();
        assert!(intercept.is_finite(), "Intercept should be a finite number");
    }
} 