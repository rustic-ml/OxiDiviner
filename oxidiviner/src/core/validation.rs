use crate::core::{OxiError, Result, TimeSeriesData};
use serde::{Deserialize, Serialize};

/// Accuracy metrics for forecast evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyReport {
    /// Mean Absolute Error
    pub mae: f64,
    /// Mean Squared Error  
    pub mse: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Percentage Error
    pub mape: f64,
    /// Symmetric Mean Absolute Percentage Error
    pub smape: f64,
    /// Mean Absolute Scaled Error (if baseline provided)
    pub mase: Option<f64>,
    /// Coefficient of Determination (R²)
    pub r_squared: f64,
    /// Number of observations
    pub n_observations: usize,
}

/// Utilities for time series validation and testing
pub struct ValidationUtils;

impl ValidationUtils {
    /// Split time series into training and testing sets
    ///
    /// # Arguments
    /// * `data` - The time series data to split
    /// * `train_ratio` - Proportion of data to use for training (0.0 to 1.0)
    ///
    /// # Returns
    /// * `Result<(TimeSeriesData, TimeSeriesData)>` - Training and test sets
    ///
    /// # Example
    /// ```rust
    /// # use crate::core::{TimeSeriesData, validation::ValidationUtils};
    /// # use chrono::Utc;
    /// # fn example() -> crate::core::Result<()> {
    /// # let data = TimeSeriesData::new(vec![Utc::now()], vec![1.0], "test")?;
    /// let (train, test) = ValidationUtils::time_split(&data, 0.8)?;
    /// assert!(train.values.len() > test.values.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn time_split(
        data: &TimeSeriesData,
        train_ratio: f64,
    ) -> Result<(TimeSeriesData, TimeSeriesData)> {
        if !(0.0..=1.0).contains(&train_ratio) {
            return Err(OxiError::InvalidParameter(
                "train_ratio must be between 0.0 and 1.0".into(),
            ));
        }

        if data.values.is_empty() {
            return Err(OxiError::DataError("Cannot split empty data".into()));
        }

        let split_index = (data.values.len() as f64 * train_ratio) as usize;

        if split_index == 0 || split_index >= data.values.len() {
            return Err(OxiError::InvalidParameter(
                "Split ratio results in empty training or test set".into(),
            ));
        }

        let train_timestamps = data.timestamps[..split_index].to_vec();
        let train_values = data.values[..split_index].to_vec();
        let train_name = format!("{}_train", data.name);

        let test_timestamps = data.timestamps[split_index..].to_vec();
        let test_values = data.values[split_index..].to_vec();
        let test_name = format!("{}_test", data.name);

        let train_data = TimeSeriesData::new(train_timestamps, train_values, &train_name)?;
        let test_data = TimeSeriesData::new(test_timestamps, test_values, &test_name)?;

        Ok((train_data, test_data))
    }

    /// Create time series cross-validation splits
    ///
    /// Uses a sliding window approach appropriate for time series data,
    /// where each split maintains temporal order.
    ///
    /// # Arguments
    /// * `data` - The time series data to split
    /// * `n_splits` - Number of CV splits to create
    /// * `min_train_size` - Minimum size for training sets
    ///
    /// # Returns
    /// * `Vec<(TimeSeriesData, TimeSeriesData)>` - Vector of (train, test) pairs
    pub fn time_series_cv(
        data: &TimeSeriesData,
        n_splits: usize,
        min_train_size: Option<usize>,
    ) -> Result<Vec<(TimeSeriesData, TimeSeriesData)>> {
        if n_splits == 0 {
            return Err(OxiError::InvalidParameter(
                "Number of splits must be greater than 0".into(),
            ));
        }

        if data.values.len() < n_splits + 1 {
            return Err(OxiError::DataError(
                "Not enough data points for the specified number of splits".into(),
            ));
        }

        let min_train = min_train_size.unwrap_or(n_splits);
        if data.values.len() <= min_train {
            return Err(OxiError::DataError(
                "Not enough data for minimum training size".into(),
            ));
        }

        let mut splits = Vec::new();
        let total_len = data.values.len();
        let test_size = (total_len - min_train) / n_splits;

        if test_size == 0 {
            return Err(OxiError::DataError(
                "Test size would be zero with current parameters".into(),
            ));
        }

        for i in 0..n_splits {
            let train_end = min_train + i * test_size;
            let test_start = train_end;
            let test_end = std::cmp::min(test_start + test_size, total_len);

            if test_start >= total_len {
                break;
            }

            let train_timestamps = data.timestamps[..train_end].to_vec();
            let train_values = data.values[..train_end].to_vec();
            let train_name = format!("{}_cv_train_{}", data.name, i);

            let test_timestamps = data.timestamps[test_start..test_end].to_vec();
            let test_values = data.values[test_start..test_end].to_vec();
            let test_name = format!("{}_cv_test_{}", data.name, i);

            let train_data = TimeSeriesData::new(train_timestamps, train_values, &train_name)?;
            let test_data = TimeSeriesData::new(test_timestamps, test_values, &test_name)?;

            splits.push((train_data, test_data));
        }

        Ok(splits)
    }

    /// Calculate comprehensive forecast accuracy metrics
    ///
    /// # Arguments
    /// * `actual` - Actual observed values
    /// * `predicted` - Predicted values
    /// * `baseline_naive` - Optional baseline naive forecast for MASE calculation
    ///
    /// # Returns
    /// * `AccuracyReport` - Comprehensive accuracy metrics
    pub fn accuracy_metrics(
        actual: &[f64],
        predicted: &[f64],
        baseline_naive: Option<&[f64]>,
    ) -> Result<AccuracyReport> {
        if actual.len() != predicted.len() {
            return Err(OxiError::InvalidParameter(
                "Actual and predicted values must have the same length".into(),
            ));
        }

        if actual.is_empty() {
            return Err(OxiError::DataError(
                "Cannot calculate metrics for empty arrays".into(),
            ));
        }

        let _n = actual.len() as f64;

        // Calculate basic metrics
        let mae = Self::mean_absolute_error(actual, predicted);
        let mse = Self::mean_squared_error(actual, predicted);
        let rmse = mse.sqrt();
        let mape = Self::mean_absolute_percentage_error(actual, predicted)?;
        let smape = Self::symmetric_mean_absolute_percentage_error(actual, predicted)?;
        let r_squared = Self::r_squared(actual, predicted)?;

        // Calculate MASE if baseline is provided
        let mase = if let Some(baseline) = baseline_naive {
            Some(Self::mean_absolute_scaled_error(
                actual, predicted, baseline,
            )?)
        } else {
            None
        };

        Ok(AccuracyReport {
            mae,
            mse,
            rmse,
            mape,
            smape,
            mase,
            r_squared,
            n_observations: actual.len(),
        })
    }

    /// Mean Absolute Error
    fn mean_absolute_error(actual: &[f64], predicted: &[f64]) -> f64 {
        actual
            .iter()
            .zip(predicted.iter())
            .map(|(a, p)| (a - p).abs())
            .sum::<f64>()
            / actual.len() as f64
    }

    /// Mean Squared Error
    fn mean_squared_error(actual: &[f64], predicted: &[f64]) -> f64 {
        actual
            .iter()
            .zip(predicted.iter())
            .map(|(a, p)| (a - p).powi(2))
            .sum::<f64>()
            / actual.len() as f64
    }

    /// Mean Absolute Percentage Error
    fn mean_absolute_percentage_error(actual: &[f64], predicted: &[f64]) -> Result<f64> {
        let mut sum = 0.0;
        let mut count = 0;

        for (a, p) in actual.iter().zip(predicted.iter()) {
            if a.abs() > f64::EPSILON {
                // Avoid division by zero
                sum += ((a - p) / a).abs() * 100.0;
                count += 1;
            }
        }

        if count == 0 {
            return Err(OxiError::DataError(
                "Cannot calculate MAPE: all actual values are zero".into(),
            ));
        }

        Ok(sum / count as f64)
    }

    /// Symmetric Mean Absolute Percentage Error
    fn symmetric_mean_absolute_percentage_error(actual: &[f64], predicted: &[f64]) -> Result<f64> {
        let mut sum = 0.0;
        let mut count = 0;

        for (a, p) in actual.iter().zip(predicted.iter()) {
            let denominator = (a.abs() + p.abs()) / 2.0;
            if denominator > f64::EPSILON {
                sum += ((a - p).abs() / denominator) * 100.0;
                count += 1;
            }
        }

        if count == 0 {
            return Err(OxiError::DataError(
                "Cannot calculate SMAPE: all values are zero".into(),
            ));
        }

        Ok(sum / count as f64)
    }

    /// Mean Absolute Scaled Error
    fn mean_absolute_scaled_error(
        actual: &[f64],
        predicted: &[f64],
        baseline: &[f64],
    ) -> Result<f64> {
        if baseline.len() != actual.len() {
            return Err(OxiError::InvalidParameter(
                "Baseline must have same length as actual values".into(),
            ));
        }

        let mae_forecast = Self::mean_absolute_error(actual, predicted);
        let mae_baseline = Self::mean_absolute_error(actual, baseline);

        if mae_baseline.abs() < f64::EPSILON {
            return Err(OxiError::DataError(
                "Cannot calculate MASE: baseline MAE is zero".into(),
            ));
        }

        Ok(mae_forecast / mae_baseline)
    }

    /// Coefficient of Determination (R²)
    fn r_squared(actual: &[f64], predicted: &[f64]) -> Result<f64> {
        let mean_actual = actual.iter().sum::<f64>() / actual.len() as f64;

        let ss_res: f64 = actual
            .iter()
            .zip(predicted.iter())
            .map(|(a, p)| (a - p).powi(2))
            .sum();

        let ss_tot: f64 = actual.iter().map(|a| (a - mean_actual).powi(2)).sum();

        if ss_tot.abs() < f64::EPSILON {
            return Err(OxiError::DataError(
                "Cannot calculate R²: total sum of squares is zero".into(),
            ));
        }

        Ok(1.0 - (ss_res / ss_tot))
    }
}

/// Configuration for backtesting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial training window size
    pub initial_window: usize,
    /// Step size for rolling forward
    pub step_size: usize,
    /// Whether to use expanding or rolling window
    pub expanding_window: bool,
    /// Forecast horizon for each step
    pub forecast_horizon: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_window: 50,
            step_size: 1,
            expanding_window: true,
            forecast_horizon: 1,
        }
    }
}

/// Result of a backtesting run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Accuracy metrics for each forecast step
    pub step_metrics: Vec<AccuracyReport>,
    /// Overall accuracy metrics
    pub overall_metrics: AccuracyReport,
    /// All forecasts made during backtesting
    pub forecasts: Vec<Vec<f64>>,
    /// Actual values corresponding to forecasts
    pub actuals: Vec<Vec<f64>>,
    /// Configuration used
    pub config: BacktestConfig,
}

/// Model parameter validator with comprehensive validation rules
pub struct ModelValidator;

impl ModelValidator {
    /// Validate ARIMA parameters
    pub fn validate_arima_params(p: usize, d: usize, q: usize) -> Result<()> {
        // Check that at least one parameter is non-zero
        if p == 0 && d == 0 && q == 0 {
            return Err(OxiError::InvalidParameter(
                "ARIMA model must have at least one non-zero parameter (p, d, or q)".to_string(),
            ));
        }

        if p > 10 {
            return Err(OxiError::InvalidParameter(
                "AR order (p) too high: maximum recommended value is 10".to_string(),
            ));
        }
        if d > 2 {
            return Err(OxiError::InvalidParameter(
                "Differencing order (d) too high: maximum recommended value is 2".to_string(),
            ));
        }
        if q > 10 {
            return Err(OxiError::InvalidParameter(
                "MA order (q) too high: maximum recommended value is 10".to_string(),
            ));
        }
        if p + d + q > 15 {
            return Err(OxiError::InvalidParameter(
                "Total model complexity (p+d+q) too high: maximum recommended value is 15"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Validate smoothing parameters (alpha, beta, gamma)
    pub fn validate_smoothing_param(param: f64, name: &str) -> Result<()> {
        if param <= 0.0 || param >= 1.0 {
            return Err(OxiError::InvalidParameter(format!(
                "Smoothing parameter {} must be between 0 and 1 (exclusive), got {}",
                name, param
            )));
        }
        Ok(())
    }

    /// Validate damping parameter (phi)
    pub fn validate_damping_param(phi: f64) -> Result<()> {
        if phi <= 0.0 || phi >= 1.0 {
            return Err(OxiError::InvalidParameter(format!(
                "Damping parameter φ must be between 0 and 1 (exclusive), got {}",
                phi
            )));
        }
        Ok(())
    }

    /// Validate seasonal period
    pub fn validate_seasonal_period(period: usize) -> Result<()> {
        if period < 2 {
            return Err(OxiError::InvalidParameter(format!(
                "Seasonal period must be at least 2, got {}",
                period
            )));
        }
        if period > 366 {
            return Err(OxiError::InvalidParameter(format!(
                "Seasonal period too large: maximum recommended value is 366, got {}",
                period
            )));
        }
        Ok(())
    }

    /// Validate GARCH parameters for stationarity
    pub fn validate_garch_stationarity(alpha: &[f64], beta: &[f64]) -> Result<()> {
        let sum: f64 = alpha.iter().sum::<f64>() + beta.iter().sum::<f64>();
        if sum >= 1.0 {
            return Err(OxiError::InvalidParameter(format!(
                "GARCH stationarity condition violated: sum of α and β parameters must be < 1, got {}",
                sum
            )));
        }
        Ok(())
    }

    /// Validate window size for moving average models
    pub fn validate_window_size(window: usize, data_length: usize) -> Result<()> {
        if window == 0 {
            return Err(OxiError::InvalidParameter(
                "Window size must be greater than 0".to_string(),
            ));
        }
        if window > data_length {
            return Err(OxiError::InvalidParameter(format!(
                "Window size ({}) cannot be larger than data length ({})",
                window, data_length
            )));
        }
        if window > data_length / 2 {
            return Err(OxiError::InvalidParameter(format!(
                "Window size ({}) is more than half the data length ({}); consider using a smaller window",
                window, data_length
            )));
        }
        Ok(())
    }

    /// Validate forecast horizon
    pub fn validate_forecast_horizon(horizon: usize, data_length: usize) -> Result<()> {
        if horizon == 0 {
            return Err(OxiError::InvalidParameter(
                "Forecast horizon must be greater than 0".to_string(),
            ));
        }
        if horizon > data_length {
            return Err(OxiError::InvalidParameter(format!(
                "Forecast horizon ({}) is longer than the training data length ({}); forecasts may be unreliable",
                horizon, data_length
            )));
        }
        Ok(())
    }

    /// Validate confidence level for confidence intervals
    pub fn validate_confidence_level(confidence: f64) -> Result<()> {
        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(OxiError::InvalidParameter(format!(
                "Confidence level must be between 0 and 1 (exclusive), got {}",
                confidence
            )));
        }
        if confidence < 0.5 {
            return Err(OxiError::InvalidParameter(format!(
                "Confidence level ({}) is less than 0.5; this is unusual for confidence intervals",
                confidence
            )));
        }
        Ok(())
    }

    /// Validate that time series data has sufficient length for a model
    pub fn validate_min_data_length(
        data_length: usize,
        min_required: usize,
        model_name: &str,
    ) -> Result<()> {
        if data_length < min_required {
            return Err(OxiError::InvalidParameter(format!(
                "{} model requires at least {} data points, got {}",
                model_name, min_required, data_length
            )));
        }
        Ok(())
    }

    /// Validate that a parameter is non-negative
    pub fn validate_non_negative(value: f64, param_name: &str) -> Result<()> {
        if value < 0.0 {
            return Err(OxiError::InvalidParameter(format!(
                "Parameter {} must be non-negative, got {}",
                param_name, value
            )));
        }
        Ok(())
    }

    /// Validate that a parameter is positive
    pub fn validate_positive(value: f64, param_name: &str) -> Result<()> {
        if value <= 0.0 {
            return Err(OxiError::InvalidParameter(format!(
                "Parameter {} must be positive, got {}",
                param_name, value
            )));
        }
        Ok(())
    }

    /// Check for reasonable values in time series data
    pub fn validate_data_quality(data: &[f64], name: &str) -> Result<()> {
        if data.is_empty() {
            return Err(OxiError::DataError(format!(
                "Time series {} is empty",
                name
            )));
        }

        // Check for NaN or infinite values
        for (i, &value) in data.iter().enumerate() {
            if value.is_nan() {
                return Err(OxiError::DataError(format!(
                    "Time series {} contains NaN at index {}",
                    name, i
                )));
            }
            if value.is_infinite() {
                return Err(OxiError::DataError(format!(
                    "Time series {} contains infinite value at index {}",
                    name, i
                )));
            }
        }

        // Check for extreme values (potentially indicating data issues)
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if max_val / min_val.abs() > 1e10 {
            return Err(OxiError::DataError(format!(
                "Time series {} has extreme value range (min: {}, max: {}); consider scaling or transforming the data",
                name, min_val, max_val
            )));
        }

        Ok(())
    }

    /// Comprehensive validation for model fitting
    pub fn validate_for_fitting(
        data: &TimeSeriesData,
        min_points: usize,
        model_name: &str,
    ) -> Result<()> {
        // Check data quality
        Self::validate_data_quality(&data.values, &data.name)?;

        // Check minimum data length
        Self::validate_min_data_length(data.values.len(), min_points, model_name)?;

        // Check timestamps and values have same length
        if data.timestamps.len() != data.values.len() {
            return Err(OxiError::DataError(format!(
                "Timestamps ({}) and values ({}) have different lengths",
                data.timestamps.len(),
                data.values.len()
            )));
        }

        // Check for monotonic timestamps
        for i in 1..data.timestamps.len() {
            if data.timestamps[i] <= data.timestamps[i - 1] {
                return Err(OxiError::DataError(format!(
                    "Timestamps are not strictly increasing at index {}",
                    i
                )));
            }
        }

        Ok(())
    }

    /// Validate moving average model parameters
    pub fn validate_ma_params(window: usize) -> Result<()> {
        if window == 0 {
            return Err(OxiError::InvalidParameter(
                "Moving average window must be positive".to_string(),
            ));
        }
        if window > 50 {
            return Err(OxiError::InvalidParameter(
                "Moving average window too large (max 50)".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate exponential smoothing model parameters
    pub fn validate_exponential_smoothing_params(
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
    ) -> Result<()> {
        Self::validate_smoothing_param(alpha, "alpha")?;

        if let Some(beta) = beta {
            Self::validate_smoothing_param(beta, "beta")?;
        }

        if let Some(gamma) = gamma {
            Self::validate_smoothing_param(gamma, "gamma")?;
        }

        Ok(())
    }

    /// Validate GARCH stationarity constraints
    pub fn validate_garch_stationarity_constraints(alpha: &[f64], beta: &[f64]) -> Result<()> {
        let sum: f64 = alpha.iter().sum::<f64>() + beta.iter().sum::<f64>();
        if sum >= 1.0 {
            return Err(OxiError::InvalidParameter(format!(
                "GARCH stationarity condition violated: sum of α and β parameters must be < 1, got {}",
                sum
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeriesData;
    use chrono::{Duration, Utc};

    fn create_test_data() -> TimeSeriesData {
        let timestamps = (0..10).map(|i| Utc::now() + Duration::days(i)).collect();
        let values = (0..10).map(|i| i as f64).collect();

        TimeSeriesData::new(timestamps, values, "test").unwrap()
    }

    #[test]
    fn test_time_split() {
        let data = create_test_data();
        let (train, test) = ValidationUtils::time_split(&data, 0.7).unwrap();

        assert_eq!(train.values.len(), 7);
        assert_eq!(test.values.len(), 3);
        assert_eq!(train.values[6], 6.0);
        assert_eq!(test.values[0], 7.0);
    }

    #[test]
    fn test_accuracy_metrics() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![1.1, 2.1, 2.9, 4.1, 4.9];

        let metrics = ValidationUtils::accuracy_metrics(&actual, &predicted, None).unwrap();

        assert!(metrics.mae < 0.2);
        assert!(metrics.rmse < 0.2);
        assert!(metrics.r_squared > 0.9);
        assert_eq!(metrics.n_observations, 5);
    }

    #[test]
    fn test_time_series_cv() {
        let data = create_test_data();
        let splits = ValidationUtils::time_series_cv(&data, 3, Some(4)).unwrap();

        assert_eq!(splits.len(), 3);
        assert!(splits[0].0.values.len() >= 4); // Minimum training size
        assert!(splits[1].0.values.len() > splits[0].0.values.len()); // Expanding
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;
    use crate::core::TimeSeriesData;
    use chrono::{TimeZone, Utc};

    fn create_test_data() -> TimeSeriesData {
        let timestamps = (0..10)
            .map(|i| Utc.timestamp_opt(1609459200 + i * 86400, 0).unwrap())
            .collect();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        TimeSeriesData::new(timestamps, values, "test_series").unwrap()
    }

    #[test]
    fn test_model_validator_arima_validation() {
        assert!(ModelValidator::validate_arima_params(1, 1, 1).is_ok());
        assert!(ModelValidator::validate_arima_params(5, 2, 3).is_ok());
        assert!(ModelValidator::validate_arima_params(11, 1, 1).is_err());
        assert!(ModelValidator::validate_arima_params(1, 11, 1).is_err());
        assert!(ModelValidator::validate_arima_params(1, 1, 11).is_err());
    }

    #[test]
    fn test_model_validator_smoothing_param() {
        assert!(ModelValidator::validate_smoothing_param(0.1, "alpha").is_ok());
        assert!(ModelValidator::validate_smoothing_param(0.5, "alpha").is_ok());
        assert!(ModelValidator::validate_smoothing_param(0.9, "alpha").is_ok());
        assert!(ModelValidator::validate_smoothing_param(0.0, "alpha").is_err());
        assert!(ModelValidator::validate_smoothing_param(1.0, "alpha").is_err());
        assert!(ModelValidator::validate_smoothing_param(-0.1, "alpha").is_err());
        assert!(ModelValidator::validate_smoothing_param(1.1, "alpha").is_err());
    }

    #[test]
    fn test_model_validator_seasonal_period() {
        assert!(ModelValidator::validate_seasonal_period(2).is_ok());
        assert!(ModelValidator::validate_seasonal_period(12).is_ok());
        assert!(ModelValidator::validate_seasonal_period(365).is_ok());
        assert!(ModelValidator::validate_seasonal_period(366).is_ok());
        assert!(ModelValidator::validate_seasonal_period(0).is_err());
        assert!(ModelValidator::validate_seasonal_period(1).is_err());
        assert!(ModelValidator::validate_seasonal_period(367).is_err());
    }

    #[test]
    fn test_model_validator_data_quality() {
        let good_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(ModelValidator::validate_data_quality(&good_data, "test").is_ok());

        let nan_data = vec![1.0, f64::NAN, 3.0];
        assert!(ModelValidator::validate_data_quality(&nan_data, "test").is_err());

        let inf_data = vec![1.0, f64::INFINITY, 3.0];
        assert!(ModelValidator::validate_data_quality(&inf_data, "test").is_err());

        let empty_data: Vec<f64> = vec![];
        assert!(ModelValidator::validate_data_quality(&empty_data, "test").is_err());
    }

    #[test]
    fn test_model_validator_for_fitting() {
        let data = create_test_data();
        assert!(ModelValidator::validate_for_fitting(&data, 5, "TestModel").is_ok());
        assert!(ModelValidator::validate_for_fitting(&data, 15, "TestModel").is_err());
    }

    #[test]
    fn test_model_validator_min_data_length() {
        assert!(ModelValidator::validate_min_data_length(10, 5, "TestModel").is_ok());
        assert!(ModelValidator::validate_min_data_length(3, 5, "TestModel").is_err());
    }

    #[test]
    fn test_model_validator_non_negative() {
        assert!(ModelValidator::validate_non_negative(0.0, "param").is_ok());
        assert!(ModelValidator::validate_non_negative(1.0, "param").is_ok());
        assert!(ModelValidator::validate_non_negative(-1.0, "param").is_err());
    }

    #[test]
    fn test_model_validator_positive() {
        assert!(ModelValidator::validate_positive(1.0, "param").is_ok());
        assert!(ModelValidator::validate_positive(0.1, "param").is_ok());
        assert!(ModelValidator::validate_positive(0.0, "param").is_err());
        assert!(ModelValidator::validate_positive(-1.0, "param").is_err());
    }

    #[test]
    fn test_validation_utils_time_split_basic() {
        let data = create_test_data();
        let (train, test) = ValidationUtils::time_split(&data, 0.8).unwrap();

        assert_eq!(train.len(), 8);
        assert_eq!(test.len(), 2);
        assert_eq!(train.values[7], 8.0);
        assert_eq!(test.values[0], 9.0);
    }

    #[test]
    fn test_validation_utils_time_split_edge_cases() {
        let data = create_test_data();

        // Invalid ratio
        assert!(ValidationUtils::time_split(&data, -0.1).is_err());
        assert!(ValidationUtils::time_split(&data, 1.1).is_err());
    }

    #[test]
    fn test_validation_utils_accuracy_metrics() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![1.1, 1.9, 3.2, 3.8, 5.1];

        let metrics = ValidationUtils::accuracy_metrics(&actual, &predicted, None).unwrap();

        assert!(metrics.mae > 0.0);
        assert!(metrics.mse > 0.0);
        assert!(metrics.rmse > 0.0);
        assert!(metrics.mape > 0.0);
        assert!(metrics.smape > 0.0);
        assert_eq!(metrics.n_observations, 5);

        // RMSE should be sqrt of MSE
        assert!((metrics.rmse - metrics.mse.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_validation_utils_accuracy_metrics_perfect() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = actual.clone();

        let metrics = ValidationUtils::accuracy_metrics(&actual, &predicted, None).unwrap();

        assert_eq!(metrics.mae, 0.0);
        assert_eq!(metrics.mse, 0.0);
        assert_eq!(metrics.rmse, 0.0);
        assert_eq!(metrics.mape, 0.0);
        assert_eq!(metrics.smape, 0.0);
    }

    #[test]
    fn test_validation_utils_accuracy_metrics_errors() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.1, 1.9]; // Different length

        assert!(ValidationUtils::accuracy_metrics(&actual, &predicted, None).is_err());

        let empty_actual: Vec<f64> = vec![];
        let empty_predicted: Vec<f64> = vec![];
        assert!(ValidationUtils::accuracy_metrics(&empty_actual, &empty_predicted, None).is_err());
    }

    #[test]
    fn test_validation_utils_time_series_cv() {
        let data = create_test_data();
        let folds = ValidationUtils::time_series_cv(&data, 3, Some(4)).unwrap();

        assert!(!folds.is_empty());

        for (_train, test) in folds {
            assert!(!test.is_empty());
        }
    }

    #[test]
    fn test_validation_utils_cv_errors() {
        let data = create_test_data();

        // Zero splits
        assert!(ValidationUtils::time_series_cv(&data, 0, None).is_err());

        // Too many splits for data size
        assert!(ValidationUtils::time_series_cv(&data, 20, None).is_err());
    }

    #[test]
    fn test_backtest_config_default() {
        let config = BacktestConfig::default();
        assert_eq!(config.initial_window, 50);
        assert_eq!(config.step_size, 1);
        assert!(config.expanding_window);
        assert_eq!(config.forecast_horizon, 1);
    }

    #[test]
    fn test_accuracy_report_creation() {
        let report = AccuracyReport {
            mae: 1.0,
            mse: 2.0,
            rmse: 1.414,
            mape: 10.0,
            smape: 15.0,
            mase: Some(1.2),
            r_squared: 0.95,
            n_observations: 100,
        };

        // Test basic report creation
        assert_eq!(report.mae, 1.0);
        assert_eq!(report.mse, 2.0);
        assert_eq!(report.mase, Some(1.2));
        assert_eq!(report.r_squared, 0.95);
        assert_eq!(report.n_observations, 100);
        assert!(report.rmse > 0.0);
    }
}
