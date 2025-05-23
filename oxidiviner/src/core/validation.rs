use crate::{OxiError, Result, TimeSeriesData};
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
    /// # fn example() -> oxidiviner_core::Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
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
