use crate::math::statistics::{mean, std_dev};

/// Calculate the simple moving average (SMA) of a slice of values
///
/// # Arguments
/// * `values` - Slice of values to compute moving average for
/// * `window_size` - Size of the moving window
///
/// # Returns
/// A vector of moving averages with length = values.len() - window_size + 1
pub fn moving_average(values: &[f64], window_size: usize) -> Vec<f64> {
    if values.len() < window_size || window_size == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(values.len() - window_size + 1);

    // Calculate the sum of the first window
    let mut window_sum = values[0..window_size].iter().sum::<f64>();
    result.push(window_sum / window_size as f64);

    // Calculate the rest of the windows using the sliding window technique
    for i in window_size..values.len() {
        window_sum = window_sum - values[i - window_size] + values[i];
        result.push(window_sum / window_size as f64);
    }

    result
}

/// Calculate the exponential moving average (EMA) of a slice of values
///
/// # Arguments
/// * `values` - Slice of values to compute EMA for
/// * `alpha` - Smoothing factor, between 0 and 1 (higher means more weight to recent observations)
///
/// # Returns
/// A vector of exponential moving averages with the same length as values
pub fn exponential_moving_average(values: &[f64], alpha: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    if alpha <= 0.0 || alpha > 1.0 {
        // Invalid alpha, return empty vector
        return Vec::new();
    }

    let mut result = Vec::with_capacity(values.len());

    // Initialize with the first value
    result.push(values[0]);

    // Calculate the rest of the EMAs
    for i in 1..values.len() {
        let ema = alpha * values[i] + (1.0 - alpha) * result[i - 1];
        result.push(ema);
    }

    result
}

/// Standardize a data vector (z-score normalization)
pub fn standardize(data: &[f64]) -> (Vec<f64>, f64, f64) {
    let mean_val = mean(data);
    let std_dev_val = std_dev(data);

    if std_dev_val == 0.0 {
        return (vec![0.0; data.len()], mean_val, std_dev_val);
    }

    let standardized = data.iter().map(|&x| (x - mean_val) / std_dev_val).collect();

    (standardized, mean_val, std_dev_val)
}

/// Revert standardization
pub fn destandardize(data: &[f64], mean: f64, std_dev: f64) -> Vec<f64> {
    data.iter().map(|&x| x * std_dev + mean).collect()
}

/// Calculate the difference of a time series (for detrending)
pub fn difference(data: &[f64]) -> Vec<f64> {
    if data.len() <= 1 {
        return Vec::new();
    }

    data.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Undo differencing given the original first value
pub fn undifference(diffs: &[f64], first_value: f64) -> Vec<f64> {
    if diffs.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(diffs.len() + 1);
    result.push(first_value);

    for &diff in diffs {
        let next_value = result.last().unwrap() + diff;
        result.push(next_value);
    }

    result
}

/// Measure stationarity of a time series (returns coefficient of variation of differences)
pub fn stationarity_measure(data: &[f64]) -> f64 {
    if data.len() <= 2 {
        return 0.0;
    }

    // Calculate first differences
    let diffs: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();

    // Calculate coefficient of variation (std_dev / mean)
    let mean_diff = mean(&diffs);

    // Avoid division by zero
    if mean_diff.abs() < 1e-10 {
        return f64::INFINITY;
    }

    let std_diff = std_dev(&diffs);
    std_diff / mean_diff.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moving_average() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Window size 3
        let ma3 = moving_average(&values, 3);
        assert_eq!(ma3.len(), 3);
        assert!((ma3[0] - 2.0).abs() < 1e-6); // (1+2+3)/3 = 2
        assert!((ma3[1] - 3.0).abs() < 1e-6); // (2+3+4)/3 = 3
        assert!((ma3[2] - 4.0).abs() < 1e-6); // (3+4+5)/3 = 4

        // Window size 2
        let ma2 = moving_average(&values, 2);
        assert_eq!(ma2.len(), 4);
        assert!((ma2[0] - 1.5).abs() < 1e-6); // (1+2)/2 = 1.5
        assert!((ma2[1] - 2.5).abs() < 1e-6); // (2+3)/2 = 2.5
        assert!((ma2[2] - 3.5).abs() < 1e-6); // (3+4)/2 = 3.5
        assert!((ma2[3] - 4.5).abs() < 1e-6); // (4+5)/2 = 4.5

        // Empty input
        let empty = moving_average(&[], 3);
        assert!(empty.is_empty());

        // Window size larger than input
        let too_large = moving_average(&values, 6);
        assert!(too_large.is_empty());

        // Zero window size
        let zero_window = moving_average(&values, 0);
        assert!(zero_window.is_empty());
    }

    #[test]
    fn test_exponential_moving_average() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Alpha = 0.3
        let ema1 = exponential_moving_average(&values, 0.3);
        assert_eq!(ema1.len(), 5);
        assert!((ema1[0] - 1.0).abs() < 1e-6); // First value is the same
        assert!((ema1[1] - 1.3).abs() < 1e-6); // 0.3*2 + 0.7*1 = 1.3

        // Alpha = 0.5
        let ema2 = exponential_moving_average(&values, 0.5);
        assert_eq!(ema2.len(), 5);
        assert!((ema2[0] - 1.0).abs() < 1e-6); // First value is the same
        assert!((ema2[1] - 1.5).abs() < 1e-6); // 0.5*2 + 0.5*1 = 1.5
        assert!((ema2[2] - 2.25).abs() < 1e-6); // 0.5*3 + 0.5*1.5 = 2.25

        // Invalid alpha
        let invalid1 = exponential_moving_average(&values, 0.0);
        assert!(invalid1.is_empty());

        let invalid2 = exponential_moving_average(&values, 1.1);
        assert!(invalid2.is_empty());

        // Empty input
        let empty = exponential_moving_average(&[], 0.3);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_standardize() {
        // Test normal case
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (standardized, mean_val, std_dev_val) = standardize(&data);

        assert_eq!(mean_val, 3.0);
        assert!((std_dev_val - std_dev(&data)).abs() < 1e-6);
        assert_eq!(standardized.len(), data.len());

        // Check standardized values (should have mean=0, std=1)
        let std_mean = mean(&standardized);
        let std_std = std_dev(&standardized);
        assert!(std_mean.abs() < 1e-6); // Should be approximately 0
        assert!((std_std - 1.0).abs() < 1e-6); // Should be approximately 1

        // Test with constant data (should handle 0 std dev)
        let constant_data = vec![2.0, 2.0, 2.0];
        let (standardized_const, mean_const, std_dev_const) = standardize(&constant_data);

        assert_eq!(mean_const, 2.0);
        assert_eq!(std_dev_const, 0.0);
        assert_eq!(standardized_const, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_destandardize() {
        // Standardize and then destandardize should give original data
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let (standardized, mean_val, std_dev_val) = standardize(&data);

        let destandardized = destandardize(&standardized, mean_val, std_dev_val);

        assert_eq!(destandardized.len(), data.len());

        // Check values are close to original (within rounding error)
        for i in 0..data.len() {
            assert!((destandardized[i] - data[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_difference() {
        // Test normal case
        let data = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let diffs = difference(&data);

        assert_eq!(diffs.len(), data.len() - 1);
        assert_eq!(diffs, vec![2.0, 3.0, 4.0, 5.0]);

        // Test empty input
        let empty = difference(&[]);
        assert!(empty.is_empty());

        // Test single value
        let single = difference(&[5.0]);
        assert!(single.is_empty());
    }

    #[test]
    fn test_undifference() {
        // Test normal case
        let diffs = vec![2.0, 3.0, 4.0, 5.0];
        let first_value = 1.0;

        let original = undifference(&diffs, first_value);

        assert_eq!(original.len(), diffs.len() + 1);
        assert_eq!(original, vec![1.0, 3.0, 6.0, 10.0, 15.0]);

        // Test empty input
        let empty = undifference(&[], 5.0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_stationarity_measure() {
        // Stationary data - constant differences
        let stationary = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Differences are all 1.0
        let stationary_measure = stationarity_measure(&stationary);
        assert_eq!(stationary_measure, 0.0); // Standard deviation of differences is 0

        // Non-stationary data - increasing differences
        let non_stationary = vec![1.0, 2.0, 4.0, 7.0, 11.0]; // Differences: 1, 2, 3, 4
        let non_stationary_measure = stationarity_measure(&non_stationary);
        assert!(non_stationary_measure > 0.0);

        // Test with too few values
        let too_small = stationarity_measure(&[1.0]);
        assert_eq!(too_small, 0.0);

        // Test with zeroes (mean of differences is zero)
        let zeros = vec![0.0, 0.0, 0.0, 0.0];
        let zeros_measure = stationarity_measure(&zeros);
        assert_eq!(zeros_measure, f64::INFINITY);
    }
}
