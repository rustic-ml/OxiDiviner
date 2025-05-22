// No need for statrs imports as we're implementing everything ourselves

/// Calculate the mean of an array of values
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    let mut sum = 0.0;
    for val in values {
        sum += val;
    }
    sum / values.len() as f64
}

/// Calculate the variance of an array of values
pub fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    let mean_val = mean(values);
    let mut sum_squared_diff = 0.0;

    for val in values {
        let diff = val - mean_val;
        sum_squared_diff += diff * diff;
    }

    sum_squared_diff / values.len() as f64
}

/// Calculate the standard deviation of an array of values
pub fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    variance(values).sqrt()
}

/// Calculate covariance between two arrays
pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    if x.is_empty() || y.is_empty() || x.len() != y.len() {
        return f64::NAN;
    }

    let n = x.len();
    let mean_x = mean(x);
    let mean_y = mean(y);

    let mut sum_xy = 0.0;
    for i in 0..n {
        sum_xy += (x[i] - mean_x) * (y[i] - mean_y);
    }

    sum_xy / (n as f64)
}

/// Calculate correlation coefficient between two arrays
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.is_empty() || y.is_empty() || x.len() != y.len() {
        return f64::NAN;
    }

    let cov_xy = covariance(x, y);
    let std_x = std_dev(x);
    let std_y = std_dev(y);

    if std_x == 0.0 || std_y == 0.0 {
        return 0.0; // No correlation if one variable is constant
    }

    cov_xy / (std_x * std_y)
}

/// Calculate autocorrelation for a specific lag
pub fn autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.is_empty() || lag >= data.len() {
        return f64::NAN;
    }

    if lag == 0 {
        return 1.0; // Autocorrelation at lag 0 is always 1
    }

    let mean_val = mean(data);
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..data.len() - lag {
        numerator += (data[i] - mean_val) * (data[i + lag] - mean_val);
    }

    for i in 0..data.len() {
        denominator += (data[i] - mean_val) * (data[i] - mean_val);
    }

    if denominator == 0.0 {
        return 0.0;
    }

    numerator / denominator
}

/// Calculate the quantile value from an array
pub fn quantile(data: &[f64], q: f64) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }

    if q < 0.0 || q > 1.0 {
        return f64::NAN;
    }

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted_data.len();

    if n == 1 {
        return sorted_data[0];
    }

    let pos = q * (n - 1) as f64;
    let idx_lower = pos.floor() as usize;
    let idx_upper = pos.ceil() as usize;

    if idx_lower == idx_upper {
        return sorted_data[idx_lower];
    }

    let weight_upper = pos - idx_lower as f64;
    let weight_lower = 1.0 - weight_upper;

    weight_lower * sorted_data[idx_lower] + weight_upper * sorted_data[idx_upper]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&data) - 3.0).abs() < 1e-10);

        let data = vec![-5.0, -3.0, 0.0, 3.0, 5.0];
        assert!((mean(&data) - 0.0).abs() < 1e-10);

        // Empty data should return NaN
        let data: Vec<f64> = vec![];
        assert!(mean(&data).is_nan());
    }

    #[test]
    fn test_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Variance = sum((x_i - mean)²) / n = 2
        assert!((variance(&data) - 2.0).abs() < 1e-10);

        let data = vec![-5.0, -3.0, 0.0, 3.0, 5.0];
        // Variance = sum((x_i - 0)²) / n = (25 + 9 + 0 + 9 + 25) / 5 = 13.6
        assert!((variance(&data) - 13.6).abs() < 1e-10);

        // Empty data should return NaN
        let data: Vec<f64> = vec![];
        assert!(variance(&data).is_nan());
    }

    #[test]
    fn test_std_dev() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // std_dev = sqrt(variance) = sqrt(2) ≈ 1.414
        assert!((std_dev(&data) - 1.414).abs() < 0.001);

        let data = vec![-5.0, -3.0, 0.0, 3.0, 5.0];
        // std_dev = sqrt(13.6) ≈ 3.688
        assert!((std_dev(&data) - 3.688).abs() < 0.001);

        // Empty data should return NaN
        let data: Vec<f64> = vec![];
        assert!(std_dev(&data).is_nan());
    }

    #[test]
    fn test_covariance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        // Perfect negative correlation
        assert!((covariance(&x, &y) + 2.0).abs() < 0.001);

        let y2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        // Perfect positive correlation
        assert!((covariance(&x, &y2) - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        // Perfect negative correlation should be -1
        assert!((correlation(&x, &y) + 1.0).abs() < 0.001);

        let y2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        // Perfect positive correlation should be 1
        assert!((correlation(&x, &y2) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_autocorrelation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Lag 0 autocorrelation should be 1
        assert!((autocorrelation(&x, 0) - 1.0).abs() < 0.001);

        // For trend data, lag 1 autocorrelation should be positive and high
        // The current implementation gives a result of about 0.4
        assert!(autocorrelation(&x, 1) > 0.3);

        // Test with oscillating data
        let y = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        assert!(autocorrelation(&y, 1) < -0.5); // Strong negative autocorrelation at lag 1
        assert!(autocorrelation(&y, 2) > 0.5); // Strong positive autocorrelation at lag 2
    }

    #[test]
    fn test_quantile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        assert!((quantile(&data, 0.5) - 5.5).abs() < 0.001); // median
        assert!((quantile(&data, 0.25) - 3.25).abs() < 0.001); // first quartile
        assert!((quantile(&data, 0.75) - 7.75).abs() < 0.001); // third quartile

        // Empty data should return NaN
        let empty: Vec<f64> = vec![];
        assert!(quantile(&empty, 0.5).is_nan());
    }
}
