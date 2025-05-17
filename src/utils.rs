#[cfg(feature = "ndarray_support")]
use ndarray::Array1;

/// Calculate the mean of an array of values
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Calculate the variance of an array of values
pub fn variance(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    
    let mean_val = mean(values);
    let sum_squared_diff = values.iter()
        .map(|&x| (x - mean_val).powi(2))
        .sum::<f64>();
    
    sum_squared_diff / (values.len() - 1) as f64
}

/// Calculate the standard deviation of an array of values
pub fn std_dev(values: &[f64]) -> f64 {
    variance(values).sqrt()
}

/// Calculate the mean absolute error (MAE) between actual and predicted values
pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    let n = actual.len().min(predicted.len());
    if n == 0 {
        return 0.0;
    }
    
    let sum_abs_errors = actual.iter()
        .zip(predicted.iter())
        .map(|(&a, &p)| (a - p).abs())
        .sum::<f64>();
    
    sum_abs_errors / n as f64
}

/// Calculate the mean squared error (MSE) between actual and predicted values
pub fn mse(actual: &[f64], predicted: &[f64]) -> f64 {
    let n = actual.len().min(predicted.len());
    if n == 0 {
        return 0.0;
    }
    
    let sum_squared_errors = actual.iter()
        .zip(predicted.iter())
        .map(|(&a, &p)| (a - p).powi(2))
        .sum::<f64>();
    
    sum_squared_errors / n as f64
}

/// Calculate the root mean squared error (RMSE) between actual and predicted values
pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    mse(actual, predicted).sqrt()
}

/// Calculate the mean absolute percentage error (MAPE) between actual and predicted values
pub fn mape(actual: &[f64], predicted: &[f64]) -> f64 {
    let n = actual.len().min(predicted.len());
    if n == 0 {
        return 0.0;
    }
    
    let mut sum_percentage_errors = 0.0;
    let mut count = 0;
    
    for (a, p) in actual.iter().zip(predicted.iter()) {
        if *a != 0.0 {
            sum_percentage_errors += ((*a - *p) / a.abs()).abs() * 100.0;
            count += 1;
        }
    }
    
    if count > 0 {
        sum_percentage_errors / count as f64
    } else {
        0.0
    }
}

/// Calculate Mean Absolute Error (MAE)
pub fn mae_old(actual: &[f64], forecast: &[f64]) -> f64 {
    if actual.len() != forecast.len() || actual.is_empty() {
        return f64::NAN;
    }
    
    let sum: f64 = actual.iter()
        .zip(forecast.iter())
        .map(|(&a, &f)| (a - f).abs())
        .sum();
    
    sum / actual.len() as f64
}

/// Calculate Mean Squared Error (MSE)
pub fn mse_old(actual: &[f64], forecast: &[f64]) -> f64 {
    if actual.len() != forecast.len() || actual.is_empty() {
        return f64::NAN;
    }
    
    let sum: f64 = actual.iter()
        .zip(forecast.iter())
        .map(|(&a, &f)| (a - f).powi(2))
        .sum();
    
    sum / actual.len() as f64
}

/// Calculate Root Mean Squared Error (RMSE)
pub fn rmse_old(actual: &[f64], forecast: &[f64]) -> f64 {
    mse_old(actual, forecast).sqrt()
}

/// Calculate Mean Absolute Percentage Error (MAPE)
pub fn mape_old(actual: &[f64], forecast: &[f64]) -> f64 {
    if actual.len() != forecast.len() || actual.is_empty() {
        return f64::NAN;
    }
    
    let sum: f64 = actual.iter()
        .zip(forecast.iter())
        .filter(|(&a, _)| a != 0.0)  // Avoid division by zero
        .map(|(&a, &f)| ((a - f).abs() / a.abs()) * 100.0)
        .sum();
    
    let non_zero_count = actual.iter().filter(|&&a| a != 0.0).count();
    
    if non_zero_count == 0 {
        f64::NAN
    } else {
        sum / non_zero_count as f64
    }
}

/// Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
pub fn smape(actual: &[f64], forecast: &[f64]) -> f64 {
    if actual.len() != forecast.len() || actual.is_empty() {
        return f64::NAN;
    }
    
    let sum: f64 = actual.iter()
        .zip(forecast.iter())
        .filter(|(&a, &f)| (a.abs() + f.abs()) > 0.0)  // Avoid division by zero
        .map(|(&a, &f)| (2.0 * (a - f).abs() / (a.abs() + f.abs())) * 100.0)
        .sum();
    
    let valid_count = actual.iter()
        .zip(forecast.iter())
        .filter(|(&a, &f)| (a.abs() + f.abs()) > 0.0)
        .count();
    
    if valid_count == 0 {
        f64::NAN
    } else {
        sum / valid_count as f64
    }
}

/// Calculate moving average of a time series
pub fn moving_average(data: &[f64], window_size: usize) -> Vec<f64> {
    if window_size == 0 || window_size > data.len() {
        return Vec::new();
    }
    
    let n = data.len();
    let mut result = Vec::with_capacity(n - window_size + 1);
    
    let mut sum = data.iter().take(window_size).sum::<f64>();
    result.push(sum / window_size as f64);
    
    for i in window_size..n {
        sum = sum - data[i - window_size] + data[i];
        result.push(sum / window_size as f64);
    }
    
    result
}

/// Calculate exponentially weighted moving average
pub fn exponential_moving_average(data: &[f64], alpha: f64) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    
    let n = data.len();
    let mut result = Vec::with_capacity(n);
    
    // Use first value as initial value
    result.push(data[0]);
    
    for i in 1..n {
        let ema = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        result.push(ema);
    }
    
    result
}

/// Normalize data to have mean 0 and standard deviation 1
pub fn standardize(data: &[f64]) -> (Vec<f64>, f64, f64) {
    if data.is_empty() {
        return (Vec::new(), 0.0, 1.0);
    }
    
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    
    let variance = data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n;
    
    let std_dev = variance.sqrt();
    
    let normalized = if std_dev > 0.0 {
        data.iter().map(|&x| (x - mean) / std_dev).collect()
    } else {
        vec![0.0; data.len()]
    };
    
    (normalized, mean, std_dev)
}

/// De-standardize data (reverse the standardization)
pub fn destandardize(data: &[f64], mean: f64, std_dev: f64) -> Vec<f64> {
    data.iter().map(|&x| x * std_dev + mean).collect()
}

/// Check if a time series is stationary (constant mean and variance over time)
/// Returns a rough measure of stationarity (lower is more stationary)
pub fn stationarity_measure(data: &[f64]) -> f64 {
    if data.len() < 10 {
        return f64::NAN;
    }
    
    // Split data into segments and compare their means and variances
    let segments = 4;
    let segment_size = data.len() / segments;
    
    let mut means = Vec::with_capacity(segments);
    let mut variances = Vec::with_capacity(segments);
    
    for i in 0..segments {
        let start = i * segment_size;
        let end = if i == segments - 1 { data.len() } else { (i + 1) * segment_size };
        
        let segment = &data[start..end];
        let mean = segment.iter().sum::<f64>() / segment.len() as f64;
        
        let variance = segment.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / segment.len() as f64;
        
        means.push(mean);
        variances.push(variance);
    }
    
    // Calculate coefficient of variation for means and standard deviations
    let mean_of_means = means.iter().sum::<f64>() / segments as f64;
    let std_of_means = (means.iter()
        .map(|&m| (m - mean_of_means).powi(2))
        .sum::<f64>() / segments as f64).sqrt();
    
    let mean_of_vars = variances.iter().sum::<f64>() / segments as f64;
    let std_of_vars = (variances.iter()
        .map(|&v| (v - mean_of_vars).powi(2))
        .sum::<f64>() / segments as f64).sqrt();
    
    // Combine measures (higher means less stationary)
    let cv_means = if mean_of_means != 0.0 { std_of_means / mean_of_means.abs() } else { 0.0 };
    let cv_vars = if mean_of_vars != 0.0 { std_of_vars / mean_of_vars } else { 0.0 };
    
    cv_means + cv_vars
}

/// First-order differencing of a time series
pub fn difference(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return Vec::new();
    }
    
    data.windows(2)
        .map(|w| w[1] - w[0])
        .collect()
}

/// Reverse differencing (integrate)
pub fn undifference(diffs: &[f64], first_value: f64) -> Vec<f64> {
    if diffs.is_empty() {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(diffs.len() + 1);
    result.push(first_value);
    
    for &diff in diffs {
        let next = result.last().unwrap() + diff;
        result.push(next);
    }
    
    result
}

/// Calculate autocorrelation function (ACF) for lags 1 to max_lag
pub fn autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
    if data.is_empty() || max_lag >= data.len() {
        return Vec::new();
    }
    
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    
    // Calculate variance
    let variance = data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n;
    
    if variance == 0.0 {
        return vec![0.0; max_lag];
    }
    
    let mut acf = Vec::with_capacity(max_lag);
    
    for lag in 1..=max_lag {
        let correlation = data.iter().take(data.len() - lag)
            .zip(data.iter().skip(lag))
            .map(|(&x1, &x2)| (x1 - mean) * (x2 - mean))
            .sum::<f64>() / ((n - lag as f64) * variance);
            
        acf.push(correlation);
    }
    
    acf
} 