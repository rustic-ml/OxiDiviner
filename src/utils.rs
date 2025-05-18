#[cfg(feature = "ndarray_support")]
use ndarray::Array1;
use statrs::statistics::Data;
use statrs::statistics::Distribution;

/// Calculate the mean of an array of values
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    Data::new(values.to_vec()).mean().unwrap_or(0.0)
}

/// Calculate the variance of an array of values
pub fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    Data::new(values.to_vec()).variance().unwrap_or(0.0)
}

/// Calculate the standard deviation of an array of values
pub fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    Data::new(values.to_vec()).std_dev().unwrap_or(0.0)
}

/// Calculate the mean absolute error (MAE) between actual and predicted values
pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return 0.0;
    }
    
    let sum: f64 = actual.iter().zip(predicted.iter())
        .map(|(&a, &p)| (a - p).abs())
        .sum();
    
    sum / actual.len() as f64
}

/// Calculate the mean squared error (MSE) between actual and predicted values
pub fn mse(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return 0.0;
    }
    
    let sum: f64 = actual.iter().zip(predicted.iter())
        .map(|(&a, &p)| (a - p).powi(2))
        .sum();
    
    sum / actual.len() as f64
}

/// Calculate the root mean squared error (RMSE) between actual and predicted values
pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    mse(actual, predicted).sqrt()
}

/// Calculate the mean absolute percentage error (MAPE) between actual and predicted values
pub fn mape(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return 0.0;
    }
    
    let mut count = 0;
    let sum: f64 = actual.iter().zip(predicted.iter())
        .filter_map(|(&a, &p)| {
            if a != 0.0 {
                count += 1;
                Some(((a - p).abs() / a.abs()) * 100.0)
            } else {
                None
            }
        })
        .sum();
    
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// Calculate Mean Absolute Error (MAE)
pub fn mae_old(actual: &[f64], forecast: &[f64]) -> f64 {
    mae(actual, forecast)
}

/// Calculate Mean Squared Error (MSE)
pub fn mse_old(actual: &[f64], forecast: &[f64]) -> f64 {
    mse(actual, forecast)
}

/// Calculate Root Mean Squared Error (RMSE)
pub fn rmse_old(actual: &[f64], forecast: &[f64]) -> f64 {
    rmse(actual, forecast)
}

/// Calculate Mean Absolute Percentage Error (MAPE)
pub fn mape_old(actual: &[f64], forecast: &[f64]) -> f64 {
    mape(actual, forecast)
}

/// Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
pub fn smape(actual: &[f64], forecast: &[f64]) -> f64 {
    if actual.len() != forecast.len() || actual.is_empty() {
        return 0.0;
    }
    
    let mut count = 0;
    let sum: f64 = actual.iter().zip(forecast.iter())
        .filter_map(|(&a, &f)| {
            let abs_a = a.abs();
            let abs_f = f.abs();
            if abs_a + abs_f > 0.0 {
                count += 1;
                Some(200.0 * (abs_a - abs_f).abs() / (abs_a + abs_f))
            } else {
                None
            }
        })
        .sum();
    
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// Calculate moving average of a time series
pub fn moving_average(data: &[f64], window_size: usize) -> Vec<f64> {
    if data.is_empty() || window_size == 0 || window_size > data.len() {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(data.len());
    // Fill with NaN values for the first window_size-1 elements
    for _ in 0..window_size-1 {
        result.push(f64::NAN);
    }
    
    // Calculate moving average
    for i in window_size-1..data.len() {
        let window_sum: f64 = data[i-(window_size-1)..=i].iter().sum();
        result.push(window_sum / window_size as f64);
    }
    
    result
}

/// Calculate exponentially weighted moving average
pub fn exponential_moving_average(data: &[f64], alpha: f64) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(data.len());
    result.push(data[0]); // First value is just the first data point
    
    // Calculate EMA
    for i in 1..data.len() {
        let ema = alpha * data[i] + (1.0 - alpha) * result[i-1];
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
    
    let standardized = data.iter()
        .map(|&x| (x - mean_val) / std_dev_val)
        .collect();
    
    (standardized, mean_val, std_dev_val)
}

/// Revert standardization
pub fn destandardize(data: &[f64], mean: f64, std_dev: f64) -> Vec<f64> {
    data.iter().map(|&x| x * std_dev + mean).collect()
}

/// Measure stationarity of a time series (returns coefficient of variation of differences)
pub fn stationarity_measure(data: &[f64]) -> f64 {
    if data.len() <= 2 {
        return 0.0;
    }
    
    // Calculate first differences
    let diffs: Vec<f64> = data.windows(2)
        .map(|w| w[1] - w[0])
        .collect();
    
    // Calculate coefficient of variation (std_dev / mean)
    let mean_diff = mean(&diffs);
    
    // Avoid division by zero
    if mean_diff.abs() < 1e-10 {
        return f64::INFINITY;
    }
    
    let std_diff = std_dev(&diffs);
    std_diff / mean_diff.abs()
}

/// Calculate the difference of a time series (for detrending)
pub fn difference(data: &[f64]) -> Vec<f64> {
    if data.len() <= 1 {
        return Vec::new();
    }
    
    data.windows(2)
        .map(|w| w[1] - w[0])
        .collect()
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

/// Calculate autocorrelation for lags 0 to max_lag
pub fn autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
    if data.is_empty() || max_lag >= data.len() {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(max_lag + 1);
    let mean_val = mean(data);
    
    // Calculate variance (denominator)
    let variance_val = data.iter()
        .map(|&x| (x - mean_val).powi(2))
        .sum::<f64>() / data.len() as f64;
    
    if variance_val == 0.0 {
        return vec![1.0]; // Only lag 0 correlation is defined for constant series
    }
    
    // Lag 0 autocorrelation is always 1
    result.push(1.0);
    
    // Calculate autocorrelation for lags 1 to max_lag
    for lag in 1..=max_lag {
        let mut sum = 0.0;
        for i in 0..data.len() - lag {
            sum += (data[i] - mean_val) * (data[i + lag] - mean_val);
        }
        let autocorr = sum / ((data.len() - lag) as f64 * variance_val);
        result.push(autocorr);
    }
    
    result
} 