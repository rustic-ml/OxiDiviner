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

/// Calculate autocorrelation for lags 0 to max_lag
pub fn autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
    if data.is_empty() || max_lag >= data.len() {
        return Vec::new();
    }
    
    let mean_val = mean(data);
    let variance_val = variance(data);
    
    if variance_val == 0.0 {
        return vec![1.0; max_lag + 1];
    }
    
    let mut result = Vec::with_capacity(max_lag + 1);
    
    for lag in 0..=max_lag {
        let mut numerator = 0.0;
        for i in 0..data.len() - lag {
            numerator += (data[i] - mean_val) * (data[i + lag] - mean_val);
        }
        result.push(numerator / ((data.len() - lag) as f64 * variance_val));
    }
    
    result
} 