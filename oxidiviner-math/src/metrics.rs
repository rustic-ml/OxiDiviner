/// Calculate the mean absolute error (MAE) between actual and predicted values
pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return 0.0;
    }

    let sum: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(&a, &p)| (a - p).abs())
        .sum();

    sum / actual.len() as f64
}

/// Calculate the mean squared error (MSE) between actual and predicted values
pub fn mse(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return 0.0;
    }

    let sum: f64 = actual
        .iter()
        .zip(predicted.iter())
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
    let sum: f64 = actual
        .iter()
        .zip(predicted.iter())
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

/// Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
pub fn smape(actual: &[f64], forecast: &[f64]) -> f64 {
    if actual.len() != forecast.len() || actual.is_empty() {
        return 0.0;
    }

    let mut count = 0;
    let sum: f64 = actual
        .iter()
        .zip(forecast.iter())
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mae() {
        // Test normal case
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![1.2, 2.3, 2.7, 4.1, 4.8];
        
        // Absolute errors: [0.2, 0.3, 0.3, 0.1, 0.2] -> sum = 1.1 -> MAE = 1.1/5 = 0.22
        let error = mae(&actual, &predicted);
        assert!((error - 0.22).abs() < 1e-6);
        
        // Test with empty vectors
        let empty_error = mae(&[], &[]);
        assert_eq!(empty_error, 0.0);
        
        // Test with mismatched lengths
        let mismatched_error = mae(&[1.0, 2.0], &[1.0]);
        assert_eq!(mismatched_error, 0.0);
    }
    
    #[test]
    fn test_mse() {
        // Test normal case
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        
        // Errors: [0, 1, -1, 1, -1] -> squares: [0, 1, 1, 1, 1] -> sum = 4 -> MSE = 4/5 = 0.8
        let error = mse(&actual, &predicted);
        assert!((error - 0.8).abs() < 1e-6);
        
        // Test with empty vectors
        let empty_error = mse(&[], &[]);
        assert_eq!(empty_error, 0.0);
        
        // Test with mismatched lengths
        let mismatched_error = mse(&[1.0, 2.0], &[1.0]);
        assert_eq!(mismatched_error, 0.0);
    }
    
    #[test]
    fn test_rmse() {
        // Test normal case
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        
        // MSE = 0.8, RMSE = sqrt(0.8) = 0.894...
        let error = rmse(&actual, &predicted);
        assert!((error - (0.8f64).sqrt()).abs() < 1e-6);
        
        // Test with empty vectors
        let empty_error = rmse(&[], &[]);
        assert_eq!(empty_error, 0.0);
    }
    
    #[test]
    fn test_mape() {
        // Test normal case
        let actual = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let predicted = vec![9.0, 21.0, 29.0, 41.0, 48.0];
        
        // Percentage errors: [10%, 5%, 3.33%, 2.5%, 4%] -> MAPE = 4.97%
        let error = mape(&actual, &predicted);
        assert!((error - 4.97).abs() < 0.1);
        
        // Test with zero values in actual
        let actual_with_zero = vec![0.0, 20.0, 30.0];
        let predicted_with_zero = vec![1.0, 21.0, 29.0];
        
        // Zero values should be ignored, so only compute for non-zero values
        let error_with_zero = mape(&actual_with_zero, &predicted_with_zero);
        assert!((error_with_zero - 4.17).abs() < 0.1);
        
        // Test with all zeros in actual
        let all_zeros = vec![0.0, 0.0, 0.0];
        let non_zeros = vec![1.0, 2.0, 3.0];
        let error_all_zeros = mape(&all_zeros, &non_zeros);
        assert_eq!(error_all_zeros, 0.0);
        
        // Test with empty vectors
        let empty_error = mape(&[], &[]);
        assert_eq!(empty_error, 0.0);
        
        // Test with mismatched lengths
        let mismatched_error = mape(&[10.0, 20.0], &[9.0]);
        assert_eq!(mismatched_error, 0.0);
    }
    
    #[test]
    fn test_smape() {
        // Test normal case
        let actual = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let predicted = vec![9.0, 21.0, 29.0, 41.0, 48.0];
        
        // Calculate manually:
        // [200*|10-9|/(10+9), 200*|20-21|/(20+21), ...] = [10.53%, 4.88%, 3.39%, 2.47%, 4.08%]
        // Average = 5.07%
        let error = smape(&actual, &predicted);
        assert!((error - 5.07).abs() < 0.1);
        
        // Test with zero values
        let actual_with_zero = vec![0.0, 20.0, 30.0];
        let predicted_with_zero = vec![0.0, 21.0, 29.0];
        
        // First pair is (0,0) which should be ignored
        let error_with_zero = smape(&actual_with_zero, &predicted_with_zero);
        assert!((error_with_zero - 4.13).abs() < 0.1);
        
        // Test with all zero pairs
        let all_zeros = vec![0.0, 0.0, 0.0];
        let error_all_zeros = smape(&all_zeros, &all_zeros);
        assert_eq!(error_all_zeros, 0.0);
        
        // Test with empty vectors
        let empty_error = smape(&[], &[]);
        assert_eq!(empty_error, 0.0);
        
        // Test with mismatched lengths
        let mismatched_error = smape(&[10.0, 20.0], &[9.0]);
        assert_eq!(mismatched_error, 0.0);
    }
}
