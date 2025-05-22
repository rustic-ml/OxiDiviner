
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
