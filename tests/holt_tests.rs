use chrono::{Duration, Utc};
use oxidiviner::{OHLCVData, HoltLinearModel, TimeSeriesData};

fn create_test_data() -> OHLCVData {
    let now = Utc::now();
    let timestamps = vec![
        now,
        now + Duration::days(1),
        now + Duration::days(2),
        now + Duration::days(3),
        now + Duration::days(4),
    ];
    
    // Data with a clear upward trend
    let open = vec![100.0, 102.0, 104.0, 106.0, 108.0];
    let high = vec![105.0, 107.0, 109.0, 111.0, 113.0];
    let low = vec![98.0, 100.0, 102.0, 104.0, 106.0];
    let close = vec![103.0, 105.0, 107.0, 109.0, 111.0];
    let volume = vec![1000.0, 1050.0, 1100.0, 1150.0, 1200.0];
    
    OHLCVData::new(
        timestamps,
        open,
        high,
        low,
        close,
        volume,
        Some("TEST".to_string())
    ).unwrap()
}

#[test]
fn test_holt_model_creation() {
    // Test valid parameter ranges
    assert!(HoltLinearModel::new(0.3, 0.1).is_ok());
    
    // Test invalid alpha values
    assert!(HoltLinearModel::new(0.0, 0.1).is_err());
    assert!(HoltLinearModel::new(1.0, 0.1).is_err());
    
    // Test invalid beta values
    assert!(HoltLinearModel::new(0.3, 0.0).is_err());
    assert!(HoltLinearModel::new(0.3, 1.0).is_err());
}

#[test]
fn test_holt_model_fit_and_forecast() {
    let ohlcv_data = create_test_data();
    
    // Convert to TimeSeriesData using close prices
    let data = TimeSeriesData::new(
        ohlcv_data.timestamps.clone(),
        ohlcv_data.close.clone(),
        "test_trend"
    ).unwrap();
    
    // Create a Holt model with linear trend
    let mut model = HoltLinearModel::new(0.3, 0.1).unwrap();
    
    // Fit the model
    assert!(model.fit(&data).is_ok());
    
    // Check if fitted values exist
    let fitted_values = model.fitted_values().unwrap();
    assert_eq!(fitted_values.len(), data.len());
    
    // Generate forecasts
    let horizon = 3;
    let forecasts = model.forecast(horizon).unwrap();
    assert_eq!(forecasts.len(), horizon);
    
    // Forecasts should follow the trend
    for i in 1..horizon {
        assert!(forecasts[i] > forecasts[i-1]);
    }
}

#[test]
fn test_holt_model_evaluation() {
    let ohlcv_data = create_test_data();
    
    // Convert to TimeSeriesData using close prices
    let data = TimeSeriesData::new(
        ohlcv_data.timestamps.clone(),
        ohlcv_data.close.clone(),
        "test_trend"
    ).unwrap();
    
    // Split data into train and test sets
    let (train_data, test_data) = data.train_test_split(0.6).unwrap();
    
    // Create and fit model
    let mut model = HoltLinearModel::new(0.3, 0.1).unwrap();
    model.fit(&train_data).unwrap();
    
    // Evaluate on test data
    let eval = model.evaluate(&test_data).unwrap();
    
    // Basic checks on evaluation metrics
    assert!(eval.mae > 0.0);
    assert!(eval.rmse > 0.0);
    assert!(eval.mape > 0.0);
    
    // Model name should be in the evaluation
    assert_eq!(eval.model_name, model.name());
}

#[test]
fn test_holt_insufficient_data() {
    // Create model
    let mut model = HoltLinearModel::new(0.3, 0.1).unwrap();
    
    // Test with single data point (not enough for trend initialization)
    let single_data = TimeSeriesData::new(
        vec![Utc::now()],
        vec![100.0],
        "single"
    ).unwrap();
    
    // Should fail because at least two points are needed for trend
    assert!(model.fit(&single_data).is_err());
} 