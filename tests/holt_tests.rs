use chrono::{Duration, Utc};
use oxidiviner::ModelsOHLCVData;
use oxidiviner::models::exponential_smoothing::holt::{HoltModel, TargetColumn};

fn create_test_data() -> ModelsOHLCVData {
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
    
    ModelsOHLCVData::new(
        timestamps,
        open,
        high,
        low,
        close,
        volume,
        "TEST"
    ).unwrap()
}

#[test]
fn test_holt_model_creation() {
    // Test valid parameter ranges
    assert!(HoltModel::new(0.3, 0.1, None, None).is_ok());
    assert!(HoltModel::new(0.3, 0.1, Some(0.9), None).is_ok());
    
    // Test invalid alpha values
    assert!(HoltModel::new(0.0, 0.1, None, None).is_err());
    assert!(HoltModel::new(1.0, 0.1, None, None).is_err());
    
    // Test invalid beta values
    assert!(HoltModel::new(0.3, 0.0, None, None).is_err());
    assert!(HoltModel::new(0.3, 1.0, None, None).is_err());
    
    // Test invalid phi values
    assert!(HoltModel::new(0.3, 0.1, Some(0.0), None).is_err());
    assert!(HoltModel::new(0.3, 0.1, Some(1.0), None).is_err());
    
    // Test with different target columns
    let model = HoltModel::new(0.3, 0.1, None, Some(TargetColumn::Open)).unwrap();
    assert!(model.name().contains("Open"));
    
    let model = HoltModel::new(0.3, 0.1, None, Some(TargetColumn::Close)).unwrap();
    assert!(model.name().contains("Close"));
    
    // Test damped trend model name
    let model = HoltModel::new(0.3, 0.1, Some(0.9), None).unwrap();
    assert!(model.name().contains("Damped"));
}

#[test]
fn test_holt_model_fit_and_forecast() {
    let data = create_test_data();
    
    // Create a Holt model with linear trend
    let mut model = HoltModel::new(0.3, 0.1, None, None).unwrap();
    
    // Fit the model
    assert!(model.fit(&data).is_ok());
    
    // Check if fitted values exist
    let fitted_values = model.fitted_values().unwrap();
    assert_eq!(fitted_values.len(), data.len());
    
    // Get level and trend
    let level = model.level().unwrap();
    let trend = model.trend().unwrap();
    
    // For our test data with linear trend, trend should be positive
    assert!(trend > 0.0);
    
    // Generate forecasts
    let horizon = 3;
    let forecasts = model.forecast(horizon).unwrap();
    assert_eq!(forecasts.len(), horizon);
    
    // Forecasts should follow the trend
    for i in 1..horizon {
        assert!(forecasts[i] > forecasts[i-1]);
    }
    
    // First forecast should be approximately level + trend
    assert!((forecasts[0] - (level + trend)).abs() < 1e-6);
}

#[test]
fn test_holt_damped_trend() {
    let data = create_test_data();
    
    // Create a Holt model with damped trend
    let mut model = HoltModel::new(0.3, 0.1, Some(0.9), None).unwrap();
    
    // Fit the model
    model.fit(&data).unwrap();
    
    // Generate forecasts with both models
    let horizon = 10;
    let damped_forecasts = model.forecast(horizon).unwrap();
    
    // Create regular Holt model for comparison
    let mut regular_model = HoltModel::new(0.3, 0.1, None, None).unwrap();
    regular_model.fit(&data).unwrap();
    let regular_forecasts = regular_model.forecast(horizon).unwrap();
    
    // For distant forecasts, damped trend should produce more conservative values
    // than regular trend when we have an upward trend
    assert!(damped_forecasts[horizon-1] < regular_forecasts[horizon-1]);
}

#[test]
fn test_holt_model_evaluation() {
    let data = create_test_data();
    
    // Split data into train and test sets
    let (train_data, test_data) = data.train_test_split(0.6).unwrap();
    
    // Create and fit model
    let mut model = HoltModel::new(0.3, 0.1, None, None).unwrap();
    model.fit(&train_data).unwrap();
    
    // Evaluate on test data
    let eval = model.evaluate(&test_data).unwrap();
    
    // Basic checks on evaluation metrics
    assert!(eval.mae > 0.0);
    assert!(eval.rmse > 0.0);
    assert!(eval.mape > 0.0);
    assert!(eval.smape > 0.0);
    
    // RMSE should be greater than or equal to MAE
    assert!(eval.rmse >= eval.mae);
    
    // Model name should be in the evaluation
    assert_eq!(eval.model_name, model.name());
}

#[test]
fn test_holt_insufficient_data() {
    // Create model
    let mut model = HoltModel::new(0.3, 0.1, None, None).unwrap();
    
    // Test with single data point (not enough for trend initialization)
    let single_data = ModelsOHLCVData::new(
        vec![Utc::now()],
        vec![100.0],
        vec![105.0],
        vec![98.0],
        vec![103.0],
        vec![1000.0],
        "SINGLE"
    ).unwrap();
    
    // Should fail because at least two points are needed for trend
    assert!(model.fit(&single_data).is_err());
} 