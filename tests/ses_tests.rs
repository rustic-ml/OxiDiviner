use chrono::{Duration, Utc};
use oxidiviner::OHLCVData;
use oxidiviner_exponential_smoothing::simple::SimpleESModel;

fn create_test_data() -> OHLCVData {
    let now = Utc::now();
    let timestamps = vec![
        now,
        now + Duration::days(1),
        now + Duration::days(2),
        now + Duration::days(3),
        now + Duration::days(4),
    ];
    
    let open = vec![100.0, 101.0, 102.0, 103.0, 104.0];
    let high = vec![105.0, 106.0, 107.0, 108.0, 109.0];
    let low = vec![98.0, 99.0, 100.0, 101.0, 102.0];
    let close = vec![103.0, 104.0, 105.0, 106.0, 107.0];
    let volume = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0];
    
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
fn test_ses_model_creation() {
    // Test valid parameter range
    assert!(SimpleESModel::new(0.3).is_ok());
    
    // Test invalid alpha values
    assert!(SimpleESModel::new(0.0).is_err());
    assert!(SimpleESModel::new(1.0).is_err());
    assert!(SimpleESModel::new(-0.1).is_err());
    assert!(SimpleESModel::new(1.1).is_err());
    
    // Test with model name
    let model = SimpleESModel::new(0.3).unwrap();
    assert!(model.name().contains("SES"));
    assert!(model.name().contains("0.3"));
}

#[test]
fn test_ses_model_fit_and_forecast() {
    let data = create_test_data();
    
    // Create a simple exponential smoothing model
    let mut model = SimpleESModel::new(0.3).unwrap();
    
    // Fit the model
    assert!(model.fit(&data).is_ok());
    
    // Check if fitted values exist
    let fitted_values = model.fitted_values().unwrap();
    assert_eq!(fitted_values.len(), data.len());
    
    // Generate forecasts
    let horizon = 3;
    let forecasts = model.forecast(horizon).unwrap();
    assert_eq!(forecasts.len(), horizon);
    
    // In SES, all forecasts should be equal to the final level
    for i in 1..horizon {
        assert!((forecasts[0] - forecasts[i]).abs() < 1e-6);
    }
}

#[test]
fn test_ses_model_evaluation() {
    let data = create_test_data();
    
    // Split data into train and test sets
    let (train_data, test_data) = data.train_test_split(0.6).unwrap();
    
    // Create and fit model
    let mut model = SimpleESModel::new(0.3).unwrap();
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
fn test_ses_empty_data_handling() {
    let empty_data = OHLCVData::new(
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        Some("EMPTY".to_string())
    ).unwrap();
    
    let mut model = SimpleESModel::new(0.3).unwrap();
    
    // Fitting on empty data should fail
    assert!(model.fit(&empty_data).is_err());
    
    // Model has not been fit, so forecast should fail
    assert!(model.forecast(5).is_err());
    
    // Model has not been fit, so evaluate should fail
    assert!(model.evaluate(&empty_data).is_err());
} 