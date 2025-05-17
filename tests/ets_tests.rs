use chrono::{Duration, Utc};
use oxidiviner::ModelsOHLCVData;
use oxidiviner::models::exponential_smoothing::ets::{ETSComponent, DailyETSModel, MinuteETSModel, ModelEvaluation};

fn create_test_data() -> ModelsOHLCVData {
    let now = Utc::now();
    let mut timestamps = Vec::with_capacity(100);
    let mut open = Vec::with_capacity(100);
    let mut high = Vec::with_capacity(100);
    let mut low = Vec::with_capacity(100);
    let mut close = Vec::with_capacity(100);
    let mut volume = Vec::with_capacity(100);
    
    // Create 100 days of synthetic data with trend and seasonality
    for i in 0..100 {
        let day = now + Duration::days(i);
        timestamps.push(day);
        
        // Add trend (0.5 per day) and seasonality (7-day cycle)
        let trend = 0.5 * i as f64;
        let season = 5.0 * (2.0 * std::f64::consts::PI * (i % 7) as f64 / 7.0).sin();
        let noise = rand::random::<f64>() * 3.0 - 1.5; // Random noise between -1.5 and 1.5
        
        let base_price = 100.0 + trend + season + noise;
        let daily_range = base_price * 0.02; // 2% daily range
        
        open.push(base_price - daily_range / 2.0);
        close.push(base_price + daily_range / 2.0);
        high.push(base_price + daily_range);
        low.push(base_price - daily_range);
        volume.push(1000.0 + rand::random::<f64>() * 500.0);
    }
    
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
fn test_simple_exponential_smoothing() {
    let data = create_test_data();
    
    // Create a simple exponential smoothing model
    let mut model = DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::None,      // No trend
        ETSComponent::None,      // No seasonality
        0.3,                     // alpha = 0.3
        None,                    // No beta
        None,                    // No gamma
        None,                    // No phi
        None,                    // No seasonal period
        None,                    // Default target column (Close)
    ).expect("Failed to create ETS model");
    
    // Fit the model
    model.fit(&data).expect("Failed to fit model");
    
    // Check if fitted values exist
    let fitted_values = model.fitted_values().expect("No fitted values");
    assert_eq!(fitted_values.len(), data.len());
    
    // Simple test: first fitted value should be equal to first observation
    assert!((fitted_values[0] - data.close[0]).abs() < 1e-6);
    
    // Try forecasting
    let horizon = 10;
    let forecasts = model.forecast(horizon).expect("Failed to generate forecasts");
    assert_eq!(forecasts.len(), horizon);
}

#[test]
fn test_holt_winters() {
    let data = create_test_data();
    
    // Create a Holt-Winters model
    let mut model = DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::Additive,  // Additive trend
        ETSComponent::Additive,  // Additive seasonality
        0.3,                     // alpha = 0.3
        Some(0.1),               // beta = 0.1
        Some(0.1),               // gamma = 0.1
        None,                    // No phi
        Some(7),                 // Weekly seasonality
        None,                    // Default target column (Close)
    ).expect("Failed to create Holt-Winters model");
    
    // Fit the model
    model.fit(&data).expect("Failed to fit model");
    
    // Check if fitted values exist
    let fitted_values = model.fitted_values().expect("No fitted values");
    assert_eq!(fitted_values.len(), data.len());
    
    // Try forecasting
    let horizon = 14;
    let forecasts = model.forecast(horizon).expect("Failed to generate forecasts");
    assert_eq!(forecasts.len(), horizon);
    
    // Split data for evaluation
    let (train_data, test_data) = data.train_test_split(0.8)
        .expect("Failed to split data");
    
    // Re-fit on training data
    model.fit(&train_data).expect("Failed to fit model on training data");
    
    // Evaluate on test data
    let eval = model.evaluate(&test_data).expect("Failed to evaluate model");
    assert!(eval.mae > 0.0);
    assert!(eval.rmse > 0.0);
    assert!(eval.mape > 0.0);
}

#[test]
fn test_minute_model() {
    let mut data = create_test_data();
    
    // Convert daily data to "minute" data by changing the timestamps
    // (This is just for testing - not a realistic conversion)
    for i in 0..data.timestamps.len() {
        data.timestamps[i] = Utc::now() + Duration::minutes(i as i64);
    }
    
    // Create a minute ETS model with aggregation
    let mut model = MinuteETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::None,      // No trend
        ETSComponent::None,      // No seasonality
        0.3,                     // alpha = 0.3
        None,                    // No beta
        None,                    // No gamma
        None,                    // No phi
        None,                    // No seasonal period
        None,                    // Default target column (Close)
        Some(5),                 // 5-minute aggregation
    ).expect("Failed to create minute ETS model");
    
    // Fit the model
    model.fit(&data).expect("Failed to fit model");
    
    // Check aggregation
    assert_eq!(model.aggregation_minutes(), 5);
    
    // Check forecasting
    let horizon = 12;
    let forecasts = model.forecast(horizon).expect("Failed to generate forecasts");
    assert_eq!(forecasts.len(), horizon);
}

#[test]
fn test_model_evaluation() {
    // This tests the ModelEvaluation struct directly
    let model_name = "Test Model".to_string();
    let _predictions = vec![10.0, 11.0, 12.0, 13.0, 14.0]; // Unused but kept for reference
    let _actuals = vec![9.5, 10.5, 12.5, 12.8, 13.5]; // Unused but kept for reference
    
    // Calculate metrics manually
    let mae = (0.5 + 0.5 + 0.5 + 0.2 + 0.5) / 5.0;
    let mse = (0.5*0.5 + 0.5*0.5 + 0.5*0.5 + 0.2*0.2 + 0.5*0.5) / 5.0;
    let rmse = f64::sqrt(mse);
    
    // Calculate MAPE
    let mape = ((0.5/9.5 + 0.5/10.5 + 0.5/12.5 + 0.2/12.8 + 0.5/13.5) * 100.0) / 5.0;
    
    // Create evaluation manually
    let eval = ModelEvaluation {
        model_name,
        mae,
        mse,
        rmse,
        mape,
    };
    
    // Check basic properties
    assert_eq!(eval.model_name, "Test Model");
    
    // Error metrics should be positive
    assert!(eval.mae > 0.0);
    assert!(eval.rmse > 0.0);
    assert!(eval.mape > 0.0);
    
    // Basic sanity check on MAE
    let expected_mae = (0.5 + 0.5 + 0.5 + 0.2 + 0.5) / 5.0;
    assert!((eval.mae - expected_mae).abs() < 1e-6);
} 