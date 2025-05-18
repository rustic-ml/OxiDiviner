use chrono::{Duration, Utc};
use oxidiviner::ModelsOHLCVData;
use oxidiviner::models::exponential_smoothing::ets::{ETSComponent, ModelEvaluation};

// Import the new DailyETSModel and MinuteETSModel from the reorganized structure
// The imports might need to be adjusted based on the actual reorganized structure
use oxidiviner::models::exponential_smoothing::ets::{DailyETSModel, MinuteETSModel};

fn create_test_data_daily() -> ModelsOHLCVData {
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
        "TEST_DAILY"
    ).unwrap()
}

fn create_test_data_minute() -> ModelsOHLCVData {
    let now = Utc::now();
    let mut timestamps = Vec::with_capacity(480); // 8 hours of minute data
    let mut open = Vec::with_capacity(480);
    let mut high = Vec::with_capacity(480);
    let mut low = Vec::with_capacity(480);
    let mut close = Vec::with_capacity(480);
    let mut volume = Vec::with_capacity(480);
    
    // Create 8 hours of synthetic minute data with trend and seasonality
    for i in 0..480 {
        let minute = now + Duration::minutes(i);
        timestamps.push(minute);
        
        // Add trend (0.001 per minute) and seasonality (60-minute cycle)
        let trend = 0.001 * i as f64;
        let season = 0.5 * (2.0 * std::f64::consts::PI * (i % 60) as f64 / 60.0).sin();
        let noise = rand::random::<f64>() * 0.2 - 0.1; // Random noise between -0.1 and 0.1
        
        let base_price = 100.0 + trend + season + noise;
        let minute_range = base_price * 0.001; // 0.1% range
        
        open.push(base_price - minute_range / 2.0);
        close.push(base_price + minute_range / 2.0);
        high.push(base_price + minute_range);
        low.push(base_price - minute_range);
        volume.push(10.0 + rand::random::<f64>() * 20.0);
    }
    
    ModelsOHLCVData::new(
        timestamps,
        open,
        high,
        low,
        close,
        volume,
        "TEST_MINUTE"
    ).unwrap()
}

#[test]
fn test_daily_ets_model_creation() {
    // Test simple exponential smoothing (ETS(A,N,N))
    assert!(DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::None,      // No trend
        ETSComponent::None,      // No seasonality
        0.3,                     // alpha = 0.3
        None,                    // No beta (no trend)
        None,                    // No gamma (no seasonality)
        None,                    // No phi (no damping)
        None,                    // No seasonal period
        None,                    // Default to Close price
    ).is_ok());
    
    // Test Holt's linear trend (ETS(A,A,N))
    assert!(DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::Additive,  // Additive trend
        ETSComponent::None,      // No seasonality
        0.3,                     // alpha = 0.3
        Some(0.1),               // beta = 0.1
        None,                    // No gamma (no seasonality)
        None,                    // No phi (no damping)
        None,                    // No seasonal period
        None,                    // Default to Close price
    ).is_ok());
    
    // Test Holt-Winters (ETS(A,A,A))
    assert!(DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::Additive,  // Additive trend
        ETSComponent::Additive,  // Additive seasonality
        0.3,                     // alpha = 0.3
        Some(0.1),               // beta = 0.1
        Some(0.1),               // gamma = 0.1
        None,                    // No phi (no damping)
        Some(7),                 // Weekly seasonality
        None,                    // Default to Close price
    ).is_ok());
    
    // Test damped trend (ETS(A,D,N))
    assert!(DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::Damped,    // Damped trend
        ETSComponent::None,      // No seasonality
        0.3,                     // alpha = 0.3
        Some(0.1),               // beta = 0.1
        None,                    // No gamma (no seasonality)
        Some(0.9),               // phi = 0.9 (damping factor)
        None,                    // No seasonal period
        None,                    // Default to Close price
    ).is_ok());
    
    // Test invalid parameters
    
    // Missing beta for trend model
    assert!(DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::Additive,  // Additive trend
        ETSComponent::None,      // No seasonality
        0.3,                     // alpha = 0.3
        None,                    // Missing beta!
        None,                    // No gamma (no seasonality)
        None,                    // No phi (no damping)
        None,                    // No seasonal period
        None,                    // Default to Close price
    ).is_err());
    
    // Missing gamma for seasonal model
    assert!(DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::None,      // No trend
        ETSComponent::Additive,  // Additive seasonality
        0.3,                     // alpha = 0.3
        None,                    // No beta (no trend)
        None,                    // Missing gamma!
        None,                    // No phi (no damping)
        Some(7),                 // Weekly seasonality
        None,                    // Default to Close price
    ).is_err());
    
    // Missing phi for damped trend
    assert!(DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::Damped,    // Damped trend
        ETSComponent::None,      // No seasonality
        0.3,                     // alpha = 0.3
        Some(0.1),               // beta = 0.1
        None,                    // No gamma (no seasonality)
        None,                    // Missing phi!
        None,                    // No seasonal period
        None,                    // Default to Close price
    ).is_err());
    
    // Missing seasonal period for seasonal model
    assert!(DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::None,      // No trend
        ETSComponent::Additive,  // Additive seasonality
        0.3,                     // alpha = 0.3
        None,                    // No beta (no trend)
        Some(0.1),               // gamma = 0.1
        None,                    // No phi (no damping)
        None,                    // Missing seasonal period!
        None,                    // Default to Close price
    ).is_err());
}

#[test]
fn test_daily_ets_model_fit_and_forecast() {
    let data = create_test_data_daily();
    
    // Create a simple ETS model
    let mut model = DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::Additive,  // Additive trend
        ETSComponent::Additive,  // Additive seasonality
        0.3,                     // alpha = 0.3
        Some(0.1),               // beta = 0.1
        Some(0.1),               // gamma = 0.1
        None,                    // No phi (no damping)
        Some(7),                 // Weekly seasonality
        None,                    // Default to Close price
    ).unwrap();
    
    // Fit the model
    assert!(model.fit(&data).is_ok());
    
    // Check if fitted values exist
    let fitted_values = model.fitted_values().unwrap();
    assert_eq!(fitted_values.len(), data.len());
    
    // Generate forecasts
    let horizon = 14;
    let forecasts = model.forecast(horizon).unwrap();
    assert_eq!(forecasts.len(), horizon);
    
    // Weekly pattern should repeat in forecasts
    for i in 7..horizon {
        let day_of_week = i % 7;
        let same_day_previous_week = i - 7;
        
        // Difference should be approximately weekly trend
        let diff = forecasts[i] - forecasts[same_day_previous_week];
        assert!(diff > 0.0); // Should have upward trend
    }
}

#[test]
fn test_daily_ets_model_evaluation() {
    let data = create_test_data_daily();
    
    // Split data into train and test sets
    let (train_data, test_data) = data.train_test_split(0.8).unwrap();
    
    // Create and fit several models
    
    // 1. Simple Exponential Smoothing
    let mut ses_model = DailyETSModel::new(
        ETSComponent::Additive, ETSComponent::None, ETSComponent::None,
        0.3, None, None, None, None, None
    ).unwrap();
    
    // 2. Holt's Linear Trend
    let mut holt_model = DailyETSModel::new(
        ETSComponent::Additive, ETSComponent::Additive, ETSComponent::None,
        0.3, Some(0.1), None, None, None, None
    ).unwrap();
    
    // 3. Holt-Winters
    let mut hw_model = DailyETSModel::new(
        ETSComponent::Additive, ETSComponent::Additive, ETSComponent::Additive,
        0.3, Some(0.1), Some(0.1), None, Some(7), None
    ).unwrap();
    
    // Fit all models
    ses_model.fit(&train_data).unwrap();
    holt_model.fit(&train_data).unwrap();
    hw_model.fit(&train_data).unwrap();
    
    // Evaluate all models
    let ses_eval = ses_model.evaluate(&test_data).unwrap();
    let holt_eval = holt_model.evaluate(&test_data).unwrap();
    let hw_eval = hw_model.evaluate(&test_data).unwrap();
    
    // Basic checks on evaluation metrics
    assert!(ses_eval.mae > 0.0);
    assert!(ses_eval.rmse > 0.0);
    assert!(ses_eval.mape > 0.0);
    
    assert!(holt_eval.mae > 0.0);
    assert!(holt_eval.rmse > 0.0);
    assert!(holt_eval.mape > 0.0);
    
    assert!(hw_eval.mae > 0.0);
    assert!(hw_eval.rmse > 0.0);
    assert!(hw_eval.mape > 0.0);
    
    // For data with trend and seasonality, Holt-Winters should perform best
    assert!(hw_eval.mae <= ses_eval.mae);
    assert!(hw_eval.rmse <= ses_eval.rmse);
}

#[test]
fn test_minute_ets_model() {
    let data = create_test_data_minute();
    
    // Test MinuteETSModel with default aggregation
    let mut model1 = MinuteETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::None,      // No trend
        ETSComponent::None,      // No seasonality
        0.3,                     // alpha = 0.3
        None,                    // No beta
        None,                    // No gamma
        None,                    // No phi
        None,                    // No seasonal period
        None,                    // Default target column (Close)
        None,                    // No aggregation
    ).unwrap();
    
    // Test MinuteETSModel with 5-minute aggregation
    let mut model2 = MinuteETSModel::new(
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
    ).unwrap();
    
    // Fit both models
    assert!(model1.fit(&data).is_ok());
    assert!(model2.fit(&data).is_ok());
    
    // Second model should be using 5-minute aggregation
    assert_eq!(model2.aggregation_minutes(), 5);
    
    // Generate forecasts with both models
    let horizon = 60; // 1 hour forecast
    let forecasts1 = model1.forecast(horizon).unwrap();
    let forecasts2 = model2.forecast(horizon).unwrap();
    
    assert_eq!(forecasts1.len(), horizon);
    assert_eq!(forecasts2.len(), horizon);
    
    // Forecasts should be different due to different aggregation levels
    let mut differences = 0;
    for i in 0..horizon {
        if (forecasts1[i] - forecasts2[i]).abs() > 1e-6 {
            differences += 1;
        }
    }
    assert!(differences > 0);
}

#[test]
fn test_minute_ets_model_with_seasonality() {
    let data = create_test_data_minute();
    
    // Create a minute ETS model with hourly seasonality
    let mut model = MinuteETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::None,      // No trend
        ETSComponent::Additive,  // Additive seasonality
        0.3,                     // alpha = 0.3
        None,                    // No beta
        Some(0.1),               // gamma = 0.1
        None,                    // No phi
        Some(60),                // 60-minute seasonal period
        None,                    // Default target column (Close)
        Some(5),                 // 5-minute aggregation
    ).unwrap();
    
    // Fit the model
    assert!(model.fit(&data).is_ok());
    
    // Generate forecasts
    let horizon = 120; // 2 hours forecast
    let forecasts = model.forecast(horizon).unwrap();
    assert_eq!(forecasts.len(), horizon);
    
    // Check for hourly pattern in forecasts
    for i in 60..horizon {
        let minute_in_hour = i % 60;
        let same_minute_previous_hour = i - 60;
        
        // Difference should be small (within a reasonable range)
        let diff = (forecasts[i] - forecasts[same_minute_previous_hour]).abs();
        assert!(diff < 1.0); // Approximate pattern should repeat
    }
} 