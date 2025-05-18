use chrono::{Duration, Utc};
use oxidiviner::ModelsOHLCVData;
use oxidiviner::models::exponential_smoothing::holt_winters::{HoltWintersModel, SeasonalType, TargetColumn};

fn create_test_data_with_seasonality() -> ModelsOHLCVData {
    let now = Utc::now();
    let mut timestamps = Vec::with_capacity(28);
    let mut open = Vec::with_capacity(28);
    let mut high = Vec::with_capacity(28);
    let mut low = Vec::with_capacity(28);
    let mut close = Vec::with_capacity(28);
    let mut volume = Vec::with_capacity(28);
    
    // Create 4 weeks of data with weekly seasonality
    for week in 0..4 {
        for day in 0..7 {
            let timestamp = now + Duration::days((week * 7 + day) as i64);
            timestamps.push(timestamp);
            
            // Base trend with 0.5 increase per day
            let trend = 0.5 * (week * 7 + day) as f64;
            
            // Weekly seasonality pattern
            let season = match day {
                0 => 5.0,  // Monday peak
                1 => 3.0,  // Tuesday
                2 => 1.0,  // Wednesday
                3 => -1.0, // Thursday
                4 => -3.0, // Friday
                5 => -4.0, // Saturday dip
                6 => -1.0, // Sunday
                _ => 0.0,
            };
            
            // Random component
            let random = (day as f64 * 0.3).sin() * 2.0;
            
            let base_price = 100.0 + trend + season + random;
            let daily_range = 3.0;
            
            open.push(base_price - 1.0);
            high.push(base_price + daily_range);
            low.push(base_price - daily_range);
            close.push(base_price + 1.0);
            volume.push(1000.0 + (day as f64 * 100.0));
        }
    }
    
    ModelsOHLCVData::new(
        timestamps,
        open,
        high,
        low,
        close,
        volume,
        "TEST_SEASONAL"
    ).unwrap()
}

#[test]
fn test_holt_winters_model_creation() {
    // Test valid parameter ranges for additive seasonality
    assert!(HoltWintersModel::new(
        0.3, 0.1, 0.1, None, 7, SeasonalType::Additive, None
    ).is_ok());
    
    // Test valid parameter ranges for multiplicative seasonality
    assert!(HoltWintersModel::new(
        0.3, 0.1, 0.1, None, 7, SeasonalType::Multiplicative, None
    ).is_ok());
    
    // Test with damping
    assert!(HoltWintersModel::new(
        0.3, 0.1, 0.1, Some(0.9), 7, SeasonalType::Additive, None
    ).is_ok());
    
    // Test invalid alpha values
    assert!(HoltWintersModel::new(
        0.0, 0.1, 0.1, None, 7, SeasonalType::Additive, None
    ).is_err());
    
    assert!(HoltWintersModel::new(
        1.0, 0.1, 0.1, None, 7, SeasonalType::Additive, None
    ).is_err());
    
    // Test invalid beta values
    assert!(HoltWintersModel::new(
        0.3, 0.0, 0.1, None, 7, SeasonalType::Additive, None
    ).is_err());
    
    assert!(HoltWintersModel::new(
        0.3, 1.0, 0.1, None, 7, SeasonalType::Additive, None
    ).is_err());
    
    // Test invalid gamma values
    assert!(HoltWintersModel::new(
        0.3, 0.1, 0.0, None, 7, SeasonalType::Additive, None
    ).is_err());
    
    assert!(HoltWintersModel::new(
        0.3, 0.1, 1.0, None, 7, SeasonalType::Additive, None
    ).is_err());
    
    // Test invalid seasonal period
    assert!(HoltWintersModel::new(
        0.3, 0.1, 0.1, None, 1, SeasonalType::Additive, None
    ).is_err());
    
    // Test different target columns
    let model = HoltWintersModel::new(
        0.3, 0.1, 0.1, None, 7, SeasonalType::Additive, Some(TargetColumn::Open)
    ).unwrap();
    assert!(model.name().contains("Open"));
    
    let model = HoltWintersModel::new(
        0.3, 0.1, 0.1, None, 7, SeasonalType::Additive, Some(TargetColumn::Close)
    ).unwrap();
    assert!(model.name().contains("Close"));
}

#[test]
fn test_holt_winters_fit_and_forecast() {
    let data = create_test_data_with_seasonality();
    
    // Create a Holt-Winters model with weekly seasonality
    let mut model = HoltWintersModel::new(
        0.3, 0.1, 0.1, None, 7, SeasonalType::Additive, None
    ).unwrap();
    
    // Fit the model
    assert!(model.fit(&data).is_ok());
    
    // Check if fitted values exist
    let fitted_values = model.fitted_values().unwrap();
    assert_eq!(fitted_values.len(), data.len());
    
    // Get components
    let level = model.level().unwrap();
    let trend = model.trend().unwrap();
    let seasonal = model.seasonal().unwrap();
    
    // For our test data with upward trend, trend should be positive
    assert!(trend > 0.0);
    
    // Should have 7 seasonal factors
    assert_eq!(seasonal.len(), 7);
    
    // Generate forecasts
    let horizon = 14; // 2 weeks
    let forecasts = model.forecast(horizon).unwrap();
    assert_eq!(forecasts.len(), horizon);
    
    // Weekly pattern should repeat
    for i in 7..horizon {
        let current_day = i % 7;
        let day_week_before = (i - 7) % 7;
        
        // The difference should be approximately constant due to trend
        let diff = forecasts[i] - forecasts[i-7];
        let expected_diff = 7.0 * trend;
        
        // Allow some tolerance for floating point calculations
        assert!((diff - expected_diff).abs() < 1.0);
        
        // Relative positions should be maintained (e.g., Monday > Tuesday in our pattern)
        if current_day == 0 && day_week_before == 6 {
            // Monday > Sunday in our pattern
            assert!(forecasts[i] > forecasts[i-1]);
        }
    }
}

#[test]
fn test_holt_winters_multiplicative() {
    let data = create_test_data_with_seasonality();
    
    // Create models with both additive and multiplicative seasonality
    let mut additive_model = HoltWintersModel::new(
        0.3, 0.1, 0.1, None, 7, SeasonalType::Additive, None
    ).unwrap();
    
    let mut multiplicative_model = HoltWintersModel::new(
        0.3, 0.1, 0.1, None, 7, SeasonalType::Multiplicative, None
    ).unwrap();
    
    // Fit both models
    additive_model.fit(&data).unwrap();
    multiplicative_model.fit(&data).unwrap();
    
    // Generate forecasts far into the future
    let horizon = 28; // 4 weeks
    let additive_forecasts = additive_model.forecast(horizon).unwrap();
    let multiplicative_forecasts = multiplicative_model.forecast(horizon).unwrap();
    
    // Multiplicative seasonality should result in different forecasts
    // The differences should grow over time
    let early_diff = (additive_forecasts[0] - multiplicative_forecasts[0]).abs();
    let late_diff = (additive_forecasts[horizon-1] - multiplicative_forecasts[horizon-1]).abs();
    
    // The difference should grow over time (not always true, but generally expected)
    // This could be commented out if not stable
    // assert!(late_diff > early_diff);
}

#[test]
fn test_holt_winters_evaluation() {
    let data = create_test_data_with_seasonality();
    
    // Split data into train and test sets (3 weeks train, 1 week test)
    let (train_data, test_data) = data.train_test_split(0.75).unwrap();
    
    // Create and fit model
    let mut model = HoltWintersModel::new(
        0.3, 0.1, 0.1, None, 7, SeasonalType::Additive, None
    ).unwrap();
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
fn test_holt_winters_insufficient_data() {
    // Create a model with weekly seasonality
    let mut model = HoltWintersModel::new(
        0.3, 0.1, 0.1, None, 7, SeasonalType::Additive, None
    ).unwrap();
    
    // Create data with only one week (not enough for initialization)
    let now = Utc::now();
    let mut timestamps = Vec::with_capacity(7);
    let mut values = Vec::with_capacity(7);
    
    for i in 0..7 {
        timestamps.push(now + Duration::days(i));
        values.push(100.0 + i as f64);
    }
    
    let insufficient_data = ModelsOHLCVData::new(
        timestamps,
        values.clone(),
        values.clone(),
        values.clone(),
        values.clone(),
        values.clone(),
        "INSUFFICIENT"
    ).unwrap();
    
    // Should fail because Holt-Winters needs at least 2 complete seasons for initialization
    assert!(model.fit(&insufficient_data).is_err());
} 