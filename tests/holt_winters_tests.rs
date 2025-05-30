use chrono::{Duration, Utc};
use oxidiviner::{OHLCVData, HoltWintersModel};

fn create_test_data_with_seasonality() -> OHLCVData {
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
    
    OHLCVData::new(
        timestamps,
        open,
        high,
        low,
        close,
        volume,
        Some("TEST_SEASONAL".to_string())
    ).unwrap()
}

#[test]
fn test_holt_winters_model_creation() {
    // Test valid parameter ranges
    assert!(HoltWintersModel::new(0.3, 0.1, 0.1, 7).is_ok());
    
    // Test invalid alpha values
    assert!(HoltWintersModel::new(0.0, 0.1, 0.1, 7).is_err());
    assert!(HoltWintersModel::new(1.0, 0.1, 0.1, 7).is_err());
    
    // Test invalid beta values
    assert!(HoltWintersModel::new(0.3, 0.0, 0.1, 7).is_err());
    assert!(HoltWintersModel::new(0.3, 1.0, 0.1, 7).is_err());
    
    // Test invalid gamma values
    assert!(HoltWintersModel::new(0.3, 0.1, 0.0, 7).is_err());
    assert!(HoltWintersModel::new(0.3, 0.1, 1.0, 7).is_err());
    
    // Test invalid seasonal period
    assert!(HoltWintersModel::new(0.3, 0.1, 0.1, 1).is_err());
}

#[test]
fn test_holt_winters_fit_and_forecast() {
    let ohlcv_data = create_test_data_with_seasonality();
    
    // Convert to TimeSeriesData using close prices
    let data = oxidiviner::TimeSeriesData::new(
        ohlcv_data.timestamps.clone(),
        ohlcv_data.close.clone(),
        "test_seasonal"
    ).unwrap();
    
    // Create a Holt-Winters model with weekly seasonality
    let mut model = HoltWintersModel::new(0.3, 0.1, 0.1, 7).unwrap();
    
    // Fit the model
    assert!(model.fit(&data).is_ok());
    
    // Check if fitted values exist
    let fitted_values = model.fitted_values().unwrap();
    assert_eq!(fitted_values.len(), data.len());
    
    // Generate forecasts
    let horizon = 14; // 2 weeks
    let forecasts = model.forecast(horizon).unwrap();
    assert_eq!(forecasts.len(), horizon);
}

#[test]
fn test_holt_winters_evaluation() {
    let ohlcv_data = create_test_data_with_seasonality();
    
    // Convert to TimeSeriesData using close prices
    let data = oxidiviner::TimeSeriesData::new(
        ohlcv_data.timestamps.clone(),
        ohlcv_data.close.clone(),
        "test_seasonal"
    ).unwrap();
    
    // Split data into train and test sets (3 weeks train, 1 week test)
    let (train_data, test_data) = data.train_test_split(0.75).unwrap();
    
    // Create and fit model
    let mut model = HoltWintersModel::new(0.3, 0.1, 0.1, 7).unwrap();
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
fn test_holt_winters_insufficient_data() {
    // Create a model with weekly seasonality
    let mut model = HoltWintersModel::new(0.3, 0.1, 0.1, 7).unwrap();
    
    // Create data with only one week (not enough for initialization)
    let now = Utc::now();
    let mut timestamps = Vec::with_capacity(7);
    let mut values = Vec::with_capacity(7);
    
    for i in 0..7 {
        timestamps.push(now + Duration::days(i));
        values.push(100.0 + i as f64);
    }
    
    let insufficient_data = oxidiviner::TimeSeriesData::new(
        timestamps,
        values,
        "insufficient"
    ).unwrap();
    
    // Should fail because Holt-Winters needs at least 2 complete seasons for initialization
    assert!(model.fit(&insufficient_data).is_err());
} 