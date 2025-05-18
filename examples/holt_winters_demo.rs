use chrono::{Duration, Utc};
use rand::Rng;
use std::error::Error;
use oxidiviner::ModelsOHLCVData;
use oxidiviner::models::exponential_smoothing::holt_winters::{HoltWintersModel, SeasonalType};

fn main() -> Result<(), Box<dyn Error>> {
    println!("Holt-Winters Seasonal Model Demo");
    println!("================================\n");
    
    // Generate synthetic data with trend and seasonality
    println!("Generating synthetic data with trend and seasonality...");
    let data = generate_synthetic_data_with_seasonality();
    println!("Generated {} data points\n", data.len());
    
    // Create Holt-Winters model with additive seasonality
    println!("Creating Holt-Winters model with weekly seasonality...");
    println!("Parameters: alpha = 0.3, beta = 0.1, gamma = 0.1");
    let mut model = HoltWintersModel::new(
        0.3,                    // alpha
        0.1,                    // beta
        0.1,                    // gamma
        None,                   // no damping
        7,                      // weekly seasonality
        SeasonalType::Additive, // additive seasonality
        None                    // default to Close price
    )?;
    
    // Split data for training and testing
    println!("Splitting data into training (80%) and testing (20%) sets...");
    let (train_data, test_data) = data.train_test_split(0.8)?;
    println!("Training set: {} points", train_data.len());
    println!("Testing set: {} points\n", test_data.len());
    
    // Fit model to training data
    println!("Fitting model to training data...");
    model.fit(&train_data)?;
    println!("Model fitted successfully");
    
    // Get components
    let level = model.level().unwrap();
    let trend = model.trend().unwrap();
    let seasonal = model.seasonal().unwrap();
    
    println!("Final components:");
    println!("Level: {:.2}", level);
    println!("Trend: {:.4} per period", trend);
    
    println!("Seasonal factors (for weekly seasonality):");
    for (i, factor) in seasonal.iter().enumerate() {
        let day = match i {
            0 => "Monday",
            1 => "Tuesday",
            2 => "Wednesday",
            3 => "Thursday",
            4 => "Friday",
            5 => "Saturday",
            6 => "Sunday",
            _ => "Unknown"
        };
        println!("  {}: {:.4}", day, factor);
    }
    println!();
    
    // Generate forecasts
    let horizon = 28; // 4 weeks
    println!("Generating {} day forecast...", horizon);
    let forecasts = model.forecast(horizon)?;
    
    // Print forecasts for the first week
    println!("Forecast for the first week:");
    for i in 0..7.min(horizon) {
        let day = match i % 7 {
            0 => "Monday",
            1 => "Tuesday",
            2 => "Wednesday",
            3 => "Thursday",
            4 => "Friday",
            5 => "Saturday",
            6 => "Sunday",
            _ => "Unknown"
        };
        println!("Day {} ({}): {:.2}", i + 1, day, forecasts[i]);
    }
    println!("...\n");
    
    // Evaluate model on test data
    println!("Evaluating model on test data...");
    let evaluation = model.evaluate(&test_data)?;
    
    // Print evaluation metrics
    println!("Evaluation metrics:");
    println!("MAE:   {:.4}", evaluation.mae);
    println!("RMSE:  {:.4}", evaluation.rmse);
    println!("MAPE:  {:.2}%", evaluation.mape);
    println!("SMAPE: {:.2}%\n", evaluation.smape);
    
    // Compare additive vs multiplicative seasonality
    println!("Comparing additive vs multiplicative seasonality...");
    compare_seasonal_types(&train_data, &test_data)?;
    
    println!("Demo completed successfully!");
    Ok(())
}

fn generate_synthetic_data_with_seasonality() -> ModelsOHLCVData {
    let mut rng = rand::thread_rng();
    let now = Utc::now();
    let n = 112; // 16 weeks of daily data
    
    let mut timestamps = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);
    
    let base = 100.0;
    let trend = 0.1; // Upward trend of 0.1 per day
    
    for i in 0..n {
        let day = now + Duration::days(i as i64);
        timestamps.push(day);
        
        // Add trend
        let trend_component = trend * i as f64;
        
        // Add weekly seasonality
        let day_of_week = i % 7;
        let seasonal_component = match day_of_week {
            0 => 5.0,  // Monday (peak)
            1 => 3.0,  // Tuesday
            2 => 0.0,  // Wednesday
            3 => -1.0, // Thursday
            4 => -2.0, // Friday
            5 => -4.0, // Saturday (trough)
            6 => -1.0, // Sunday
            _ => 0.0,
        };
        
        // Add random noise
        let noise = rng.gen_range(-2.0..2.0);
        
        // Combine components
        let value = base + trend_component + seasonal_component + noise;
        
        // Generate OHLC data
        let daily_range = value * 0.02; // 2% daily range
        let open_val = value - daily_range / 2.0 + rng.gen_range(-0.5..0.5);
        let close_val = value + daily_range / 2.0 + rng.gen_range(-0.5..0.5);
        let high_val = f64::max(value, f64::max(open_val, close_val)) + rng.gen_range(0.0..1.0);
        let low_val = f64::min(value, f64::min(open_val, close_val)) - rng.gen_range(0.0..1.0);
        
        open.push(open_val);
        high.push(high_val);
        low.push(low_val);
        close.push(close_val);
        
        // Volume with weekly pattern (higher on Monday, lower on weekend)
        let vol_seasonal = match day_of_week {
            0 => 1.3,  // Monday
            5 => 0.7,  // Saturday
            6 => 0.6,  // Sunday
            _ => 1.0,
        };
        
        volume.push(1000.0 * vol_seasonal + rng.gen_range(-100.0..100.0));
    }
    
    ModelsOHLCVData {
        timestamps,
        open,
        high,
        low,
        close,
        volume,
        name: "Synthetic data with trend and seasonality for Holt-Winters demo".to_string(),
    }
}

fn compare_seasonal_types(train_data: &ModelsOHLCVData, test_data: &ModelsOHLCVData) -> Result<(), Box<dyn Error>> {
    // Create additive seasonality model
    let mut additive_model = HoltWintersModel::new(
        0.3, 0.1, 0.1, None, 7, SeasonalType::Additive, None
    )?;
    additive_model.fit(train_data)?;
    let additive_eval = additive_model.evaluate(test_data)?;
    
    // Create multiplicative seasonality model
    let mut mult_model = HoltWintersModel::new(
        0.3, 0.1, 0.1, None, 7, SeasonalType::Multiplicative, None
    )?;
    mult_model.fit(train_data)?;
    let mult_eval = mult_model.evaluate(test_data)?;
    
    println!("Model comparison:");
    println!("| Seasonal Type   | MAE     | RMSE    | MAPE    |");
    println!("|-----------------|---------|---------|---------|");
    println!("| Additive        | {:.4} | {:.4} | {:.2}% |", 
             additive_eval.mae, additive_eval.rmse, additive_eval.mape);
    println!("| Multiplicative  | {:.4} | {:.4} | {:.2}% |", 
             mult_eval.mae, mult_eval.rmse, mult_eval.mape);
    
    println!("\nAdditive seasonality: Effects have constant magnitude regardless of level");
    println!("Multiplicative seasonality: Effects scale with level (percentage change)\n");
    
    // Generate long-term forecasts with both models
    let horizon = 28; // 4 weeks
    let additive_forecasts = additive_model.forecast(horizon)?;
    let mult_forecasts = mult_model.forecast(horizon)?;
    
    println!("Weekly pattern in forecasts:");
    println!("| Day       | Additive | Multiplicative |");
    println!("|-----------|----------|----------------|");
    
    // Show forecast for the last week
    let start_day = horizon - 7;
    for i in 0..7 {
        let day_idx = start_day + i;
        let day = match i {
            0 => "Monday",
            1 => "Tuesday",
            2 => "Wednesday",
            3 => "Thursday",
            4 => "Friday",
            5 => "Saturday",
            6 => "Sunday",
            _ => "Unknown"
        };
        
        println!("| {:9} | {:8.2} | {:14.2} |", 
                 day, additive_forecasts[day_idx], mult_forecasts[day_idx]);
    }
    
    Ok(())
} 