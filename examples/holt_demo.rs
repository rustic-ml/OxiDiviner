use chrono::{Duration, Utc};
use rand::Rng;
use std::error::Error;
use oxidiviner::ModelsOHLCVData;
use oxidiviner::models::exponential_smoothing::holt::HoltModel;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Holt Linear Trend Model Demo");
    println!("============================\n");
    
    // Generate synthetic data with trend
    println!("Generating synthetic data with trend...");
    let data = generate_synthetic_data_with_trend();
    println!("Generated {} data points\n", data.len());
    
    // Create Holt model
    println!("Creating Holt model with alpha = 0.3, beta = 0.1...");
    let mut model = HoltModel::new(0.3, 0.1, None, None)?;
    
    // Split data for training and testing
    println!("Splitting data into training (80%) and testing (20%) sets...");
    let (train_data, test_data) = data.train_test_split(0.8)?;
    println!("Training set: {} points", train_data.len());
    println!("Testing set: {} points\n", test_data.len());
    
    // Fit model to training data
    println!("Fitting model to training data...");
    model.fit(&train_data)?;
    println!("Model fitted successfully");
    
    // Get the final level and trend values
    let level = model.level().unwrap();
    let trend = model.trend().unwrap();
    println!("Final level: {:.2}", level);
    println!("Final trend: {:.2} per period\n", trend);
    
    // Generate forecasts
    let horizon = 30;
    println!("Generating {} day forecast...", horizon);
    let forecasts = model.forecast(horizon)?;
    
    // Print a few forecasts
    println!("Forecast preview:");
    for i in 0..5.min(horizon) {
        println!("Day {}: {:.2}", i + 1, forecasts[i]);
    }
    println!("...");
    println!("Day {}: {:.2}\n", horizon, forecasts[horizon - 1]);
    
    // Evaluate model on test data
    println!("Evaluating model on test data...");
    let evaluation = model.evaluate(&test_data)?;
    
    // Print evaluation metrics
    println!("Evaluation metrics:");
    println!("MAE:   {:.4}", evaluation.mae);
    println!("RMSE:  {:.4}", evaluation.rmse);
    println!("MAPE:  {:.2}%", evaluation.mape);
    println!("SMAPE: {:.2}%\n", evaluation.smape);
    
    // Compare regular vs damped trend
    println!("Comparing regular trend vs damped trend...");
    compare_regular_vs_damped(&train_data, &test_data)?;
    
    println!("Demo completed successfully!");
    Ok(())
}

fn generate_synthetic_data_with_trend() -> ModelsOHLCVData {
    let mut rng = rand::thread_rng();
    let now = Utc::now();
    let n = 200; // 200 days of data
    
    let mut timestamps = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);
    
    let base = 100.0;
    let trend = 0.5; // Upward trend of 0.5 per day
    
    for i in 0..n {
        let day = now + Duration::days(i as i64);
        timestamps.push(day);
        
        // Add trend and random noise
        let noise = rng.gen_range(-3.0..3.0);
        let value = base + trend * i as f64 + noise;
        
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
        
        // Volume with slight uptrend
        volume.push(1000.0 + i as f64 * 5.0 + rng.gen_range(-200.0..200.0));
    }
    
    ModelsOHLCVData {
        timestamps,
        open,
        high,
        low,
        close,
        volume,
        name: "Synthetic data with trend for Holt demo".to_string(),
    }
}

fn compare_regular_vs_damped(train_data: &ModelsOHLCVData, test_data: &ModelsOHLCVData) -> Result<(), Box<dyn Error>> {
    // Create regular Holt model
    let mut regular_model = HoltModel::new(0.3, 0.1, None, None)?;
    regular_model.fit(train_data)?;
    let regular_eval = regular_model.evaluate(test_data)?;
    let regular_trend = regular_model.trend().unwrap();
    
    // Create damped trend Holt model
    let mut damped_model = HoltModel::new(0.3, 0.1, Some(0.9), None)?;
    damped_model.fit(train_data)?;
    let damped_eval = damped_model.evaluate(test_data)?;
    let damped_trend = damped_model.trend().unwrap();
    
    println!("Model comparison:");
    println!("| Model Type  | Trend    | MAE     | RMSE    | MAPE    |");
    println!("|------------|----------|---------|---------|---------|");
    println!("| Regular     | {:.4}   | {:.4} | {:.4} | {:.2}% |", 
             regular_trend, regular_eval.mae, regular_eval.rmse, regular_eval.mape);
    println!("| Damped (φ=0.9) | {:.4}   | {:.4} | {:.4} | {:.2}% |", 
             damped_trend, damped_eval.mae, damped_eval.rmse, damped_eval.mape);
    
    // Generate long-term forecasts with both models
    let horizon = 100;
    let regular_forecasts = regular_model.forecast(horizon)?;
    let damped_forecasts = damped_model.forecast(horizon)?;
    
    println!("\nLong-term forecast comparison:");
    println!("| Day | Regular  | Damped   |");
    println!("|-----|----------|----------|");
    for day in [1, 10, 25, 50, 75, 100] {
        if day <= horizon {
            println!("| {:3} | {:8.2} | {:8.2} |", 
                     day, regular_forecasts[day-1], damped_forecasts[day-1]);
        }
    }
    
    println!("\nNote: Damped trend models prevent the trend from continuing indefinitely,");
    println!("which can provide more realistic long-term forecasts.");
    
    Ok(())
} 