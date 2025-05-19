use chrono::{Duration, Utc};
use std::error::Error;
use oxidiviner_core::{TimeSeriesData};
use oxidiviner_moving_average::MAModel;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Moving Average (MA) Model Demo");
    println!("==============================\n");
    
    // Generate synthetic data
    println!("Generating synthetic data with linear trend and noise...");
    let data = generate_synthetic_data();
    println!("Generated {} data points\n", data.len());
    
    // Create MA model with window size 3
    println!("Creating MA model with window size = 3...");
    let mut model = MAModel::new(3)?;
    
    // Split data for training and testing
    println!("Splitting data into training (80%) and testing (20%) sets...");
    let (train_data, test_data) = data.train_test_split(0.8)?;
    println!("Training set: {} points", train_data.len());
    println!("Testing set: {} points\n", test_data.len());
    
    // Fit model to training data
    println!("Fitting model to training data...");
    model.fit(&train_data)?;
    println!("Model fitted successfully\n");
    
    // Generate forecasts
    let horizon = 10;
    println!("Generating {} day forecast...", horizon);
    let forecasts = model.forecast(horizon)?;
    
    // Print forecasts
    println!("Forecast preview:");
    for i in 0..horizon {
        println!("Day {}: {:.2}", i + 1, forecasts[i]);
    }
    println!();
    
    // Evaluate model on test data
    println!("Evaluating model on test data...");
    let evaluation = model.evaluate(&test_data)?;
    
    // Print evaluation metrics
    println!("Evaluation metrics:");
    println!("MAE:   {:.4}", evaluation.mae);
    println!("RMSE:  {:.4}", evaluation.rmse);
    println!("MAPE:  {:.2}%", evaluation.mape);
    println!("SMAPE: {:.2}%\n", evaluation.smape);
    
    // Compare models with different window sizes
    println!("Comparing MA models with different window sizes...");
    compare_window_sizes(&train_data, &test_data)?;
    
    println!("Demo completed successfully!");
    Ok(())
}

fn generate_synthetic_data() -> TimeSeriesData {
    let now = Utc::now();
    let n = 100; // 100 days of data
    
    let mut timestamps = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    
    for i in 0..n {
        let day = now + Duration::days(i as i64);
        timestamps.push(day);
        
        // Linear trend with some noise
        let trend = 100.0 + i as f64 * 0.5;
        let noise = (i as f64 * 0.2).sin() * 5.0; // Sinusoidal noise
        
        values.push(trend + noise);
    }
    
    TimeSeriesData::new(timestamps, values, "test_series").unwrap()
}

fn compare_window_sizes(train_data: &TimeSeriesData, test_data: &TimeSeriesData) -> Result<(), Box<dyn Error>> {
    println!("| Window | MAE     | RMSE    | MAPE    |");
    println!("|--------|---------|---------|---------|");
    
    // Try different window sizes
    for window_size in [2, 3, 5, 7, 10, 14].iter() {
        let mut model = MAModel::new(*window_size)?;
        model.fit(train_data)?;
        let eval = model.evaluate(test_data)?;
        
        println!("| {:6} | {:.4} | {:.4} | {:.2}% |", window_size, eval.mae, eval.rmse, eval.mape);
    }
    
    println!();
    println!("A smaller window is more responsive to recent changes.");
    println!("A larger window produces smoother forecasts but may lag behind trends.");
    
    Ok(())
} 