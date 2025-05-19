use chrono::{Duration, Utc};
use rand::Rng;
use std::error::Error;
use oxidiviner_core::{TimeSeriesData};
use oxidiviner_exponential_smoothing::SimpleESModel;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Simple Exponential Smoothing (SES) Model Demo");
    println!("=============================================\n");
    
    // Generate synthetic data
    println!("Generating synthetic data with random noise...");
    let data = generate_synthetic_data();
    println!("Generated {} data points\n", data.len());
    
    // Create SES model
    println!("Creating SES model with alpha = 0.3...");
    let mut model = SimpleESModel::new(0.3)?;
    
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
    
    // Compare models with different alphas
    println!("Comparing SES models with different alpha values...");
    compare_alpha_values(&train_data, &test_data)?;
    
    println!("Demo completed successfully!");
    Ok(())
}

fn generate_synthetic_data() -> TimeSeriesData {
    let mut rng = rand::thread_rng();
    let now = Utc::now();
    let n = 200; // 200 days of data
    
    let mut timestamps = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    
    let mut level = 100.0;
    
    for i in 0..n {
        let day = now + Duration::days(i as i64);
        timestamps.push(day);
        
        // Add random noise
        let noise = rng.gen_range(-2.0..2.0);
        
        // Slowly vary the level (mean-reverting)
        level = level * 0.99 + 100.0 * 0.01 + noise;
        
        values.push(level);
    }
    
    TimeSeriesData::new(
        timestamps,
        values,
        "Synthetic data for SES demo"
    ).unwrap()
}

fn compare_alpha_values(train_data: &TimeSeriesData, test_data: &TimeSeriesData) -> Result<(), Box<dyn Error>> {
    println!("| Alpha | MAE     | RMSE    | MAPE    |");
    println!("|-------|---------|---------|---------|");
    
    // Try different alpha values
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9].iter() {
        let mut model = SimpleESModel::new(*alpha)?;
        model.fit(train_data)?;
        let eval = model.evaluate(test_data)?;
        
        println!("| {:.1} | {:.4} | {:.4} | {:.2}% |", alpha, eval.mae, eval.rmse, eval.mape);
    }
    
    println!();
    println!("A lower alpha gives more weight to historical data (smoother forecasts).");
    println!("A higher alpha gives more weight to recent data (more responsive forecasts).");
    
    Ok(())
} 