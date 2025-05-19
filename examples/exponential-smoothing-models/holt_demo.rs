use chrono::{Duration, Utc};
use std::error::Error;
use oxidiviner_core::{TimeSeriesData};
use oxidiviner_exponential_smoothing::HoltLinearModel;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Holt Linear (Double Exponential Smoothing) Model Demo");
    println!("====================================================\n");
    
    // Generate synthetic data
    println!("Generating synthetic data with linear trend and noise...");
    let data = generate_synthetic_data();
    println!("Generated {} data points\n", data.len());
    
    // Create Holt model
    println!("Creating Holt model with alpha = 0.8, beta = 0.2...");
    let mut model = HoltLinearModel::new(0.8, 0.2)?;
    
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
    let horizon = 20;
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
    
    // Compare models with different parameter values
    println!("Comparing Holt models with different alpha and beta values...");
    compare_parameter_values(&train_data, &test_data)?;
    
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
        
        // Linear trend with increasing slope
        let trend = 100.0 + 0.5 * i as f64 + 0.01 * (i as f64).powi(2);
        
        // Add small random noise
        let noise = (i as f64 * 0.1).sin() * 3.0;
        
        values.push(trend + noise);
    }
    
    TimeSeriesData::new(
        timestamps,
        values,
        "Synthetic data with trend for Holt demo"
    ).unwrap()
}

fn compare_parameter_values(train_data: &TimeSeriesData, test_data: &TimeSeriesData) -> Result<(), Box<dyn Error>> {
    println!("| Alpha | Beta | MAE     | RMSE    | MAPE    |");
    println!("|-------|------|---------|---------|---------|");
    
    // Try different alpha and beta values
    for alpha in [0.2, 0.5, 0.8].iter() {
        for beta in [0.1, 0.3, 0.5].iter() {
            let mut model = HoltLinearModel::new(*alpha, *beta)?;
            model.fit(train_data)?;
            let eval = model.evaluate(test_data)?;
            
            println!("| {:.1}   | {:.1}  | {:.4} | {:.4} | {:.2}% |", 
                     alpha, beta, eval.mae, eval.rmse, eval.mape);
        }
    }
    
    println!();
    println!("A higher alpha gives more weight to recent observations for level estimation.");
    println!("A higher beta gives more weight to recent changes for trend estimation.");
    
    Ok(())
} 