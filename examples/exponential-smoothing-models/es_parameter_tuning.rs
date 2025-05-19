use chrono::{DateTime, TimeZone, Utc};
use oxidiviner_core::prelude::*;
use oxidiviner_core::models::Forecaster;
use oxidiviner_core::models::exponential_smoothing::{SESModel, HoltModel};
use std::error::Error;

fn main() -> std::result::Result<(), Box<dyn Error>> {
    println!("OxiDiviner - Exponential Smoothing Parameter Tuning Example");
    println!("========================================================\n");

    // Generate sample time series data with trend
    let time_series = generate_sample_data();
    
    // Split data into training and testing sets (80% train, 20% test)
    let (train_data, test_data) = time_series.train_test_split(0.8)?;
    let ohlcv_train = train_data.as_ohlcv()?;
    let ohlcv_test = test_data.as_ohlcv()?;
    
    println!("Data split: {} total points ({} train, {} test)",
             time_series.len(), train_data.len(), test_data.len());
             
    // Set forecast horizon
    let horizon = 10;
    
    // 1. Parameter Tuning for SES Model
    // ---------------------------------
    println!("\n1. Parameter Tuning for Simple Exponential Smoothing");
    println!("------------------------------------------------");
    println!("Testing different alpha values (smoothing parameter):");
    println!("Alpha    RMSE     MAE      MAPE");
    println!("--------------------------------");
    
    // Try different alpha values (smoothing parameters)
    let alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let mut best_alpha = 0.0;
    let mut best_rmse = f64::MAX;
    
    for &alpha in &alpha_values {
        // Create and train SES model with this alpha
        let mut model = SESModel::new(alpha, None)?;
        model.fit(&train_data)?;
        
        // Evaluate on test data
        let output = model.predict(horizon, Some(&test_data))?;
        
        if let Some(eval) = output.evaluation {
            println!("{:.1}     {:.4}  {:.4}  {:.4}%", 
                     alpha, eval.rmse, eval.mae, eval.mape);
                     
            // Track best parameter
            if eval.rmse < best_rmse {
                best_rmse = eval.rmse;
                best_alpha = alpha;
            }
        }
    }
    
    println!("\nBest alpha value: {:.1} (RMSE: {:.4})", best_alpha, best_rmse);
    
    // 2. Parameter Tuning for Holt's Model
    // ------------------------------------
    println!("\n2. Parameter Grid Search for Holt's Model");
    println!("--------------------------------------");
    println!("Testing combinations of alpha and beta values:");
    println!("Alpha  Beta   RMSE     MAE");
    println!("---------------------------");
    
    // Grid search for alpha and beta parameters
    let param_values = [0.1, 0.3, 0.5, 0.7, 0.9];
    let mut best_params = (0.0, 0.0);
    let mut best_rmse_holt = f64::MAX;
    
    for &alpha in &param_values {
        for &beta in &param_values {
            // Create and train Holt model with these parameters
            let mut model = HoltModel::new(alpha, beta, None, None)?;
            model.fit(&ohlcv_train)?;
            
            // Evaluate on test data
            let eval = model.evaluate(&ohlcv_test)?;
            
            println!("{:.1}    {:.1}    {:.4}  {:.4}", 
                    alpha, beta, eval.rmse, eval.mae);
                    
            // Track best parameters
            if eval.rmse < best_rmse_holt {
                best_rmse_holt = eval.rmse;
                best_params = (alpha, beta);
            }
        }
    }
    
    println!("\nBest parameters for Holt's model:");
    println!("Alpha: {:.1}, Beta: {:.1} (RMSE: {:.4})", 
             best_params.0, best_params.1, best_rmse_holt);
    
    // 3. Damped vs. Non-Damped Holt's Model Comparison
    // -----------------------------------------------
    println!("\n3. Comparing Damped vs. Non-Damped Trend");
    println!("-------------------------------------");
    
    // Create Holt model with best parameters (no damping)
    let mut model_no_damping = HoltModel::new(
        best_params.0, best_params.1, None, None)?;
    
    model_no_damping.fit(&ohlcv_train)?;
    let eval_no_damping = model_no_damping.evaluate(&ohlcv_test)?;
    
    // Create Holt model with best parameters and damping
    let mut model_with_damping = HoltModel::new(
        best_params.0, best_params.1, Some(0.9), None)?;
    
    model_with_damping.fit(&ohlcv_train)?;
    let eval_with_damping = model_with_damping.evaluate(&ohlcv_test)?;
    
    println!("Non-Damped Model:");
    println!("  RMSE: {:.4}", eval_no_damping.rmse);
    println!("  MAE: {:.4}", eval_no_damping.mae);
    
    println!("\nDamped Model (phi = 0.9):");
    println!("  RMSE: {:.4}", eval_with_damping.rmse);
    println!("  MAE: {:.4}", eval_with_damping.mae);
    
    // 4. Forecasting with the Best Model
    // ---------------------------------
    println!("\n4. Forecasting with the Best Model");
    println!("--------------------------------");
    
    // Determine which model performed best
    let (best_model_name, best_rmse_value) = if best_rmse <= best_rmse_holt &&
                                               best_rmse <= eval_with_damping.rmse {
        ("SES", best_rmse)
    } else if best_rmse_holt <= eval_with_damping.rmse {
        ("Holt (Non-Damped)", best_rmse_holt)
    } else {
        ("Holt (Damped)", eval_with_damping.rmse)
    };
    
    println!("Best model: {} (RMSE: {:.4})", best_model_name, best_rmse_value);
    
    // Create and train the best model for final forecasting
    if best_model_name == "SES" {
        let mut best_model = SESModel::new(best_alpha, None)?;
        best_model.fit(&train_data)?;
        
        // Make forecasts for future periods
        let output = best_model.predict(horizon, None)?;
        
        println!("\nForecasts for the next {} periods:", horizon);
        for (i, value) in output.forecasts.iter().enumerate() {
            println!("  Period t+{}: {:.4}", i+1, value);
        }
    } else if best_model_name == "Holt (Non-Damped)" {
        let mut best_model = HoltModel::new(
            best_params.0, best_params.1, None, None)?;
        best_model.fit(&ohlcv_train)?;
        
        // Make forecasts for future periods
        let forecasts = best_model.forecast(horizon)?;
        
        println!("\nForecasts for the next {} periods:", horizon);
        for (i, value) in forecasts.iter().enumerate() {
            println!("  Period t+{}: {:.4}", i+1, value);
        }
    } else {
        let mut best_model = HoltModel::new(
            best_params.0, best_params.1, Some(0.9), None)?;
        best_model.fit(&ohlcv_train)?;
        
        // Make forecasts for future periods
        let forecasts = best_model.forecast(horizon)?;
        
        println!("\nForecasts for the next {} periods:", horizon);
        for (i, value) in forecasts.iter().enumerate() {
            println!("  Period t+{}: {:.4}", i+1, value);
        }
    }
    
    println!("\nNote on Parameter Tuning:");
    println!("- For real applications, use cross-validation with multiple train/test splits");
    println!("- Try a finer grid of parameters around the best values found in the initial search");
    println!("- Consider optimizing for the appropriate metric for your specific application");
    
    Ok(())
}

// Generate a sample time series with trend and some noise
fn generate_sample_data() -> TimeSeriesData {
    let now = Utc::now();
    let n = 100;
    
    // Create timestamps (daily intervals)
    let timestamps: Vec<DateTime<Utc>> = (0..n)
        .map(|i| Utc.timestamp_opt(now.timestamp() + i as i64 * 86400, 0).unwrap())
        .collect();
    
    // Create values with a linear trend and some noise
    let mut values = Vec::with_capacity(n);
    
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for i in 0..n {
        // Linear trend with slope 0.5
        let trend = 10.0 + 0.5 * (i as f64);
        
        // Add random noise
        let noise = rng.gen::<f64>() * 3.0 - 1.5;
        values.push(trend + noise);
    }
    
    TimeSeriesData::new(
        timestamps,
        values,
        "sample_data_with_trend"
    ).unwrap()
} 