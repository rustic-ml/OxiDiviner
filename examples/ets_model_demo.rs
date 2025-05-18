use chrono::{Duration, Utc};
use rand::Rng;
use std::error::Error;
use oxidiviner::ModelsOHLCVData;
use oxidiviner::models::exponential_smoothing::ets::{ETSComponent, DailyETSModel, MinuteETSModel};

fn main() -> Result<(), Box<dyn Error>> {
    println!("ETS (Error-Trend-Seasonality) Model Demo");
    println!("========================================\n");
    
    // Generate synthetic data with trend and seasonality
    println!("Generating synthetic daily data with trend and seasonality...");
    let data = generate_synthetic_daily_data();
    println!("Generated {} data points\n", data.len());
    
    // Split data for training and testing
    println!("Splitting data into training (80%) and testing (20%) sets...");
    let (train_data, test_data) = data.train_test_split(0.8)?;
    println!("Training set: {} points", train_data.len());
    println!("Testing set: {} points\n", test_data.len());
    
    // Demonstrate different ETS model configurations
    println!("Comparing different ETS model configurations...");
    
    // 1. ETS(A,N,N) - Simple Exponential Smoothing with additive errors
    println!("\n1. ETS(A,N,N) - Simple Exponential Smoothing");
    let mut model1 = DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::None,      // No trend
        ETSComponent::None,      // No seasonality
        0.3,                     // alpha = 0.3
        None,                    // No beta (no trend)
        None,                    // No gamma (no seasonality)
        None,                    // No phi (no damping)
        None,                    // No seasonal period
        None,                    // Default to Close price
    )?;
    
    model1.fit(&train_data)?;
    let eval1 = model1.evaluate(&test_data)?;
    println!("Model: {}", eval1.model_name);
    println!("MAE: {:.4}", eval1.mae);
    println!("RMSE: {:.4}", eval1.rmse);
    println!("MAPE: {:.2}%", eval1.mape);
    
    // 2. ETS(A,A,N) - Holt's Linear Trend with additive errors
    println!("\n2. ETS(A,A,N) - Holt's Linear Trend");
    let mut model2 = DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::Additive,  // Additive trend
        ETSComponent::None,      // No seasonality
        0.3,                     // alpha = 0.3
        Some(0.1),               // beta = 0.1
        None,                    // No gamma (no seasonality)
        None,                    // No phi (no damping)
        None,                    // No seasonal period
        None,                    // Default to Close price
    )?;
    
    model2.fit(&train_data)?;
    let eval2 = model2.evaluate(&test_data)?;
    println!("Model: {}", eval2.model_name);
    println!("MAE: {:.4}", eval2.mae);
    println!("RMSE: {:.4}", eval2.rmse);
    println!("MAPE: {:.2}%", eval2.mape);
    
    // 3. ETS(A,A,A) - Holt-Winters Additive Seasonal
    println!("\n3. ETS(A,A,A) - Holt-Winters Additive Seasonal");
    let mut model3 = DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::Additive,  // Additive trend
        ETSComponent::Additive,  // Additive seasonality
        0.3,                     // alpha = 0.3
        Some(0.1),               // beta = 0.1
        Some(0.1),               // gamma = 0.1
        None,                    // No phi (no damping)
        Some(7),                 // Weekly seasonality
        None,                    // Default to Close price
    )?;
    
    model3.fit(&train_data)?;
    let eval3 = model3.evaluate(&test_data)?;
    println!("Model: {}", eval3.model_name);
    println!("MAE: {:.4}", eval3.mae);
    println!("RMSE: {:.4}", eval3.rmse);
    println!("MAPE: {:.2}%", eval3.mape);
    
    // 4. ETS(A,Ad,A) - Holt-Winters with Damped Trend
    println!("\n4. ETS(A,Ad,A) - Holt-Winters with Damped Trend");
    let mut model4 = DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::Damped,    // Damped trend
        ETSComponent::Additive,  // Additive seasonality
        0.3,                     // alpha = 0.3
        Some(0.1),               // beta = 0.1
        Some(0.1),               // gamma = 0.1
        Some(0.9),               // phi = 0.9 (damping factor)
        Some(7),                 // Weekly seasonality
        None,                    // Default to Close price
    )?;
    
    model4.fit(&train_data)?;
    let eval4 = model4.evaluate(&test_data)?;
    println!("Model: {}", eval4.model_name);
    println!("MAE: {:.4}", eval4.mae);
    println!("RMSE: {:.4}", eval4.rmse);
    println!("MAPE: {:.2}%", eval4.mape);
    
    // Generate forecasts from the best model
    println!("\nGenerating forecasts from the best performing model...");
    
    // Choose the best model based on MAE
    let (mut best_model, best_eval) = [(model1, eval1), (model2, eval2), (model3, eval3), (model4, eval4)]
        .into_iter()
        .min_by(|(_, eval_a), (_, eval_b)| 
            eval_a.mae.partial_cmp(&eval_b.mae).unwrap_or(std::cmp::Ordering::Equal)
        )
        .unwrap();
    
    println!("Best model: {}", best_eval.model_name);
    println!("MAE: {:.4}", best_eval.mae);
    
    // Generate forecasts
    let horizon = 28; // 4 weeks
    println!("Generating {} day forecast...", horizon);
    let forecasts = best_model.forecast(horizon)?;
    
    // Print weekly aggregated forecasts
    println!("Forecasts by week (weekly means):");
    println!("| Week | Forecast |");
    println!("|------|----------|");
    
    for week in 0..4 {
        let start_idx = week * 7;
        let end_idx = start_idx + 7;
        if end_idx <= forecasts.len() {
            let week_mean: f64 = forecasts[start_idx..end_idx].iter().sum::<f64>() / 7.0;
            println!("| {:4} | {:8.2} |", week + 1, week_mean);
        }
    }
    
    println!("\nDemo for minute-level models...");
    minute_model_demo()?;
    
    println!("\nDemo completed successfully!");
    Ok(())
}

fn minute_model_demo() -> Result<(), Box<dyn Error>> {
    println!("Generating synthetic minute data...");
    let data = generate_synthetic_minute_data();
    println!("Generated {} data points", data.len());
    
    // Split data for training and testing
    let (train_data, test_data) = data.train_test_split(0.8)?;
    
    // Create and compare different minute aggregation levels
    println!("\nComparing different minute aggregation levels:");
    println!("| Aggregation | MAE     | RMSE    | MAPE    |");
    println!("|------------|---------|---------|---------|");
    
    // Only try aggregation levels that would provide enough data
    // For a 60-minute seasonality, we need at least 2 complete cycles (120 points)
    // With 5-min aggregation, we need 600 original points, etc.
    let viable_aggregations = [1]; // Only 1-minute aggregation is safe with our dataset size
    
    for agg_minutes in viable_aggregations {
        // ETS model with hourly seasonality and variable aggregation
        let mut model = MinuteETSModel::new(
            ETSComponent::Additive,  // Error type
            ETSComponent::Additive,  // Additive trend
            ETSComponent::Additive,  // Additive seasonality
            0.3,                     // alpha = 0.3
            Some(0.1),               // beta = 0.1
            Some(0.1),               // gamma = 0.1
            None,                    // No phi (no damping)
            Some(60),                // 60-minute (hourly) seasonality
            None,                    // Default to Close price
            Some(agg_minutes),       // Aggregation minutes
        )?;
        
        model.fit(&train_data)?;
        let eval = model.evaluate(&test_data)?;
        
        println!("| {:10} | {:.4} | {:.4} | {:.2}% |", 
            format!("{} min", agg_minutes), eval.mae, eval.rmse, eval.mape);
    }
    
    println!("\nNote on aggregation: Higher aggregation reduces noise but may lose detail");
    println!("The best aggregation depends on the specific use case and data characteristics");
    println!("For this demo, we're only using 1-minute aggregation due to the limited dataset size");
    println!("In practice, with larger datasets, you can experiment with 5, 15, 30, or 60-minute aggregations");
    
    Ok(())
}

fn generate_synthetic_daily_data() -> ModelsOHLCVData {
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
    let trend = 0.2; // Upward trend of 0.2 per day
    
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
            2 => 1.0,  // Wednesday
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
        
        // Volume with weekly pattern
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
        name: "Synthetic daily data for ETS demo".to_string(),
    }
}

fn generate_synthetic_minute_data() -> ModelsOHLCVData {
    let mut rng = rand::thread_rng();
    let now = Utc::now();
    // Increase to 960 minutes (16 hours) to have enough data for seasonal patterns
    let n = 960; // 16 hours of minute data
    
    let mut timestamps = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);
    
    let base = 100.0;
    let trend = 0.001; // Small upward trend per minute
    
    for i in 0..n {
        let minute = now + Duration::minutes(i as i64);
        timestamps.push(minute);
        
        // Add trend
        let trend_component = trend * i as f64;
        
        // Add hourly seasonality
        let minute_of_hour = i % 60;
        let seasonal_component = 0.5 * (2.0 * std::f64::consts::PI * (minute_of_hour as f64 / 60.0)).sin();
        
        // Add random noise
        let noise = rng.gen_range(-0.1..0.1);
        
        // Combine components
        let value = base + trend_component + seasonal_component + noise;
        
        // Generate OHLC data
        let minute_range = value * 0.001; // 0.1% range per minute
        let open_val = value - minute_range / 2.0 + rng.gen_range(-0.01..0.01);
        let close_val = value + minute_range / 2.0 + rng.gen_range(-0.01..0.01);
        let high_val = f64::max(value, f64::max(open_val, close_val)) + rng.gen_range(0.0..0.02);
        let low_val = f64::min(value, f64::min(open_val, close_val)) - rng.gen_range(0.0..0.02);
        
        open.push(open_val);
        high.push(high_val);
        low.push(low_val);
        close.push(close_val);
        
        // Volume with hourly pattern
        let vol_seasonal = 1.0 + 0.3 * (2.0 * std::f64::consts::PI * (minute_of_hour as f64 / 60.0)).cos();
        volume.push(10.0 * vol_seasonal + rng.gen_range(-1.0..1.0));
    }
    
    ModelsOHLCVData {
        timestamps,
        open,
        high,
        low,
        close,
        volume,
        name: "Synthetic minute data for ETS demo".to_string(),
    }
} 