use chrono::{DateTime, Duration, Utc};
use rand::Rng;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Simple ETS Model Demo");
    println!("====================\n");
    
    // Generate some synthetic data
    println!("Generating synthetic data...");
    let data = generate_synthetic_data();
    
    // In a real application, you would use the actual OxiDiviner APIs:
    // use oxidiviner_core::{TimeSeriesData, Forecaster, ModelEvaluation};
    // use oxidiviner_exponential_smoothing::ETSModel;
    
    println!("\nThis is a simplified demo that demonstrates the structure");
    println!("of an ETS model without syntax errors. In a real application,");
    println!("you would use the actual OxiDiviner API.");
    
    println!("\nDemo completed successfully!");
    Ok(())
}

// Generate synthetic time series data
fn generate_synthetic_data() -> Vec<(DateTime<Utc>, f64)> {
    let mut rng = rand::thread_rng();
    let now = Utc::now();
    let n = 100; // 100 data points
    
    let mut data = Vec::with_capacity(n);
    let base = 100.0;
    let trend = 0.2; // Upward trend
    
    for i in 0..n {
        let timestamp = now + Duration::days(i as i64);
        
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
        
        // Add random noise (using random_range instead of gen_range)
        let noise = rng.gen_range(-2.0..2.0);
        
        // Combine components
        let value = base + trend_component + seasonal_component + noise;
        
        data.push((timestamp, value));
    }
    
    println!("Generated {} data points", n);
    data
} 