//! Quick test of the new enhanced API modules
//!
//! This is a minimal test to see if the basic functionality works

use chrono::{Duration, Utc};
use oxidiviner::quick;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing enhanced API modules...");

    // Generate simple test data
    let start_date = Utc::now() - Duration::days(20);
    let _timestamps: Vec<_> = (0..20).map(|i| start_date + Duration::days(i)).collect();
    let values: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64) * 1.5).collect();

    println!("Generated {} data points", values.len());

    // Test the quick module
    match quick::values_only_forecast(values, 3) {
        Ok((forecast, model)) => {
            println!("✓ Quick forecast successful!");
            println!("  Model used: {}", model);
            println!("  Forecast: {:?}", forecast);
        }
        Err(e) => {
            println!("✗ Quick forecast failed: {}", e);
        }
    }

    Ok(())
}
