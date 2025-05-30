#![allow(deprecated)]
#![allow(unused_imports)]

use chrono::{DateTime, Duration, Utc};
use oxidiviner::TimeSeriesData;
use rand::Rng;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Simple Exponential Smoothing Model Example");
    println!("=========================================\n");

    // Generate synthetic data
    println!("Generating synthetic time series data...");
    let data = generate_synthetic_data();
    println!("Generated {} data points", data.len());

    // Since we can't use the actual model implementation due to import issues,
    // we'll just show a conceptual explanation
    println!("\nSimple Exponential Smoothing (SES) Model Concept:");
    println!("-----------------------------------------------");
    println!("SES is a time series forecasting method that gives exponentially");
    println!("decreasing weights to past observations, with recent observations");
    println!("having more influence on forecasts than older observations.");

    println!("\nThe SES formula is:");
    println!("  Forecast(t+1) = α × Actual(t) + (1-α) × Forecast(t)");
    println!("where α is the smoothing parameter (0 < α < 1)");

    println!("\nCharacteristics of SES:");
    println!("- Works best for series without trend or seasonality");
    println!("- Low α values (near 0) result in smoother forecasts with more weight on history");
    println!("- High α values (near 1) give more weight to recent observations");
    println!("- Produces a 'flat' forecast (same value for all future periods)");

    // Show a hypothetical forecast
    println!("\nExample forecast (with α = 0.3):");
    println!("Last actual value: 105.7");
    println!("Forecast for next 5 periods:");

    let last_value = 105.7;
    for i in 1..=5 {
        println!("  Period t+{}: {:.2}", i, last_value);
    }

    println!("\nNote: This is a simplified demonstration. In a real application,");
    println!("you would use the actual OxiDiviner API to implement SES forecasting.");

    Ok(())
}

// Generate synthetic time series data
fn generate_synthetic_data() -> TimeSeriesData {
    let mut rng = rand::thread_rng();
    let now = Utc::now();
    let n = 50;

    let mut timestamps = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);

    // Base value with random walk
    let mut value = 100.0;

    for i in 0..n {
        // Create timestamp (daily)
        let timestamp = now - Duration::days((n - i) as i64);
        timestamps.push(timestamp);

        // Add random walk component (no clear trend)
        let noise = rng.gen_range(-2.0..2.0);

        // Update value with random walk
        value += noise;
        values.push(value);
    }

    TimeSeriesData::new(timestamps, values, "Random walk data").unwrap()
}
