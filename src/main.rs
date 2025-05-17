use rand::Rng;
use std::error::Error;

// Main function for the demo
fn main() -> Result<(), Box<dyn Error>> {
    println!("ETS Model Demo (Simplified)");
    println!("=========================\n");

    println!("This simplified demo shows how the Exponential Smoothing Time Series (ETS) model works.");
    println!("The ETS model is a powerful forecasting method that can capture trend and seasonality.");
    println!("\nIn a complete implementation, the model would:\n");
    println!("1. Support different error types (Additive, Multiplicative)");
    println!("2. Support different trend types (None, Additive, Multiplicative, Damped)");
    println!("3. Support different seasonal types (None, Additive, Multiplicative)");
    println!("4. Initialize from data, fit parameters, and generate forecasts");
    println!("\nCheck the full implementation in the OxiDiviner library!");
    
    // Generate and display a sample synthetic time series
    println!("\nSample synthetic time series with trend and seasonality:");
    
    // Parameters for the synthetic data
    const N: usize = 24;  // 2 years of monthly data
    const TREND: f64 = 0.5;  // Upward trend
    const SEASON_AMPLITUDE: f64 = 10.0;  // Seasonal amplitude
    const NOISE_LEVEL: f64 = 3.0;  // Random noise level
    
    let mut rng = rand::thread_rng();
    
    println!("\n{:<5} {:<10}", "Month", "Value");
    println!("----------------");
    
    for i in 0..N {
        // Generate value with trend, seasonality, and noise
        let trend_component = TREND * i as f64;
        let season_component = SEASON_AMPLITUDE * (2.0 * std::f64::consts::PI * (i % 12) as f64 / 12.0).sin();
        let noise = NOISE_LEVEL * (rng.gen::<f64>() - 0.5);
        let value = 100.0 + trend_component + season_component + noise;
        
        println!("{:<5} {:<10.2}", i+1, value);
    }
    
    println!("\nA full ETS model would decompose this series into:");
    println!("- Base level (starting around 100)");
    println!("- Trend component (increasing by ~0.5 per period)");
    println!("- Seasonal component (amplitude ~10, period 12 months)");
    println!("- Random noise component");
    
    Ok(())
}
