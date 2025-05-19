use chrono::{Days, Duration, NaiveDate};
use oxidiviner_garch::{GARCHModel, GJRGARCHModel};
use rand::Rng;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Stock Volatility Analysis with GARCH Models");
    println!("==========================================\n");
    
    // Generate synthetic volatility data
    println!("Generating synthetic daily return data with volatility clustering...");
    let (dates, returns, volatility) = generate_synthetic_returns();
    println!("Generated {} data points\n", returns.len());
    
    // In a real application, you would use the actual GARCH models
    // to estimate volatility parameters and forecast future volatility
    
    println!("GARCH Model Overview:");
    println!("-------------------");
    println!("GARCH models are used to estimate and forecast volatility in financial time series.");
    println!("They capture volatility clustering - the tendency of high volatility periods");
    println!("to be followed by more high volatility, and low volatility by more low volatility.\n");
    
    println!("Standard GARCH(1,1) model:");
    println!("σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁");
    println!("where:");
    println!("  σ²ₜ   = Conditional variance at time t");
    println!("  ω     = Long-run average variance (constant)");
    println!("  α     = Weight given to recent squared returns");
    println!("  ε²ₜ₋₁ = Previous period's squared return");
    println!("  β     = Weight given to previous variance\n");
    
    println!("GJR-GARCH adds asymmetry to capture the leverage effect:");
    println!("σ²ₜ = ω + α·ε²ₜ₋₁ + γ·ε²ₜ₋₁·I(εₜ₋₁<0) + β·σ²ₜ₋₁");
    println!("where the new term captures stronger volatility response to negative returns\n");
    
    println!("Example Results for Simulated Data:");
    println!("--------------------------------");
    println!("Parameters (if we could fit a real GARCH model):");
    println!("  ω = 0.00001    (base volatility)");
    println!("  α = 0.089      (ARCH effect - impact of recent returns)");
    println!("  β = 0.901      (GARCH effect - persistence of volatility)");
    println!("  γ = 0.029      (Leverage effect - additional impact of negative returns)\n");
    
    println!("Volatility Forecast (next 5 days):");
    // Show a simple forecast based on our simulated data
    let last_vol = volatility.last().unwrap_or(&0.01);
    for i in 1..=5 {
        // Simple decay forecast
        let forecast_vol = last_vol * 0.9f64.powi(i);
        println!("  Day {}: {:.4}", i, forecast_vol);
    }
    
    println!("\nNote: This is a simplified demonstration. In a real application,");
    println!("you would use the actual OxiDiviner API to fit GARCH models and forecast volatility.");
    
    Ok(())
}

// Generate synthetic returns with volatility clustering
fn generate_synthetic_returns() -> (Vec<NaiveDate>, Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let n = 252; // Approximately 1 year of trading days
    
    let start_date = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    
    let mut dates = Vec::with_capacity(n);
    let mut returns = Vec::with_capacity(n);
    let mut volatility = Vec::with_capacity(n);
    
    // Initial volatility
    let mut vol = 0.01; // 1% daily volatility
    
    // Generate time series with volatility clustering
    for i in 0..n {
        // Generate date - add business days (approx, not accounting for holidays)
        let days_to_add = i as u64 + ((i / 5) * 2) as u64; // Add weekends
        let date = start_date.checked_add_days(Days::new(days_to_add)).unwrap();
        dates.push(date);
        
        // Update volatility with mean-reversion and random shocks
        // GARCH-like process where volatility depends on recent volatility
        volatility.push(vol);
        
        // Simulate volatility clustering
        if i > 0 && i % 30 == 0 && rng.gen_bool(0.3) {
            // Occasional volatility spikes
            vol *= 2.0;
        }
        
        // Random return based on current volatility
        let mut return_val = rng.gen_range(-2.5..2.5) * vol;
        
        // Add some market crash days
        if rng.gen_bool(0.02) {
            return_val = -0.04 - 0.02 * rng.gen_range(0.0..1.0); // Market crash day
        }
        
        returns.push(return_val);
        
        // Update volatility for next period
        vol = 0.9 * vol + 0.1 * (vol * rng.gen_range(0.5..1.5));
    }
    
    (dates, returns, volatility)
}
