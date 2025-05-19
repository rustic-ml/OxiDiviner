use oxidiviner_garch::{GARCHModel, GJRGARCHModel, EGARCHModel};
use std::fs::File;
use std::io::{BufRead, BufReader};
use chrono::{NaiveDate, Utc, TimeZone};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Stock Market Volatility Analysis with GARCH Models");
    println!("------------------------------------------------\n");

    // For demonstration purposes, we'll use a synthetic price series
    // In a real application, you would load actual stock price data
    let (dates, prices) = load_sample_stock_data()?;
    
    // Calculate log returns
    let mut returns = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        returns.push((prices[i] / prices[i-1]).ln() * 100.0); // Percentage returns
    }
    
    println!("Loaded {} days of price data, {} returns", prices.len(), returns.len());
    println!("Sample period: {} to {}", dates.first().unwrap(), dates.last().unwrap());
    
    // Calculate descriptive statistics
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std_dev = variance.sqrt();
    let skewness = returns.iter().map(|&r| (r - mean).powi(3)).sum::<f64>() / (returns.len() as f64 * std_dev.powi(3));
    let kurtosis = returns.iter().map(|&r| (r - mean).powi(4)).sum::<f64>() / (returns.len() as f64 * variance.powi(2)) - 3.0;
    
    println!("\nDescriptive Statistics:");
    println!("Mean: {:.4}%", mean);
    println!("Standard Deviation: {:.4}%", std_dev);
    println!("Annualized Volatility: {:.2}%", std_dev * (252.0_f64).sqrt());
    println!("Skewness: {:.4}", skewness);
    println!("Excess Kurtosis: {:.4}", kurtosis);
    
    // Fit GARCH(1,1) model
    println!("\nFitting GARCH(1,1) model...");
    let mut garch = GARCHModel::new(1, 1, None)?;
    garch.fit(&returns, None)?;
    println!("{}", garch);
    
    // Get GARCH-implied volatility series
    let garch_variance = garch.fitted_variance.as_ref().unwrap();
    let annualized_volatility: Vec<f64> = garch_variance
        .iter()
        .map(|&v| (v * 252.0).sqrt() * 100.0) // Annualized volatility in percentage
        .collect();
    
    println!("\nVolatility Periods:");
    println!("Average Annualized Volatility: {:.2}%", 
             annualized_volatility.iter().sum::<f64>() / annualized_volatility.len() as f64);
    
    // Find high volatility periods
    let high_vol_threshold = annualized_volatility.iter().sum::<f64>() / annualized_volatility.len() as f64 * 1.5;
    println!("High Volatility Periods (> {:.2}%):", high_vol_threshold);
    
    for i in 0..annualized_volatility.len() {
        if annualized_volatility[i] > high_vol_threshold {
            // Skip printing consecutive days for brevity
            if i > 0 && annualized_volatility[i-1] > high_vol_threshold {
                continue;
            }
            println!("  {} - Volatility: {:.2}%", dates[i+1], annualized_volatility[i]);
        }
    }
    
    // Fit GJR-GARCH for leverage effect
    println!("\nFitting GJR-GARCH(1,1) model to detect leverage effect...");
    let mut gjr_garch = GJRGARCHModel::new(1, 1, None)?;
    gjr_garch.fit(&returns, None)?;
    println!("{}", gjr_garch);
    
    if gjr_garch.gamma[0] > 0.0 {
        println!("\nLeverage effect detected! Gamma = {:.4}", gjr_garch.gamma[0]);
        println!("Negative returns have {:.2}x more impact on volatility than positive returns", 
                 1.0 + gjr_garch.gamma[0] / gjr_garch.alpha[0]);
    } else {
        println!("\nNo significant leverage effect detected.");
    }
    
    // Forecast future volatility
    let forecast_horizon = 10;
    let garch_forecast = garch.forecast_variance(forecast_horizon)?;
    let gjr_forecast = gjr_garch.forecast_variance(forecast_horizon)?;
    
    println!("\nVolatility Forecast (Next {} Days):", forecast_horizon);
    println!("Day    GARCH     GJR-GARCH");
    println!("--- --------- -----------");
    for i in 0..forecast_horizon {
        let garch_annualized = (garch_forecast[i] * 252.0).sqrt() * 100.0;
        let gjr_annualized = (gjr_forecast[i] * 252.0).sqrt() * 100.0;
        println!("{:3} {:9.2}% {:11.2}%", i+1, garch_annualized, gjr_annualized);
    }
    
    println!("\nAnalysis Complete!");
    
    Ok(())
}

// Function to generate sample stock price data
// In a real application, replace this with code to load actual data
fn load_sample_stock_data() -> Result<(Vec<NaiveDate>, Vec<f64>), Box<dyn Error>> {
    // Generate synthetic price series with volatility changes
    let start_date = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
    let trading_days = 500;
    
    let mut dates = Vec::with_capacity(trading_days);
    let mut prices = Vec::with_capacity(trading_days);
    let mut price = 100.0; // Starting price
    
    use rand::prelude::*;
    use rand_distr::Normal;
    
    let mut rng = rand::thread_rng();
    
    // Create a price series with changing volatility regimes
    let mut volatility = 0.01;
    
    for i in 0..trading_days {
        // Add the date (skip weekends for simplicity)
        let date = start_date.checked_add_days(chrono::Days::new(i as u64 + (i / 5) * 2)).unwrap();
        dates.push(date);
        
        // Change volatility regime at certain points
        if i == 100 {
            volatility = 0.02; // Increase volatility
        } else if i == 200 {
            volatility = 0.015; // Decrease volatility
        } else if i == 300 {
            volatility = 0.03; // Market stress period
        } else if i == 400 {
            volatility = 0.01; // Return to normal
        }
        
        // Add some volatility clustering
        volatility = 0.9 * volatility + 0.1 * (volatility * rng.gen_range(0.5..1.5));
        
        // Create asymmetric returns (negative returns have larger impact)
        let normal = Normal::new(0.0003, volatility).unwrap(); // Small positive drift
        let mut return_val = normal.sample(&mut rng);
        
        // Introduce some large negative shocks
        if i == 150 || i == 320 || i == 321 {
            return_val = -0.04 - 0.02 * rng.gen::<f64>(); // Market crash days
        }
        
        // Update price with log return
        price *= (1.0 + return_val);
        prices.push(price);
    }
    
    Ok((dates, prices))
} 