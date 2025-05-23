#![allow(deprecated)]
#![allow(unused_imports)]

use chrono::{DateTime, Duration, Utc};
use oxidiviner_garch::{EGARCHModel, GARCHMModel, GARCHModel, GJRGARCHModel, RiskPremiumType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiDiviner GARCH Models Example");
    println!("-------------------------------\n");

    // Generate some synthetic returns data (random walk with some volatility clusters)
    let data = generate_sample_data(1000);
    println!("Generated {} sample data points", data.len());

    // Demonstrate GARCH(1,1) model
    println!("\n1. Standard GARCH(1,1) Model");
    println!("---------------------------");
    let mut garch = GARCHModel::new(1, 1, None)?;
    garch.fit(&data, None)?;
    println!("{}", garch);

    let forecast = garch.forecast_variance(5)?;
    println!("5-step ahead variance forecast: {:?}", forecast);

    // Demonstrate GJR-GARCH(1,1) model with asymmetric effects
    println!("\n2. GJR-GARCH(1,1) Model with Asymmetric Effects");
    println!("---------------------------------------------");
    let mut gjr_garch = GJRGARCHModel::new(1, 1, None)?;
    gjr_garch.fit(&data, None)?;
    println!("{}", gjr_garch);

    let (shock_values, variance_values) = gjr_garch.news_impact_curve(10, 2.0);
    println!("News Impact Curve (10 points):");
    for i in 0..shock_values.len() {
        println!(
            "  Shock: {:.2}, Variance: {:.4}",
            shock_values[i], variance_values[i]
        );
    }

    // Demonstrate EGARCH(1,1) model
    println!("\n3. EGARCH(1,1) Model");
    println!("------------------");
    let mut egarch = EGARCHModel::new(1, 1, None)?;
    egarch.fit(&data, None)?;
    println!("{}", egarch);

    // Demonstrate GARCH-M(1,1) model with risk premium
    println!("\n4. GARCH-M(1,1) Model with Risk Premium");
    println!("-------------------------------------");
    let mut garch_m = GARCHMModel::new(1, 1, RiskPremiumType::StdDev, None)?;
    garch_m.fit(&data, None)?;
    println!("{}", garch_m);

    let (mean_forecast, var_forecast) = garch_m.forecast(5)?;
    println!("5-step ahead mean and variance forecast:");
    for i in 0..mean_forecast.len() {
        println!(
            "  Step {}: Mean = {:.4}, Variance = {:.4}",
            i + 1,
            mean_forecast[i],
            var_forecast[i]
        );
    }

    Ok(())
}

// Function to generate sample data with volatility clustering
fn generate_sample_data(n: usize) -> Vec<f64> {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = rand::thread_rng();
    let mut returns = Vec::with_capacity(n);

    let mut volatility = 0.01;
    let mean = 0.0001; // Small positive drift

    for _ in 0..n {
        // Create volatility clustering by making volatility autocorrelated
        volatility = 0.9 * volatility + 0.1 * (0.005 + 0.015 * rng.gen::<f64>());

        // Add a safety check to prevent volatility from getting too large
        volatility = volatility.min(0.05);

        // Generate return with current volatility
        let normal = Normal::new(mean, volatility).unwrap();
        let return_value = normal.sample(&mut rng);

        returns.push(return_value);
    }

    returns
}
