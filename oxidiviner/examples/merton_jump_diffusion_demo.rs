//! Merton Jump-Diffusion Model Demo
//!
//! This example demonstrates the advanced financial modeling capabilities
//! of the Merton Jump-Diffusion Model for:
//! - Market crash modeling and extreme event analysis
//! - Risk management with VaR calculations
//! - Options pricing with jump risk
//! - Monte Carlo simulation for scenario analysis

use oxidiviner::core::{Forecaster, TimeSeriesData};
use oxidiviner::models::financial::MertonJumpDiffusionModel;
use std::collections::HashMap;

fn main() -> oxidiviner::Result<()> {
    println!("🚀 Merton Jump-Diffusion Model Demo");
    println!("=====================================\n");

    // Example 1: Model Creation and Parameter Analysis
    println!("1. 📊 Model Creation and Parameter Analysis");
    println!("------------------------------------------");

    // Create model with typical equity market parameters
    let model = MertonJumpDiffusionModel::new_equity_default()?;
    println!("Created equity market model with default parameters:");
    println!("  • Drift (μ): {:.1}% annually", model.drift * 100.0);
    println!(
        "  • Volatility (σ): {:.1}% annually",
        model.volatility * 100.0
    );
    println!(
        "  • Jump intensity (λ): {:.1} jumps/year",
        model.jump_intensity
    );
    println!("  • Average jump size: {:.1}%", model.jump_mean * 100.0);
    println!("  • Jump volatility: {:.1}%", model.jump_volatility * 100.0);

    // Example 2: Monte Carlo Simulation and Path Analysis
    println!("\n2. 🎲 Monte Carlo Simulation and Path Analysis");
    println!("---------------------------------------------");

    let initial_price = 100.0;
    let horizon_days = 252; // 1 year
    let num_paths = 1000;

    println!(
        "Simulating {} price paths over {} trading days...",
        num_paths, horizon_days
    );

    let paths = model.simulate_paths(initial_price, horizon_days, num_paths, Some(42))?;

    // Analyze path statistics
    let final_prices: Vec<f64> = paths.iter().map(|path| *path.last().unwrap()).collect();
    let returns: Vec<f64> = final_prices
        .iter()
        .map(|&price| price / initial_price - 1.0)
        .collect();

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let std_return = (returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64)
        .sqrt();

    println!("  📈 Simulation Results:");
    println!("    Mean annual return: {:.2}%", mean_return * 100.0);
    println!("    Annual volatility: {:.2}%", std_return * 100.0);
    println!(
        "    Final price range: ${:.2} - ${:.2}",
        final_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        final_prices
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Example 3: Jump Event Analysis
    println!("\n3. ⚡ Jump Event Analysis");
    println!("------------------------");

    // Analyze jump events in first few paths
    let jump_threshold = 0.03; // 3% threshold for jump detection
    let mut total_jumps = 0;
    let mut large_crashes = 0;
    let mut large_rallies = 0;

    for (i, path) in paths.iter().take(100).enumerate() {
        let jump_events = model.extract_jump_events(path, jump_threshold);
        total_jumps += jump_events.len();

        for event in &jump_events {
            if event.size < -0.05 {
                // Crashes > 5%
                large_crashes += 1;
            } else if event.size > 0.05 {
                // Rallies > 5%
                large_rallies += 1;
            }
        }

        if i < 3 && !jump_events.is_empty() {
            println!("  📊 Path {} jump events:", i + 1);
            for event in jump_events {
                println!(
                    "    Time: {:.0} days, Size: {:.2}%, Impact: {:.1}σ",
                    event.time * 252.0,
                    event.size * 100.0,
                    event.relative_impact
                );
            }
        }
    }

    println!("  🎯 Jump Statistics (100 paths):");
    println!("    Total jumps detected: {}", total_jumps);
    println!("    Large crashes (>5%): {}", large_crashes);
    println!("    Large rallies (>5%): {}", large_rallies);
    println!(
        "    Average jumps per path: {:.1}",
        total_jumps as f64 / 100.0
    );

    // Example 4: Risk Management - VaR Calculation
    println!("\n4. 📉 Risk Management - Value at Risk (VaR)");
    println!("------------------------------------------");

    let portfolio_value = 1_000_000.0; // $1M portfolio
    let confidence_levels = vec![0.90, 0.95, 0.99];
    let time_horizons = vec![1.0 / 252.0, 1.0 / 52.0, 1.0 / 12.0]; // 1 day, 1 week, 1 month

    println!("  💼 Portfolio Value: $1,000,000");
    println!("  🎯 VaR Analysis with Jump Risk:");

    for &confidence in &confidence_levels {
        println!("\n    📊 {}% Confidence Level:", (confidence * 100.0) as u8);
        for &horizon in &time_horizons {
            let horizon_name = if horizon < 1.0 / 100.0 {
                format!("{} day", (horizon * 252.0_f64).round() as i32)
            } else if horizon < 1.0 / 10.0 {
                format!("{} week", (horizon * 52.0_f64).round() as i32)
            } else {
                format!("{} month", (horizon * 12.0_f64).round() as i32)
            };

            let var = model.calculate_var(portfolio_value, confidence, horizon, 5000)?;
            let var_percentage = var / portfolio_value * 100.0;

            println!(
                "      {} VaR: ${:.0} ({:.2}%)",
                horizon_name, var, var_percentage
            );
        }
    }

    // Example 5: Options Pricing with Jump Risk
    println!("\n5. 📈 Options Pricing with Jump Risk");
    println!("-----------------------------------");

    let spot_price = 100.0;
    let risk_free_rate = 0.05;
    let time_to_expiry = 0.25; // 3 months
    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];

    println!("  🎯 Option Pricing Parameters:");
    println!("    Spot Price: ${:.0}", spot_price);
    println!("    Risk-free Rate: {:.1}%", risk_free_rate * 100.0);
    println!("    Time to Expiry: {:.0} months", time_to_expiry * 12.0);

    println!("\n  📊 Call Option Prices:");
    println!("    Strike    Merton Price    Jump Premium");
    println!("    ------    ------------    ------------");

    for &strike in &strikes {
        let merton_price =
            model.option_price(spot_price, strike, time_to_expiry, risk_free_rate, true, 15)?;

        // Calculate Black-Scholes price for comparison (no jumps)
        let bs_price = black_scholes_call(
            spot_price,
            strike,
            time_to_expiry,
            risk_free_rate,
            model.volatility,
        );
        let jump_premium = merton_price - bs_price;

        println!(
            "    ${:3.0}       ${:7.2}        ${:6.2}",
            strike, merton_price, jump_premium
        );
    }

    // Example 6: Model Calibration to Market Data
    println!("\n6. 🔧 Model Calibration to Market Data");
    println!("-------------------------------------");

    // Generate synthetic market data with realistic characteristics
    let market_data = generate_realistic_market_data(252 * 2, initial_price).unwrap(); // 2 years of daily data

    println!("  📊 Calibrating to {} observations...", market_data.len());

    // Create a new model for calibration
    let mut calibration_model = MertonJumpDiffusionModel::new(
        0.08,        // Initial guess: 8% drift
        0.25,        // Initial guess: 25% volatility
        3.0,         // Initial guess: 3 jumps/year
        -0.02,       // Initial guess: -2% jump size
        0.04,        // Initial guess: 4% jump volatility
        1.0 / 252.0, // Daily time step
    )?;

    // Fit the model using the Forecaster trait
    let now = chrono::Utc::now();
    let timestamps: Vec<chrono::DateTime<chrono::Utc>> = (0..market_data.len())
        .map(|i| now - chrono::Duration::days((market_data.len() - i - 1) as i64))
        .collect();

    let ts_data = TimeSeriesData {
        name: "Market Data".to_string(),
        values: market_data,
        timestamps,
    };
    calibration_model.fit(&ts_data)?;

    if let Some(params) = calibration_model.get_estimated_parameters() {
        println!("  ✅ Calibration Results:");
        println!("    Estimated drift: {:.2}% annually", params.drift * 100.0);
        println!(
            "    Estimated volatility: {:.2}% annually",
            params.volatility * 100.0
        );
        println!(
            "    Estimated jump intensity: {:.2} jumps/year",
            params.jump_intensity
        );
        println!("    Estimated jump mean: {:.2}%", params.jump_mean * 100.0);
        println!(
            "    Estimated jump volatility: {:.2}%",
            params.jump_volatility * 100.0
        );
        println!("    Log-likelihood: {:.2}", params.log_likelihood);
    }

    if let Some(diagnostics) = calibration_model.get_diagnostics() {
        println!("  📈 Model Diagnostics:");
        println!("    AIC: {:.2}", diagnostics.aic);
        println!("    BIC: {:.2}", diagnostics.bic);
        println!("    Sample size: {}", diagnostics.sample_size);
        println!("    Converged: {}", diagnostics.converged);
    }

    // Example 7: Stress Testing and Scenario Analysis
    println!("\n7. 💥 Stress Testing and Scenario Analysis");
    println!("-----------------------------------------");

    let stress_scenarios = create_stress_scenarios();

    println!("  🎯 Stress Testing Scenarios:");
    for (name, scenario_model) in stress_scenarios {
        println!("\n    📊 {}", name);

        // Calculate 1-day 99% VaR for each scenario
        let stress_var = scenario_model.calculate_var(portfolio_value, 0.99, 1.0 / 252.0, 2000)?;
        let stress_percentage = stress_var / portfolio_value * 100.0;

        println!(
            "      1-day 99% VaR: ${:.0} ({:.2}%)",
            stress_var, stress_percentage
        );

        // Simulate a few paths to show extreme scenarios
        let stress_paths = scenario_model.simulate_paths(100.0, 5, 10, Some(123))?;
        let min_price = stress_paths
            .iter()
            .flat_map(|path| path.iter())
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = stress_paths
            .iter()
            .flat_map(|path| path.iter())
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        println!(
            "      5-day price range: ${:.2} - ${:.2}",
            min_price, max_price
        );
    }

    // Example 8: Performance Summary
    println!("\n8. 📊 Performance Summary");
    println!("------------------------");

    println!("  ✅ Model Capabilities Demonstrated:");
    println!("    • Market crash modeling with jump processes");
    println!("    • Risk management through VaR calculations");
    println!("    • Options pricing with jump risk premiums");
    println!("    • Parameter calibration from market data");
    println!("    • Stress testing for extreme scenarios");
    println!("    • Monte Carlo simulation for scenario analysis");

    println!("\n  🎯 Key Applications:");
    println!("    • Portfolio risk management");
    println!("    • Derivatives pricing and hedging");
    println!("    • Regulatory capital calculations");
    println!("    • Stress testing and scenario planning");
    println!("    • High-frequency trading strategies");

    Ok(())
}

/// Generate realistic market data with volatility clustering and occasional jumps
fn generate_realistic_market_data(n: usize, initial_price: f64) -> Result<Vec<f64>, &'static str> {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = rand::thread_rng();
    let mut prices = vec![initial_price];

    // Parameters for realistic market simulation
    let daily_drift = 0.0002; // ~5% annual
    let base_vol = 0.015; // ~24% annual
    let vol_persistence = 0.95;
    let vol_mean_reversion = 0.02;

    let mut current_vol = base_vol;

    for i in 1..n {
        // Volatility clustering (simplified GARCH-like)
        let vol_shock: f64 = rng.gen_range(-0.001..0.001);
        current_vol = vol_persistence * current_vol
            + (1.0 - vol_persistence) * base_vol
            + vol_mean_reversion * vol_shock;
        current_vol = current_vol.clamp(0.005, 0.08); // Bounds

        // Occasional jump events (simplified)
        let jump_prob = 0.02; // 2% chance per day
        let mut return_val = rng.sample(
            Normal::new(daily_drift, current_vol)
                .map_err(|_| "Failed to create normal distribution")?,
        );

        if rng.gen::<f64>() < jump_prob {
            let jump_size = rng.sample(
                Normal::new(-0.03, 0.02)
                    .map_err(|_| "Failed to create jump normal distribution")?,
            ); // Negative bias
            return_val += jump_size;
        }

        let next_price = prices[i - 1] * (1.0 + return_val);
        prices.push(next_price.max(0.01)); // Prevent negative prices
    }

    Ok(prices)
}

/// Simple Black-Scholes call option pricing for comparison
fn black_scholes_call(spot: f64, strike: f64, time: f64, rate: f64, volatility: f64) -> f64 {
    let d1 = ((spot / strike).ln() + (rate + 0.5 * volatility.powi(2)) * time)
        / (volatility * time.sqrt());
    let d2 = d1 - volatility * time.sqrt();

    let n_d1 = standard_normal_cdf(d1);
    let n_d2 = standard_normal_cdf(d2);

    spot * n_d1 - strike * (-rate * time).exp() * n_d2
}

/// Standard normal CDF approximation
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Create stress testing scenarios
fn create_stress_scenarios() -> HashMap<String, MertonJumpDiffusionModel> {
    let mut scenarios = HashMap::new();

    // Normal market conditions
    if let Ok(normal) = MertonJumpDiffusionModel::new(0.05, 0.20, 1.0, -0.02, 0.03, 1.0 / 252.0) {
        scenarios.insert("Normal Market".to_string(), normal);
    }

    // High volatility regime
    if let Ok(high_vol) = MertonJumpDiffusionModel::new(0.03, 0.35, 2.0, -0.03, 0.05, 1.0 / 252.0) {
        scenarios.insert("High Volatility".to_string(), high_vol);
    }

    // Crisis scenario with frequent crashes
    if let Ok(crisis) = MertonJumpDiffusionModel::new(-0.02, 0.45, 5.0, -0.08, 0.10, 1.0 / 252.0) {
        scenarios.insert("Financial Crisis".to_string(), crisis);
    }

    // Flash crash scenario
    if let Ok(flash_crash) =
        MertonJumpDiffusionModel::new(0.02, 0.25, 10.0, -0.15, 0.08, 1.0 / 252.0)
    {
        scenarios.insert("Flash Crash Risk".to_string(), flash_crash);
    }

    // Low volatility bull market
    if let Ok(bull) = MertonJumpDiffusionModel::new(0.12, 0.12, 0.5, 0.01, 0.02, 1.0 / 252.0) {
        scenarios.insert("Bull Market".to_string(), bull);
    }

    scenarios
}
