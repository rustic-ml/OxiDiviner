//! Kou Double-Exponential Jump-Diffusion Model Demo
//!
//! This example demonstrates the advanced asymmetric jump modeling capabilities
//! of the Kou Double-Exponential Jump-Diffusion Model for:
//! - Asymmetric market crash and rally modeling
//! - Enhanced risk management with asymmetric tail distributions
//! - Comparison with Merton Jump-Diffusion model
//! - Advanced options pricing with realistic jump distributions

use chrono;
use oxidiviner::core::{Forecaster, TimeSeriesData};
use oxidiviner::models::financial::{KouJumpDiffusionModel, MertonJumpDiffusionModel};
use std::collections::HashMap;

fn main() -> oxidiviner::Result<()> {
    println!("🚀 Kou Double-Exponential Jump-Diffusion Model Demo");
    println!("===================================================\n");

    // Example 1: Model Creation and Asymmetric Parameter Analysis
    println!("1. 📊 Asymmetric Jump Model Creation and Analysis");
    println!("------------------------------------------------");

    // Create Kou model with asymmetric jump parameters
    let mut kou_model = KouJumpDiffusionModel::new_equity_default()?;
    println!("Created Kou equity market model with asymmetric parameters:");
    println!("  • Drift (μ): {:.1}% annually", kou_model.drift * 100.0);
    println!(
        "  • Volatility (σ): {:.1}% annually",
        kou_model.volatility * 100.0
    );
    println!(
        "  • Jump intensity (λ): {:.1} jumps/year",
        kou_model.jump_intensity
    );
    println!(
        "  • Upward jump probability: {:.1}%",
        kou_model.upward_jump_prob * 100.0
    );
    println!(
        "  • Upward jump rate (η₁): {:.1} (small rallies)",
        kou_model.upward_jump_rate
    );
    println!(
        "  • Downward jump rate (η₂): {:.1} (large crashes)",
        kou_model.downward_jump_rate
    );

    // Calculate expected jump sizes
    let expected_upward_jump = 1.0 / kou_model.upward_jump_rate;
    let expected_downward_jump = 1.0 / kou_model.downward_jump_rate;
    println!("\n  🎯 Expected Jump Characteristics:");
    println!(
        "    Expected upward jump: +{:.2}%",
        expected_upward_jump * 100.0
    );
    println!(
        "    Expected downward jump: -{:.2}%",
        expected_downward_jump * 100.0
    );
    println!(
        "    Asymmetry ratio: {:.1}x (crashes {:.1}x larger than rallies)",
        expected_upward_jump / expected_downward_jump,
        expected_upward_jump / expected_downward_jump
    );

    // Example 2: Monte Carlo Simulation with Asymmetric Analysis
    println!("\n2. 🎲 Monte Carlo Simulation with Asymmetric Jump Analysis");
    println!("----------------------------------------------------------");

    let initial_price = 100.0;
    let horizon_days = 252; // 1 year
    let num_paths = 1000;

    println!(
        "Simulating {} price paths over {} trading days...",
        num_paths, horizon_days
    );

    let kou_paths = kou_model.simulate_paths(initial_price, horizon_days, num_paths, Some(42))?;

    // Analyze path statistics
    let final_prices: Vec<f64> = kou_paths.iter().map(|path| *path.last().unwrap()).collect();
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

    // Calculate skewness (asymmetry measure)
    let skewness = returns
        .iter()
        .map(|r| ((r - mean_return) / std_return).powi(3))
        .sum::<f64>()
        / returns.len() as f64;

    println!("  📈 Kou Model Simulation Results:");
    println!("    Mean annual return: {:.2}%", mean_return * 100.0);
    println!("    Annual volatility: {:.2}%", std_return * 100.0);
    println!("    Skewness: {:.3} (negative = left tail bias)", skewness);
    println!(
        "    Final price range: ${:.2} - ${:.2}",
        final_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        final_prices
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Example 3: Asymmetric Jump Event Analysis
    println!("\n3. ⚡ Asymmetric Jump Event Analysis");
    println!("-----------------------------------");

    let jump_threshold = 0.03; // 3% threshold for jump detection
    let mut total_jumps = 0;
    let mut upward_jumps = 0;
    let mut downward_jumps = 0;
    let mut large_crashes = 0;
    let mut large_rallies = 0;

    for (i, path) in kou_paths.iter().take(100).enumerate() {
        let jump_events = kou_model.extract_jump_events(path, jump_threshold);
        total_jumps += jump_events.len();

        for event in &jump_events {
            if event.is_upward {
                upward_jumps += 1;
                if event.size > 0.05 {
                    // Rallies > 5%
                    large_rallies += 1;
                }
            } else {
                downward_jumps += 1;
                if event.size < -0.05 {
                    // Crashes > 5%
                    large_crashes += 1;
                }
            }
        }

        if i < 3 && !jump_events.is_empty() {
            println!("  📊 Path {} asymmetric jump events:", i + 1);
            for event in jump_events {
                let direction = if event.is_upward { "Rally" } else { "Crash" };
                println!(
                    "    {} at day {:.0}: {:.2}%, Impact: {:.1}σ",
                    direction,
                    event.time * 252.0,
                    event.size * 100.0,
                    event.relative_impact
                );
            }
        }
    }

    println!("  🎯 Asymmetric Jump Statistics (100 paths):");
    println!("    Total jumps detected: {}", total_jumps);
    println!(
        "    Upward jumps (rallies): {} ({:.1}%)",
        upward_jumps,
        upward_jumps as f64 / total_jumps as f64 * 100.0
    );
    println!(
        "    Downward jumps (crashes): {} ({:.1}%)",
        downward_jumps,
        downward_jumps as f64 / total_jumps as f64 * 100.0
    );
    println!("    Large crashes (>5%): {}", large_crashes);
    println!("    Large rallies (>5%): {}", large_rallies);
    println!(
        "    Crash/Rally ratio: {:.1}x",
        large_crashes as f64 / large_rallies.max(1) as f64
    );

    // Example 4: Model Comparison - Kou vs Merton
    println!("\n4. ⚖️  Model Comparison: Kou vs Merton Jump-Diffusion");
    println!("-----------------------------------------------------");

    // Create comparable Merton model
    let mut merton_model = MertonJumpDiffusionModel::new_equity_default()?;

    println!("  🔄 Comparing models with same basic parameters...");

    // Simulate paths with both models
    let merton_paths = merton_model.simulate_paths(initial_price, horizon_days, 500, Some(123))?;
    let kou_paths_comp = kou_model.simulate_paths(initial_price, horizon_days, 500, Some(123))?;

    // Calculate statistics for comparison
    let merton_final_prices: Vec<f64> = merton_paths
        .iter()
        .map(|path| *path.last().unwrap())
        .collect();
    let kou_final_prices: Vec<f64> = kou_paths_comp
        .iter()
        .map(|path| *path.last().unwrap())
        .collect();

    let merton_returns: Vec<f64> = merton_final_prices
        .iter()
        .map(|&price| price / initial_price - 1.0)
        .collect();
    let kou_returns: Vec<f64> = kou_final_prices
        .iter()
        .map(|&price| price / initial_price - 1.0)
        .collect();

    let merton_skew = calculate_skewness(&merton_returns);
    let kou_skew = calculate_skewness(&kou_returns);

    let merton_var_95 = calculate_percentile(&merton_returns, 5.0) * initial_price;
    let kou_var_95 = calculate_percentile(&kou_returns, 5.0) * initial_price;

    println!("  📊 Model Comparison Results:");
    println!("    {:20} {:>10} {:>10}", "Metric", "Merton", "Kou");
    println!("    {:20} {:>10} {:>10}", "----", "------", "---");
    println!(
        "    {:20} {:>9.3} {:>9.3}",
        "Skewness", merton_skew, kou_skew
    );
    println!(
        "    {:20} {:>9.1} {:>9.1}",
        "95% VaR ($)", -merton_var_95, -kou_var_95
    );
    println!(
        "    {:20} {:>9.2} {:>9.2}",
        "Min Price ($)",
        merton_final_prices
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b)),
        kou_final_prices
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b))
    );

    // Example 5: Enhanced Risk Management with Asymmetric VaR
    println!("\n5. 📉 Enhanced Risk Management with Asymmetric VaR");
    println!("-------------------------------------------------");

    let portfolio_value = 1_000_000.0; // $1M portfolio
    let confidence_levels = vec![0.90, 0.95, 0.99];
    let time_horizons = vec![1.0 / 252.0, 1.0 / 52.0, 1.0 / 12.0]; // 1 day, 1 week, 1 month

    println!("  💼 Portfolio Value: $1,000,000");
    println!("  🎯 Asymmetric VaR Analysis:");

    for &confidence in &confidence_levels {
        println!("\n    📊 {}% Confidence Level:", (confidence * 100.0) as u8);
        println!(
            "      {:10} {:>12} {:>12} {:>10}",
            "Horizon", "Kou VaR", "Merton VaR", "Difference"
        );
        println!(
            "      {:10} {:>12} {:>12} {:>10}",
            "-------", "-------", "----------", "----------"
        );

        for &horizon in &time_horizons {
            let horizon_name = if horizon < 1.0 / 100.0 {
                format!("{} day", (horizon * 252.0_f64).round() as i32)
            } else if horizon < 1.0 / 10.0 {
                format!("{} week", (horizon * 52.0_f64).round() as i32)
            } else {
                format!("{} month", (horizon * 12.0_f64).round() as i32)
            };

            let kou_var = kou_model.calculate_var(portfolio_value, confidence, horizon, 5000)?;
            let merton_var =
                merton_model.calculate_var(portfolio_value, confidence, horizon, 5000)?;
            let difference = ((kou_var - merton_var) / merton_var) * 100.0;

            println!(
                "      {:10} ${:>10.0} ${:>10.0} {:>8.1}%",
                horizon_name, kou_var, merton_var, difference
            );
        }
    }

    // Example 6: Advanced Options Pricing Comparison
    println!("\n6. 📈 Advanced Options Pricing with Asymmetric Jumps");
    println!("----------------------------------------------------");

    let spot_price = 100.0;
    let risk_free_rate = 0.05;
    let time_to_expiry = 0.25; // 3 months
    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];

    println!("  🎯 Option Pricing Parameters:");
    println!("    Spot Price: ${:.0}", spot_price);
    println!("    Risk-free Rate: {:.1}%", risk_free_rate * 100.0);
    println!("    Time to Expiry: {:.0} months", time_to_expiry * 12.0);

    println!("\n  📊 Call Option Prices Comparison:");
    println!(
        "    {:6} {:>12} {:>12} {:>10} {:>12}",
        "Strike", "Kou Price", "Merton Price", "Difference", "Asymm Premium"
    );
    println!(
        "    {:6} {:>12} {:>12} {:>10} {:>12}",
        "------", "---------", "------------", "----------", "-------------"
    );

    for &strike in &strikes {
        let kou_price =
            kou_model.option_price(spot_price, strike, time_to_expiry, risk_free_rate, true, 15)?;
        let merton_price = merton_model.option_price(
            spot_price,
            strike,
            time_to_expiry,
            risk_free_rate,
            true,
            15,
        )?;

        let difference = kou_price - merton_price;
        let bs_price = black_scholes_call(
            spot_price,
            strike,
            time_to_expiry,
            risk_free_rate,
            kou_model.volatility,
        );
        let asymm_premium = kou_price - bs_price;

        println!(
            "    ${:3.0}    ${:>9.2}    ${:>9.2}    ${:>7.2}    ${:>9.2}",
            strike, kou_price, merton_price, difference, asymm_premium
        );
    }

    // Example 7: Model Calibration and Parameter Estimation
    println!("\n7. 🔧 Asymmetric Model Calibration");
    println!("----------------------------------");

    // Generate synthetic market data with asymmetric characteristics
    let market_data = generate_asymmetric_market_data(252 * 2, initial_price).unwrap(); // 2 years of daily data

    println!(
        "  📊 Calibrating to {} observations with asymmetric jumps...",
        market_data.len()
    );

    // Create models for calibration
    let mut kou_calibration = KouJumpDiffusionModel::new(
        0.08,        // Initial guess: 8% drift
        0.25,        // Initial guess: 25% volatility
        4.0,         // Initial guess: 4 jumps/year
        0.25,        // Initial guess: 25% upward probability
        15.0,        // Initial guess: η₁ = 15
        35.0,        // Initial guess: η₂ = 35
        1.0 / 252.0, // Daily time step
    )?;

    let mut merton_calibration =
        MertonJumpDiffusionModel::new(0.08, 0.25, 4.0, -0.02, 0.04, 1.0 / 252.0)?;

    // Fit both models
    let now = chrono::Utc::now();
    let timestamps: Vec<chrono::DateTime<chrono::Utc>> = (0..market_data.len())
        .map(|i| now - chrono::Duration::days((market_data.len() - i - 1) as i64))
        .collect();

    let ts_data = TimeSeriesData {
        name: "Asymmetric Market Data".to_string(),
        values: market_data,
        timestamps,
    };

    kou_calibration.fit(&ts_data)?;
    merton_calibration.fit(&ts_data)?;

    if let Some(kou_params) = kou_calibration.get_estimated_parameters() {
        println!("  ✅ Kou Model Calibration Results:");
        println!(
            "    Estimated drift: {:.2}% annually",
            kou_params.drift * 100.0
        );
        println!(
            "    Estimated volatility: {:.2}% annually",
            kou_params.volatility * 100.0
        );
        println!(
            "    Estimated jump intensity: {:.2} jumps/year",
            kou_params.jump_intensity
        );
        println!(
            "    Upward jump probability: {:.1}%",
            kou_params.upward_jump_prob * 100.0
        );
        println!(
            "    Upward jump rate (η₁): {:.1}",
            kou_params.upward_jump_rate
        );
        println!(
            "    Downward jump rate (η₂): {:.1}",
            kou_params.downward_jump_rate
        );
        println!("    Log-likelihood: {:.2}", kou_params.log_likelihood);
    }

    if let Some(merton_params) = merton_calibration.get_estimated_parameters() {
        println!("\n  ✅ Merton Model Calibration Results:");
        println!("    Log-likelihood: {:.2}", merton_params.log_likelihood);
    }

    // Model comparison
    if let (Some(kou_diag), Some(merton_diag)) = (
        kou_calibration.get_diagnostics(),
        merton_calibration.get_diagnostics(),
    ) {
        println!("\n  📈 Model Comparison:");
        println!("    {:15} {:>10} {:>10}", "Metric", "Kou", "Merton");
        println!("    {:15} {:>10} {:>10}", "------", "---", "------");
        println!(
            "    {:15} {:>10.2} {:>10.2}",
            "AIC", kou_diag.aic, merton_diag.aic
        );
        println!(
            "    {:15} {:>10.2} {:>10.2}",
            "BIC", kou_diag.bic, merton_diag.bic
        );
        println!(
            "    {:15} {:>10} {:>10}",
            "Parameters", kou_diag.num_params, merton_diag.num_params
        );

        let aic_improvement = merton_diag.aic - kou_diag.aic;
        println!(
            "    AIC Improvement: {:.2} (Kou {} better)",
            aic_improvement.abs(),
            if aic_improvement > 0.0 {
                "is"
            } else {
                "is not"
            }
        );
    }

    // Example 8: Performance Summary
    println!("\n8. 📊 Asymmetric Model Performance Summary");
    println!("------------------------------------------");

    println!("  ✅ Kou Model Capabilities Demonstrated:");
    println!("    • Asymmetric jump distributions (different crash/rally characteristics)");
    println!("    • Enhanced risk management with tail asymmetry");
    println!("    • Superior options pricing for realistic market conditions");
    println!("    • Advanced parameter calibration (6 parameters vs 5 for Merton)");
    println!("    • Stress testing with asymmetric scenarios");
    println!("    • Model comparison and selection capabilities");

    println!("\n  🎯 Key Advantages over Merton Model:");
    println!("    • Captures market crash asymmetry (larger downward jumps)");
    println!("    • More accurate tail risk modeling");
    println!("    • Better fit to real market data with skewed returns");
    println!("    • Enhanced derivatives pricing accuracy");
    println!("    • Superior stress testing for extreme scenarios");

    println!("\n  💼 Professional Applications:");
    println!("    • Regulatory capital calculations with asymmetric risk");
    println!("    • Derivatives trading and market making");
    println!("    • Portfolio optimization with crash protection");
    println!("    • Algorithmic trading with jump-aware strategies");
    println!("    • Risk management for institutional portfolios");

    Ok(())
}

/// Calculate skewness of a return series
fn calculate_skewness(returns: &[f64]) -> f64 {
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return 0.0;
    }

    let skewness = returns
        .iter()
        .map(|r| ((r - mean) / std_dev).powi(3))
        .sum::<f64>()
        / n;

    skewness
}

/// Calculate percentile of a return series
fn calculate_percentile(returns: &[f64], percentile: f64) -> f64 {
    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let index = (percentile / 100.0 * sorted_returns.len() as f64) as usize;
    sorted_returns[index.min(sorted_returns.len() - 1)]
}

/// Generate asymmetric market data with realistic crash/rally patterns
fn generate_asymmetric_market_data(n: usize, initial_price: f64) -> Result<Vec<f64>, &'static str> {
    use rand::prelude::*;
    use rand_distr::{Exp, Normal};

    let mut rng = rand::thread_rng();
    let mut prices = vec![initial_price];

    // Parameters for asymmetric market simulation
    let daily_drift = 0.0003; // ~7.5% annual
    let base_vol = 0.018; // ~28% annual
    let vol_persistence = 0.92;
    let vol_mean_reversion = 0.025;

    // Asymmetric jump parameters
    let jump_prob = 0.025; // 2.5% chance per day
    let upward_jump_prob = 0.3; // 30% of jumps are upward
    let upward_exp = Exp::new(12.0).map_err(|_| "Failed to create upward exponential")?;
    let downward_exp = Exp::new(20.0).map_err(|_| "Failed to create downward exponential")?;

    let mut current_vol = base_vol;

    for i in 1..n {
        // Volatility clustering
        let vol_shock: f64 = rng.gen_range(-0.002..0.002);
        current_vol = vol_persistence * current_vol
            + (1.0 - vol_persistence) * base_vol
            + vol_mean_reversion * vol_shock;
        current_vol = current_vol.max(0.008).min(0.12); // Bounds

        // Normal return component
        let mut return_val = rng
            .sample(Normal::new(daily_drift, current_vol).map_err(|_| "Failed to create normal")?);

        // Asymmetric jump events
        if rng.gen::<f64>() < jump_prob {
            let is_upward = rng.gen::<f64>() < upward_jump_prob;
            let jump_size = if is_upward {
                upward_exp.sample(&mut rng) * 0.01 // Small positive jumps
            } else {
                -downward_exp.sample(&mut rng) * 0.01 // Large negative jumps
            };
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
