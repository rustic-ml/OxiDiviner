//! Heston Stochastic Volatility Model - Comprehensive Demo
//!
//! This demo showcases the Heston Stochastic Volatility Model's capabilities for:
//! - Monte Carlo simulation with correlated price and volatility processes
//! - European options pricing with stochastic volatility
//! - Value-at-Risk (VaR) calculation incorporating volatility clustering
//! - Implied volatility surface generation
//! - Model diagnostics and parameter analysis
//!
//! The Heston model is the gold standard for stochastic volatility modeling in quantitative finance.

use oxidiviner::models::financial::HestonStochasticVolatilityModel;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("🔬 HESTON STOCHASTIC VOLATILITY MODEL - COMPREHENSIVE DEMO");
    println!("{}", "=".repeat(80));

    // 1. Model Creation and Parameter Analysis
    println!("\n1️⃣  MODEL CREATION AND PARAMETER ANALYSIS");
    println!("{}", "-".repeat(50));

    let model = HestonStochasticVolatilityModel::new(
        0.05,        // 5% annual drift (μ)
        2.0,         // Mean reversion speed (κ)
        0.04,        // Long-term variance (θ) = 20% long-term volatility
        0.3,         // Volatility of volatility (σᵥ)
        -0.7,        // Correlation (ρ) - leverage effect
        0.04,        // Initial variance (V₀) = 20% initial volatility
        1.0 / 252.0, // Daily time step
    )?;

    println!("Model Parameters:");
    println!("  • Drift (μ): {:.1}%", model.drift * 100.0);
    println!("  • Mean reversion speed (κ): {:.2}", model.kappa);
    println!(
        "  • Long-term volatility: {:.1}%",
        (model.theta.sqrt() * 100.0)
    );
    println!(
        "  • Volatility of volatility (σᵥ): {:.1}%",
        model.vol_of_vol * 100.0
    );
    println!("  • Correlation (ρ): {:.1}%", model.correlation * 100.0);
    println!(
        "  • Initial volatility: {:.1}%",
        (model.initial_variance.sqrt() * 100.0)
    );

    // Check Feller condition
    let feller_ok = model.check_feller_condition();
    println!(
        "  • Feller condition (2κθ > σᵥ²): {} ({})",
        if feller_ok {
            "✅ Satisfied"
        } else {
            "❌ Violated"
        },
        if feller_ok {
            "variance stays positive"
        } else {
            "variance may hit zero"
        }
    );

    // 2. Monte Carlo Simulation
    println!("\n2️⃣  MONTE CARLO SIMULATION");
    println!("{}", "-".repeat(50));

    let initial_price = 100.0;
    let horizon_days = 252; // 1 year
    let num_paths = 5000;

    println!(
        "Simulating {} paths over {} trading days...",
        num_paths, horizon_days
    );

    let paths = model.simulate_paths(initial_price, horizon_days, num_paths, Some(42))?;

    // Analyze simulation results
    let final_prices: Vec<f64> = paths.iter().map(|p| *p.prices.last().unwrap()).collect();
    let final_volatilities: Vec<f64> = paths
        .iter()
        .map(|p| p.variances.last().unwrap().sqrt())
        .collect();

    let mean_final_price = final_prices.iter().sum::<f64>() / num_paths as f64;
    let std_final_price = {
        let variance = final_prices
            .iter()
            .map(|p| (p - mean_final_price).powi(2))
            .sum::<f64>()
            / num_paths as f64;
        variance.sqrt()
    };

    let mean_final_vol = final_volatilities.iter().sum::<f64>() / num_paths as f64;
    let std_final_vol = {
        let variance = final_volatilities
            .iter()
            .map(|v| (v - mean_final_vol).powi(2))
            .sum::<f64>()
            / num_paths as f64;
        variance.sqrt()
    };

    println!("Final Price Statistics:");
    println!("  • Mean: ${:.2}", mean_final_price);
    println!("  • Standard Deviation: ${:.2}", std_final_price);
    println!(
        "  • Range: ${:.2} - ${:.2}",
        final_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        final_prices
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    println!("Final Volatility Statistics:");
    println!("  • Mean: {:.1}%", mean_final_vol * 100.0);
    println!("  • Standard Deviation: {:.1}%", std_final_vol * 100.0);
    println!(
        "  • Range: {:.1}% - {:.1}%",
        final_volatilities
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b))
            * 100.0,
        final_volatilities
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            * 100.0
    );

    // Analyze path statistics
    println!("\nPath Analysis:");
    let sample_path = &paths[0];
    let price_returns: Vec<f64> = sample_path
        .prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    let realized_vol = {
        let mean_return = price_returns.iter().sum::<f64>() / price_returns.len() as f64;
        let variance = price_returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / price_returns.len() as f64;
        (variance * 252.0).sqrt() // Annualized
    };

    println!(
        "  • Realized volatility (sample path): {:.1}%",
        realized_vol * 100.0
    );
    println!(
        "  • Average volatility over time: {:.1}%",
        sample_path.variances.iter().map(|v| v.sqrt()).sum::<f64>()
            / sample_path.variances.len() as f64
            * 100.0
    );

    // 3. European Options Pricing
    println!("\n3️⃣  EUROPEAN OPTIONS PRICING");
    println!("{}", "-".repeat(50));

    let spot_price = 100.0;
    let risk_free_rate = 0.05;
    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let expiries = vec![1.0 / 12.0, 3.0 / 12.0, 6.0 / 12.0, 1.0]; // 1M, 3M, 6M, 1Y

    println!(
        "Options Pricing (Spot = ${:.0}, r = {:.1}%):",
        spot_price,
        risk_free_rate * 100.0
    );
    println!("Strike    1M      3M      6M      1Y");
    println!("{}", "-".repeat(40));

    for &strike in &strikes {
        print!("{:>6.0}", strike);
        for &expiry in &expiries {
            let option_price =
                model.option_price(spot_price, strike, expiry, risk_free_rate, true)?;
            print!("   {:>6.2}", option_price);
        }
        println!();
    }

    // Compare ATM options for different expiries
    println!("\nAt-The-Money (ATM) Call Options:");
    for &expiry in &expiries {
        let atm_price = model.option_price(spot_price, spot_price, expiry, risk_free_rate, true)?;
        let time_value = atm_price;
        println!(
            "  • {:.0}M expiry: ${:.2} (time value: ${:.2})",
            expiry * 12.0,
            atm_price,
            time_value
        );
    }

    // 4. Risk Management - Value at Risk (VaR)
    println!("\n4️⃣  RISK MANAGEMENT - VALUE AT RISK (VaR)");
    println!("{}", "-".repeat(50));

    let portfolio_value = 1_000_000.0;
    let confidence_levels = vec![0.90, 0.95, 0.99];
    let time_horizons = vec![1.0 / 252.0, 5.0 / 252.0, 21.0 / 252.0]; // 1 day, 1 week, 1 month

    println!("Portfolio Value: ${:.0}", portfolio_value);
    println!("Time Horizon    90% VaR    95% VaR    99% VaR");
    println!("{}", "-".repeat(50));

    for &horizon in &time_horizons {
        let horizon_days = (horizon * 252.0_f64).round() as i32;
        print!(
            "{:>8} day{}",
            horizon_days,
            if horizon_days == 1 { " " } else { "s" }
        );

        for &confidence in &confidence_levels {
            let var = model.calculate_var(portfolio_value, confidence, horizon, 5000)?;
            print!("   ${:>7.0}", var);
        }
        println!();
    }

    // VaR analysis
    println!("\nVaR Analysis with Stochastic Volatility:");
    let daily_var_95 = model.calculate_var(portfolio_value, 0.95, 1.0 / 252.0, 10000)?;
    let weekly_var_95 = model.calculate_var(portfolio_value, 0.95, 5.0 / 252.0, 10000)?;
    let scaling_factor = weekly_var_95 / daily_var_95;
    let sqrt_5 = 5.0_f64.sqrt();

    println!("  • Daily 95% VaR: ${:.0}", daily_var_95);
    println!("  • Weekly 95% VaR: ${:.0}", weekly_var_95);
    println!("  • Actual scaling factor: {:.2}x", scaling_factor);
    println!("  • √5 scaling (constant vol): {:.2}x", sqrt_5);
    println!(
        "  • Stochastic vol effect: {:.1}% {} than constant vol",
        (scaling_factor / sqrt_5 - 1.0).abs() * 100.0,
        if scaling_factor > sqrt_5 {
            "higher"
        } else {
            "lower"
        }
    );

    // 5. Implied Volatility Surface
    println!("\n5️⃣  IMPLIED VOLATILITY SURFACE");
    println!("{}", "-".repeat(50));

    let surface_strikes = vec![85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0];
    let surface_expiries = vec![0.25, 0.5, 1.0]; // 3M, 6M, 1Y

    println!("Generating implied volatility surface...");
    let vol_surface = model.volatility_surface(
        spot_price,
        risk_free_rate,
        &surface_strikes,
        &surface_expiries,
    )?;

    println!("Implied Volatility Surface (%):");
    println!("Strike   3M     6M     1Y");
    println!("{}", "-".repeat(30));

    for &strike in &surface_strikes {
        print!("{:>6.0}", strike);
        for &expiry in &surface_expiries {
            let point = vol_surface
                .iter()
                .find(|p| (p.strike - strike).abs() < 0.1 && (p.expiry - expiry).abs() < 0.01)
                .unwrap();
            print!("   {:>5.1}", point.implied_volatility * 100.0);
        }
        println!();
    }

    // Analyze volatility smile/skew
    println!("\nVolatility Smile Analysis (6M expiry):");
    let six_month_vols: Vec<_> = surface_strikes
        .iter()
        .map(|&strike| {
            let point = vol_surface
                .iter()
                .find(|p| (p.strike - strike).abs() < 0.1 && (p.expiry - 0.5).abs() < 0.01)
                .unwrap();
            (strike, point.implied_volatility)
        })
        .collect();

    let atm_vol = six_month_vols
        .iter()
        .find(|(s, _)| (*s - spot_price).abs() < 0.1)
        .map(|(_, v)| *v)
        .unwrap_or(0.2);

    for (strike, vol) in &six_month_vols {
        let moneyness = strike / spot_price;
        let vol_diff = (vol - atm_vol) * 100.0;
        println!(
            "  • K={:.0} (M={:.2}): {:.1}% (ATM{:+.1}%)",
            strike,
            moneyness,
            vol * 100.0,
            vol_diff
        );
    }

    // 6. Model Characteristics and Market Insights
    println!("\n6️⃣  MODEL CHARACTERISTICS & MARKET INSIGHTS");
    println!("{}", "-".repeat(50));

    println!("Heston Model Features:");
    println!(
        "  • Mean Reversion: Volatility reverts to {:.1}% with speed κ={:.1}",
        (model.theta.sqrt() * 100.0),
        model.kappa
    );
    println!(
        "  • Leverage Effect: {:.0}% correlation captures asymmetric volatility response",
        model.correlation * 100.0
    );
    println!(
        "  • Vol of Vol: {:.0}% captures volatility clustering and jumps",
        model.vol_of_vol * 100.0
    );
    println!(
        "  • Feller Condition: {} variance process behavior",
        if feller_ok {
            "Well-behaved"
        } else {
            "May hit zero boundary"
        }
    );

    println!("\nMarket Implications:");
    println!("  • Options Prices: Incorporate realistic volatility dynamics");
    println!("  • Risk Management: Capture volatility clustering effects");
    println!("  • Hedging: Dynamic hedging strategies with vol exposure");
    println!("  • Term Structure: Model volatility term structure evolution");

    // Calculate some theoretical quantities
    let long_term_vol = model.theta.sqrt();
    let current_vol = model.initial_variance.sqrt();
    let vol_half_life = (2.0_f64.ln()) / model.kappa;

    println!("\nModel Analytics:");
    println!("  • Current volatility: {:.1}%", current_vol * 100.0);
    println!("  • Long-term volatility: {:.1}%", long_term_vol * 100.0);
    println!(
        "  • Volatility half-life: {:.1} years ({:.0} days)",
        vol_half_life,
        vol_half_life * 252.0
    );
    println!(
        "  • Mean reversion strength: {} (κ={:.1})",
        if model.kappa > 2.0 {
            "Strong"
        } else if model.kappa > 1.0 {
            "Moderate"
        } else {
            "Weak"
        },
        model.kappa
    );

    // 7. Performance Summary
    println!("\n7️⃣  PERFORMANCE SUMMARY");
    println!("{}", "-".repeat(50));

    println!("Model Performance:");
    println!(
        "  • Monte Carlo simulation: {} paths × {} steps = {} total simulations",
        num_paths,
        horizon_days,
        num_paths * horizon_days
    );
    println!(
        "  • Options pricing: {} strikes × {} expiries = {} option prices",
        strikes.len(),
        expiries.len(),
        strikes.len() * expiries.len()
    );
    println!(
        "  • VaR calculations: {} confidence levels × {} horizons = {} VaR estimates",
        confidence_levels.len(),
        time_horizons.len(),
        confidence_levels.len() * time_horizons.len()
    );
    println!(
        "  • Volatility surface: {} strikes × {} expiries = {} surface points",
        surface_strikes.len(),
        surface_expiries.len(),
        vol_surface.len()
    );

    println!("\nKey Results:");
    println!(
        "  • Expected 1-year return: {:.1}% (drift)",
        model.drift * 100.0
    );
    println!(
        "  • 1-day 95% VaR: ${:.0} ({:.1}% of portfolio)",
        daily_var_95,
        daily_var_95 / portfolio_value * 100.0
    );
    println!(
        "  • ATM 6M option: ${:.2} ({:.1}% of spot)",
        model.option_price(spot_price, spot_price, 0.5, risk_free_rate, true)?,
        model.option_price(spot_price, spot_price, 0.5, risk_free_rate, true)? / spot_price * 100.0
    );
    println!("  • Vol smile: Captures realistic asymmetric volatility patterns");

    println!("\n🎯 HESTON MODEL DEMO COMPLETED SUCCESSFULLY!");
    println!("{}", "=".repeat(80));

    Ok(())
}
