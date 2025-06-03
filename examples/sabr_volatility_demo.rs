//! SABR Volatility Model - Comprehensive Demo
//!
//! This demo showcases the SABR (Stochastic Alpha Beta Rho) Volatility Model's capabilities for:
//! - Monte Carlo simulation with CEV-type forward dynamics and stochastic volatility
//! - SABR implied volatility calculation using Hagan's approximation
//! - European options pricing with SABR stochastic volatility
//! - Value-at-Risk (VaR) calculation for FX and interest rate products
//! - Volatility surface generation for derivatives trading
//! - Model type analysis (Normal, Square-Root, Log-Normal models)
//!
//! The SABR model is the industry standard for volatility surface modeling in FX and rates markets.

use oxidiviner::models::financial::SABRVolatilityModel;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("📊 SABR VOLATILITY MODEL - COMPREHENSIVE DEMO");
    println!("{}", "=".repeat(80));

    // 1. Model Creation and Parameter Analysis
    println!("\n1️⃣  MODEL CREATION AND PARAMETER ANALYSIS");
    println!("{}", "-".repeat(50));

    // Create different SABR models for different market types
    let fx_model = SABRVolatilityModel::new_fx_default()?;
    let equity_model = SABRVolatilityModel::new_equity_default()?;

    // Custom rates model
    let rates_model = SABRVolatilityModel::new(
        0.05,        // 5% forward rate
        0.30,        // 30% initial volatility (rates are more volatile)
        0.50,        // 50% vol of vol (high volatility clustering)
        0.2,         // β = 0.2 (closer to normal model for rates)
        -0.4,        // -40% correlation
        1.0 / 252.0, // Daily time step
    )?;

    println!("FX Model Parameters (EUR/USD):");
    println!("  • Forward (F₀): {:.4}", fx_model.initial_forward);
    println!(
        "  • Initial volatility (σ₀): {:.1}%",
        fx_model.initial_volatility * 100.0
    );
    println!("  • Vol of vol (α): {:.1}%", fx_model.vol_of_vol * 100.0);
    println!(
        "  • Beta (β): {:.1} - {}",
        fx_model.beta,
        fx_model.get_model_type()
    );
    println!("  • Correlation (ρ): {:.1}%", fx_model.correlation * 100.0);

    println!("\nEquity Index Model Parameters (S&P 500):");
    println!("  • Forward (F₀): {:.0}", equity_model.initial_forward);
    println!(
        "  • Initial volatility (σ₀): {:.1}%",
        equity_model.initial_volatility * 100.0
    );
    println!(
        "  • Vol of vol (α): {:.1}%",
        equity_model.vol_of_vol * 100.0
    );
    println!(
        "  • Beta (β): {:.1} - {}",
        equity_model.beta,
        equity_model.get_model_type()
    );
    println!(
        "  • Correlation (ρ): {:.1}%",
        equity_model.correlation * 100.0
    );

    println!("\nInterest Rates Model Parameters (5Y Swap Rate):");
    println!(
        "  • Forward (F₀): {:.1}%",
        rates_model.initial_forward * 100.0
    );
    println!(
        "  • Initial volatility (σ₀): {:.1}%",
        rates_model.initial_volatility * 100.0
    );
    println!("  • Vol of vol (α): {:.1}%", rates_model.vol_of_vol * 100.0);
    println!(
        "  • Beta (β): {:.1} - {}",
        rates_model.beta,
        rates_model.get_model_type()
    );
    println!(
        "  • Correlation (ρ): {:.1}%",
        rates_model.correlation * 100.0
    );

    // 2. Monte Carlo Simulation - FX Focus
    println!("\n2️⃣  MONTE CARLO SIMULATION (EUR/USD FX)");
    println!("{}", "-".repeat(50));

    let horizon_days = 252; // 1 year
    let num_paths = 3000;

    println!(
        "Simulating {} paths over {} trading days...",
        num_paths, horizon_days
    );

    let fx_paths = fx_model.simulate_paths(horizon_days, num_paths, Some(42))?;

    // Analyze simulation results
    let final_forwards: Vec<f64> = fx_paths
        .iter()
        .map(|p| *p.forwards.last().unwrap())
        .collect();
    let final_volatilities: Vec<f64> = fx_paths
        .iter()
        .map(|p| *p.volatilities.last().unwrap())
        .collect();

    let mean_final_forward = final_forwards.iter().sum::<f64>() / num_paths as f64;
    let std_final_forward = {
        let variance = final_forwards
            .iter()
            .map(|f| (f - mean_final_forward).powi(2))
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

    println!("Final EUR/USD Rate Statistics:");
    println!("  • Mean: {:.4}", mean_final_forward);
    println!("  • Standard Deviation: {:.4}", std_final_forward);
    println!(
        "  • Range: {:.4} - {:.4}",
        final_forwards.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        final_forwards
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!(
        "  • 1-year move: {:.1}% ({:.0} pips)",
        (mean_final_forward / fx_model.initial_forward - 1.0) * 100.0,
        (mean_final_forward - fx_model.initial_forward) * 10000.0
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

    // 3. SABR Implied Volatility Analysis
    println!("\n3️⃣  SABR IMPLIED VOLATILITY ANALYSIS");
    println!("{}", "-".repeat(50));

    let forward = 1.0;
    let expiries = vec![0.25, 0.5, 1.0, 2.0]; // 3M, 6M, 1Y, 2Y
    let strikes = vec![0.90, 0.95, 1.00, 1.05, 1.10];

    println!("SABR Implied Volatilities (FX Model):");
    println!("Strike   3M     6M     1Y     2Y");
    println!("{}", "-".repeat(40));

    for &strike in &strikes {
        print!("{:>6.2}", strike);
        for &expiry in &expiries {
            let sabr_vol = fx_model.sabr_implied_volatility(forward, strike, expiry)?;
            print!("   {:>5.1}", sabr_vol * 100.0);
        }
        println!();
    }

    // Analyze volatility smile properties
    println!("\nVolatility Smile Analysis (1Y expiry):");
    let one_year_vols: Vec<_> = strikes
        .iter()
        .map(|&strike| {
            let vol = fx_model
                .sabr_implied_volatility(forward, strike, 1.0)
                .unwrap();
            (strike, vol)
        })
        .collect();

    let atm_vol = one_year_vols
        .iter()
        .find(|(s, _)| (*s - forward).abs() < 0.01)
        .map(|(_, v)| *v)
        .unwrap_or(0.1);

    for (strike, vol) in &one_year_vols {
        let moneyness = (strike / forward).ln() * 100.0; // Log-moneyness in %
        let vol_diff = (vol - atm_vol) * 100.0;
        println!(
            "  • K={:.2} (ln(K/F)={:+.1}%): {:.1}% (ATM{:+.1}%)",
            strike,
            moneyness,
            vol * 100.0,
            vol_diff
        );
    }

    // 4. Cross-Model Comparison
    println!("\n4️⃣  CROSS-MODEL COMPARISON (β PARAMETER EFFECTS)");
    println!("{}", "-".repeat(50));

    // Create models with different beta values
    let beta_models = vec![
        (
            "Normal (β=0.0)",
            SABRVolatilityModel::new(1.0, 0.10, 0.30, 0.0, -0.3, 1.0 / 252.0)?,
        ),
        (
            "Square-Root (β=0.5)",
            SABRVolatilityModel::new(1.0, 0.10, 0.30, 0.5, -0.3, 1.0 / 252.0)?,
        ),
        (
            "Log-Normal (β=1.0)",
            SABRVolatilityModel::new(1.0, 0.10, 0.30, 1.0, -0.3, 1.0 / 252.0)?,
        ),
    ];

    println!("ATM Implied Volatility across different β values (6M expiry):");
    for (name, model) in &beta_models {
        let atm_vol = model.sabr_implied_volatility(1.0, 1.0, 0.5)?;
        println!("  • {}: {:.1}%", name, atm_vol * 100.0);
    }

    println!("\nOTM Call (K=1.10) Implied Volatility:");
    for (name, model) in &beta_models {
        let otm_vol = model.sabr_implied_volatility(1.0, 1.10, 0.5)?;
        println!("  • {}: {:.1}%", name, otm_vol * 100.0);
    }

    // 5. European Options Pricing
    println!("\n5️⃣  EUROPEAN OPTIONS PRICING");
    println!("{}", "-".repeat(50));

    let spot_forward = 1.0;
    let discount_factor = 0.98; // 2% discount rate
    let option_strikes = vec![0.95, 1.00, 1.05, 1.10];
    let option_expiries = vec![0.25, 0.5, 1.0]; // 3M, 6M, 1Y

    println!(
        "European Call Options (SABR Model, Forward = {:.2}):",
        spot_forward
    );
    println!("Strike   3M      6M      1Y");
    println!("{}", "-".repeat(35));

    for &strike in &option_strikes {
        print!("{:>6.2}", strike);
        for &expiry in &option_expiries {
            let option_price =
                fx_model.option_price(spot_forward, strike, expiry, discount_factor, true)?;
            print!("   {:>6.4}", option_price);
        }
        println!();
    }

    // Compare call vs put parity
    println!("\nCall-Put Parity Check (6M, K=1.00):");
    let call_price = fx_model.option_price(spot_forward, 1.00, 0.5, discount_factor, true)?;
    let put_price = fx_model.option_price(spot_forward, 1.00, 0.5, discount_factor, false)?;
    let parity_diff = call_price - put_price - discount_factor * (spot_forward - 1.00);
    println!("  • Call price: {:.6}", call_price);
    println!("  • Put price: {:.6}", put_price);
    println!("  • Parity difference: {:.8} (should be ~0)", parity_diff);

    // 6. Risk Management - Value at Risk
    println!("\n6️⃣  RISK MANAGEMENT - VALUE AT RISK (VaR)");
    println!("{}", "-".repeat(50));

    let portfolio_value = 10_000_000.0; // $10M EUR/USD position
    let confidence_levels = vec![0.90, 0.95, 0.99];
    let time_horizons = vec![1.0 / 252.0, 5.0 / 252.0, 21.0 / 252.0]; // 1 day, 1 week, 1 month

    println!(
        "Portfolio Value: ${:.0} (EUR/USD position)",
        portfolio_value
    );
    println!("Time Horizon    90% VaR    95% VaR    99% VaR");
    println!("{}", "-".repeat(50));

    for &horizon in &time_horizons {
        let horizon_days = (horizon * 252.0_f64).round() as i32;
        let horizon_label = match horizon_days {
            1 => "1 day".to_string(),
            5 => "1 week".to_string(),
            21 => "1 month".to_string(),
            _ => format!("{} days", horizon_days),
        };
        print!("{:>12}", horizon_label);

        for &confidence in &confidence_levels {
            let var = fx_model.calculate_var(portfolio_value, confidence, horizon, 3000)?;
            print!("   ${:>7.0}", var);
        }
        println!();
    }

    // VaR decomposition analysis
    println!("\nVaR Analysis with SABR Features:");
    let daily_var_95 = fx_model.calculate_var(portfolio_value, 0.95, 1.0 / 252.0, 5000)?;
    let weekly_var_95 = fx_model.calculate_var(portfolio_value, 0.95, 5.0 / 252.0, 5000)?;
    let monthly_var_95 = fx_model.calculate_var(portfolio_value, 0.95, 21.0 / 252.0, 5000)?;

    println!(
        "  • Daily 95% VaR: ${:.0} ({:.2}% of portfolio)",
        daily_var_95,
        daily_var_95 / portfolio_value * 100.0
    );
    println!(
        "  • Weekly vs Daily scaling: {:.2}x (√5 = {:.2}x)",
        weekly_var_95 / daily_var_95,
        5.0_f64.sqrt()
    );
    println!(
        "  • Monthly vs Daily scaling: {:.2}x (√21 = {:.2}x)",
        monthly_var_95 / daily_var_95,
        21.0_f64.sqrt()
    );

    // 7. Volatility Surface Generation
    println!("\n7️⃣  VOLATILITY SURFACE GENERATION");
    println!("{}", "-".repeat(50));

    let surface_strikes = vec![0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15];
    let surface_expiries = vec![0.083, 0.25, 0.5, 1.0, 2.0]; // 1M, 3M, 6M, 1Y, 2Y

    println!("Generating SABR volatility surface...");
    let vol_surface = fx_model.volatility_surface(1.0, &surface_strikes, &surface_expiries)?;

    println!("SABR Implied Volatility Surface (%):");
    println!("Strike   1M     3M     6M     1Y     2Y");
    println!("{}", "-".repeat(45));

    for &strike in &surface_strikes {
        print!("{:>6.2}", strike);
        for &expiry in &surface_expiries {
            let point = vol_surface
                .iter()
                .find(|p| (p.strike - strike).abs() < 0.01 && (p.expiry - expiry).abs() < 0.01)
                .unwrap();
            print!("   {:>5.1}", point.implied_volatility * 100.0);
        }
        println!();
    }

    // Surface analytics
    println!("\nVolatility Surface Analytics:");
    let atm_1y = vol_surface
        .iter()
        .find(|p| (p.strike - 1.0).abs() < 0.01 && (p.expiry - 1.0).abs() < 0.01)
        .unwrap();
    let otm_call_1y = vol_surface
        .iter()
        .find(|p| (p.strike - 1.10).abs() < 0.01 && (p.expiry - 1.0).abs() < 0.01)
        .unwrap();
    let otm_put_1y = vol_surface
        .iter()
        .find(|p| (p.strike - 0.90).abs() < 0.01 && (p.expiry - 1.0).abs() < 0.01)
        .unwrap();

    let call_skew = otm_call_1y.implied_volatility - atm_1y.implied_volatility;
    let put_skew = otm_put_1y.implied_volatility - atm_1y.implied_volatility;

    println!(
        "  • ATM 1Y volatility: {:.1}%",
        atm_1y.implied_volatility * 100.0
    );
    println!(
        "  • 10Δ Call skew: {:+.1}% ({} vs ATM)",
        call_skew * 100.0,
        if call_skew > 0.0 { "higher" } else { "lower" }
    );
    println!(
        "  • 10Δ Put skew: {:+.1}% ({} vs ATM)",
        put_skew * 100.0,
        if put_skew > 0.0 { "higher" } else { "lower" }
    );
    println!(
        "  • Risk reversal (Call-Put): {:+.1}%",
        (call_skew - put_skew) * 100.0
    );

    // 8. Model Applications and Market Insights
    println!("\n8️⃣  MODEL APPLICATIONS & MARKET INSIGHTS");
    println!("{}", "-".repeat(50));

    println!("SABR Model Market Applications:");
    println!("  • FX Options: Industry standard for EUR/USD, GBP/USD volatility surfaces");
    println!("  • Interest Rate Derivatives: Swaption and cap/floor pricing");
    println!("  • Commodity Options: Energy and agricultural derivatives");
    println!("  • Credit Derivatives: Credit default swaption modeling");

    println!("\nParameter Interpretation:");
    println!("  • α (vol of vol): Controls volatility clustering and smile intensity");
    println!("  • β (backbone): Controls forward price dynamics:");
    println!("    - β = 0: Normal model (good for low/negative rates)");
    println!("    - β = 0.5: Square-root model (balanced choice)");
    println!("    - β = 1: Log-normal model (traditional Black-Scholes)");
    println!("  • ρ (correlation): Controls volatility skew direction");

    println!("\nMarket Regime Analysis:");
    println!(
        "  • Normal Volatility (β=0.0): {:.1}% ATM vol",
        beta_models[0].1.sabr_implied_volatility(1.0, 1.0, 0.5)? * 100.0
    );
    println!(
        "  • Stochastic Volatility (β=0.5): {:.1}% ATM vol",
        beta_models[1].1.sabr_implied_volatility(1.0, 1.0, 0.5)? * 100.0
    );
    println!(
        "  • Log-Normal Regime (β=1.0): {:.1}% ATM vol",
        beta_models[2].1.sabr_implied_volatility(1.0, 1.0, 0.5)? * 100.0
    );

    // 9. Performance and Calibration Insights
    println!("\n9️⃣  PERFORMANCE SUMMARY & CALIBRATION INSIGHTS");
    println!("{}", "-".repeat(50));

    println!("Model Performance:");
    println!(
        "  • Monte Carlo simulation: {} paths × {} steps = {} total simulations",
        num_paths,
        horizon_days,
        num_paths * horizon_days
    );
    println!(
        "  • SABR implied vol calculations: {} strikes × {} expiries = {} calculations",
        surface_strikes.len(),
        surface_expiries.len(),
        vol_surface.len()
    );
    println!(
        "  • Options pricing: {} strikes × {} expiries = {} option prices",
        option_strikes.len(),
        option_expiries.len(),
        option_strikes.len() * option_expiries.len()
    );
    println!(
        "  • VaR calculations: {} confidence levels × {} horizons = {} VaR estimates",
        confidence_levels.len(),
        time_horizons.len(),
        confidence_levels.len() * time_horizons.len()
    );

    println!("\nCalibration Best Practices:");
    println!("  • Use liquid ATM options to calibrate α (vol of vol)");
    println!("  • Calibrate β to match forward price distribution characteristics");
    println!("  • Use skew information to calibrate ρ (correlation)");
    println!("  • Ensure market-observed smile is well-reproduced");

    println!("\nKey Model Insights:");
    println!("  • SABR provides analytical implied volatility formulas");
    println!("  • Beta parameter allows model flexibility for different markets");
    println!("  • Correlation parameter controls volatility skew asymmetry");
    println!("  • Model degenerates to Black-Scholes when α = 0");

    println!("\nProduction Usage:");
    let fx_1y_atm_vol = fx_model.sabr_implied_volatility(1.0, 1.0, 1.0)?;
    let fx_1y_otm_vol = fx_model.sabr_implied_volatility(1.0, 1.05, 1.0)?;
    println!(
        "  • 1Y EUR/USD ATM volatility: {:.1}%",
        fx_1y_atm_vol * 100.0
    );
    println!(
        "  • 1Y EUR/USD 5% OTM call vol: {:.1}%",
        fx_1y_otm_vol * 100.0
    );
    println!(
        "  • Volatility smile: {:.1}% differential for 5% OTM",
        (fx_1y_otm_vol - fx_1y_atm_vol) * 100.0
    );
    println!("  • Daily 95% VaR: ${:.0} for $10M position", daily_var_95);

    println!("\n🎯 SABR VOLATILITY MODEL DEMO COMPLETED SUCCESSFULLY!");
    println!("{}", "=".repeat(80));

    Ok(())
}
