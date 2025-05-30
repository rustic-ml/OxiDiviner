//! GARCH Models Example
//!
//! This example demonstrates GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models
//! for volatility modeling in financial time series. GARCH models are essential for:
//! - Risk management and VaR calculations
//! - Options pricing and volatility forecasting
//! - Portfolio optimization
//! - Financial econometric analysis

use chrono::{DateTime, Duration, Utc};
use oxidiviner::models::garch::GARCHModel;
use oxidiviner::prelude::*;

fn main() -> oxidiviner::Result<()> {
    println!("=== GARCH Models Example ===\n");

    // Generate sample financial returns data with volatility clustering
    let start_date = Utc::now() - Duration::days(500);
    let timestamps: Vec<DateTime<Utc>> = (0..500).map(|i| start_date + Duration::days(i)).collect();

    // Simulate realistic financial returns with volatility clustering
    let mut returns = Vec::with_capacity(500);
    let mut volatility: f64 = 0.02; // Initial volatility (2%)

    for i in 0..500 {
        // GARCH(1,1) simulation for realistic volatility clustering
        let random_shock = rand::random::<f64>() - 0.5;
        let normalized_shock = random_shock * 2.0; // Scale to [-1, 1]

        // Update volatility with GARCH(1,1) dynamics
        let omega: f64 = 0.0001; // Long-term variance
        let alpha: f64 = 0.05; // ARCH effect (impact of past shocks)
        let beta: f64 = 0.9; // GARCH effect (persistence of volatility)

        let previous_return: f64 = if i > 0 { returns[i - 1] } else { 0.0 };
        volatility = (omega + alpha * previous_return.powi(2) + beta * volatility.powi(2)).sqrt();

        // Generate return with current volatility
        let return_value = volatility * normalized_shock;
        returns.push(return_value);
    }

    println!(
        "Generated {} financial returns with volatility clustering",
        returns.len()
    );
    println!("Return statistics:");
    println!(
        "  Mean return: {:.4}",
        returns.iter().sum::<f64>() / returns.len() as f64
    );
    println!("  Volatility (std): {:.4}", {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        variance.sqrt()
    });
    println!(
        "  Min return: {:.4}",
        returns.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    );
    println!(
        "  Max return: {:.4}",
        returns.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Create time series data from returns
    let returns_data = TimeSeriesData::new(timestamps.clone(), returns.clone(), "daily_returns")?;

    // Split data into train/test (80/20 split)
    let split_idx = (returns.len() as f64 * 0.8) as usize;
    let train_returns = TimeSeriesData::new(
        timestamps[..split_idx].to_vec(),
        returns[..split_idx].to_vec(),
        "train_returns",
    )?;
    let test_returns = TimeSeriesData::new(
        timestamps[split_idx..].to_vec(),
        returns[split_idx..].to_vec(),
        "test_returns",
    )?;

    println!(
        "\nData split: {} training, {} testing observations",
        train_returns.len(),
        test_returns.len()
    );

    // Example 1: Basic GARCH(1,1) Model
    println!("\n1. Basic GARCH(1,1) Model");
    println!("=========================");

    let mut garch_11 = GARCHModel::new(1, 1, None)?;
    println!("Created GARCH(1,1) model");

    // Fit the model
    garch_11.fit(&train_returns.values, Some(&train_returns.timestamps))?;
    println!("✓ Model fitted successfully");

    // Generate volatility forecasts
    let volatility_forecast = garch_11.forecast_variance(test_returns.len())?;
    println!(
        "✓ Generated {} volatility forecasts",
        volatility_forecast.len()
    );

    // Calculate evaluation metrics for volatility forecasting
    // Note: For GARCH, we evaluate how well the model predicts volatility, not returns
    let actual_squared_returns: Vec<f64> = test_returns.values.iter().map(|r| r.powi(2)).collect();
    let predicted_variances: Vec<f64> = volatility_forecast.iter().map(|v| v.powi(2)).collect();

    // Simple MSE for variance prediction
    let mse = actual_squared_returns
        .iter()
        .zip(&predicted_variances)
        .map(|(actual, pred)| (actual - pred).powi(2))
        .sum::<f64>()
        / actual_squared_returns.len() as f64;

    println!("Model Performance:");
    println!("  Volatility forecast MSE: {:.6}", mse);
    println!(
        "  Average predicted volatility: {:.4}",
        volatility_forecast.iter().sum::<f64>() / volatility_forecast.len() as f64
    );

    // Example 2: Different GARCH Orders
    println!("\n2. Comparing Different GARCH Orders");
    println!("===================================");

    let garch_configs = vec![
        (1, 0, "ARCH(1)"),
        (1, 1, "GARCH(1,1)"),
        (2, 1, "GARCH(2,1)"),
        (1, 2, "GARCH(1,2)"),
        (2, 2, "GARCH(2,2)"),
    ];

    for (p, q, description) in garch_configs {
        match GARCHModel::new(p, q, None) {
            Ok(mut model) => {
                match model.fit(&train_returns.values, Some(&train_returns.timestamps)) {
                    Ok(_) => match model.forecast_variance(test_returns.len()) {
                        Ok(vol_forecast) => {
                            let pred_vars: Vec<f64> =
                                vol_forecast.iter().map(|v| v.powi(2)).collect();
                            let vol_mse = actual_squared_returns
                                .iter()
                                .zip(&pred_vars)
                                .map(|(actual, pred)| (actual - pred).powi(2))
                                .sum::<f64>()
                                / actual_squared_returns.len() as f64;
                            println!(
                                "  {}: MSE = {:.6}, Avg Vol = {:.4}",
                                description,
                                vol_mse,
                                vol_forecast.iter().sum::<f64>() / vol_forecast.len() as f64
                            );
                        }
                        Err(_) => println!("  {}: Forecast failed", description),
                    },
                    Err(_) => println!("  {}: Fit failed", description),
                }
            }
            Err(_) => println!("  {}: Model creation failed", description),
        }
    }

    // Example 3: Volatility Forecasting and Risk Management
    println!("\n3. Volatility Forecasting and Risk Management");
    println!("=============================================");

    // Use the full dataset to fit the model for risk management
    let mut risk_model = GARCHModel::new(1, 1, None)?;
    risk_model.fit(&returns_data.values, Some(&returns_data.timestamps))?;

    // Forecast volatility for next 30 days
    let future_volatility = risk_model.forecast_variance(30)?;

    println!("30-day volatility forecast:");
    for (i, &vol) in future_volatility.iter().take(10).enumerate() {
        println!(
            "  Day {}: {:.4} ({:.2}% annualized)",
            i + 1,
            vol,
            vol * (252.0_f64).sqrt() * 100.0
        ); // Annualized volatility
    }

    // Calculate Value at Risk (VaR) estimates
    println!("\nValue at Risk (VaR) Analysis:");
    let confidence_levels = vec![0.95, 0.99, 0.995];
    let portfolio_value = 1_000_000.0; // $1M portfolio

    for &confidence in &confidence_levels {
        // Normal distribution critical values
        let z_score = match confidence {
            0.95 => 1.645,
            0.99 => 2.326,
            0.995 => 2.576,
            _ => 1.96,
        };

        let avg_predicted_vol = future_volatility.iter().take(1).next().unwrap_or(&0.02);
        let var_1_day = portfolio_value * z_score * avg_predicted_vol;

        println!(
            "  {:.1}% VaR (1-day): ${:.0}",
            confidence * 100.0,
            var_1_day
        );
    }

    // Example 4: Volatility Regime Analysis
    println!("\n4. Volatility Regime Analysis");
    println!("=============================");

    // Analyze historical volatility regimes
    let window_size = 20; // 20-day rolling volatility
    let mut rolling_vols = Vec::new();

    for i in window_size..returns.len() {
        let window_returns = &returns[i - window_size..i];
        let mean = window_returns.iter().sum::<f64>() / window_size as f64;
        let variance = window_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / window_size as f64;
        rolling_vols.push(variance.sqrt());
    }

    // Define volatility regimes
    let vol_percentiles = {
        let mut sorted_vols = rolling_vols.clone();
        sorted_vols.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (
            sorted_vols[sorted_vols.len() * 25 / 100], // 25th percentile
            sorted_vols[sorted_vols.len() * 75 / 100], // 75th percentile
        )
    };

    println!("Historical volatility regimes:");
    println!(
        "  Low volatility (< {:.4}): {}% of time",
        vol_percentiles.0,
        rolling_vols
            .iter()
            .filter(|&&v| v < vol_percentiles.0)
            .count()
            * 100
            / rolling_vols.len()
    );
    println!(
        "  Normal volatility ({:.4} - {:.4}): {}% of time",
        vol_percentiles.0,
        vol_percentiles.1,
        rolling_vols
            .iter()
            .filter(|&&v| v >= vol_percentiles.0 && v <= vol_percentiles.1)
            .count()
            * 100
            / rolling_vols.len()
    );
    println!(
        "  High volatility (> {:.4}): {}% of time",
        vol_percentiles.1,
        rolling_vols
            .iter()
            .filter(|&&v| v > vol_percentiles.1)
            .count()
            * 100
            / rolling_vols.len()
    );

    // Example 5: Options Pricing Application
    println!("\n5. Options Pricing Application");
    println!("==============================");

    // Use GARCH volatility for options pricing (simplified Black-Scholes)
    let spot_price: f64 = 100.0;
    let strike_price: f64 = 105.0;
    let risk_free_rate: f64 = 0.02;
    let time_to_expiry: f64 = 30.0 / 365.0; // 30 days

    let garch_volatility = future_volatility[0]; // Use 1-day ahead forecast
    let annualized_vol = garch_volatility * (252.0_f64).sqrt();

    println!("Options pricing with GARCH volatility:");
    println!("  Spot price: ${:.2}", spot_price);
    println!("  Strike price: ${:.2}", strike_price);
    println!("  Time to expiry: {:.0} days", time_to_expiry * 365.0);
    println!("  GARCH volatility: {:.2}%", annualized_vol * 100.0);

    // Simplified option value calculation (for demonstration)
    let d1 = ((spot_price / strike_price).ln()
        + (risk_free_rate + 0.5 * annualized_vol.powi(2)) * time_to_expiry)
        / (annualized_vol * time_to_expiry.sqrt());
    println!("  Option moneyness (d1): {:.4}", d1);

    // Example 6: Stress Testing
    println!("\n6. Stress Testing Scenarios");
    println!("============================");

    // Simulate stress scenarios
    let stress_scenarios = vec![
        (2.0, "Market stress (2x volatility)"),
        (3.0, "Crisis scenario (3x volatility)"),
        (0.5, "Low volatility regime"),
    ];

    for (multiplier, scenario) in stress_scenarios {
        let stressed_vol = future_volatility[0] * multiplier;
        let stressed_var_95 = portfolio_value * 1.645 * stressed_vol;
        let stressed_var_99 = portfolio_value * 2.326 * stressed_vol;

        println!("  {}:", scenario);
        println!(
            "    Daily volatility: {:.4} ({:.2}% annualized)",
            stressed_vol,
            stressed_vol * (252.0_f64).sqrt() * 100.0
        );
        println!("    95% VaR: ${:.0}", stressed_var_95);
        println!("    99% VaR: ${:.0}", stressed_var_99);
    }

    // Example 7: Model Diagnostics
    println!("\n7. Model Diagnostics");
    println!("====================");

    // Analyze model residuals (standardized residuals should be approximately standard normal)
    let fitted_volatilities = risk_model.forecast_variance(returns_data.len())?;
    let standardized_residuals: Vec<f64> = returns_data
        .values
        .iter()
        .zip(&fitted_volatilities)
        .map(|(ret, vol)| ret / vol)
        .collect();

    let residual_mean =
        standardized_residuals.iter().sum::<f64>() / standardized_residuals.len() as f64;
    let residual_variance = standardized_residuals
        .iter()
        .map(|r| (r - residual_mean).powi(2))
        .sum::<f64>()
        / standardized_residuals.len() as f64;

    println!("Standardized residuals diagnostics:");
    println!("  Mean: {:.4} (should be ≈ 0)", residual_mean);
    println!("  Variance: {:.4} (should be ≈ 1)", residual_variance);
    println!(
        "  Min: {:.4}",
        standardized_residuals
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b))
    );
    println!(
        "  Max: {:.4}",
        standardized_residuals
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    println!("\n=== GARCH Example Complete ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_garch_example() {
        let result = main();
        assert!(
            result.is_ok(),
            "GARCH example should run successfully: {:?}",
            result
        );
    }
}
