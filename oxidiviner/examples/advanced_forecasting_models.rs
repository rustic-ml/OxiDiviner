//! Advanced Forecasting Models Examples
//!
//! This example demonstrates the usage of all newly implemented forecasting models
//! in OxiDiviner, including state-space models, regime-switching models, cointegration
//! models, nonlinear models, decomposition models, and copula models.

use chrono::{Duration, Utc};
use oxidiviner::core::{Forecaster, TimeSeriesData};
use oxidiviner::models::cointegration::VECMModel;
use oxidiviner::models::decomposition::STLModel;
use oxidiviner::models::nonlinear::TARModel;
use oxidiviner::models::regime_switching::MarkovSwitchingModel;
use oxidiviner::models::state_space::KalmanFilter;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiDiviner Advanced Forecasting Models Examples ===\n");

    // Generate sample data
    let (trending_data, seasonal_data, volatile_data, cointegrated_data) = generate_sample_data();

    // Example 1: Kalman Filter Models
    kalman_filter_examples(&trending_data, &seasonal_data)?;

    // Example 2: Markov Regime-Switching Models
    markov_switching_examples(&volatile_data)?;

    // Example 3: VECM Cointegration Models
    vecm_examples(&cointegrated_data)?;

    // Example 4: TAR Nonlinear Models
    tar_examples(&volatile_data)?;

    // Example 5: STL Decomposition Models
    stl_examples(&seasonal_data)?;

    println!("\n=== All Examples Completed Successfully! ===");
    Ok(())
}

/// Demonstrate Kalman Filter state-space models
fn kalman_filter_examples(
    trending_data: &TimeSeriesData,
    seasonal_data: &TimeSeriesData,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("1. KALMAN FILTER STATE-SPACE MODELS");
    println!("===================================");

    // Example 1.1: Local Level Model (Random Walk with Noise)
    println!("\n1.1 Local Level Model (Random Walk + Noise)");
    println!("--------------------------------------------");

    let mut local_level = KalmanFilter::local_level(1.0, 0.5)?;
    println!("Model: {}", local_level.name());

    local_level.fit(trending_data)?;
    let forecasts = local_level.forecast(10)?;

    println!(
        "Last 5 observations: {:?}",
        &trending_data.values[trending_data.values.len() - 5..]
    );
    println!("Next 10 forecasts: {:?}", forecasts);

    if let Some(state) = local_level.get_state() {
        println!("Current state estimate: {:?}", state);
    }

    // Example 1.2: Local Linear Trend Model
    println!("\n1.2 Local Linear Trend Model (Level + Trend)");
    println!("---------------------------------------------");

    let mut linear_trend = KalmanFilter::local_linear_trend(0.5, 0.1, 0.3)?;
    println!("Model: {}", linear_trend.name());

    linear_trend.fit(trending_data)?;
    let trend_forecasts = linear_trend.forecast(10)?;

    println!("Trend forecasts: {:?}", trend_forecasts);

    if let Some(state) = linear_trend.get_state() {
        println!("Level: {:.2}, Trend: {:.2}", state[0], state[1]);
    }

    // Example 1.3: Seasonal Model
    println!("\n1.3 Seasonal Model (Level + Trend + Seasonality)");
    println!("------------------------------------------------");

    let mut seasonal_model = KalmanFilter::seasonal_model(0.5, 0.1, 0.3, 0.2, 12)?;
    println!("Model: {}", seasonal_model.name());

    seasonal_model.fit(seasonal_data)?;
    let seasonal_forecasts = seasonal_model.forecast(12)?;

    println!("Seasonal forecasts: {:?}", seasonal_forecasts);

    // Example 1.4: Forecasting with Confidence Intervals
    println!("\n1.4 Forecasting with Confidence Intervals");
    println!("------------------------------------------");

    let (forecasts, lower, upper) = local_level.forecast_with_intervals(5, 0.95)?;
    println!("Point forecasts: {:?}", forecasts);
    println!("95% Lower bounds: {:?}", lower);
    println!("95% Upper bounds: {:?}", upper);

    // Example 1.5: Model Diagnostics
    println!("\n1.5 Model Diagnostics");
    println!("---------------------");

    let (ljung_box_stat, critical_value) = local_level.ljung_box_test(10)?;
    println!("Ljung-Box statistic: {:.3}", ljung_box_stat);
    println!("Critical value (95%): {:.3}", critical_value);
    println!(
        "White noise test: {}",
        if ljung_box_stat < critical_value {
            "PASS"
        } else {
            "FAIL"
        }
    );

    Ok(())
}

/// Demonstrate Markov Regime-Switching models
fn markov_switching_examples(
    volatile_data: &TimeSeriesData,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n2. MARKOV REGIME-SWITCHING MODELS");
    println!("=================================");

    // Example 2.1: Two-Regime Model (Bull/Bear Markets)
    println!("\n2.1 Two-Regime Model (Bull/Bear Markets)");
    println!("----------------------------------------");

    let mut two_regime = MarkovSwitchingModel::two_regime(Some(100), Some(1e-4));
    println!("Model: {}", two_regime.name());

    two_regime.fit(volatile_data)?;
    let regime_forecasts = two_regime.forecast(10)?;

    println!("Regime-weighted forecasts: {:?}", regime_forecasts);

    // Example 2.2: Regime Classification and Probabilities
    println!("\n2.2 Regime Analysis");
    println!("-------------------");

    let (current_regime, regime_prob) = two_regime.classify_current_regime()?;
    println!(
        "Current most likely regime: {} (probability: {:.3})",
        current_regime, regime_prob
    );

    if let Some(regime_probs) = two_regime.get_regime_probabilities() {
        let last_probs = regime_probs.last().unwrap();
        println!("Final regime probabilities: {:?}", last_probs);
    }

    // Example 2.3: Regime Parameters
    println!("\n2.3 Regime Parameters");
    println!("---------------------");

    if let Some((means, std_devs)) = two_regime.get_regime_parameters() {
        for (i, (mean, std)) in means.iter().zip(std_devs.iter()).enumerate() {
            println!("Regime {}: mean = {:.3}, std = {:.3}", i, mean, std);
        }
    }

    // Example 2.4: Transition Matrix and Duration
    println!("\n2.4 Transition Dynamics");
    println!("-----------------------");

    if let Some(transition_matrix) = two_regime.get_transition_matrix() {
        println!("Transition matrix:");
        for (i, row) in transition_matrix.iter().enumerate() {
            println!("  From regime {}: {:?}", i, row);
        }
    }

    let durations = two_regime.regime_duration_stats()?;
    println!("Expected regime durations: {:?}", durations);

    // Example 2.5: Three-Regime Model (Bear/Neutral/Bull)
    println!("\n2.5 Three-Regime Model (Bear/Neutral/Bull)");
    println!("------------------------------------------");

    let mut three_regime = MarkovSwitchingModel::three_regime(Some(50), Some(1e-3));
    println!("Model: {}", three_regime.name());

    three_regime.fit(volatile_data)?;
    let three_regime_forecasts = three_regime.forecast_by_regime(5)?;

    println!("Regime-specific forecasts:");
    for (regime, forecasts) in three_regime_forecasts.iter().enumerate() {
        println!("  Regime {}: {:?}", regime, forecasts);
    }

    Ok(())
}

/// Demonstrate VECM cointegration models
fn vecm_examples(cointegrated_data: &[TimeSeriesData]) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n3. VECM COINTEGRATION MODELS");
    println!("============================");

    // Example 3.1: Two-Variable VECM
    println!("\n3.1 Two-Variable VECM (Pairs Trading)");
    println!("-------------------------------------");

    let mut vecm = VECMModel::new(1, 2, true, false)?;
    println!("Model: {}", vecm.name());

    vecm.fit_multiple(cointegrated_data)?;
    let vecm_forecasts = vecm.forecast_multiple(10)?;

    println!("Multi-variate forecasts:");
    for (i, forecasts) in vecm_forecasts.iter().enumerate() {
        println!("  Series {}: {:?}", i, &forecasts[..5]); // Show first 5
    }

    // Example 3.2: Cointegrating Relationships
    println!("\n3.2 Cointegrating Relationships");
    println!("-------------------------------");

    if let Some(coint_vectors) = vecm.get_cointegrating_vectors() {
        println!("Cointegrating vectors:");
        for (i, vector) in coint_vectors.iter().enumerate() {
            println!("  Vector {}: {:?}", i, vector);
        }
    }

    if let Some(adjustment_coeffs) = vecm.get_adjustment_coefficients() {
        println!("Adjustment coefficients (speed of convergence):");
        for (i, coeffs) in adjustment_coeffs.iter().enumerate() {
            println!("  Variable {}: {:?}", i, coeffs);
        }
    }

    // Example 3.3: Error Correction Terms
    println!("\n3.3 Error Correction Analysis");
    println!("-----------------------------");

    if let Some(ect) = vecm.get_error_correction_terms() {
        let last_ect = ect
            .iter()
            .map(|series| series.last().unwrap_or(&0.0))
            .collect::<Vec<_>>();
        println!("Latest error correction terms: {:?}", last_ect);

        for (i, &error) in last_ect.iter().enumerate() {
            if error.abs() > 0.1 {
                println!(
                    "  Variable {} is {} from long-run equilibrium",
                    i,
                    if *error > 0.0 { "above" } else { "below" }
                );
            }
        }
    }

    Ok(())
}

/// Demonstrate TAR nonlinear models
fn tar_examples(volatile_data: &TimeSeriesData) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n4. TAR NONLINEAR MODELS");
    println!("=======================");

    // Example 4.1: Two-Regime TAR Model
    println!("\n4.1 Two-Regime TAR Model");
    println!("------------------------");

    let mut tar = TARModel::new(vec![2, 3], 1)?; // AR(2) and AR(3) regimes, delay=1
    println!("Model: {}", tar.name());

    tar.fit(volatile_data)?;
    let tar_forecasts = tar.forecast(10)?;

    println!("TAR forecasts: {:?}", tar_forecasts);

    // Example 4.2: Threshold Analysis
    println!("\n4.2 Threshold Analysis");
    println!("----------------------");

    if let Some(threshold) = tar.get_threshold() {
        println!("Estimated threshold: {:.3}", threshold);

        // Analyze regime switching behavior
        if let Some(regime_sequence) = tar.get_regime_sequence() {
            let regime_0_count = regime_sequence.iter().filter(|&&x| x == 0).count();
            let regime_1_count = regime_sequence.len() - regime_0_count;

            println!(
                "Regime 0 observations: {} ({:.1}%)",
                regime_0_count,
                100.0 * regime_0_count as f64 / regime_sequence.len() as f64
            );
            println!(
                "Regime 1 observations: {} ({:.1}%)",
                regime_1_count,
                100.0 * regime_1_count as f64 / regime_sequence.len() as f64
            );
        }
    }

    // Example 4.3: Regime-Dependent Dynamics
    println!("\n4.3 Nonlinear Dynamics Analysis");
    println!("-------------------------------");

    println!("TAR models capture threshold effects where the autoregressive");
    println!("dynamics change based on past values crossing a threshold.");
    println!("This is useful for modeling:");
    println!("- Momentum vs. mean-reversion regimes");
    println!("- Bull vs. bear market dynamics");
    println!("- High vs. low volatility periods");

    Ok(())
}

/// Demonstrate STL decomposition models
fn stl_examples(seasonal_data: &TimeSeriesData) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n5. STL DECOMPOSITION MODELS");
    println!("===========================");

    // Example 5.1: Basic STL Decomposition
    println!("\n5.1 Basic STL Decomposition");
    println!("---------------------------");

    let mut stl = STLModel::new(12, Some(7), Some(21))?; // Monthly data
    println!("Model: {}", stl.name());

    stl.fit(seasonal_data)?;
    let stl_forecasts = stl.forecast(12)?;

    println!("STL forecasts (next 12 months): {:?}", stl_forecasts);

    // Example 5.2: Component Analysis
    println!("\n5.2 Component Analysis");
    println!("---------------------");

    if let Some((trend, seasonal, remainder)) = stl.get_components() {
        let n = trend.len();
        println!("Last 5 trend values: {:?}", &trend[n - 5..]);
        println!("Last 5 seasonal values: {:?}", &seasonal[n - 5..]);
        println!("Last 5 remainder values: {:?}", &remainder[n - 5..]);

        // Verify decomposition
        let last_idx = n - 1;
        let reconstructed = trend[last_idx] + seasonal[last_idx] + remainder[last_idx];
        let original = seasonal_data.values[last_idx];
        println!(
            "Decomposition check - Original: {:.3}, Reconstructed: {:.3}, Diff: {:.6}",
            original,
            reconstructed,
            (original - reconstructed).abs()
        );
    }

    // Example 5.3: Seasonal and Trend Strength
    println!("\n5.3 Strength Measures");
    println!("---------------------");

    let seasonal_strength = stl.seasonal_strength()?;
    let trend_strength = stl.trend_strength()?;

    println!(
        "Seasonal strength: {:.3} (0=none, 1=pure seasonal)",
        seasonal_strength
    );
    println!(
        "Trend strength: {:.3} (0=none, 1=pure trend)",
        trend_strength
    );

    // Interpretation
    match seasonal_strength {
        s if s > 0.7 => println!("Strong seasonal pattern detected"),
        s if s > 0.3 => println!("Moderate seasonal pattern detected"),
        _ => println!("Weak seasonal pattern detected"),
    }

    match trend_strength {
        t if t > 0.7 => println!("Strong trend detected"),
        t if t > 0.3 => println!("Moderate trend detected"),
        _ => println!("Weak trend detected"),
    }

    // Example 5.4: Forecasting Applications
    println!("\n5.4 STL Forecasting Applications");
    println!("--------------------------------");

    println!("STL decomposition is particularly useful for:");
    println!("- Seasonal adjustment of economic data");
    println!("- Trend analysis and structural break detection");
    println!("- Outlier detection in seasonal time series");
    println!("- Forecasting with explicit seasonal patterns");

    Ok(())
}

/// Generate various types of sample data for demonstration
fn generate_sample_data() -> (
    TimeSeriesData,
    TimeSeriesData,
    TimeSeriesData,
    Vec<TimeSeriesData>,
) {
    let start_time = Utc::now();
    let n = 200_usize;

    // Generate timestamps
    let timestamps: Vec<chrono::DateTime<chrono::Utc>> = (0..n)
        .map(|i| start_time + Duration::days(i as i64))
        .collect();

    // 1. Trending data (for Kalman filters)
    let trending_values: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 100.0 + 0.5 * i as f64;
            let noise = 2.0 * (rand::random::<f64>() - 0.5);
            trend + noise
        })
        .collect();

    // 2. Seasonal data (for STL)
    let seasonal_values: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 100.0 + 0.2 * i as f64;
            let seasonal = 10.0 * (2.0 * PI * i as f64 / 12.0).sin();
            let noise = 1.5 * (rand::random::<f64>() - 0.5);
            trend + seasonal + noise
        })
        .collect();

    // 3. Volatile data with regime changes (for Markov Switching and TAR)
    let mut volatile_values = Vec::with_capacity(n);
    let mut current_regime = 0;
    let mut current_value = 100.0;

    for i in 0..n {
        // Switch regimes occasionally
        if i > 0 && rand::random::<f64>() < 0.05 {
            current_regime = 1 - current_regime;
        }

        let (mean_change, volatility) = match current_regime {
            0 => (0.02, 1.0),  // Low volatility regime
            _ => (-0.01, 3.0), // High volatility regime
        };

        let change = mean_change + volatility * (rand::random::<f64>() - 0.5);
        current_value += change;
        volatile_values.push(current_value);
    }

    // 4. Cointegrated data (for VECM)
    let mut common_trend = 100.0;
    let mut series1_values = Vec::with_capacity(n);
    let mut series2_values = Vec::with_capacity(n);

    for _ in 0..n {
        // Common stochastic trend
        common_trend += 0.1 * (rand::random::<f64>() - 0.5);

        // Series 1: follows common trend
        let series1 = common_trend + 2.0 * (rand::random::<f64>() - 0.5);
        series1_values.push(series1);

        // Series 2: cointegrated with series 1 (long-run relationship: S2 ≈ 0.8*S1 + 20)
        let equilibrium_error = series1 * 0.8 + 20.0;
        let series2 = equilibrium_error + 1.5 * (rand::random::<f64>() - 0.5);
        series2_values.push(series2);
    }

    let trending_data =
        TimeSeriesData::new(timestamps.clone(), trending_values, "trending_series").unwrap();

    let seasonal_data =
        TimeSeriesData::new(timestamps.clone(), seasonal_values, "seasonal_series").unwrap();

    let volatile_data =
        TimeSeriesData::new(timestamps.clone(), volatile_values, "volatile_series").unwrap();

    let cointegrated_data = vec![
        TimeSeriesData::new(timestamps.clone(), series1_values, "series1").unwrap(),
        TimeSeriesData::new(timestamps, series2_values, "series2").unwrap(),
    ];

    (
        trending_data,
        seasonal_data,
        volatile_data,
        cointegrated_data,
    )
}
