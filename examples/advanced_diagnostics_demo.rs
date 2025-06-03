/*!
# Advanced Model Diagnostics Demo

This example demonstrates the comprehensive diagnostic capabilities of OxiDiviner,
including residual analysis, model specification tests, forecast diagnostics,
and quality assessment.

Features demonstrated:
- Comprehensive residual analysis (normality, autocorrelation, heteroskedasticity)
- Model specification tests (information criteria, adequacy tests)
- Forecast performance diagnostics
- Outlier detection and quality scoring
- Diagnostic recommendations

Usage:
```bash
cargo run --example advanced_diagnostics_demo
```
*/

use chrono::{TimeZone, Utc};
use oxidiviner::core::{ModelDiagnostics, Result, TimeSeriesData};
use oxidiviner::models::autoregressive::ARIMAModel;
use oxidiviner::models::exponential_smoothing::SimpleESModel;
use std::f64::consts::PI;

fn main() -> Result<()> {
    println!("🔬 OxiDiviner Advanced Model Diagnostics Demo");
    println!("=============================================\n");

    // Generate synthetic time series data with known properties
    let training_data = generate_complex_time_series(120)?;
    let test_data = generate_complex_time_series(30)?;

    println!("📊 Dataset Information:");
    println!("  • Training samples: {}", training_data.len());
    println!("  • Test samples: {}", test_data.len());
    println!();

    // Demo 1: ARIMA Model Diagnostics
    println!("🎯 Demo 1: ARIMA Model Comprehensive Diagnostics");
    println!("================================================");

    let arima_report = analyze_arima_model(&training_data, &test_data)?;
    print_diagnostic_summary(&arima_report);

    // Demo 2: Exponential Smoothing Diagnostics
    println!("\n🎯 Demo 2: Exponential Smoothing Model Diagnostics");
    println!("===================================================");

    let es_report = analyze_exponential_smoothing_model(&training_data, &test_data)?;
    print_diagnostic_summary(&es_report);

    // Demo 3: Model Comparison using Diagnostics
    println!("\n🎯 Demo 3: Model Comparison & Selection");
    println!("=======================================");

    compare_models(&[("ARIMA(2,1,1)", arima_report), ("Simple ES", es_report)]);

    // Demo 4: Residual Analysis Deep Dive
    println!("\n🎯 Demo 4: Detailed Residual Analysis");
    println!("=====================================");

    perform_detailed_residual_analysis(&training_data)?;

    println!("\n✅ Advanced Diagnostics Demo Complete!");
    Ok(())
}

/// Generate a complex synthetic time series with trend, seasonality, and noise
fn generate_complex_time_series(n_points: usize) -> Result<TimeSeriesData> {
    let timestamps = (0..n_points)
        .map(|i| {
            Utc.timestamp_opt(1_640_995_200 + i as i64 * 86400, 0)
                .unwrap()
        })
        .collect();

    let mut values = Vec::with_capacity(n_points);
    let mut rng_state = 42u64; // Simple PRNG state

    for i in 0..n_points {
        // Trend component
        let trend = 100.0 + (i as f64) * 0.1;

        // Seasonal component (12-period cycle)
        let seasonal = 15.0 * (2.0 * PI * (i as f64) / 12.0).sin();

        // Cyclic component (longer 50-period cycle)
        let cyclic = 5.0 * (2.0 * PI * (i as f64) / 50.0).cos();

        // Noise component (pseudo-random)
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let noise = ((rng_state as f64 / u64::MAX as f64) - 0.5) * 10.0;

        // Occasional outliers (5% chance)
        let outlier = if (rng_state % 100) < 5 {
            ((rng_state as f64 / u64::MAX as f64) - 0.5) * 50.0
        } else {
            0.0
        };

        let value = trend + seasonal + cyclic + noise + outlier;
        values.push(value);
    }

    TimeSeriesData::new(timestamps, values, "complex_synthetic_series")
}

/// Analyze ARIMA model and generate comprehensive diagnostics
fn analyze_arima_model(
    training_data: &TimeSeriesData,
    test_data: &TimeSeriesData,
) -> Result<oxidiviner::core::DiagnosticReport> {
    // Fit ARIMA(2,1,1) model
    let mut arima = ARIMAModel::new(2, 1, 1, true)?;
    arima.fit(training_data)?;

    // Generate forecasts
    let forecasts = arima.forecast(test_data.len())?;

    // Calculate forecast errors
    let forecast_errors: Vec<f64> = test_data
        .values
        .iter()
        .zip(&forecasts)
        .map(|(actual, forecast)| actual - forecast)
        .collect();

    // Calculate fitted values and residuals (simplified approach)
    let fitted_values: Vec<f64> = training_data
        .values
        .iter()
        .enumerate()
        .map(|(i, &actual)| {
            // Simple approximation for fitted values
            if i > 3 {
                let window: Vec<f64> = training_data.values[(i - 3)..i].to_vec();
                window.iter().sum::<f64>() / window.len() as f64
            } else {
                actual
            }
        })
        .collect();

    let residuals: Vec<f64> = training_data
        .values
        .iter()
        .zip(&fitted_values)
        .map(|(actual, fitted)| actual - fitted)
        .collect();

    // Perform comprehensive diagnostic analysis
    ModelDiagnostics::analyze_model(
        "ARIMA(2,1,1)",
        &residuals,
        &fitted_values,
        &training_data.values,
        Some(&forecasts),
        Some(&forecast_errors),
    )
}

/// Analyze Exponential Smoothing model and generate diagnostics
fn analyze_exponential_smoothing_model(
    training_data: &TimeSeriesData,
    test_data: &TimeSeriesData,
) -> Result<oxidiviner::core::DiagnosticReport> {
    // Fit Simple Exponential Smoothing model
    let mut es = SimpleESModel::new(0.3)?;
    es.fit(training_data)?;

    // Generate forecasts
    let forecasts = es.forecast(test_data.len())?;

    // Calculate forecast errors
    let forecast_errors: Vec<f64> = test_data
        .values
        .iter()
        .zip(&forecasts)
        .map(|(actual, forecast)| actual - forecast)
        .collect();

    // Get fitted values and calculate residuals
    let fitted_values = es
        .fitted_values()
        .unwrap_or(&vec![0.0; training_data.len()])
        .clone();
    let residuals: Vec<f64> = training_data
        .values
        .iter()
        .zip(&fitted_values)
        .map(|(actual, fitted)| actual - fitted)
        .collect();

    // Perform comprehensive diagnostic analysis
    ModelDiagnostics::analyze_model(
        "Simple Exponential Smoothing",
        &residuals,
        &fitted_values,
        &training_data.values,
        Some(&forecasts),
        Some(&forecast_errors),
    )
}

/// Print a summary of diagnostic results
fn print_diagnostic_summary(report: &oxidiviner::core::DiagnosticReport) {
    println!("📋 Model: {}", report.model_name);
    println!("🏆 Quality Score: {:.1}/100", report.quality_score);
    println!();

    // Residual Analysis Summary
    println!("📊 Residual Analysis:");
    let stats = &report.residual_analysis.statistics;
    println!("  • Mean: {:.4}", stats.mean);
    println!("  • Std Dev: {:.4}", stats.std_dev);
    println!("  • Skewness: {:.4}", stats.skewness);
    println!("  • Kurtosis: {:.4}", stats.kurtosis);

    // Normality Tests
    let jb = &report.residual_analysis.normality_tests.jarque_bera;
    println!(
        "  • Jarque-Bera Test: {} (p={:.4})",
        if jb.is_significant {
            "❌ Reject Normality"
        } else {
            "✅ Normal"
        },
        jb.p_value
    );

    // Autocorrelation Tests
    let lb = &report.residual_analysis.autocorrelation.ljung_box;
    println!(
        "  • Ljung-Box Test: {} (p={:.4})",
        if lb.is_significant {
            "❌ Autocorrelation"
        } else {
            "✅ No Autocorrelation"
        },
        lb.p_value
    );

    // ARCH Effects
    let arch = &report.residual_analysis.heteroskedasticity_tests.arch_test;
    println!(
        "  • ARCH Test: {} (p={:.4})",
        if arch.is_significant {
            "❌ Heteroskedasticity"
        } else {
            "✅ Homoskedastic"
        },
        arch.p_value
    );

    // Outliers
    let outliers = &report.residual_analysis.outliers;
    println!(
        "  • Outliers: {:.1}% ({} points)",
        outliers.outlier_percentage,
        outliers.outlier_indices.len()
    );

    // Information Criteria
    println!("\n📈 Model Selection Criteria:");
    let ic = &report.specification_tests.information_criteria;
    println!("  • AIC: {:.2}", ic.aic);
    println!("  • BIC: {:.2}", ic.bic);
    println!("  • HQC: {:.2}", ic.hqc);

    // Forecast Performance
    if report.forecast_diagnostics.error_analysis.mae > 0.0 {
        println!("\n🎯 Forecast Performance:");
        let fa = &report.forecast_diagnostics.error_analysis;
        println!("  • MAE: {:.4}", fa.mae);
        println!("  • RMSE: {:.4}", fa.rmse);
        println!("  • MAPE: {:.2}%", fa.mape);
        println!("  • Theil's U: {:.4}", fa.theil_u);

        let bias = &report.forecast_diagnostics.bias_analysis;
        println!(
            "  • Forecast Bias: {} (bias={:.4})",
            if bias.bias_test.is_significant {
                "❌ Significant"
            } else {
                "✅ Unbiased"
            },
            bias.bias
        );
    }

    // Recommendations
    println!("\n💡 Recommendations:");
    for (i, rec) in report.recommendations.iter().enumerate() {
        println!("  {}. {}", i + 1, rec);
    }
}

/// Compare multiple models using diagnostic scores
fn compare_models(models: &[(&str, oxidiviner::core::DiagnosticReport)]) {
    println!("Model Comparison Results:");
    println!(
        "{:<25} | {:>8} | {:>8} | {:>8} | {:>10}",
        "Model", "Quality", "AIC", "RMSE", "Status"
    );
    println!("{:-<70}", "");

    let mut best_model = "";
    let mut best_score = 0.0;

    for (name, report) in models {
        let quality = report.quality_score;
        let aic = report.specification_tests.information_criteria.aic;
        let rmse = if report.forecast_diagnostics.error_analysis.rmse > 0.0 {
            format!("{:.3}", report.forecast_diagnostics.error_analysis.rmse)
        } else {
            "N/A".to_string()
        };

        let status = if quality > 80.0 {
            "✅ Excellent"
        } else if quality > 60.0 {
            "🟡 Good"
        } else {
            "❌ Poor"
        };

        println!(
            "{:<25} | {:>8.1} | {:>8.1} | {:>8} | {:>10}",
            name, quality, aic, rmse, status
        );

        if quality > best_score {
            best_score = quality;
            best_model = name;
        }
    }

    println!("{:-<70}", "");
    println!(
        "🏆 Best Model: {} (Quality Score: {:.1})",
        best_model, best_score
    );
}

/// Perform detailed residual analysis demonstration
fn perform_detailed_residual_analysis(data: &TimeSeriesData) -> Result<()> {
    println!("Detailed Residual Analysis:");

    // Fit a simple model to get residuals
    let mut model = SimpleESModel::new(0.2)?;
    model.fit(data)?;

    let fitted_values = model
        .fitted_values()
        .unwrap_or(&vec![0.0; data.len()])
        .clone();
    let residuals: Vec<f64> = data
        .values
        .iter()
        .zip(&fitted_values)
        .map(|(actual, fitted)| actual - fitted)
        .collect();

    // Analyze just the residuals
    let residual_analysis = oxidiviner::core::diagnostics::ModelDiagnostics::analyze_model(
        "Residual Analysis",
        &residuals,
        &fitted_values,
        &data.values,
        None,
        None,
    )?;

    // Display detailed autocorrelation information
    println!("\n📊 Autocorrelation Function (first 10 lags):");
    for (i, &acf) in residual_analysis
        .residual_analysis
        .autocorrelation
        .acf_values
        .iter()
        .take(10)
        .enumerate()
    {
        let lag = i + 1;
        let bounds = residual_analysis
            .residual_analysis
            .autocorrelation
            .acf_confidence_bounds;
        let significant = acf.abs() > bounds.1.abs();

        println!(
            "  Lag {:2}: {:6.3} {}",
            lag,
            acf,
            if significant {
                " ❌ Significant"
            } else {
                " ✅"
            }
        );
    }

    // Display outlier details
    println!("\n🎯 Outlier Analysis:");
    let outliers = &residual_analysis.residual_analysis.outliers;
    if !outliers.outlier_indices.is_empty() {
        println!("  Detected {} outliers:", outliers.outlier_indices.len());
        for &idx in outliers.outlier_indices.iter().take(5) {
            let z_score = outliers.z_scores.get(idx).unwrap_or(&0.0);
            let mod_z_score = outliers.modified_z_scores.get(idx).unwrap_or(&0.0);
            println!(
                "    Index {}: Z-score={:.2}, Modified Z-score={:.2}",
                idx, z_score, mod_z_score
            );
        }
        if outliers.outlier_indices.len() > 5 {
            println!("    ... and {} more", outliers.outlier_indices.len() - 5);
        }
    } else {
        println!("  ✅ No outliers detected");
    }

    Ok(())
}
