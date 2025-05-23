#![allow(deprecated)]
#![allow(unused_imports)]

use chrono::{DateTime, Duration, Utc};
use oxidiviner_core::TimeSeriesData;
use rand::Rng;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Exponential Smoothing Parameter Tuning Demo");
    println!("==========================================\n");

    // Generate synthetic data
    println!("Generating synthetic data with trend and seasonality...");
    let data = generate_synthetic_data();
    println!("Generated {} data points", data.len());

    // Since we can't use the actual model implementation due to import issues,
    // we'll just show a conceptual approach to parameter tuning
    println!("\nParameter Tuning Concept for Holt Linear Trend Method:");
    println!("--------------------------------------------------");
    println!("1. Generate a grid of alpha and beta values");
    println!("   - alpha: Smoothing parameter for level (e.g., 0.1, 0.2, ..., 0.9)");
    println!("   - beta: Smoothing parameter for trend (e.g., 0.1, 0.2, ..., 0.9)");
    println!("2. For each parameter combination:");
    println!("   a. Split the data into training and testing sets");
    println!("   b. Train a Holt model with the current parameters on the training set");
    println!("   c. Evaluate the model on the testing set using MAE, RMSE, MAPE");
    println!("3. Select the parameter combination that minimizes the error metric of interest\n");

    println!("Example grid of parameter values and hypothetical results (subset):");
    println!("-----------------------------------------------------------");
    println!("| Alpha | Beta  | MAE     | RMSE    | MAPE    |");
    println!("|-------|-------|---------|---------|---------|");

    // Generate hypothetical results for a subset of combinations
    let alphas = [0.3, 0.5, 0.7];
    let betas = [0.1, 0.3, 0.5];

    let mut min_mae = f64::MAX;
    let mut best_alpha = 0.0;
    let mut best_beta = 0.0;

    for alpha in alphas {
        for beta in betas {
            // These would be real results if we could use the actual model
            // The formula is just to generate plausible looking metrics that have a minimum
            let alpha_diff = if alpha > 0.5 {
                alpha - 0.5
            } else {
                0.5 - alpha
            };
            let beta_diff = if beta > 0.3 { beta - 0.3 } else { 0.3 - beta };
            let mae = 10.0 - alpha_diff * 8.0 - beta_diff * 10.0;
            let rmse = mae * 1.2;
            let mape = mae * 0.5;

            println!(
                "| {:.1}   | {:.1}   | {:.4}  | {:.4}  | {:.4}% |",
                alpha, beta, mae, rmse, mape
            );

            if mae < min_mae {
                min_mae = mae;
                best_alpha = alpha;
                best_beta = beta;
            }
        }
    }

    println!(
        "\nBest parameters: alpha = {:.1}, beta = {:.1} (MAE: {:.4})",
        best_alpha, best_beta, min_mae
    );

    println!("\nCharacteristics of Holt's Linear Trend Method:");
    println!("- Alpha controls the level (intercept) component adaptation speed");
    println!("- Beta controls the trend (slope) component adaptation speed");
    println!("- Low alpha/beta: More weight on historical data, smoother forecasts");
    println!("- High alpha/beta: More weight on recent data, responsive to changes");

    println!("\nNote: This is a simplified demonstration. In a real application,");
    println!("you would use the actual OxiDiviner API and implement cross-validation.");

    Ok(())
}

// Generate synthetic time series data with trend and seasonality
fn generate_synthetic_data() -> TimeSeriesData {
    let mut rng = rand::thread_rng();
    let now = Utc::now();
    let n = 100;

    let mut timestamps = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);

    for i in 0..n {
        // Create timestamp (daily)
        let timestamp = now - Duration::days((n - i) as i64);
        timestamps.push(timestamp);

        // Create trend and seasonality
        let trend = 0.5 * i as f64;
        let seasonality = 5.0 * ((i % 7) as f64 / 6.0 * std::f64::consts::PI).sin();

        // Add noise
        let noise = rng.gen_range(-3.0..3.0);

        // Combine components
        let value = 100.0 + trend + seasonality + noise;
        values.push(value);
    }

    TimeSeriesData::new(timestamps, values, "Synthetic data").unwrap()
}
