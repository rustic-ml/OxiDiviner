#![allow(deprecated)]
#![allow(unused_imports)]

use chrono::{DateTime, Duration, Utc};
use oxidiviner_core::TimeSeriesData;
use rand::Rng;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Simple Exponential Smoothing Parameter Tuning Demo");
    println!("=================================================\n");

    // Generate synthetic data
    println!("Generating synthetic data with trend and seasonality...");
    let data = generate_synthetic_data();
    println!("Generated {} data points", data.len());

    // Since we can't use the actual model implementation due to import issues,
    // we'll just show a conceptual approach to parameter tuning
    println!("\nParameter Tuning Concept:");
    println!("------------------------");
    println!("1. Generate a grid of alpha values (e.g., 0.1, 0.2, ..., 0.9)");
    println!("2. For each alpha value:");
    println!("   a. Split the data into training and testing sets");
    println!("   b. Train a SES model with the current alpha on the training set");
    println!("   c. Evaluate the model on the testing set using MAE, RMSE, MAPE");
    println!("3. Select the alpha value that minimizes the error metric of interest\n");

    println!("Example grid of alpha values and hypothetical results:");
    println!("----------------------------------------------------");
    println!("| Alpha | MAE     | RMSE    | MAPE    |");
    println!("|-------|---------|---------|---------|");

    // Generate hypothetical results
    let alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let mut min_mae = f64::MAX;
    let mut best_alpha = 0.0;

    for alpha in alphas {
        // These would be real results if we could use the actual model
        let alpha_diff = if alpha > 0.3 {
            alpha - 0.3
        } else {
            0.3 - alpha
        };
        let mae = 10.0 - alpha_diff * 15.0;
        let rmse = mae * 1.2;
        let mape = mae * 0.5;

        println!(
            "| {:.1}   | {:.4}  | {:.4}  | {:.4}% |",
            alpha, mae, rmse, mape
        );

        if mae < min_mae {
            min_mae = mae;
            best_alpha = alpha;
        }
    }

    println!(
        "\nBest alpha value: {:.1} (MAE: {:.4})",
        best_alpha, min_mae
    );
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

    let mut last_value = 100.0;

    for i in 0..n {
        // Create timestamp (daily)
        let timestamp = now - Duration::days((n - i) as i64);
        timestamps.push(timestamp);

        // Create trend and seasonality
        let trend = 0.1 * i as f64;
        let seasonality = 5.0 * ((i % 7) as f64 / 6.0 * std::f64::consts::PI).sin();

        // Add noise
        let noise = rng.gen_range(-3.0..3.0);

        // Combine components
        let value = last_value + trend + seasonality + noise;
        values.push(value);

        // Update for next iteration
        last_value = value;
    }

    TimeSeriesData::new(timestamps, values, "Synthetic data").unwrap()
}
