use chrono::{Duration, Utc};
use oxidiviner::HoltWintersModel;
use oxidiviner::TimeSeriesData;
use std::error::Error;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Holt-Winters (Triple Exponential Smoothing) Model Demo");
    println!("=====================================================\n");

    // Generate synthetic data with trend and seasonality
    println!("Generating synthetic data with trend and seasonality...");
    let data = generate_synthetic_data();
    println!("Generated {} data points\n", data.len());

    // Define seasonal period
    let period = 7; // Weekly seasonality

    // Create Holt-Winters model
    println!(
        "Creating Holt-Winters model with alpha = 0.2, beta = 0.1, gamma = 0.3, period = {}...",
        period
    );
    let mut model = HoltWintersModel::new(0.2, 0.1, 0.3, period)?;

    // Split data for training and testing
    println!("Splitting data into training (80%) and testing (20%) sets...");
    let (train_data, test_data) = data.train_test_split(0.8)?;
    println!("Training set: {} points", train_data.len());
    println!("Testing set: {} points\n", test_data.len());

    // Fit model to training data
    println!("Fitting model to training data...");
    model.fit(&train_data)?;
    println!("Model fitted successfully\n");

    // Display seasonal components
    if let Some(seasonal) = model.seasonal_components() {
        println!("Estimated seasonal components:");
        for (i, value) in seasonal.iter().enumerate() {
            println!("  Period {}: {:.2}", i + 1, value);
        }
        println!();
    }

    // Generate forecasts
    let horizon = 3 * period; // 3 full seasonal cycles
    println!("Generating {} day forecast...", horizon);
    let forecasts = model.forecast(horizon)?;

    // Print forecasts by seasonal period
    println!("Forecast preview:");
    for p in 0..3 {
        // Print 3 periods
        println!("  Period {}:", p + 1);
        for i in 0..period {
            let idx = p * period + i;
            if idx < horizon {
                println!("    Day {}: {:.2}", idx + 1, forecasts[idx]);
            }
        }
    }
    println!();

    // Evaluate model on test data
    println!("Evaluating model on test data...");
    let evaluation = model.evaluate(&test_data)?;

    // Print evaluation metrics
    println!("Evaluation metrics:");
    println!("MAE:   {:.4}", evaluation.mae);
    println!("RMSE:  {:.4}", evaluation.rmse);
    println!("MAPE:  {:.2}%", evaluation.mape);
    println!("SMAPE: {:.2}%\n", evaluation.smape);

    // Compare with different gamma values
    println!("Comparing Holt-Winters models with different gamma values...");
    compare_gamma_values(&train_data, &test_data, period)?;

    println!("Demo completed successfully!");
    Ok(())
}

fn generate_synthetic_data() -> TimeSeriesData {
    let now = Utc::now();
    let n = 140; // 20 weeks of daily data
    let period = 7; // Weekly seasonality

    let mut timestamps = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);

    for i in 0..n {
        let day = now + Duration::days(i as i64);
        timestamps.push(day);

        // Base level with linear trend
        let base = 100.0;
        let trend = 0.2 * i as f64;

        // Weekly seasonal pattern (higher on weekends, lower mid-week)
        let seasonal = 15.0 * (2.0 * PI * (i % period) as f64 / period as f64).sin();

        // Add small random noise
        let noise = (i as f64 * 0.3).cos() * 3.0;

        values.push(base + trend + seasonal + noise);
    }

    TimeSeriesData::new(
        timestamps,
        values,
        "Synthetic data with trend and seasonality for Holt-Winters demo",
    )
    .unwrap()
}

fn compare_gamma_values(
    train_data: &TimeSeriesData,
    test_data: &TimeSeriesData,
    period: usize,
) -> Result<(), Box<dyn Error>> {
    println!("| Alpha | Beta | Gamma | MAE     | RMSE    | MAPE    |");
    println!("|-------|------|-------|---------|---------|---------|");

    // Fix alpha and beta, vary gamma
    let alpha = 0.2;
    let beta = 0.1;

    for gamma in [0.1, 0.3, 0.5, 0.7, 0.9].iter() {
        let mut model = HoltWintersModel::new(alpha, beta, *gamma, period)?;
        model.fit(train_data)?;
        let eval = model.evaluate(test_data)?;

        println!(
            "| {:.1}   | {:.1}  | {:.1}   | {:.4} | {:.4} | {:.2}% |",
            alpha, beta, gamma, eval.mae, eval.rmse, eval.mape
        );
    }

    println!();
    println!("Alpha controls the level smoothing.");
    println!("Beta controls the trend smoothing.");
    println!("Gamma controls the seasonal component smoothing (higher values adapt faster to seasonal changes).");

    Ok(())
}
