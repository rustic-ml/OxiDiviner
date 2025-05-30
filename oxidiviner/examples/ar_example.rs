//! AR (AutoRegressive) Model Example
//!
//! This example demonstrates how to use AR models for time series forecasting.
//! AR models are useful for data with autocorrelation patterns where current values
//! depend on previous values.

use chrono::{DateTime, Duration, Utc};
use oxidiviner::core::Forecaster;
use oxidiviner::models::autoregressive::ARModel;
use oxidiviner::prelude::*;

fn main() -> oxidiviner::Result<()> {
    println!("=== AR Model Example ===\n");

    // Generate sample AR(2) data
    let start_date = Utc::now() - Duration::days(100);
    let timestamps: Vec<DateTime<Utc>> = (0..100).map(|i| start_date + Duration::days(i)).collect();

    // Create AR(2) process: y_t = 0.6*y_{t-1} + 0.3*y_{t-2} + noise
    let mut values = vec![0.0, 1.0]; // Initial values
    for i in 2..100 {
        let ar_component = 0.6 * values[i - 1] + 0.3 * values[i - 2];
        let noise = (rand::random::<f64>() - 0.5) * 2.0;
        values.push(ar_component + noise);
    }

    println!("Generated {} AR(2) data points", values.len());
    println!(
        "Data range: {:.2} to {:.2}",
        values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Create time series data
    let ts_data = TimeSeriesData::new(timestamps.clone(), values.clone(), "ar_series")?;

    // Split data into train/test (80/20 split)
    let split_idx = (values.len() as f64 * 0.8) as usize;
    let train_data = TimeSeriesData::new(
        timestamps[..split_idx].to_vec(),
        values[..split_idx].to_vec(),
        "train_data",
    )?;
    let test_data = TimeSeriesData::new(
        timestamps[split_idx..].to_vec(),
        values[split_idx..].to_vec(),
        "test_data",
    )?;

    println!(
        "\nData split: {} training, {} testing observations",
        train_data.len(),
        test_data.len()
    );

    // Example 1: AR(1) model
    println!("\n1. AR(1) Model");
    println!("==============");

    let mut ar1_model = ARModel::new(1, true)?;
    ar1_model.fit(&train_data)?;
    let _ar1_forecast = ar1_model.forecast(test_data.len())?;
    let ar1_eval = ar1_model.evaluate(&test_data)?;

    println!("AR(1) Performance:");
    println!("  MAE:  {:.3}", ar1_eval.mae);
    println!("  RMSE: {:.3}", ar1_eval.rmse);
    println!("  MAPE: {:.2}%", ar1_eval.mape);

    // Example 2: AR(2) model (should perform better on our AR(2) data)
    println!("\n2. AR(2) Model");
    println!("==============");

    let mut ar2_model = ARModel::new(2, true)?;
    ar2_model.fit(&train_data)?;
    let _ar2_forecast = ar2_model.forecast(test_data.len())?;
    let ar2_eval = ar2_model.evaluate(&test_data)?;

    println!("AR(2) Performance:");
    println!("  MAE:  {:.3}", ar2_eval.mae);
    println!("  RMSE: {:.3}", ar2_eval.rmse);
    println!("  MAPE: {:.2}%", ar2_eval.mape);

    // Example 3: Compare different AR orders
    println!("\n3. Comparing Different AR Orders");
    println!("================================");

    for order in 1..=5 {
        match ARModel::new(order, true) {
            Ok(mut model) => match model.fit(&train_data) {
                Ok(_) => match model.evaluate(&test_data) {
                    Ok(eval) => {
                        println!(
                            "  AR({}): RMSE = {:.3}, MAE = {:.3}",
                            order, eval.rmse, eval.mae
                        );
                    }
                    Err(_) => println!("  AR({}): Evaluation failed", order),
                },
                Err(_) => println!("  AR({}): Fit failed", order),
            },
            Err(_) => println!("  AR({}): Model creation failed", order),
        }
    }

    // Example 4: Future forecasting
    println!("\n4. Future Forecasting");
    println!("=====================");

    let mut final_model = ARModel::new(2, true)?;
    final_model.fit(&ts_data)?;
    let future_forecast = final_model.forecast(10)?;

    println!("Next 10-period forecast:");
    for (i, &forecast_val) in future_forecast.iter().enumerate() {
        println!("  Period {}: {:.2}", i + 1, forecast_val);
    }

    // Example 5: Model diagnostics
    println!("\n5. Model Diagnostics");
    println!("====================");

    if let Some(coeffs) = final_model.coefficients() {
        println!("AR coefficients: {:?}", coeffs);
    }

    // Check model info
    println!("Model name: {}", final_model.name());

    println!("\n=== AR Example Complete ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ar_example() {
        let result = main();
        assert!(
            result.is_ok(),
            "AR example should run successfully: {:?}",
            result
        );
    }
}
