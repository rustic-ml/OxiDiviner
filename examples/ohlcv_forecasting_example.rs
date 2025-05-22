use chrono::{DateTime, TimeZone, Utc};
use oxidiviner::prelude::*;
use std::error::Error;
use std::path::Path;
use std::num::ParseFloatError;

// Main function that demonstrates OHLCV data loading and forecasting
fn main() -> std::result::Result<(), Box<dyn Error>> {
    println!("OxiDiviner - OHLCV Data Forecasting Example");
    println!("============================================\n");

    // Load OHLCV data from CSV
    println!("Loading AAPL daily OHLCV data...");
    let data = load_ohlcv_data("csv/AAPL_daily_ohlcv.csv")?;
    
    println!("Data loaded: {} rows from {} to {}", 
             data.timestamps.len(),
             data.timestamps.first().unwrap().date(),
             data.timestamps.last().unwrap().date());
    
    // Convert to time series for forecasting (using close prices)
    let time_series = data.to_time_series(false);  // false = use regular close, not adjusted
    
    // Split into training and test sets (last 30 days for testing)
    let train_size = time_series.len() - 30;
    let (train_data, test_data) = split_time_series(&time_series, train_size)?;
    
    println!("\nSplit into training ({} points) and test ({} points) sets", 
             train_data.len(), test_data.len());
    
    // Forecast using different models
    let forecast_horizon = 30;  // Forecast the next 30 days
    
    // Create and fit models
    println!("\nTraining and evaluating models...");
    
    // 1. Moving Average model
    println!("\n1. Moving Average Model (MA):");
    let mut ma_model = MAModel::new(7)?;  // 7-day moving average
    ma_model.fit(&train_data)?;
    let ma_output = ma_model.predict(forecast_horizon, Some(&test_data))?;
    print_model_evaluation(&ma_output);
    
    // 2. Simple Exponential Smoothing model
    println!("\n2. Simple Exponential Smoothing Model (SES):");
    let mut ses_model = SESModel::new(0.3)?;  // Only the alpha parameter
    ses_model.fit(&train_data)?;
    let ses_output = ses_model.predict(forecast_horizon, Some(&test_data))?;
    print_model_evaluation(&ses_output);
    
    // 3. Holt-Winters model with weekly seasonality
    println!("\n3. Holt-Winters Model with Weekly Seasonality:");
    let mut hw_model = HoltWintersModel::new(0.2, 0.1, 0.1, 5)?;
    hw_model.fit(&train_data)?;
    let hw_output = hw_model.predict(forecast_horizon, Some(&test_data))?;
    print_model_evaluation(&hw_output);
    
    // 4. Autoregressive model
    println!("\n4. Autoregressive Model (AR):");
    let mut ar_model = ARModel::new(5, true)?;  // AR(5) model with intercept
    ar_model.fit(&train_data)?;
    let ar_output = ar_model.predict(forecast_horizon, Some(&test_data))?;
    print_model_evaluation(&ar_output);
    
    // Compare the first few days of forecasts
    println!("\nComparison of first 7 days of forecasts:");
    println!("Day | Actual    | MA        | SES       | HW        | AR");
    println!("----|-----------|-----------|-----------|-----------|----------");
    
    for i in 0..7.min(test_data.len()) {
        println!("{:3} | {:9.2} | {:9.2} | {:9.2} | {:9.2} | {:9.2}",
                 i + 1,
                 test_data.values[i],
                 ma_output.forecasts[i],
                 ses_output.forecasts[i],
                 hw_output.forecasts[i],
                 ar_output.forecasts[i]);
    }
    
    // Calculate a simple ensemble forecast (average of all models)
    println!("\nCalculating ensemble forecast (average of all models)...");
    
    let mut ensemble_forecast = vec![0.0; forecast_horizon.min(test_data.len())];
    for i in 0..ensemble_forecast.len() {
        ensemble_forecast[i] = (
            ma_output.forecasts[i] +
            ses_output.forecasts[i] +
            hw_output.forecasts[i] +
            ar_output.forecasts[i]
        ) / 4.0;
    }
    
    // Calculate ensemble error metrics
    let ensemble_mae = calculate_mae(&test_data.values[0..ensemble_forecast.len()], &ensemble_forecast);
    let ensemble_rmse = calculate_rmse(&test_data.values[0..ensemble_forecast.len()], &ensemble_forecast);
    
    println!("\nEnsemble Model Performance:");
    println!("  MAE:  {:.4}", ensemble_mae);
    println!("  RMSE: {:.4}", ensemble_rmse);
    
    // Compare the best individual model with the ensemble
    println!("\nEnsemble vs Individual Models:");
    
    let models = vec![
        ("MA", ma_output.evaluation.as_ref().unwrap().mae, ma_output.evaluation.as_ref().unwrap().rmse),
        ("SES", ses_output.evaluation.as_ref().unwrap().mae, ses_output.evaluation.as_ref().unwrap().rmse),
        ("HW", hw_output.evaluation.as_ref().unwrap().mae, hw_output.evaluation.as_ref().unwrap().rmse),
        ("AR", ar_output.evaluation.as_ref().unwrap().mae, ar_output.evaluation.as_ref().unwrap().rmse),
        ("Ensemble", ensemble_mae, ensemble_rmse),
    ];
    
    // Find the best model based on MAE
    let best_model = models.iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    
    println!("Best model by MAE: {} (MAE: {:.4}, RMSE: {:.4})", 
             best_model.0, best_model.1, best_model.2);
    
    Ok(())
}

// Helper function to calculate Mean Absolute Error
fn calculate_mae(actual: &[f64], forecast: &[f64]) -> f64 {
    let sum: f64 = actual.iter().zip(forecast.iter())
        .map(|(a, f)| (a - f).abs())
        .sum();
    
    sum / actual.len() as f64
}

// Helper function to calculate Root Mean Squared Error
fn calculate_rmse(actual: &[f64], forecast: &[f64]) -> f64 {
    let sum: f64 = actual.iter().zip(forecast.iter())
        .map(|(a, f)| (a - f).powi(2))
        .sum();
    
    (sum / actual.len() as f64).sqrt()
}

// Helper function to load OHLCV data from a CSV file
fn load_ohlcv_data(file_path: &str) -> std::result::Result<OHLCVData, Box<dyn Error>> {
    // Extract symbol from filename
    let path = Path::new(file_path);
    let file_stem = path.file_stem().unwrap().to_str().unwrap();
    let symbol = file_stem.split('_').next().unwrap_or("UNKNOWN");
    
    // Read the content of the CSV file
    let content = std::fs::read_to_string(file_path)?;
    let lines: Vec<&str> = content.lines().collect();
    
    // Skip header row
    let data_lines = &lines[1..];
    
    let mut timestamps = Vec::with_capacity(data_lines.len());
    let mut open = Vec::with_capacity(data_lines.len());
    let mut high = Vec::with_capacity(data_lines.len());
    let mut low = Vec::with_capacity(data_lines.len());
    let mut close = Vec::with_capacity(data_lines.len());
    let mut volume = Vec::with_capacity(data_lines.len());
    
    for line in data_lines {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;  // Skip malformed lines
        }
        
        // Parse timestamp: format is "YYYY-MM-DD HH:MM:SS UTC"
        let timestamp_str = parts[0].trim();
        let timestamp = match Utc.datetime_from_str(timestamp_str, "%Y-%m-%d %H:%M:%S %Z") {
            Ok(dt) => dt,
            Err(_) => continue,  // Skip lines with invalid timestamps
        };
        
        // Parse price and volume data
        let open_val = match parts[1].trim().parse::<f64>() {
            Ok(val) => val,
            Err(_) => continue,
        };
        let high_val = match parts[2].trim().parse::<f64>() {
            Ok(val) => val,
            Err(_) => continue,
        };
        let low_val = match parts[3].trim().parse::<f64>() {
            Ok(val) => val,
            Err(_) => continue,
        };
        let close_val = match parts[4].trim().parse::<f64>() {
            Ok(val) => val,
            Err(_) => continue,
        };
        let volume_val = match parts[5].trim().parse::<f64>() {
            Ok(val) => val,
            Err(_) => continue,
        };
        
        timestamps.push(timestamp);
        open.push(open_val);
        high.push(high_val);
        low.push(low_val);
        close.push(close_val);
        volume.push(volume_val);
    }
    
    if timestamps.is_empty() {
        return Err("No valid data found in CSV file".into());
    }
    
    Ok(OHLCVData {
        symbol: symbol.to_string(),
        timestamps,
        open,
        high,
        low,
        close,
        volume,
        adjusted_close: None,
    })
}

// Helper function to split a time series into training and test sets
fn split_time_series(data: &TimeSeriesData, train_size: usize) -> std::result::Result<(TimeSeriesData, TimeSeriesData), Box<dyn Error>> {
    if train_size >= data.len() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Train size must be less than the total data length"
        )));
    }
    
    let train = TimeSeriesData {
        timestamps: data.timestamps[0..train_size].to_vec(),
        values: data.values[0..train_size].to_vec(),
        name: format!("{}_train", data.name),
    };
    
    let test = TimeSeriesData {
        timestamps: data.timestamps[train_size..].to_vec(),
        values: data.values[train_size..].to_vec(),
        name: format!("{}_test", data.name),
    };
    
    Ok((train, test))
}

// Helper function to print model evaluation results
fn print_model_evaluation(output: &ModelOutput) {
    if let Some(eval) = &output.evaluation {
        println!("  Model: {}", output.model_name);
        println!("  MAE:  {:.4}", eval.mae);
        println!("  RMSE: {:.4}", eval.rmse);
        println!("  MAPE: {:.2}%", eval.mape);
    } else {
        println!("  No evaluation available for {}", output.model_name);
    }
} 