use std::path::Path;
use oxidiviner::data::{OHLCVData, TimeSeriesData};
use oxidiviner::models::ets::{SimpleExponentialSmoothing, HoltLinearTrend, HoltWintersAdditive};
use oxidiviner::models::Forecaster;
use oxidiviner::prelude::*;
use polars::prelude::*;
use rustalib::util::file_utils;

fn main() -> Result<()> {
    // Load sample OHLCV data
    let file_path = "examples/csv/AAPL_daily_ohlcv.csv";
    println!("Loading data from {}...", file_path);
    
    // Load and preprocess CSV manually since the format is different from expected
    let ohlcv_data = load_ohlcv_from_csv(file_path)?;
    
    // Convert to time series (using close prices)
    let time_series = ohlcv_data.to_time_series(false);
    
    println!("Loaded {} data points for {} from {} to {}", 
             time_series.len(),
             ohlcv_data.symbol,
             time_series.timestamps.first().unwrap().date_naive(),
             time_series.timestamps.last().unwrap().date_naive());
    
    // Split into training and test sets (80% train, 20% test)
    let (train_data, test_data) = time_series.train_test_split(0.8)?;
    println!("Split data into {} training points and {} test points", 
             train_data.len(), test_data.len());
    
    // Create models
    let models = create_models()?;
    
    // Train and evaluate models
    println!("\n===== MODEL COMPARISON =====");
    for mut model in models {
        // Train model
        println!("\nTraining model: {}", model.name());
        model.fit(&train_data)?;
        
        // Evaluate model
        let evaluation = model.evaluate(&test_data)?;
        print_evaluation(&evaluation);
        
        // Generate forecast for next 30 days
        let horizon = 30;
        let forecasts = model.forecast(horizon)?;
        
        println!("Forecast for next {} days:", horizon);
        println!("  First 5 days: {:.2}, {:.2}, {:.2}, {:.2}, {:.2}", 
                 forecasts[0], forecasts[1], forecasts[2], forecasts[3], forecasts[4]);
        println!("  Last 5 days:  {:.2}, {:.2}, {:.2}, {:.2}, {:.2}", 
                 forecasts[horizon-5], forecasts[horizon-4], forecasts[horizon-3], 
                 forecasts[horizon-2], forecasts[horizon-1]);
    }
    
    Ok(())
}

fn load_ohlcv_from_csv(path: &str) -> Result<OHLCVData> {
    OHLCVData::from_csv(path)
}

fn create_models() -> Result<Vec<Box<dyn Forecaster>>> {
    let mut models: Vec<Box<dyn Forecaster>> = Vec::new();
    
    // Simple Exponential Smoothing with different alpha values
    models.push(Box::new(SimpleExponentialSmoothing::new(0.1)?));
    models.push(Box::new(SimpleExponentialSmoothing::new(0.3)?));
    models.push(Box::new(SimpleExponentialSmoothing::new(0.5)?));
    
    // Holt's Linear Trend models
    models.push(Box::new(HoltLinearTrend::new(0.3, 0.1)?));
    models.push(Box::new(HoltLinearTrend::with_damped_trend(0.3, 0.1, 0.9)?));
    
    // Holt-Winters' Seasonal Method (assuming daily data with weekly seasonality)
    models.push(Box::new(HoltWintersAdditive::new(0.3, 0.1, 0.1, 7)?));
    
    Ok(models)
}

fn print_evaluation(eval: &ModelEvaluation) {
    println!("Model Evaluation:");
    println!("  MAE:   {:.4}", eval.mae);
    println!("  RMSE:  {:.4}", eval.rmse);
    println!("  MAPE:  {:.4}%", eval.mape);
} 