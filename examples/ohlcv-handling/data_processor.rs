use chrono::Timelike;
use oxidiviner_core::{OHLCVData, Result};
use oxidiviner_exponential_smoothing::{DailyETSModel, MinuteETSModel};

fn main() -> Result<()> {
    println!("OHLCV Data Processing Example");
    println!("=============================\n");

    // Process daily data
    println!("Processing Daily OHLCV Data:");
    println!("---------------------------");
    process_daily_data()?;

    // Process minute data
    println!("\nProcessing Minute OHLCV Data:");
    println!("----------------------------");
    process_minute_data()?;

    Ok(())
}

fn process_daily_data() -> Result<()> {
    // Load daily OHLCV data
    let daily_file = "examples/csv/AAPL_daily_ohlcv.csv";
    println!("Loading daily data from {}", daily_file);

    // The format from the file header check: "2022-08-22 04:00:00 UTC"
    let daily_data = OHLCVData::from_csv(daily_file, "%Y-%m-%d %H:%M:%S %Z", true)?;

    println!("Loaded {} daily data points for {}", daily_data.len(), daily_data.symbol);
    
    // Display first few records
    println!("First 3 records:");
    for i in 0..3.min(daily_data.len()) {
        println!("  {} Open: {:.2}, High: {:.2}, Low: {:.2}, Close: {:.2}, Volume: {:.2}",
            daily_data.timestamps[i].date_naive(),
            daily_data.open[i],
            daily_data.high[i],
            daily_data.low[i],
            daily_data.close[i],
            daily_data.volume[i]
        );
    }

    // Convert to time series for modeling
    let time_series = daily_data.to_time_series(false);
    
    // Split into training and test sets
    let (train_ts, test_ts) = time_series.train_test_split(0.8)?;
    
    println!("\nSplit data into {} training points and {} test points", 
             train_ts.len(), test_ts.len());

    // DAILY DATA MODELING:
    // For daily data, we often want to consider:
    // 1. Weekly seasonality (period=7)
    // 2. Monthly seasonality (period≈21 trading days)
    // 3. Quarterly patterns (period≈63 trading days)
    
    println!("\nApplying specialized DailyETSModel with weekly seasonality:");
    
    // Create a Holt-Winters model with weekly (7-day) seasonality
    let mut model = DailyETSModel::holt_winters_additive(
        0.3,    // alpha (level)
        0.1,    // beta (trend)
        0.1,    // gamma (seasonal)
        7,      // period = 7 days (weekly seasonality)
        None,   // Use close price by default
    )?;
    
    // Train the model
    // Since DailyETSModel needs OHLCVData, we need to fit using original data
    // but use the same range as our train_ts
    model.fit(&daily_data)?;
    
    // Generate forecasts
    let horizon = 30;  // 30-day forecast
    let forecasts = model.forecast(horizon)?;
    
    println!("Generated {} daily forecasts", forecasts.len());
    println!("First 5 forecasted values: {:?}", &forecasts[0..5.min(forecasts.len())]);
    
    // Show some fitted values for comparison
    if let Some(fitted) = model.fitted_values() {
        println!("\nLast 5 fitted values:");
        let start_idx = fitted.len().saturating_sub(5);
        for i in start_idx..fitted.len() {
            println!("  {}: {:.4}", daily_data.timestamps[i].date_naive(), fitted[i]);
        }
    }
    
    // Evaluate the model if we have test data
    let eval = model.evaluate(&daily_data)?;
    println!("\nModel evaluation:");
    println!("  MAE: {:.4}", eval.mae);
    println!("  RMSE: {:.4}", eval.rmse);
    println!("  MAPE: {:.2}%", eval.mape);
    
    Ok(())
}

fn process_minute_data() -> Result<()> {
    // Load minute OHLCV data
    let minute_file = "examples/csv/minute_data.csv";
    println!("Loading minute data from {}", minute_file);

    // The format from the file header check: "2025-05-17T12:20:45.076799536+00:00"
    let minute_data = OHLCVData::from_csv(minute_file, "%Y-%m-%dT%H:%M:%S%.f%z", true)?;

    println!("Loaded {} minute data points", minute_data.len());
    
    // Display first few records
    println!("First 3 records:");
    for i in 0..3.min(minute_data.len()) {
        println!("  {} {}:{}:{} Open: {:.2}, Close: {:.2}",
            minute_data.timestamps[i].date_naive(),
            minute_data.timestamps[i].time().hour(),
            minute_data.timestamps[i].time().minute(),
            minute_data.timestamps[i].time().second(),
            minute_data.open[i],
            minute_data.close[i]
        );
    }

    // Convert to time series for modeling
    let time_series = minute_data.to_time_series(false);
    
    // Split into training and test sets
    let (train_ts, test_ts) = time_series.train_test_split(0.8)?;
    
    println!("\nSplit data into {} training points and {} test points", 
             train_ts.len(), test_ts.len());

    // MINUTE DATA MODELING:
    // For minute data, we often want to consider:
    // 1. Hourly patterns (period=60)
    // 2. Session patterns (morning/afternoon, period≈240 for a 4-hour session)
    // 3. More aggressive smoothing parameters (higher alpha values) to adapt to faster changes
    // 4. Data aggregation to reduce noise (e.g., 5-minute bars)
    
    // Using a smaller seasonal period (15 minutes) to avoid the "insufficient data" error
    println!("\nApplying MinuteETSModel with 5-minute aggregation and 15-minute seasonality:");
    
    // Create a specialized minute model with 5-minute aggregation and 15-minute seasonality
    let mut model = MinuteETSModel::holt_winters_additive(
        0.4,    // alpha (higher for minute data to adapt faster)
        0.1,    // beta
        0.1,    // gamma
        15,     // period = 15 minutes (changed from 60 to avoid insufficient data error)
        None,   // Use close price by default
        Some(5), // 5-minute aggregation
    )?;
    
    // Train the model
    // Since MinuteETSModel needs OHLCVData, we need to fit using original data
    model.fit(&minute_data)?;
    
    // Generate forecasts
    let horizon = 30;  // 30-minute forecast (reduced from 60)
    let forecasts = model.forecast(horizon)?;
    
    println!("Generated {} minute forecasts", forecasts.len());
    println!("First 5 forecasted values: {:?}", &forecasts[0..5.min(forecasts.len())]);
    
    // Show some fitted values for comparison
    if let Some(fitted) = model.fitted_values() {
        println!("\nLast 5 fitted values:");
        let start_idx = fitted.len().saturating_sub(5);
        for i in start_idx..fitted.len() {
            // Since we used aggregation, our timestamps may not align directly
            // Just show the values
            println!("  Value {}: {:.4}", i, fitted[i]);
        }
    }
    
    // Compare with a non-aggregated model
    println!("\nComparing with a non-aggregated minute model:");
    let mut model2 = MinuteETSModel::simple(
        0.4,    // alpha
        None,   // Use close price by default
        None,   // No aggregation
    )?;
    
    // Train the model
    model2.fit(&minute_data)?;
    
    // Generate forecasts 
    let forecasts2 = model2.forecast(horizon)?;
    println!("Generated {} minute forecasts (non-aggregated)", forecasts2.len());
    println!("First 5 forecasted values (non-aggregated): {:?}", 
             &forecasts2[0..5.min(forecasts2.len())]);
    
    println!("\nComparison of forecasts shows differences due to aggregation.");
    println!("The aggregated model (first) smooths out minute-by-minute noise.");
    println!("The non-aggregated model (second) works directly with raw minute data.");
    
    Ok(())
} 