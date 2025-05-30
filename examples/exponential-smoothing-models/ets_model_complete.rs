use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use oxidiviner::TimeSeriesData;
use oxidiviner::{DampedTrendModel, ETSModel, HoltLinearModel, HoltWintersModel, SimpleESModel};
use std::error::Error;

/// This example demonstrates the complete ETS (Error-Trend-Seasonal) model framework
/// which includes all the exponential smoothing variants as special cases.
///
/// The demonstration includes:
/// 1. Creating different ETS models for different time series patterns
/// 2. Fitting the models to training data
/// 3. Generating forecasts
/// 4. Evaluating model performance
///
/// This is especially useful for traders who need to forecast financial time series
/// with different characteristics.
fn main() -> Result<(), Box<dyn Error>> {
    // Create simulated price data (could be stock prices, forex rates, etc.)
    println!("=== Exponential Smoothing Models for Trading Data ===\n");

    // ---- Simple trend data (e.g., steadily rising asset) ----
    println!("Example 1: Simple trending data (steadily rising asset)");
    let trending_data = create_trending_data()?;

    // Use Simple Exponential Smoothing, Holt Linear, and Damped Trend models
    println!("Comparing different forecasting models:");

    // Simple ES (good baseline but doesn't handle trend well)
    let mut ses_model = SimpleESModel::new(0.3)?;
    ses_model.fit(&trending_data)?;
    let ses_forecast = ses_model.forecast(10)?;
    println!("  Simple ES forecast (10 periods): {:.2?}", ses_forecast);

    // Holt Linear (handles trend well)
    let mut holt_model = HoltLinearModel::new(0.8, 0.2)?;
    holt_model.fit(&trending_data)?;
    let holt_forecast = holt_model.forecast(10)?;
    println!("  Holt Linear forecast (10 periods): {:.2?}", holt_forecast);

    // Damped Trend (conservative trend forecasts)
    let mut damped_model = DampedTrendModel::new(0.8, 0.2, 0.9)?;
    damped_model.fit(&trending_data)?;
    let damped_forecast = damped_model.forecast(10)?;
    println!(
        "  Damped Trend forecast (10 periods): {:.2?}",
        damped_forecast
    );

    // ETS(A,A,N) (equivalent to Holt Linear)
    let mut ets_holt = ETSModel::holt(0.8, 0.2)?;
    ets_holt.fit(&trending_data)?;
    let ets_holt_forecast = ets_holt.forecast(10)?;
    println!(
        "  ETS(A,A,N) forecast (10 periods): {:.2?}",
        ets_holt_forecast
    );

    println!("\nFor trending assets, Holt Linear and ETS(A,A,N) usually produce the most accurate forecasts.\n");

    // ---- Seasonal data (e.g., asset with monthly seasonality) ----
    println!("Example 2: Seasonal data (asset with monthly seasonality)");
    let seasonal_data = create_seasonal_data()?;

    // Holt-Winters (handles both trend and seasonality)
    let mut hw_model = HoltWintersModel::new(0.4, 0.2, 0.3, 12)?; // 12 = monthly seasonality
    hw_model.fit(&seasonal_data)?;
    let hw_forecast = hw_model.forecast(24)?; // forecast 2 years ahead

    // Only print the first 12 months to keep output readable
    println!(
        "  Holt-Winters forecast (first 12 periods): {:.2?}",
        &hw_forecast[..12]
    );

    // ETS with multiplicative seasonality
    let mut ets_hw_mult = ETSModel::holt_winters_multiplicative(0.4, 0.2, 0.3, 12)?;
    ets_hw_mult.fit(&seasonal_data)?;
    let ets_mult_forecast = ets_hw_mult.forecast(24)?;

    println!(
        "  ETS(A,A,M) forecast (first 12 periods): {:.2?}",
        &ets_mult_forecast[..12]
    );

    println!(
        "\nFor seasonal assets, Holt-Winters and ETS with seasonality component should be used.\n"
    );

    // ---- Mean-reverting data (e.g., trading ranges) ----
    println!("Example 3: Mean-reverting data (trading range assets)");
    let ranging_data = create_ranging_data()?;

    // Simple ES (works well for mean-reverting series)
    let mut ses_range_model = SimpleESModel::new(0.2)?; // Low alpha for smoother forecasts
    ses_range_model.fit(&ranging_data)?;
    let ses_range_forecast = ses_range_model.forecast(10)?;
    println!(
        "  Simple ES forecast (10 periods): {:.2?}",
        ses_range_forecast
    );

    // ETS simple model
    let mut ets_simple = ETSModel::simple(0.2)?;
    ets_simple.fit(&ranging_data)?;
    let ets_simple_forecast = ets_simple.forecast(10)?;
    println!(
        "  ETS(A,N,N) forecast (10 periods): {:.2?}",
        ets_simple_forecast
    );

    println!("\nFor ranging/mean-reverting assets, Simple ES and ETS(A,N,N) often work best.\n");

    // ---- Volatile data with damped trend (e.g., crypto assets) ----
    println!("Example 4: Volatile data with temporary trends (crypto assets)");
    let volatile_data = create_volatile_data()?;

    // Damped Trend (good for temporary trends that don't persist)
    let mut damped_volatile = DampedTrendModel::new(0.6, 0.1, 0.85)?;
    damped_volatile.fit(&volatile_data)?;
    let damped_volatile_forecast = damped_volatile.forecast(10)?;
    println!(
        "  Damped Trend forecast (10 periods): {:.2?}",
        damped_volatile_forecast
    );

    // ETS with damped trend
    let mut ets_damped = ETSModel::damped_trend(0.6, 0.1, 0.85)?;
    ets_damped.fit(&volatile_data)?;
    let ets_damped_forecast = ets_damped.forecast(10)?;
    println!(
        "  ETS(A,Ad,N) forecast (10 periods): {:.2?}",
        ets_damped_forecast
    );

    println!("\nFor volatile assets with temporary trends, Damped Trend and ETS(A,Ad,N) are often more reliable.\n");

    println!("=== Trading Strategy Recommendations ===");
    println!("1. For trending markets: Use Holt Linear or ETS(A,A,N)");
    println!("2. For seasonal markets: Use Holt-Winters or ETS with seasonality");
    println!("3. For range-bound markets: Use Simple ES or ETS(A,N,N)");
    println!("4. For volatile markets: Use Damped Trend or ETS(A,Ad,N)");
    println!("5. Consider ensemble approaches combining multiple models for more robust forecasts");

    Ok(())
}

// Create a trending time series (steadily rising asset)
fn create_trending_data() -> Result<TimeSeriesData, Box<dyn Error>> {
    let base_date = NaiveDate::from_ymd_opt(2022, 1, 1).unwrap();
    let timestamps: Vec<DateTime<Utc>> = (0..100)
        .map(|i| {
            Utc.from_utc_datetime(
                &base_date
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .checked_add_signed(chrono::Duration::days(i))
                    .unwrap(),
            )
        })
        .collect();

    // Linear trend with small noise
    let mut values = Vec::with_capacity(100);
    for i in 0..100 {
        let trend = 100.0 + 0.5 * (i as f64);
        let noise = (i % 7) as f64 * 0.3 - 1.0; // small noise
        values.push(trend + noise);
    }

    Ok(TimeSeriesData::new(timestamps, values, "trending_asset")?)
}

// Create a seasonal time series (asset with monthly pattern)
fn create_seasonal_data() -> Result<TimeSeriesData, Box<dyn Error>> {
    let base_date = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
    let timestamps: Vec<DateTime<Utc>> = (0..120) // 10 years of monthly data
        .map(|i| {
            Utc.from_utc_datetime(
                &base_date
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .checked_add_signed(chrono::Duration::days(30 * i))
                    .unwrap(),
            )
        })
        .collect();

    // Trend with seasonality
    let mut values = Vec::with_capacity(120);
    for i in 0..120 {
        let trend = 100.0 + 0.2 * (i as f64);
        let season = 5.0 * (2.0 * std::f64::consts::PI * ((i % 12) as f64) / 12.0).sin();
        let noise = ((i * 17) % 7) as f64 * 0.2 - 0.6; // small random noise
        values.push(trend + season + noise);
    }

    Ok(TimeSeriesData::new(timestamps, values, "seasonal_asset")?)
}

// Create a mean-reverting time series (trading range asset)
fn create_ranging_data() -> Result<TimeSeriesData, Box<dyn Error>> {
    let base_date = NaiveDate::from_ymd_opt(2022, 1, 1).unwrap();
    let timestamps: Vec<DateTime<Utc>> = (0..100)
        .map(|i| {
            Utc.from_utc_datetime(
                &base_date
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .checked_add_signed(chrono::Duration::days(i))
                    .unwrap(),
            )
        })
        .collect();

    // Mean-reverting process (Ornstein-Uhlenbeck process simplification)
    let mean = 100.0;
    let mut value = mean;
    let mut values = Vec::with_capacity(100);

    for i in 0..100 {
        let reversion_speed = 0.1;
        let volatility = 1.0;
        let noise = ((i * 83) % 17) as f64 * 0.2 - 1.7; // quasi-random noise

        // Mean-reverting formula: next = current + reversion_speed * (mean - current) + noise
        value = value + reversion_speed * (mean - value) + volatility * noise;
        values.push(value);
    }

    Ok(TimeSeriesData::new(timestamps, values, "ranging_asset")?)
}

// Create a volatile time series with temporary trends (crypto-like asset)
fn create_volatile_data() -> Result<TimeSeriesData, Box<dyn Error>> {
    let base_date = NaiveDate::from_ymd_opt(2022, 1, 1).unwrap();
    let timestamps: Vec<DateTime<Utc>> = (0..100)
        .map(|i| {
            Utc.from_utc_datetime(
                &base_date
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .checked_add_signed(chrono::Duration::days(i))
                    .unwrap(),
            )
        })
        .collect();

    // Volatile data with regime changes
    let mut values = Vec::with_capacity(100);
    let mut value = 100.0;

    for i in 0..100 {
        let regime = (i / 20) % 3; // Change trend direction every 20 days
        let trend = match regime {
            0 => 0.8,  // Strong up
            1 => -0.5, // Down
            _ => 0.1,  // Weak up
        };

        let volatility = 3.0;
        let noise = ((i * 73) % 19) as f64 * 0.3 - 2.5; // high noise

        value = value + trend + volatility * noise;
        if value < 50.0 {
            value = 50.0;
        } // Set a floor
        values.push(value);
    }

    Ok(TimeSeriesData::new(timestamps, values, "volatile_asset")?)
}
