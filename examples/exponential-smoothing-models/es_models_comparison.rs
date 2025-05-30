use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use oxidiviner::TimeSeriesData;
use oxidiviner::{DampedTrendModel, ETSModel, HoltLinearModel, HoltWintersModel, SimpleESModel};
use std::error::Error;

/// This example compares the forecast accuracy of different exponential smoothing models
/// on various types of time series data typical in financial markets.
///
/// It helps traders understand which model is most appropriate for different market conditions.
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Exponential Smoothing Model Comparison for Traders ===\n");

    // Create a data structure to hold our results
    #[derive(Debug, Clone)]
    struct ModelEvaluation {
        model_name: String,
        mae: f64,
        rmse: f64,
    }

    let mut eval_results: Vec<Vec<ModelEvaluation>> = Vec::new();
    let scenarios = [
        "Trending Market",
        "Sideways Market",
        "Seasonal Market",
        "Volatile Market",
    ];

    // 1. Trending Market (e.g., Bull Market)
    println!("Scenario 1: Trending Market (Bull Market)");
    let (train_data, test_data) = create_and_split_trending_data()?;

    // Create and evaluate different models
    let mut trend_results = Vec::new();

    // Simple ES
    let mut ses_model = SimpleESModel::new(0.3)?;
    ses_model.fit(&train_data)?;
    let ses_eval = ses_model.evaluate(&test_data)?;
    trend_results.push(ModelEvaluation {
        model_name: "Simple ES".to_string(),
        mae: ses_eval.mae,
        rmse: ses_eval.rmse,
    });

    // Holt Linear
    let mut holt_model = HoltLinearModel::new(0.8, 0.2)?;
    holt_model.fit(&train_data)?;
    let holt_eval = holt_model.evaluate(&test_data)?;
    trend_results.push(ModelEvaluation {
        model_name: "Holt Linear".to_string(),
        mae: holt_eval.mae,
        rmse: holt_eval.rmse,
    });

    // Damped Trend
    let mut damped_model = DampedTrendModel::new(0.8, 0.2, 0.9)?;
    damped_model.fit(&train_data)?;
    let damped_eval = damped_model.evaluate(&test_data)?;
    trend_results.push(ModelEvaluation {
        model_name: "Damped Trend".to_string(),
        mae: damped_eval.mae,
        rmse: damped_eval.rmse,
    });

    // ETS
    let mut ets_model = ETSModel::holt(0.8, 0.2)?;
    ets_model.fit(&train_data)?;
    let ets_eval = ets_model.evaluate(&test_data)?;
    trend_results.push(ModelEvaluation {
        model_name: "ETS(A,A,N)".to_string(),
        mae: ets_eval.mae,
        rmse: ets_eval.rmse,
    });

    eval_results.push(trend_results);

    // 2. Sideways Market (Range-bound)
    println!("Scenario 2: Sideways Market (Range-bound)");
    let (train_data, test_data) = create_and_split_sideways_data()?;

    // Create and evaluate different models
    let mut sideways_results = Vec::new();

    // Simple ES
    let mut ses_model = SimpleESModel::new(0.2)?;
    ses_model.fit(&train_data)?;
    let ses_eval = ses_model.evaluate(&test_data)?;
    sideways_results.push(ModelEvaluation {
        model_name: "Simple ES".to_string(),
        mae: ses_eval.mae,
        rmse: ses_eval.rmse,
    });

    // Holt Linear
    let mut holt_model = HoltLinearModel::new(0.8, 0.2)?;
    holt_model.fit(&train_data)?;
    let holt_eval = holt_model.evaluate(&test_data)?;
    sideways_results.push(ModelEvaluation {
        model_name: "Holt Linear".to_string(),
        mae: holt_eval.mae,
        rmse: holt_eval.rmse,
    });

    // Damped Trend
    let mut damped_model = DampedTrendModel::new(0.8, 0.2, 0.9)?;
    damped_model.fit(&train_data)?;
    let damped_eval = damped_model.evaluate(&test_data)?;
    sideways_results.push(ModelEvaluation {
        model_name: "Damped Trend".to_string(),
        mae: damped_eval.mae,
        rmse: damped_eval.rmse,
    });

    // ETS
    let mut ets_model = ETSModel::simple(0.2)?;
    ets_model.fit(&train_data)?;
    let ets_eval = ets_model.evaluate(&test_data)?;
    sideways_results.push(ModelEvaluation {
        model_name: "ETS(A,N,N)".to_string(),
        mae: ets_eval.mae,
        rmse: ets_eval.rmse,
    });

    eval_results.push(sideways_results);

    // 3. Seasonal Market
    println!("Scenario 3: Seasonal Market (e.g., Assets with yearly patterns)");
    let (train_data, test_data) = create_and_split_seasonal_data()?;

    // Create and evaluate different models
    let mut seasonal_results = Vec::new();

    // Simple ES
    let mut ses_model = SimpleESModel::new(0.3)?;
    ses_model.fit(&train_data)?;
    let ses_eval = ses_model.evaluate(&test_data)?;
    seasonal_results.push(ModelEvaluation {
        model_name: "Simple ES".to_string(),
        mae: ses_eval.mae,
        rmse: ses_eval.rmse,
    });

    // Holt-Winters
    let mut hw_model = HoltWintersModel::new(0.4, 0.2, 0.3, 12)?;
    hw_model.fit(&train_data)?;
    let hw_eval = hw_model.evaluate(&test_data)?;
    seasonal_results.push(ModelEvaluation {
        model_name: "Holt-Winters".to_string(),
        mae: hw_eval.mae,
        rmse: hw_eval.rmse,
    });

    // ETS with multiplicative seasonality
    let mut ets_model = ETSModel::holt_winters_multiplicative(0.4, 0.2, 0.3, 12)?;
    ets_model.fit(&train_data)?;
    let ets_eval = ets_model.evaluate(&test_data)?;
    seasonal_results.push(ModelEvaluation {
        model_name: "ETS(A,A,M)".to_string(),
        mae: ets_eval.mae,
        rmse: ets_eval.rmse,
    });

    eval_results.push(seasonal_results);

    // 4. Volatile Market
    println!("Scenario 4: Volatile Market (e.g., Crypto or High Beta Stocks)");
    let (train_data, test_data) = create_and_split_volatile_data()?;

    // Create and evaluate different models
    let mut volatile_results = Vec::new();

    // Simple ES
    let mut ses_model = SimpleESModel::new(0.3)?;
    ses_model.fit(&train_data)?;
    let ses_eval = ses_model.evaluate(&test_data)?;
    volatile_results.push(ModelEvaluation {
        model_name: "Simple ES".to_string(),
        mae: ses_eval.mae,
        rmse: ses_eval.rmse,
    });

    // Damped Trend
    let mut damped_model = DampedTrendModel::new(0.6, 0.1, 0.85)?;
    damped_model.fit(&train_data)?;
    let damped_eval = damped_model.evaluate(&test_data)?;
    volatile_results.push(ModelEvaluation {
        model_name: "Damped Trend".to_string(),
        mae: damped_eval.mae,
        rmse: damped_eval.rmse,
    });

    // ETS damped trend
    let mut ets_model = ETSModel::damped_trend(0.6, 0.1, 0.85)?;
    ets_model.fit(&train_data)?;
    let ets_eval = ets_model.evaluate(&test_data)?;
    volatile_results.push(ModelEvaluation {
        model_name: "ETS(A,Ad,N)".to_string(),
        mae: ets_eval.mae,
        rmse: ets_eval.rmse,
    });

    eval_results.push(volatile_results);

    // Display results table for each scenario
    println!("\n=== Model Comparison Results ===");
    for (i, results) in eval_results.iter().enumerate() {
        println!("\nScenario {}: {}", i + 1, scenarios[i]);
        println!("{:<15} {:<10} {:<10}", "Model", "MAE", "RMSE");
        println!("{:-<37}", "");

        // Sort results by MAE
        let mut sorted_results = results.clone();
        sorted_results.sort_by(|a, b| a.mae.partial_cmp(&b.mae).unwrap());

        for eval in &sorted_results {
            println!(
                "{:<15} {:<10.4} {:<10.4}",
                eval.model_name, eval.mae, eval.rmse
            );
        }

        // Highlight best model
        println!(
            "\nBest model: {} (Lowest MAE: {:.4})",
            sorted_results[0].model_name, sorted_results[0].mae
        );
    }

    // Trading recommendations
    println!("\n=== Trading Strategy Recommendations ===");
    println!("1. For trending markets (bull/bear): Prefer Holt Linear or ETS(A,A,N)");
    println!("2. For sideways/range-bound markets: Prefer Simple ES or ETS(A,N,N)");
    println!("3. For seasonal markets: Prefer Holt-Winters or ETS with seasonality component");
    println!("4. For volatile markets: Prefer Damped Trend or ETS(A,Ad,N)");
    println!("5. Parameter optimization is crucial - lower alpha for volatile markets");
    println!("6. Consider ensemble approaches for improved robustness");

    Ok(())
}

// Helper function to create and split trending data
fn create_and_split_trending_data() -> Result<(TimeSeriesData, TimeSeriesData), Box<dyn Error>> {
    let base_date = NaiveDate::from_ymd_opt(2022, 1, 1).unwrap();

    // Training data: 80 data points
    let train_timestamps: Vec<DateTime<Utc>> = (0..80)
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

    let mut train_values = Vec::with_capacity(80);
    for i in 0..80 {
        let trend = 100.0 + 0.5 * (i as f64);
        let noise = (i % 7) as f64 * 0.3 - 1.0;
        train_values.push(trend + noise);
    }

    // Test data: 20 data points
    let test_timestamps: Vec<DateTime<Utc>> = (80..100)
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

    let mut test_values = Vec::with_capacity(20);
    for i in 80..100 {
        let trend = 100.0 + 0.5 * (i as f64);
        let noise = (i % 7) as f64 * 0.3 - 1.0;
        test_values.push(trend + noise);
    }

    let train_data = TimeSeriesData::new(train_timestamps, train_values, "trending_train")?;
    let test_data = TimeSeriesData::new(test_timestamps, test_values, "trending_test")?;

    Ok((train_data, test_data))
}

// Helper function to create and split sideways/ranging data
fn create_and_split_sideways_data() -> Result<(TimeSeriesData, TimeSeriesData), Box<dyn Error>> {
    let base_date = NaiveDate::from_ymd_opt(2022, 1, 1).unwrap();

    // Mean-reverting process parameters
    let mean = 100.0;
    let reversion_speed = 0.1;
    let volatility = 1.0;

    // Training data: 80 data points
    let train_timestamps: Vec<DateTime<Utc>> = (0..80)
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

    let mut value = mean;
    let mut train_values = Vec::with_capacity(80);

    for i in 0..80 {
        let noise = ((i * 83) % 17) as f64 * 0.2 - 1.7;
        value = value + reversion_speed * (mean - value) + volatility * noise;
        train_values.push(value);
    }

    // Test data: 20 data points
    let test_timestamps: Vec<DateTime<Utc>> = (80..100)
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

    let mut test_values = Vec::with_capacity(20);

    for i in 80..100 {
        let noise = ((i * 83) % 17) as f64 * 0.2 - 1.7;
        value = value + reversion_speed * (mean - value) + volatility * noise;
        test_values.push(value);
    }

    let train_data = TimeSeriesData::new(train_timestamps, train_values, "sideways_train")?;
    let test_data = TimeSeriesData::new(test_timestamps, test_values, "sideways_test")?;

    Ok((train_data, test_data))
}

// Helper function to create and split seasonal data
fn create_and_split_seasonal_data() -> Result<(TimeSeriesData, TimeSeriesData), Box<dyn Error>> {
    let base_date = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();

    // Training data: 96 data points (8 years of monthly data)
    let train_timestamps: Vec<DateTime<Utc>> = (0..96)
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

    let mut train_values = Vec::with_capacity(96);
    for i in 0..96 {
        let trend = 100.0 + 0.2 * (i as f64);
        let season = 5.0 * (2.0 * std::f64::consts::PI * ((i % 12) as f64) / 12.0).sin();
        let noise = ((i * 17) % 7) as f64 * 0.2 - 0.6;
        train_values.push(trend + season + noise);
    }

    // Test data: 24 data points (2 years of monthly data)
    let test_timestamps: Vec<DateTime<Utc>> = (96..120)
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

    let mut test_values = Vec::with_capacity(24);
    for i in 96..120 {
        let trend = 100.0 + 0.2 * (i as f64);
        let season = 5.0 * (2.0 * std::f64::consts::PI * ((i % 12) as f64) / 12.0).sin();
        let noise = ((i * 17) % 7) as f64 * 0.2 - 0.6;
        test_values.push(trend + season + noise);
    }

    let train_data = TimeSeriesData::new(train_timestamps, train_values, "seasonal_train")?;
    let test_data = TimeSeriesData::new(test_timestamps, test_values, "seasonal_test")?;

    Ok((train_data, test_data))
}

// Helper function to create and split volatile data
fn create_and_split_volatile_data() -> Result<(TimeSeriesData, TimeSeriesData), Box<dyn Error>> {
    let base_date = NaiveDate::from_ymd_opt(2022, 1, 1).unwrap();

    // Volatile data with regime changes
    let train_timestamps: Vec<DateTime<Utc>> = (0..80)
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

    let mut value = 100.0;
    let mut train_values = Vec::with_capacity(80);

    for i in 0..80 {
        let regime = (i / 20) % 3; // Change trend direction every 20 days
        let trend = match regime {
            0 => 0.8,  // Strong up
            1 => -0.5, // Down
            _ => 0.1,  // Weak up
        };

        let volatility = 3.0;
        let noise = ((i * 73) % 19) as f64 * 0.3 - 2.5;

        value = value + trend + volatility * noise;
        if value < 50.0 {
            value = 50.0;
        } // Set a floor
        train_values.push(value);
    }

    // Test data
    let test_timestamps: Vec<DateTime<Utc>> = (80..100)
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

    let mut test_values = Vec::with_capacity(20);

    for i in 80..100 {
        let regime = (i / 20) % 3;
        let trend = match regime {
            0 => 0.8,
            1 => -0.5,
            _ => 0.1,
        };

        let volatility = 3.0;
        let noise = ((i * 73) % 19) as f64 * 0.3 - 2.5;

        value = value + trend + volatility * noise;
        if value < 50.0 {
            value = 50.0;
        }
        test_values.push(value);
    }

    let train_data = TimeSeriesData::new(train_timestamps, train_values, "volatile_train")?;
    let test_data = TimeSeriesData::new(test_timestamps, test_values, "volatile_test")?;

    Ok((train_data, test_data))
}
