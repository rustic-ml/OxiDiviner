#![allow(deprecated)]
#![allow(dead_code)]
#![allow(unused_imports)]

use chrono::{DateTime, Duration, Utc};
use oxidiviner_core::TimeSeriesData;
use rand::Rng;
use std::error::Error;

// Define our own enum for the demo since we can't import from ets module (it's private)
#[derive(Debug, Clone, Copy, PartialEq)]
enum ETSComponent {
    None,
    Additive,
    Multiplicative,
    Damped,
}

impl std::fmt::Display for ETSComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ETSComponent::None => write!(f, "N"),
            ETSComponent::Additive => write!(f, "A"),
            ETSComponent::Multiplicative => write!(f, "M"),
            ETSComponent::Damped => write!(f, "D"),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("ETS (Error-Trend-Seasonality) Demo");
    println!("===================================\n");

    // Generate synthetic data
    println!("Generating synthetic daily data...");
    let daily_data = generate_synthetic_daily_data();
    println!("Generated {} daily data points", daily_data.len());

    println!("\nGenerating synthetic minute data...");
    let minute_data = generate_synthetic_minute_data();
    println!("Generated {} minute data points", minute_data.len());

    // Since we can't use the actual ETS model implementations due to import issues,
    // we'll just demonstrate the concept with a simplified explanation

    println!("\nETS Model Overview");
    println!("-----------------");
    println!("ETS models represent time series with three components:");
    println!("1. Error (E): How random fluctuations are modeled (Additive or Multiplicative)");
    println!("2. Trend (T): How the series changes over time (None, Additive, Multiplicative, or Damped)");
    println!("3. Seasonal (S): How seasonal patterns affect the series (None, Additive, or Multiplicative)");

    // Demo interpretation functions
    println!("\nExample Interpretation");
    println!("---------------------");

    // Example parameters for interpretation
    let error_type = ETSComponent::Additive;
    let trend_type = ETSComponent::Additive;
    let seasonal_type = ETSComponent::Additive;
    let alpha = 0.3;
    let beta = 0.1;
    let gamma = 0.1;
    let phi = 0.9;
    let mae = 1.5;
    let rmse = 2.3;
    let mape = 5.2;

    println!(
        "Model: ETS({},{},{})",
        error_type, trend_type, seasonal_type
    );
    println!(
        "Parameters: α={:.2}, β={:.2}, γ={:.2}, φ={:.2}",
        alpha, beta, gamma, phi
    );

    // Call the interpretation functions
    println!("\nTrend Analysis:");
    println!(
        "{}",
        interpret_trend("Example", trend_type, Some(beta), Some(phi), mae)
    );

    println!("\nSeasonality Analysis:");
    println!(
        "{}",
        interpret_seasonality(seasonal_type, Some(7), Some(gamma))
    );

    println!("\nVolatility Analysis:");
    println!("{}", interpret_volatility(error_type, alpha, rmse));

    println!("\nTrading Insights:");
    println!("{}", trading_insights(trend_type, seasonal_type, mape));

    println!("\nRisk Assessment:");
    println!("{}", risk_assessment(error_type, trend_type, mape, rmse));

    println!("\nNote: This is a simplified demo. For actual implementations, use the OxiDiviner library directly.");
    Ok(())
}

// Generate synthetic daily time series data
fn generate_synthetic_daily_data() -> TimeSeriesData {
    let mut rng = rand::thread_rng();
    let now = Utc::now();
    let n = 365; // 1 year of daily data

    let base_date = now - Duration::days(n as i64);

    let mut timestamps = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);

    for i in 0..n {
        // Generate timestamp (daily data)
        let timestamp = base_date + Duration::days(i as i64);
        timestamps.push(timestamp);

        // Generate value with trend, seasonality, and noise
        let trend_component = 0.5 * i as f64;
        let season_component = 10.0 * (2.0 * std::f64::consts::PI * (i % 30) as f64 / 30.0).sin();
        let noise = 3.0 * (rng.gen_range(0.0..1.0) - 0.5);

        let value = 100.0 + trend_component + season_component + noise;
        values.push(value);
    }

    TimeSeriesData::new(timestamps, values, "Synthetic daily data").unwrap() // Unwrapping is safe here since we ensure lengths are equal
}

// Generate synthetic minute time series data
fn generate_synthetic_minute_data() -> TimeSeriesData {
    let mut rng = rand::thread_rng();
    let now = Utc::now();
    let n = 60 * 8; // 8 hours of minute data

    let base_date = now - Duration::minutes(n as i64);

    let mut timestamps = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);

    for i in 0..n {
        // Generate timestamp (minute data)
        let timestamp = base_date + Duration::minutes(i as i64);
        timestamps.push(timestamp);

        // Generate value with trend, seasonality, and noise
        let trend_component = 0.001 * i as f64;
        let season_component = 1.0 * (2.0 * std::f64::consts::PI * (i % 60) as f64 / 60.0).sin();
        let noise = 0.5 * (rng.gen_range(0.0..1.0) - 0.5);

        let value = 100.0 + trend_component + season_component + noise;
        values.push(value);
    }

    TimeSeriesData::new(timestamps, values, "Synthetic minute data").unwrap() // Unwrapping is safe here since we ensure lengths are equal
}

/// Interpret trend characteristics from ETS model
fn interpret_trend(
    _model_name: &str,
    trend_type: ETSComponent,
    beta: Option<f64>,
    phi: Option<f64>,
    mae: f64,
) -> String {
    let mut analysis = String::new();

    match trend_type {
        ETSComponent::None => {
            analysis.push_str("No significant trend detected. ");
            analysis.push_str(
                "The series appears to fluctuate around a relatively stable level without ",
            );
            analysis.push_str("persistent directional movement.");
        }
        ETSComponent::Additive => {
            analysis.push_str("Clear additive trend detected. ");
            if let Some(b) = beta {
                analysis.push_str(&format!("With β={:.3}, the model indicates ", b));
                if b > 0.2 {
                    analysis.push_str("strong responsiveness to recent trend changes. ");
                } else {
                    analysis.push_str("moderate responsiveness to recent trend changes. ");
                }
            }
            analysis.push_str("This suggests predictable linear growth patterns.");
        }
        ETSComponent::Multiplicative => {
            analysis.push_str("Multiplicative trend detected. ");
            analysis.push_str("The growth rate is proportional to the level, ");
            analysis.push_str("suggesting exponential rather than linear movement.");
        }
        ETSComponent::Damped => {
            analysis.push_str("Damped trend detected. ");
            if let Some(p) = phi {
                analysis.push_str(&format!("With damping factor φ={:.3}, ", p));
                if p > 0.8 {
                    analysis.push_str("the trend diminishes slowly over time. ");
                } else {
                    analysis.push_str("the trend diminishes quickly over time. ");
                }
            }
            analysis.push_str("This suggests short-term momentum that gradually weakens.");
        }
    }

    // Add error-based assessment
    if mae < 2.0 {
        analysis.push_str(" The low forecast error suggests high trend predictability.");
    } else if mae < 5.0 {
        analysis
            .push_str(" The moderate forecast error indicates reasonable trend predictability.");
    } else {
        analysis.push_str(" The high forecast error suggests the trend may be frequently disrupted by other factors.");
    }

    analysis
}

/// Interpret seasonal patterns from ETS model
fn interpret_seasonality(
    seasonal_type: ETSComponent,
    seasonal_period: Option<usize>,
    gamma: Option<f64>,
) -> String {
    let mut analysis = String::new();

    match seasonal_type {
        ETSComponent::None => {
            analysis.push_str("No significant seasonal patterns detected. ");
            analysis.push_str(
                "The series doesn't show regular, repeating cycles within the analyzed timeframe.",
            );
        }
        ETSComponent::Additive => {
            analysis.push_str("Additive seasonal pattern detected. ");
            if let Some(period) = seasonal_period {
                if period == 7 {
                    analysis.push_str("Weekly cycles are present with consistent magnitude. ");
                } else if period == 30 || period == 31 {
                    analysis.push_str("Monthly cycles are present with consistent magnitude. ");
                } else if period == 60 {
                    analysis.push_str("Hourly cycles are present with consistent magnitude. ");
                } else {
                    analysis.push_str(&format!("Regular {}-period cycles are present. ", period));
                }
            }
            if let Some(g) = gamma {
                if g > 0.2 {
                    analysis.push_str(&format!(
                        "With γ={:.3}, the model quickly adapts to changing seasonal patterns.",
                        g
                    ));
                } else {
                    analysis.push_str(&format!(
                        "With γ={:.3}, the model maintains stable seasonal patterns.",
                        g
                    ));
                }
            }
        }
        ETSComponent::Multiplicative => {
            analysis.push_str("Multiplicative seasonal pattern detected. ");
            analysis.push_str("The magnitude of seasonal effects scales with the level. ");
            if let Some(period) = seasonal_period {
                if period == 7 {
                    analysis.push_str(" The series exhibits weekly cycles that become more pronounced at higher levels.");
                } else if period == 30 || period == 31 {
                    analysis.push_str(" The series exhibits monthly cycles that become more pronounced at higher levels.");
                } else if period == 60 {
                    analysis.push_str(" The series exhibits hourly cycles that become more pronounced at higher levels.");
                } else {
                    analysis.push_str(&format!(" The series exhibits {}-period cycles that become more pronounced at higher levels.", period));
                }
            }
        }
        ETSComponent::Damped => {
            // This is technically not used for seasonality but included for completeness
            analysis.push_str("Unusual damped seasonal pattern detected. ");
            analysis.push_str("This suggests seasonal effects that gradually change over time.");
        }
    }

    analysis
}

/// Interpret volatility characteristics from ETS model
fn interpret_volatility(error_type: ETSComponent, alpha: f64, rmse: f64) -> String {
    let mut analysis = String::new();

    analysis.push_str("Volatility Analysis: ");

    // Analyze alpha
    if alpha > 0.5 {
        analysis.push_str(&format!(
            "With high α={:.3}, the model heavily weights recent observations. ",
            alpha
        ));
        analysis.push_str("This suggests high sensitivity to changes. ");
    } else if alpha > 0.2 {
        analysis.push_str(&format!(
            "With moderate α={:.3}, the model balances recent and historical data. ",
            alpha
        ));
        analysis.push_str("This indicates moderate responsiveness to new information. ");
    } else {
        analysis.push_str(&format!(
            "With low α={:.3}, the model emphasizes historical patterns. ",
            alpha
        ));
        analysis.push_str("This points to slow-changing dynamics. ");
    }

    // Analyze error type
    match error_type {
        ETSComponent::Additive => {
            analysis.push_str(
                "The additive error structure suggests that volatility is generally consistent ",
            );
            analysis.push_str("regardless of the level. ");
        }
        ETSComponent::Multiplicative => {
            analysis.push_str(
                "The multiplicative error structure indicates that volatility scales with level. ",
            );
            analysis.push_str("Higher values correspond to larger fluctuations. ");
        }
        _ => {
            // These shouldn't occur for error type, but included for completeness
            analysis.push_str("The error structure suggests complex volatility dynamics. ");
        }
    }

    // RMSE-based assessment
    if rmse < 2.0 {
        analysis.push_str(&format!(
            "With RMSE={:.3}, overall volatility is relatively low.",
            rmse
        ));
    } else if rmse < 5.0 {
        analysis.push_str(&format!(
            "With RMSE={:.3}, overall volatility is moderate.",
            rmse
        ));
    } else {
        analysis.push_str(&format!(
            "With RMSE={:.3}, overall volatility is high.",
            rmse
        ));
    }

    analysis
}

/// Provide trading insights based on ETS model parameters
fn trading_insights(trend_type: ETSComponent, seasonal_type: ETSComponent, mape: f64) -> String {
    let mut analysis = String::new();

    analysis.push_str("Trading Insights: ");

    // Combined trend and seasonality analysis
    match (trend_type, seasonal_type) {
        (ETSComponent::None, ETSComponent::None) => {
            analysis.push_str("This series shows neither strong trend nor seasonality, ");
            analysis.push_str("suggesting mean-reverting characteristics. ");
            analysis.push_str("Trading strategies could focus on range-bound approaches. ");
        }
        (ETSComponent::None, _) => {
            analysis
                .push_str("With seasonality but no trend, this series shows cyclical patterns ");
            analysis.push_str("around a stable mean. Calendar-based strategies ");
            analysis.push_str("that anticipate seasonal moves could be effective. ");
        }
        (_, ETSComponent::None) => {
            analysis.push_str("With trend but no seasonality, this series shows persistent ");
            analysis.push_str("directional movement without regular cycles. ");
            analysis.push_str("Trend-following strategies could be appropriate. ");
        }
        (_, _) => {
            analysis.push_str("This series exhibits both trend and seasonality, ");
            analysis.push_str("suggesting complex but potentially predictable behavior. ");
            analysis.push_str(
                "Consider strategies that combine trend-following with seasonal adjustments. ",
            );
        }
    }

    // MAPE-based assessment of predictability
    if mape < 5.0 {
        analysis.push_str(&format!(
            "With low MAPE={:.3}%, the series shows high forecasting accuracy, ",
            mape
        ));
        analysis.push_str("suggesting quantitative prediction-based strategies may be effective.");
    } else if mape < 15.0 {
        analysis.push_str(&format!(
            "With moderate MAPE={:.3}%, forecasts have reasonable accuracy, ",
            mape
        ));
        analysis.push_str("though significant deviations can occur.");
    } else {
        analysis.push_str(&format!(
            "With high MAPE={:.3}%, forecasting accuracy is limited, ",
            mape
        ));
        analysis.push_str("suggesting wider margins of safety in strategies.");
    }

    analysis
}

/// Assess risk profile based on ETS model results
fn risk_assessment(
    error_type: ETSComponent,
    trend_type: ETSComponent,
    mape: f64,
    rmse: f64,
) -> String {
    let mut analysis = String::new();

    analysis.push_str("Risk Assessment: ");

    // Error structure assessment
    match error_type {
        ETSComponent::Additive => {
            analysis.push_str("The additive error structure suggests relatively consistent ");
            analysis.push_str("absolute risk regardless of level. ");
        }
        ETSComponent::Multiplicative => {
            analysis.push_str("The multiplicative error structure indicates that risk ");
            analysis.push_str("scales proportionally with level. ");
        }
        _ => { /* Other types not typically used for error component */ }
    }

    // Trend-based risk
    match trend_type {
        ETSComponent::None => {
            analysis.push_str("Without a significant trend component, the primary risk ");
            analysis.push_str("comes from temporary deviations and volatility rather than ");
            analysis.push_str("sustained adverse movements. ");
        }
        ETSComponent::Additive => {
            analysis.push_str("The linear trend component suggests consistent directional ");
            analysis.push_str("risk that can compound over time. ");
        }
        ETSComponent::Multiplicative => {
            analysis.push_str("The multiplicative trend indicates accelerating movement, ");
            analysis.push_str("which can create rapidly increasing risk exposure. ");
        }
        ETSComponent::Damped => {
            analysis.push_str("The damped trend suggests initial directional movement that ");
            analysis.push_str("gradually weakens, reducing long-term directional risk. ");
        }
    }

    // Overall risk level based on error metrics
    let risk_level = if rmse > 5.0 || mape > 15.0 {
        "high"
    } else if rmse > 2.0 || mape > 8.0 {
        "moderate"
    } else {
        "low"
    };

    analysis.push_str(&format!(
        "Overall, this series shows {} risk characteristics ",
        risk_level
    ));
    analysis.push_str(&format!("with RMSE={:.3} and MAPE={:.3}%. ", rmse, mape));

    analysis
}
