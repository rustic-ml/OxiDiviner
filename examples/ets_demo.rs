use chrono::{Duration, Utc, TimeZone, NaiveDateTime};
use rand::Rng;
use std::error::Error;
use std::path::Path;
use std::env;
use oxdiviner::ModelsOHLCVData;
use oxdiviner::models::ets::{ETSComponent, DailyETSModel, MinuteETSModel};

// Generate synthetic daily OHLCV data
fn generate_synthetic_daily_data() -> ModelsOHLCVData {
    // Parameters
    const N: usize = 365;  // 1 year of daily data
    const TREND: f64 = 0.5;  // Upward trend
    const SEASON_AMPLITUDE: f64 = 10.0;  // Seasonal amplitude
    const NOISE_LEVEL: f64 = 3.0;  // Random noise level
    
    let mut rng = rand::thread_rng();
    let base_date = Utc::now() - Duration::days(N as i64);
    
    let mut timestamps = Vec::with_capacity(N);
    let mut open = Vec::with_capacity(N);
    let mut high = Vec::with_capacity(N);
    let mut low = Vec::with_capacity(N);
    let mut close = Vec::with_capacity(N);
    let mut volume = Vec::with_capacity(N);
    
    let mut last_close = 100.0;
    
    for i in 0..N {
        // Generate timestamp (daily data)
        let timestamp = base_date + Duration::days(i as i64);
        timestamps.push(timestamp);
        
        // Generate value with trend, seasonality, and noise
        let trend_component = TREND * i as f64;
        let season_component = SEASON_AMPLITUDE * (2.0 * std::f64::consts::PI * (i % 30) as f64 / 30.0).sin();
        let noise = NOISE_LEVEL * (rng.gen::<f64>() - 0.5);
        
        let close_price = 100.0 + trend_component + season_component + noise;
        let daily_volatility = close_price * 0.02 * rng.gen::<f64>();
        
        // OHLC based on previous close
        let open_price = last_close * (1.0 + 0.01 * (rng.gen::<f64>() - 0.5));
        let high_price = close_price.max(open_price) + daily_volatility;
        let low_price = close_price.min(open_price) - daily_volatility;
        
        // Random volume between 1000 and 10000
        let vol = 1000.0 + rng.gen::<f64>() * 9000.0;
        
        open.push(open_price);
        high.push(high_price);
        low.push(low_price);
        close.push(close_price);
        volume.push(vol);
        
        last_close = close_price;
    }
    
    ModelsOHLCVData {
        timestamps,
        open,
        high,
        low,
        close,
        volume,
        name: "Synthetic daily data".to_string(),
    }
}

// Generate synthetic minute OHLCV data
fn generate_synthetic_minute_data() -> ModelsOHLCVData {
    // Parameters
    const N: usize = 60 * 8;  // 8 hours of minute data
    const TREND: f64 = 0.001;  // Upward trend per minute
    const SEASON_AMPLITUDE: f64 = 1.0;  // Seasonal amplitude
    const NOISE_LEVEL: f64 = 0.5;  // Random noise level
    
    let mut rng = rand::thread_rng();
    let base_date = Utc::now() - Duration::minutes(N as i64);
    
    let mut timestamps = Vec::with_capacity(N);
    let mut open = Vec::with_capacity(N);
    let mut high = Vec::with_capacity(N);
    let mut low = Vec::with_capacity(N);
    let mut close = Vec::with_capacity(N);
    let mut volume = Vec::with_capacity(N);
    
    let mut last_close = 100.0;
    
    for i in 0..N {
        // Generate timestamp (minute data)
        let timestamp = base_date + Duration::minutes(i as i64);
        timestamps.push(timestamp);
        
        // Generate value with trend, seasonality, and noise
        let trend_component = TREND * i as f64;
        // 60-minute seasonality pattern
        let season_component = SEASON_AMPLITUDE * (2.0 * std::f64::consts::PI * (i % 60) as f64 / 60.0).sin();
        let noise = NOISE_LEVEL * (rng.gen::<f64>() - 0.5);
        
        let close_price = 100.0 + trend_component + season_component + noise;
        let minute_volatility = close_price * 0.001 * rng.gen::<f64>();
        
        // OHLC based on previous close
        let open_price = last_close * (1.0 + 0.002 * (rng.gen::<f64>() - 0.5));
        let high_price = close_price.max(open_price) + minute_volatility;
        let low_price = close_price.min(open_price) - minute_volatility;
        
        // Random volume between 10 and 1000
        let vol = 10.0 + rng.gen::<f64>() * 990.0;
        
        open.push(open_price);
        high.push(high_price);
        low.push(low_price);
        close.push(close_price);
        volume.push(vol);
        
        last_close = close_price;
    }
    
    ModelsOHLCVData {
        timestamps,
        open,
        high,
        low,
        close,
        volume,
        name: "Synthetic minute data".to_string(),
    }
}

// Load OHLCV data from CSV file
fn load_stock_data_from_csv(ticker: &str, data_type: &str) -> Result<ModelsOHLCVData, Box<dyn Error>> {
    let filename = format!("{}_{}_ohlcv.csv", ticker, data_type);
    let path = Path::new("examples").join("csv").join(&filename);
    
    // Check if file exists
    if !path.exists() {
        return Err(format!("CSV file not found: {}", path.display()).into());
    }
    
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    
    let file = File::open(&path)?;
    let reader = BufReader::new(file);
    
    let mut timestamps = Vec::new();
    let mut open = Vec::new();
    let mut high = Vec::new();
    let mut low = Vec::new();
    let mut close = Vec::new();
    let mut volume = Vec::new();
    
    // Skip the header
    let mut lines = reader.lines();
    if let Some(Ok(_)) = lines.next() {
        // Header skipped
    }
    
    for line in lines {
        let line = line?;
        let fields: Vec<&str> = line.split(',').collect();
        
        if fields.len() < 6 {
            continue;  // Skip malformed lines
        }
        
        // Parse timestamp using NaiveDateTime first, then convert to UTC
        // Format example: "2022-08-22 04:00:00 UTC"
        let timestamp_str = fields[0];
        let naive_datetime = match NaiveDateTime::parse_from_str(
            timestamp_str.trim_end_matches(" UTC"), 
            "%Y-%m-%d %H:%M:%S"
        ) {
            Ok(dt) => dt,
            Err(e) => {
                println!("Error parsing date '{}': {}", timestamp_str, e);
                continue;  // Skip lines with invalid dates
            }
        };
        
        // Convert to UTC
        let timestamp = Utc.from_utc_datetime(&naive_datetime);
        
        // Parse OHLCV values, skipping any that can't be parsed
        let open_val = match fields[1].parse::<f64>() {
            Ok(val) => val,
            Err(_) => continue,
        };
        
        let high_val = match fields[2].parse::<f64>() {
            Ok(val) => val,
            Err(_) => continue,
        };
        
        let low_val = match fields[3].parse::<f64>() {
            Ok(val) => val,
            Err(_) => continue,
        };
        
        let close_val = match fields[4].parse::<f64>() {
            Ok(val) => val,
            Err(_) => continue,
        };
        
        let volume_val = match fields[5].parse::<f64>() {
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
    
    // Check if we successfully loaded any data
    if timestamps.is_empty() {
        return Err(format!("No valid data found in file: {}", path.display()).into());
    }
    
    println!("Successfully loaded {} data points for {}", timestamps.len(), ticker);
    
    Ok(ModelsOHLCVData {
        timestamps,
        open,
        high,
        low,
        close,
        volume,
        name: format!("{} {} data", ticker, data_type),
    })
}

// Save OHLCV data to CSV file
fn save_to_csv(data: &ModelsOHLCVData, filename: &str) -> Result<(), Box<dyn Error>> {
    use std::fs::File;
    use std::io::Write;
    
    let path = Path::new("examples").join("csv").join(filename);
    let mut file = File::create(path)?;
    
    // Write header
    writeln!(file, "timestamp,open,high,low,close,volume")?;
    
    // Write data rows
    for i in 0..data.len() {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            data.timestamps[i].to_rfc3339(),
            data.open[i],
            data.high[i],
            data.low[i],
            data.close[i],
            data.volume[i]
        )?;
    }
    
    Ok(())
}

/// Interpret trend characteristics from ETS model
fn interpret_trend(_model_name: &str, trend_type: ETSComponent, beta: Option<f64>, phi: Option<f64>, mae: f64) -> String {
    let mut analysis = String::new();
    
    match trend_type {
        ETSComponent::None => {
            analysis.push_str("Trend Analysis: No significant trend detected. ");
            analysis.push_str("The stock appears to fluctuate around a relatively stable level without ");
            analysis.push_str("persistent directional movement. This suggests mean-reverting behavior ");
            analysis.push_str("where prices tend to revert to their average level over time.");
        },
        ETSComponent::Additive => {
            analysis.push_str("Trend Analysis: Clear additive trend detected. ");
            if let Some(b) = beta {
                analysis.push_str(&format!("With β={:.3}, the model indicates ", b));
                if b > 0.2 {
                    analysis.push_str("strong responsiveness to recent trend changes. ");
                    analysis.push_str("The stock shows momentum characteristics with ");
                } else {
                    analysis.push_str("moderate responsiveness to recent trend changes. ");
                    analysis.push_str("The stock shows a steady, persistent ");
                }
            }
            analysis.push_str("directional movement independent of the price level. ");
            analysis.push_str("This constant rate of change suggests predictable linear growth/decline patterns.");
        },
        ETSComponent::Multiplicative => {
            analysis.push_str("Trend Analysis: Multiplicative trend detected. ");
            analysis.push_str("The stock's growth rate is proportional to its price level, ");
            analysis.push_str("suggesting exponential rather than linear movement. ");
            analysis.push_str("This is characteristic of stocks with compound growth dynamics, ");
            analysis.push_str("where percentage changes remain relatively constant.");
        },
        ETSComponent::Damped => {
            analysis.push_str("Trend Analysis: Damped trend detected. ");
            if let Some(p) = phi {
                analysis.push_str(&format!("With damping factor φ={:.3}, ", p));
                if p > 0.8 {
                    analysis.push_str("the trend diminishes slowly over time. ");
                } else {
                    analysis.push_str("the trend diminishes quickly over time. ");
                }
            }
            analysis.push_str("This suggests the stock has short-term momentum that gradually weakens, ");
            analysis.push_str("with prices eventually stabilizing rather than continuing indefinitely in one direction.");
        },
    }
    
    // Add error-based assessment
    if mae < 10.0 {
        analysis.push_str(" The low forecast error suggests high trend predictability.");
    } else if mae < 25.0 {
        analysis.push_str(" The moderate forecast error indicates reasonable trend predictability.");
    } else {
        analysis.push_str(" The high forecast error suggests the trend may be frequently disrupted by other factors.");
    }
    
    analysis
}

/// Interpret seasonal patterns from ETS model
fn interpret_seasonality(seasonal_type: ETSComponent, seasonal_period: Option<usize>, gamma: Option<f64>) -> String {
    let mut analysis = String::new();
    
    match seasonal_type {
        ETSComponent::None => {
            analysis.push_str("Seasonality Analysis: No significant seasonal patterns detected. ");
            analysis.push_str("The stock doesn't show regular, repeating cycles within the analyzed timeframe. ");
            analysis.push_str("This suggests price movements are more influenced by trend and random factors ");
            analysis.push_str("than by calendar-based or cyclical patterns.");
        },
        ETSComponent::Additive => {
            analysis.push_str("Seasonality Analysis: Additive seasonal pattern detected. ");
            if let Some(period) = seasonal_period {
                if period == 7 {
                    analysis.push_str("Weekly cycles are present with consistent magnitude regardless of price level. ");
                } else if period == 30 || period == 31 {
                    analysis.push_str("Monthly cycles are present with consistent magnitude regardless of price level. ");
                } else if period == 90 {
                    analysis.push_str("Quarterly cycles are present with consistent magnitude regardless of price level. ");
                } else if period == 60 {
                    analysis.push_str("Hourly cycles are present with consistent magnitude regardless of price level. ");
                } else {
                    analysis.push_str(&format!("Regular {}-period cycles are present with consistent magnitude regardless of price level. ", period));
                }
            }
            if let Some(g) = gamma {
                if g > 0.2 {
                    analysis.push_str(&format!("With γ={:.3}, the model quickly adapts to changing seasonal patterns. ", g));
                    analysis.push_str("This suggests that the timing or magnitude of seasonal effects may vary over time.");
                } else {
                    analysis.push_str(&format!("With γ={:.3}, the model maintains stable seasonal patterns. ", g));
                    analysis.push_str("This suggests highly predictable cyclic behavior that persists consistently.");
                }
            }
        },
        ETSComponent::Multiplicative => {
            analysis.push_str("Seasonality Analysis: Multiplicative seasonal pattern detected. ");
            analysis.push_str("The magnitude of seasonal effects scales with the price level. ");
            analysis.push_str("This means percentage changes due to seasonality remain constant, ");
            analysis.push_str("which is typical for stocks with growing volatility as prices increase.");
            if let Some(period) = seasonal_period {
                if period == 7 {
                    analysis.push_str(" The stock exhibits weekly cycles that become more pronounced at higher price levels.");
                } else if period == 30 || period == 31 {
                    analysis.push_str(" The stock exhibits monthly cycles that become more pronounced at higher price levels.");
                } else if period == 90 {
                    analysis.push_str(" The stock exhibits quarterly cycles that become more pronounced at higher price levels.");
                } else if period == 60 {
                    analysis.push_str(" The stock exhibits hourly cycles that become more pronounced at higher price levels.");
                } else {
                    analysis.push_str(&format!(" The stock exhibits {}-period cycles that become more pronounced at higher price levels.", period));
                }
            }
        },
        ETSComponent::Damped => {
            // This is technically not used for seasonality but included for completeness
            analysis.push_str("Seasonality Analysis: Unusual damped seasonal pattern detected. ");
            analysis.push_str("This suggests seasonal effects that gradually change over time, ");
            analysis.push_str("potentially indicating evolving market dynamics or structural changes in the stock.");
        },
    }
    
    analysis
}

/// Interpret volatility characteristics from ETS model
fn interpret_volatility(error_type: ETSComponent, alpha: f64, rmse: f64) -> String {
    let mut analysis = String::new();
    
    analysis.push_str("Volatility Analysis: ");
    
    // Analyze alpha
    if alpha > 0.5 {
        analysis.push_str(&format!("With high α={:.3}, the model heavily weights recent observations. ", alpha));
        analysis.push_str("This suggests high sensitivity to news and market changes, ");
        analysis.push_str("characteristic of stocks with rapidly changing volatility profiles. ");
    } else if alpha > 0.2 {
        analysis.push_str(&format!("With moderate α={:.3}, the model balances recent and historical data. ", alpha));
        analysis.push_str("This indicates moderate responsiveness to new information, ");
        analysis.push_str("typical of stocks with stable but adaptive volatility patterns. ");
    } else {
        analysis.push_str(&format!("With low α={:.3}, the model emphasizes historical patterns over recent changes. ", alpha));
        analysis.push_str("This points to slow-changing volatility dynamics, ");
        analysis.push_str("common in stable stocks with persistent volatility characteristics. ");
    }
    
    // Analyze error type
    match error_type {
        ETSComponent::Additive => {
            analysis.push_str("The additive error structure suggests that volatility is generally consistent ");
            analysis.push_str("regardless of the price level. This implies that absolute price movements ");
            analysis.push_str("(rather than percentage changes) tend to be stable over time. ");
        },
        ETSComponent::Multiplicative => {
            analysis.push_str("The multiplicative error structure indicates that volatility scales with price level. ");
            analysis.push_str("Higher prices correspond to larger absolute price movements but similar percentage changes. ");
            analysis.push_str("This is common in stocks where risk is proportional to investment size. ");
        },
        _ => {
            // These shouldn't occur for error type, but included for completeness
            analysis.push_str("The error structure suggests complex volatility dynamics ");
            analysis.push_str("that may require more sophisticated modeling approaches. ");
        }
    }
    
    // RMSE-based assessment
    if rmse < 10.0 {
        analysis.push_str(&format!("With RMSE={:.3}, overall volatility is relatively low, ", rmse));
        analysis.push_str("suggesting a more predictable stock with smaller unexpected price movements.");
    } else if rmse < 30.0 {
        analysis.push_str(&format!("With RMSE={:.3}, overall volatility is moderate, ", rmse));
        analysis.push_str("indicating average unpredictability compared to typical stocks.");
    } else {
        analysis.push_str(&format!("With RMSE={:.3}, overall volatility is high, ", rmse));
        analysis.push_str("suggesting a stock with large, difficult-to-predict price movements.");
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
            analysis.push_str("This stock shows neither strong trend nor seasonality, ");
            analysis.push_str("suggesting mean-reverting characteristics. ");
            analysis.push_str("Trading strategies could focus on range-bound approaches, ");
            analysis.push_str("buying at statistical lows and selling at statistical highs. ");
            analysis.push_str("Volatility-based strategies like straddles or strangles may also be appropriate.");
        },
        (ETSComponent::None, _) => {
            analysis.push_str("With seasonality but no trend, this stock shows cyclical patterns ");
            analysis.push_str("around a stable mean. Calendar-based trading strategies ");
            analysis.push_str("that anticipate seasonal moves could be effective. ");
            analysis.push_str("Consider positions that benefit from expected seasonal changes ");
            analysis.push_str("while hedging against unexpected directional movements.");
        },
        (_, ETSComponent::None) => {
            analysis.push_str("With trend but no seasonality, this stock shows persistent ");
            analysis.push_str("directional movement without regular cycles. ");
            analysis.push_str("Trend-following strategies like moving average systems ");
            analysis.push_str("or breakout trading could be appropriate. ");
            analysis.push_str("Position sizing should account for potential acceleration or deceleration of the trend.");
        },
        (_, _) => {
            analysis.push_str("This stock exhibits both trend and seasonality, ");
            analysis.push_str("suggesting complex but potentially predictable behavior. ");
            analysis.push_str("Consider strategies that combine trend-following during dominant trends ");
            analysis.push_str("with seasonal adjustments or reversals at key cyclical turning points. ");
            analysis.push_str("Multi-timeframe analysis may help distinguish between short-term seasonal ");
            analysis.push_str("fluctuations and longer-term trend developments.");
        }
    }
    
    // MAPE-based assessment of predictability
    if mape < 5.0 {
        analysis.push_str(&format!("With low MAPE={:.3}%, the stock shows high forecasting accuracy, ", mape));
        analysis.push_str("suggesting strategies based on quantitative predictions may be particularly effective.");
    } else if mape < 15.0 {
        analysis.push_str(&format!("With moderate MAPE={:.3}%, forecasts have reasonable accuracy, ", mape));
        analysis.push_str("though significant deviations can occur. Risk management remains important.");
    } else {
        analysis.push_str(&format!("With high MAPE={:.3}%, forecasting accuracy is limited, ", mape));
        analysis.push_str("suggesting strategies should incorporate wider margins of safety and robust risk controls.");
    }
    
    analysis
}

/// Assess risk profile based on ETS model results
fn risk_assessment(error_type: ETSComponent, trend_type: ETSComponent, mape: f64, rmse: f64) -> String {
    let mut analysis = String::new();
    
    analysis.push_str("Risk Assessment: ");
    
    // Error structure assessment
    match error_type {
        ETSComponent::Additive => {
            analysis.push_str("The additive error structure suggests relatively consistent ");
            analysis.push_str("absolute risk regardless of price levels. ");
            analysis.push_str("This may indicate more predictable drawdown magnitudes ");
            analysis.push_str("and easier position sizing based on fixed dollar amounts. ");
        },
        ETSComponent::Multiplicative => {
            analysis.push_str("The multiplicative error structure indicates that risk ");
            analysis.push_str("scales proportionally with price levels. ");
            analysis.push_str("This suggests a need for percentage-based position sizing ");
            analysis.push_str("and risk management approaches that adjust with price changes. ");
        },
        _ => { /* Other types not typically used for error component */ }
    }
    
    // Trend-based risk
    match trend_type {
        ETSComponent::None => {
            analysis.push_str("Without a significant trend component, the primary risk ");
            analysis.push_str("comes from temporary deviations and volatility rather than ");
            analysis.push_str("sustained adverse price movements. ");
            analysis.push_str("This favors mean-reversion strategies but carries the risk ");
            analysis.push_str("of unexpected trend development. ");
        },
        ETSComponent::Additive => {
            analysis.push_str("The linear trend component suggests consistent directional ");
            analysis.push_str("risk that can compound over time. ");
            analysis.push_str("Positions against the trend face steadily increasing adverse exposure, ");
            analysis.push_str("while trend-aligned positions benefit from persistent favorable movement. ");
        },
        ETSComponent::Multiplicative => {
            analysis.push_str("The multiplicative trend indicates accelerating price movement, ");
            analysis.push_str("which can create rapidly increasing risk exposure in counter-trend positions ");
            analysis.push_str("or quickly building profits in trend-aligned trades. ");
            analysis.push_str("This dynamic typically requires more active management and tighter monitoring. ");
        },
        ETSComponent::Damped => {
            analysis.push_str("The damped trend suggests initial directional movement that ");
            analysis.push_str("gradually weakens, reducing long-term directional risk. ");
            analysis.push_str("This pattern may lull traders into false security as trends flatten, ");
            analysis.push_str("potentially missing new emerging patterns. ");
        }
    }
    
    // Overall risk level based on error metrics
    let risk_level = if rmse > 30.0 || mape > 20.0 {
        "high"
    } else if rmse > 10.0 || mape > 10.0 {
        "moderate"
    } else {
        "low"
    };
    
    analysis.push_str(&format!("Overall, this stock shows {} risk characteristics ", risk_level));
    analysis.push_str(&format!("with RMSE={:.3} and MAPE={:.3}%. ", rmse, mape));
    
    if risk_level == "high" {
        analysis.push_str("This suggests a need for tighter risk controls, smaller position sizes, ");
        analysis.push_str("and potentially wider stop levels to accommodate larger price swings.");
    } else if risk_level == "moderate" {
        analysis.push_str("This suggests standard risk management approaches with balanced position sizing ");
        analysis.push_str("and moderate stop levels based on recent volatility patterns.");
    } else {
        analysis.push_str("This allows for larger position sizes with tighter stop levels, ");
        analysis.push_str("potentially improving risk-adjusted returns through higher capital efficiency.");
    }
    
    analysis
}

/// Print detailed model interpretation
fn print_model_interpretation(
    name: &str, 
    error_type: ETSComponent,
    trend_type: ETSComponent, 
    seasonal_type: ETSComponent,
    alpha: f64,
    beta: Option<f64>,
    gamma: Option<f64>,
    phi: Option<f64>,
    seasonal_period: Option<usize>,
    mae: f64,
    rmse: f64,
    mape: f64
) {
    println!("\n{}", "=".repeat(100));
    println!("DETAILED INTERPRETATION FOR: {}\n", name);
    
    // Print trend analysis
    println!("{}\n", interpret_trend(name, trend_type, beta, phi, mae));
    
    // Print seasonality analysis
    println!("{}\n", interpret_seasonality(seasonal_type, seasonal_period, gamma));
    
    // Print volatility analysis
    println!("{}\n", interpret_volatility(error_type, alpha, rmse));
    
    // Print trading insights
    println!("{}\n", trading_insights(trend_type, seasonal_type, mape));
    
    // Print risk assessment
    println!("{}\n", risk_assessment(error_type, trend_type, mape, rmse));
    
    println!("{}", "=".repeat(100));
}

/// Generate a concise trading summary based on all model results
fn generate_trading_summary(
    ticker: &str,
    data_type: &str,
    evaluations: Vec<(&str, f64, f64, f64, ETSComponent, ETSComponent)>, // (name, mae, rmse, mape, trend, seasonality)
) {
    println!("\nTRADER'S ACTIONABLE SUMMARY FOR {}", ticker.to_uppercase());
    println!("{}", "=".repeat(80));
    
    // Find the best model based on MAPE (or use RMSE/MAE if preferred)
    let mut best_model_idx = 0;
    let mut best_mape = f64::MAX;
    
    for (i, (_, _, _, mape, _, _)) in evaluations.iter().enumerate() {
        if *mape < best_mape {
            best_mape = *mape;
            best_model_idx = i;
        }
    }
    
    let (best_name, _, best_rmse, best_mape, best_trend, best_seasonality) = evaluations[best_model_idx];
    
    // Overall volatility rating
    let volatility_rating = if best_rmse > 40.0 {
        "HIGH"
    } else if best_rmse > 20.0 {
        "MODERATE"
    } else {
        "LOW"
    };
    
    // Overall predictability rating
    let predictability_rating = if best_mape < 8.0 {
        "HIGH"
    } else if best_mape < 15.0 {
        "MODERATE" 
    } else {
        "LOW"
    };
    
    // Trend direction assessment
    let trend_assessment = match best_trend {
        ETSComponent::None => "NO CLEAR TREND detected - mean reversion is likely",
        ETSComponent::Additive => "UPWARD TREND detected - prices showing steady increase",
        ETSComponent::Multiplicative => "ACCELERATING TREND detected - potential momentum play",
        ETSComponent::Damped => "SLOWING TREND detected - trend losing momentum and likely to stabilize",
    };
    
    // Seasonality assessment
    let seasonality_assessment = match best_seasonality {
        ETSComponent::None => "NO SEASONALITY detected - no cyclical patterns apparent",
        _ => "SEASONAL PATTERNS detected - cyclical movements likely to repeat",
    };
    
    // Entry/exit signal
    let entry_signal = match (best_trend, best_seasonality, volatility_rating, predictability_rating) {
        // Strong Buy
        (ETSComponent::Additive, _, "LOW", "HIGH") | 
        (ETSComponent::Additive, ETSComponent::Additive, "MODERATE", "HIGH") => "STRONG BUY - Trending upward with high predictability",
        
        // Buy
        (ETSComponent::Additive, _, _, _) |
        (ETSComponent::None, ETSComponent::Additive, "LOW", _) => "BUY - Positive factors outweigh negative ones",
        
        // Hold/Neutral
        (ETSComponent::None, ETSComponent::None, _, _) |
        (ETSComponent::Damped, _, _, _) => "NEUTRAL/HOLD - No clear directional opportunity",
        
        // Sell
        (ETSComponent::Multiplicative, _, "HIGH", "LOW") => "SELL - High volatility with unpredictable movements",
        
        // Default case - depends on volatility
        _ => {
            if volatility_rating == "HIGH" && predictability_rating == "LOW" {
                "CAUTION - High risk profile suggests staying on sidelines"
            } else if predictability_rating == "HIGH" {
                "SELECTIVE ENTRY - Look for technical confirmation signals"
            } else {
                "MONITOR - Wait for clearer signals before taking positions"
            }
        }
    };
    
    // Risk management advice
    let risk_advice = match (volatility_rating, predictability_rating) {
        ("HIGH", _) => "Use smaller position sizes and wider stops due to high volatility",
        (_, "LOW") => "Consider options strategies or reduced exposure to manage unpredictability",
        ("MODERATE", "MODERATE") => "Standard position sizing with moderate stops based on recent volatility",
        ("LOW", "HIGH") => "Can use larger positions with tighter stops given the predictable nature",
        _ => "Use balanced risk management tailored to your risk tolerance", 
    };
    
    // Trading strategy suggestion
    let strategy_suggestion = match (best_trend, best_seasonality) {
        (ETSComponent::None, ETSComponent::None) => "Range-bound strategies: Consider mean-reversion trades, selling high and buying low within established ranges.",
        (ETSComponent::None, _) => "Calendar-based trading: Look for seasonal patterns to anticipate cyclical price movements while maintaining range awareness.",
        (ETSComponent::Additive, ETSComponent::None) => "Trend-following: Use breakouts, moving average systems, or momentum indicators to capture the trend direction.",
        (ETSComponent::Additive, _) => "Trend with seasonal adjustments: Follow the primary trend but adjust entries/exits based on seasonal factors that may temporarily accelerate or slow the trend.",
        (ETSComponent::Multiplicative, _) => "Momentum strategy: Look for acceleration in price movement and use trailing stops to capture extended moves while protecting gains.",
        (ETSComponent::Damped, _) => "Transition strategy: Prepare for trend exhaustion and potential reversal; consider taking profits on trend positions and look for reversal signals.",
    };
    
    // Time horizon recommendation
    let time_horizon = if data_type == "minute" {
        "SHORT-TERM (intraday) trading"
    } else if best_trend == ETSComponent::None {
        "MEDIUM-TERM (weeks) time horizon"
    } else {
        "LONGER-TERM (months) position appropriate"
    };
    
    // Output the summary
    println!("Best performing model: {}", best_name);
    println!("Forecast accuracy: {:.2}% error (MAPE)", best_mape);
    println!("\nKEY INSIGHTS:");
    println!("• Trend assessment: {}", trend_assessment);
    println!("• Seasonality: {}", seasonality_assessment);
    println!("• Volatility: {} (RMSE: {:.2})", volatility_rating, best_rmse);
    println!("• Predictability: {}", predictability_rating);
    
    println!("\nACTIONABLE ADVICE:");
    println!("• Signal: {}", entry_signal);
    println!("• Risk management: {}", risk_advice);
    println!("• Suggested strategy: {}", strategy_suggestion);
    println!("• Appropriate time horizon: {}", time_horizon);
    
    println!("\nDISCLAIMER: This is algorithmic analysis for educational purposes only.");
    println!("Always conduct your own research and consider your investment goals before trading.");
    println!("{}", "=".repeat(80));
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("OxiDiviner ETS Model Demo");
    println!("========================\n");
    
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    
    // Default settings
    let mut ticker = "SYNTHETIC";
    let mut data_type = "daily";
    
    // Process arguments if provided
    if args.len() > 1 {
        ticker = &args[1];
    }
    
    if args.len() > 2 {
        data_type = &args[2];
        if data_type != "daily" && data_type != "minute" {
            println!("Invalid data type: {}. Using 'daily' instead.", data_type);
            data_type = "daily";
        }
    }
    
    // Help message
    if ticker == "-h" || ticker == "--help" {
        println!("Usage: ets_demo [TICKER] [DATA_TYPE]");
        println!("  TICKER    - Stock ticker symbol (e.g., AAPL, MSFT) or SYNTHETIC for synthetic data");
        println!("  DATA_TYPE - Type of data to use: 'daily' or 'minute' (default: 'daily')");
        println!("\nAvailable tickers:");
        println!("  SYNTHETIC - Uses synthetic data generated with known patterns");
        println!("  AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA, TSM - Real stock data");
        println!("\nExamples:");
        println!("  ets_demo AAPL daily    - Analyze Apple daily stock data");
        println!("  ets_demo MSFT minute   - Analyze Microsoft minute stock data");
        println!("  ets_demo SYNTHETIC     - Use synthetic data with known patterns");
        return Ok(());
    }
    
    let data = if ticker == "SYNTHETIC" {
        println!("Generating synthetic {} data...", data_type);
        
        if data_type == "minute" {
            generate_synthetic_minute_data()
        } else {
            generate_synthetic_daily_data()
        }
    } else {
        println!("Loading {} {} data from CSV...", ticker, data_type);
        match load_stock_data_from_csv(ticker, data_type) {
            Ok(data) => {
                println!("Successfully loaded {} data points", data.len());
                data
            },
            Err(e) => {
                println!("Error loading data: {}", e);
                println!("Falling back to synthetic data");
                
                if data_type == "minute" {
                    generate_synthetic_minute_data()
                } else {
                    generate_synthetic_daily_data()
                }
            }
        }
    };
    
    // Create directory for CSV output if it doesn't exist
    let csv_dir = Path::new("examples").join("csv");
    if !csv_dir.exists() {
        std::fs::create_dir_all(&csv_dir)?;
    }
    
    // Save data to CSV (useful for verification)
    let output_filename = format!("{}_{}.csv", ticker.to_lowercase(), data_type);
    save_to_csv(&data, &output_filename)?;
    println!("Data saved to {}", csv_dir.join(&output_filename).display());
    
    // Create train/test splits
    let (train_data, test_data) = data.train_test_split(0.8)?;
    
    println!("\n{} data: {} observations (train: {}, test: {})",
             if data_type == "daily" { "Daily" } else { "Minute" },
             data.len(), train_data.len(), test_data.len());
    
    println!("\nThis demo will run various ETS (Error-Trend-Seasonality) models");
    println!("on the data and provide detailed analysis and interpretation.\n");
    
    // Run appropriate models based on data type
    if data_type == "daily" {
        run_daily_models(&train_data, &test_data, ticker)?;
    } else {
        run_minute_models(&train_data, &test_data, ticker)?;
    }
    
    println!("\nDemo completed successfully!");
    
    Ok(())
}

// Function to run and analyze daily models
fn run_daily_models(train_data: &ModelsOHLCVData, test_data: &ModelsOHLCVData, ticker: &str) 
    -> Result<(), Box<dyn Error>> {
    
    println!("Running ETS models for daily {} data...", ticker);
    
    // 1. Simple Exponential Smoothing (ETS(A,N,N))
    println!("\nModel 1: ETS(A,N,N) - Simple Exponential Smoothing");
    let mut daily_model1 = DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::None,      // No trend
        ETSComponent::None,      // No seasonality
        0.3,                     // alpha = 0.3
        None,                    // No beta (no trend)
        None,                    // No gamma (no seasonality)
        None,                    // No phi (no damping)
        None,                    // No seasonal period
        None,                    // Default to Close price
    )?;
    
    daily_model1.fit(train_data)?;
    let daily_eval1 = daily_model1.evaluate(test_data)?;
    
    println!("Model: {}", daily_eval1.model_name);
    println!("MAE: {:.4}", daily_eval1.mae);
    println!("RMSE: {:.4}", daily_eval1.rmse);
    println!("MAPE: {:.4}%\n", daily_eval1.mape);
    
    print_model_interpretation(
        &daily_eval1.model_name,
        ETSComponent::Additive,     // Error type
        ETSComponent::None,         // Trend type
        ETSComponent::None,         // Seasonal type
        0.3,                        // alpha
        None,                       // beta
        None,                       // gamma
        None,                       // phi
        None,                       // seasonal_period
        daily_eval1.mae,
        daily_eval1.rmse,
        daily_eval1.mape
    );
    
    // 2. Holt's Linear Trend (ETS(A,A,N))
    println!("\nModel 2: ETS(A,A,N) - Holt's Linear Trend");
    let mut daily_model2 = DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::Additive,  // Additive trend
        ETSComponent::None,      // No seasonality
        0.3,                     // alpha = 0.3
        Some(0.1),               // beta = 0.1
        None,                    // No gamma (no seasonality)
        None,                    // No phi (no damping)
        None,                    // No seasonal period
        None,                    // Default to Close price
    )?;
    
    daily_model2.fit(train_data)?;
    let daily_eval2 = daily_model2.evaluate(test_data)?;
    
    println!("Model: {}", daily_eval2.model_name);
    println!("MAE: {:.4}", daily_eval2.mae);
    println!("RMSE: {:.4}", daily_eval2.rmse);
    println!("MAPE: {:.4}%\n", daily_eval2.mape);
    
    print_model_interpretation(
        &daily_eval2.model_name,
        ETSComponent::Additive,     // Error type
        ETSComponent::Additive,     // Trend type
        ETSComponent::None,         // Seasonal type
        0.3,                        // alpha
        Some(0.1),                  // beta
        None,                       // gamma
        None,                       // phi
        None,                       // seasonal_period
        daily_eval2.mae,
        daily_eval2.rmse,
        daily_eval2.mape
    );
    
    // 3. Seasonal Model (ETS(A,N,A))
    println!("\nModel 3: ETS(A,N,A) - Seasonal without Trend");
    let mut daily_model3 = DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::None,      // No trend
        ETSComponent::Additive,  // Additive seasonality
        0.3,                     // alpha = 0.3
        None,                    // No beta (no trend)
        Some(0.1),               // gamma = 0.1
        None,                    // No phi (no damping)
        Some(30),                // Seasonal period = 30 days
        None,                    // Default to Close price
    )?;
    
    daily_model3.fit(train_data)?;
    let daily_eval3 = daily_model3.evaluate(test_data)?;
    
    println!("Model: {}", daily_eval3.model_name);
    println!("MAE: {:.4}", daily_eval3.mae);
    println!("RMSE: {:.4}", daily_eval3.rmse);
    println!("MAPE: {:.4}%\n", daily_eval3.mape);
    
    print_model_interpretation(
        &daily_eval3.model_name,
        ETSComponent::Additive,     // Error type
        ETSComponent::None,         // Trend type
        ETSComponent::Additive,     // Seasonal type
        0.3,                        // alpha
        None,                       // beta
        Some(0.1),                  // gamma
        None,                       // phi
        Some(30),                   // seasonal_period
        daily_eval3.mae,
        daily_eval3.rmse,
        daily_eval3.mape
    );
    
    // 4. Holt-Winters (ETS(A,A,A))
    println!("\nModel 4: ETS(A,A,A) - Holt-Winters");
    let mut daily_model4 = DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::Additive,  // Additive trend
        ETSComponent::Additive,  // Additive seasonality
        0.3,                     // alpha = 0.3
        Some(0.1),               // beta = 0.1 
        Some(0.1),               // gamma = 0.1
        None,                    // No phi (no damping)
        Some(30),                // Seasonal period = 30 days
        None,                    // Default to Close price
    )?;
    
    daily_model4.fit(train_data)?;
    let daily_eval4 = daily_model4.evaluate(test_data)?;
    
    println!("Model: {}", daily_eval4.model_name);
    println!("MAE: {:.4}", daily_eval4.mae);
    println!("RMSE: {:.4}", daily_eval4.rmse);
    println!("MAPE: {:.4}%\n", daily_eval4.mape);
    
    print_model_interpretation(
        &daily_eval4.model_name,
        ETSComponent::Additive,     // Error type
        ETSComponent::Additive,     // Trend type
        ETSComponent::Additive,     // Seasonal type
        0.3,                        // alpha
        Some(0.1),                  // beta
        Some(0.1),                  // gamma
        None,                       // phi
        Some(30),                   // seasonal_period
        daily_eval4.mae,
        daily_eval4.rmse,
        daily_eval4.mape
    );
    
    // 5. Damped Trend (ETS(A,D,A))
    println!("\nModel 5: ETS(A,D,A) - Damped Trend with Seasonality");
    let mut daily_model5 = DailyETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::Damped,    // Damped trend
        ETSComponent::Additive,  // Additive seasonality
        0.3,                     // alpha = 0.3
        Some(0.1),               // beta = 0.1
        Some(0.1),               // gamma = 0.1
        Some(0.9),               // phi = 0.9 (damping factor)
        Some(30),                // Seasonal period = 30 days
        None,                    // Default to Close price
    )?;
    
    daily_model5.fit(train_data)?;
    let daily_eval5 = daily_model5.evaluate(test_data)?;
    
    println!("Model: {}", daily_eval5.model_name);
    println!("MAE: {:.4}", daily_eval5.mae);
    println!("RMSE: {:.4}", daily_eval5.rmse);
    println!("MAPE: {:.4}%\n", daily_eval5.mape);
    
    print_model_interpretation(
        &daily_eval5.model_name,
        ETSComponent::Additive,     // Error type
        ETSComponent::Damped,       // Trend type
        ETSComponent::Additive,     // Seasonal type
        0.3,                        // alpha
        Some(0.1),                  // beta
        Some(0.1),                  // gamma
        Some(0.9),                  // phi
        Some(30),                   // seasonal_period
        daily_eval5.mae,
        daily_eval5.rmse,
        daily_eval5.mape
    );
    
    // Collect results for trading summary
    let model_evaluations = vec![
        (daily_eval1.model_name.as_str(), daily_eval1.mae, daily_eval1.rmse, daily_eval1.mape, 
            ETSComponent::None, ETSComponent::None),
        (daily_eval2.model_name.as_str(), daily_eval2.mae, daily_eval2.rmse, daily_eval2.mape, 
            ETSComponent::Additive, ETSComponent::None),
        (daily_eval3.model_name.as_str(), daily_eval3.mae, daily_eval3.rmse, daily_eval3.mape, 
            ETSComponent::None, ETSComponent::Additive),
        (daily_eval4.model_name.as_str(), daily_eval4.mae, daily_eval4.rmse, daily_eval4.mape, 
            ETSComponent::Additive, ETSComponent::Additive),
        (daily_eval5.model_name.as_str(), daily_eval5.mae, daily_eval5.rmse, daily_eval5.mape, 
            ETSComponent::Damped, ETSComponent::Additive),
    ];
    
    // Generate trader summary
    generate_trading_summary(ticker, "daily", model_evaluations);
    
    Ok(())
}

// Function to run and analyze minute models
fn run_minute_models(train_data: &ModelsOHLCVData, test_data: &ModelsOHLCVData, ticker: &str) 
    -> Result<(), Box<dyn Error>> {
    
    println!("Running ETS models for minute {} data...", ticker);
    
    // 1. Simple Exponential Smoothing (ETS(A,N,N))
    println!("\nMinute Model 1: ETS(A,N,N) - Simple Exponential Smoothing");
    let mut minute_model1 = MinuteETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::None,      // No trend
        ETSComponent::None,      // No seasonality
        0.3,                     // alpha = 0.3
        None,                    // No beta (no trend)
        None,                    // No gamma (no seasonality)
        None,                    // No phi (no damping)
        None,                    // No seasonal period
        None,                    // Default to Close price
        None,                    // Default aggregation (1 minute)
    )?;
    
    minute_model1.fit(train_data)?;
    let minute_eval1 = minute_model1.evaluate(test_data)?;
    
    println!("Model: {}", minute_eval1.model_name);
    println!("MAE: {:.4}", minute_eval1.mae);
    println!("RMSE: {:.4}", minute_eval1.rmse);
    println!("MAPE: {:.4}%\n", minute_eval1.mape);
    
    print_model_interpretation(
        &minute_eval1.model_name,
        ETSComponent::Additive,     // Error type
        ETSComponent::None,         // Trend type
        ETSComponent::None,         // Seasonal type
        0.3,                        // alpha
        None,                       // beta
        None,                       // gamma
        None,                       // phi
        None,                       // seasonal_period
        minute_eval1.mae,
        minute_eval1.rmse,
        minute_eval1.mape
    );
    
    // 2. Aggregated Minute Data (ETS(A,N,N) with 5-minute aggregation)
    println!("\nMinute Model 2: ETS(A,N,N) with 5-minute aggregation");
    let mut minute_model2 = MinuteETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::None,      // No trend
        ETSComponent::None,      // No seasonality
        0.3,                     // alpha = 0.3
        None,                    // No beta (no trend)
        None,                    // No gamma (no seasonality)
        None,                    // No phi (no damping)
        None,                    // No seasonal period
        None,                    // Default to Close price
        Some(5),                 // 5-minute aggregation
    )?;
    
    minute_model2.fit(train_data)?;
    let minute_eval2 = minute_model2.evaluate(test_data)?;
    
    println!("Model: {}", minute_eval2.model_name);
    println!("MAE: {:.4}", minute_eval2.mae);
    println!("RMSE: {:.4}", minute_eval2.rmse);
    println!("MAPE: {:.4}%\n", minute_eval2.mape);
    
    print_model_interpretation(
        &minute_eval2.model_name,
        ETSComponent::Additive,     // Error type
        ETSComponent::None,         // Trend type
        ETSComponent::None,         // Seasonal type
        0.3,                        // alpha
        None,                       // beta
        None,                       // gamma
        None,                       // phi
        None,                       // seasonal_period
        minute_eval2.mae,
        minute_eval2.rmse,
        minute_eval2.mape
    );
    
    // 3. Seasonal Minute Model (ETS(A,N,A) with 60-minute seasonality)
    println!("\nMinute Model 3: ETS(A,N,A) - Seasonal Model with 60-minute period");
    let mut minute_model3 = MinuteETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::None,      // No trend
        ETSComponent::Additive,  // Additive seasonality
        0.3,                     // alpha = 0.3
        None,                    // No beta (no trend)
        Some(0.1),               // gamma = 0.1
        None,                    // No phi (no damping)
        Some(60),                // Seasonal period = 60 minutes
        None,                    // Default to Close price
        None,                    // Default aggregation (1 minute)
    )?;
    
    minute_model3.fit(train_data)?;
    let minute_eval3 = minute_model3.evaluate(test_data)?;
    
    println!("Model: {}", minute_eval3.model_name);
    println!("MAE: {:.4}", minute_eval3.mae);
    println!("RMSE: {:.4}", minute_eval3.rmse);
    println!("MAPE: {:.4}%\n", minute_eval3.mape);
    
    print_model_interpretation(
        &minute_eval3.model_name,
        ETSComponent::Additive,     // Error type
        ETSComponent::None,         // Trend type
        ETSComponent::Additive,     // Seasonal type
        0.3,                        // alpha
        None,                       // beta
        Some(0.1),                  // gamma
        None,                       // phi
        Some(60),                   // seasonal_period
        minute_eval3.mae,
        minute_eval3.rmse,
        minute_eval3.mape
    );
    
    // 4. Holt-Winters for Minute Data (ETS(A,A,A))
    println!("\nMinute Model 4: ETS(A,A,A) - Holt-Winters with 60-minute seasonality and 15-min aggregation");
    let mut minute_model4 = MinuteETSModel::new(
        ETSComponent::Additive,  // Error type
        ETSComponent::Additive,  // Additive trend
        ETSComponent::Additive,  // Additive seasonality
        0.3,                     // alpha = 0.3
        Some(0.1),               // beta = 0.1
        Some(0.1),               // gamma = 0.1
        None,                    // No phi (no damping)
        Some(60),                // Seasonal period = 60 minutes (adjusted for 15-min aggregation = 4 periods per hour)
        None,                    // Default to Close price
        Some(15),                // 15-minute aggregation
    )?;
    
    minute_model4.fit(train_data)?;
    let minute_eval4 = minute_model4.evaluate(test_data)?;
    
    println!("Model: {}", minute_eval4.model_name);
    println!("MAE: {:.4}", minute_eval4.mae);
    println!("RMSE: {:.4}", minute_eval4.rmse);
    println!("MAPE: {:.4}%\n", minute_eval4.mape);
    
    print_model_interpretation(
        &minute_eval4.model_name,
        ETSComponent::Additive,     // Error type
        ETSComponent::Additive,     // Trend type
        ETSComponent::Additive,     // Seasonal type
        0.3,                        // alpha
        Some(0.1),                  // beta
        Some(0.1),                  // gamma
        None,                       // phi
        Some(60),                   // seasonal_period
        minute_eval4.mae,
        minute_eval4.rmse,
        minute_eval4.mape
    );
    
    // Collect results for trading summary
    let model_evaluations = vec![
        (minute_eval1.model_name.as_str(), minute_eval1.mae, minute_eval1.rmse, minute_eval1.mape, 
            ETSComponent::None, ETSComponent::None),
        (minute_eval2.model_name.as_str(), minute_eval2.mae, minute_eval2.rmse, minute_eval2.mape, 
            ETSComponent::None, ETSComponent::None),
        (minute_eval3.model_name.as_str(), minute_eval3.mae, minute_eval3.rmse, minute_eval3.mape, 
            ETSComponent::None, ETSComponent::Additive),
        (minute_eval4.model_name.as_str(), minute_eval4.mae, minute_eval4.rmse, minute_eval4.mape, 
            ETSComponent::Additive, ETSComponent::Additive),
    ];
    
    // Generate trader summary
    generate_trading_summary(ticker, "minute", model_evaluations);
    
    Ok(())
} 