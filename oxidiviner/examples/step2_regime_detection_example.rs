//! STEP 2: Regime Detection Foundation Example
//!
//! This example demonstrates the regime detection capabilities implemented in STEP 2
//! using real market data from CSV files.

use chrono::{DateTime, Utc};
use oxidiviner::adaptive::{AdaptiveConfig, MarketRegime, RegimeDetector};
use oxidiviner::core::{Result, TimeSeriesData};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn main() -> Result<()> {
    println!("🎯 STEP 2: Regime Detection Foundation Example");
    println!("==============================================");
    println!("📊 Using real market data from CSV files\n");

    // Example 1: Basic regime detection with Apple stock data
    basic_regime_detection_example()?;

    // Example 2: Multi-stock regime comparison
    multi_stock_regime_analysis()?;

    // Example 3: Performance validation with high-frequency data
    performance_validation_example()?;

    println!("\n✅ All STEP 2 examples completed successfully!");
    println!("📊 Regime detection system validated with real market data");

    Ok(())
}

/// Example 1: Basic Regime Detection with Apple Stock Data
fn basic_regime_detection_example() -> Result<()> {
    println!("📈 Example 1: Basic Regime Detection (AAPL Daily Data)");
    println!("------------------------------------------------------");

    // Load Apple daily OHLCV data
    let aapl_data = load_ohlcv_csv("../examples/csv/AAPL_daily_ohlcv.csv")?;
    println!("📊 Loaded {} AAPL data points", aapl_data.values.len());

    // Create and fit detector
    let config = AdaptiveConfig::default();
    let mut detector = RegimeDetector::new(config)?;

    println!("🔧 Fitting regime detector to AAPL historical data...");
    let fit_start = Instant::now();
    detector.fit(&aapl_data)?;
    let fit_time = fit_start.elapsed();
    println!("✅ Fitted in {:.2}ms", fit_time.as_millis());

    // Test regime detection with different market scenarios
    let test_scenarios = vec![
        (120.0, "Bear market scenario (significant drop)"),
        (160.0, "Neutral market scenario (average price)"),
        (200.0, "Bull market scenario (strong growth)"),
        (250.0, "Extreme bull scenario (major breakout)"),
    ];

    println!("\n📈 Testing regime detection scenarios:");
    for (price, description) in test_scenarios {
        let start = Instant::now();
        let result = detector.detect_regime(price)?;
        let latency = start.elapsed();

        println!("  {} (Price: ${:.2})", description, price);
        println!("    Regime: {:?}", result.current_regime);
        println!("    Confidence: {:.1}%", result.confidence * 100.0);
        println!("    Duration: {} periods", result.duration_in_regime);
        println!("    Change Prob: {:.1}%", result.change_probability * 100.0);
        println!("    Latency: {}ms", latency.as_millis());

        // Verify latency requirement
        assert!(
            latency.as_millis() < 50,
            "Latency requirement not met: {}ms",
            latency.as_millis()
        );
    }

    // Display performance metrics
    let metrics = detector.get_metrics();
    println!("\n📊 Detector Performance Metrics:");
    println!("  Accuracy: {:.1}%", metrics.accuracy * 100.0);
    println!("  Avg Latency: {:.2}ms", metrics.avg_latency_ms);
    println!("  Max Latency: {}ms", metrics.max_latency_ms);
    println!("  Regime Changes: {}", metrics.regime_changes_detected);

    println!("✅ Basic regime detection validated with real AAPL data\n");
    Ok(())
}

/// Example 2: Multi-Stock Regime Comparison
fn multi_stock_regime_analysis() -> Result<()> {
    println!("🔍 Example 2: Multi-Stock Regime Comparison");
    println!("-------------------------------------------");

    let stocks = vec![
        ("AAPL", "../examples/csv/AAPL_daily_ohlcv.csv"),
        ("NVDA", "../examples/csv/NVDA_daily_ohlcv.csv"),
        ("TSLA", "../examples/csv/TSLA_daily_ohlcv.csv"),
    ];

    println!("📊 Analyzing regime patterns across multiple stocks...\n");

    for (symbol, file_path) in stocks {
        println!("  📈 {} Stock Analysis:", symbol);

        // Load and analyze stock data
        let data = load_ohlcv_csv(file_path)?;
        let config = AdaptiveConfig::default();
        let mut detector = RegimeDetector::new(config)?;

        detector.fit(&data)?;

        // Get current market price (last close price)
        let current_price = data.values[data.values.len() - 1];

        let result = detector.detect_regime(current_price)?;

        println!("    Current Price: ${:.2}", current_price);
        println!("    Current Regime: {:?}", result.current_regime);
        println!("    Confidence: {:.1}%", result.confidence * 100.0);
        println!(
            "    Regime Description: {}",
            result.current_regime.description()
        );

        // Analyze regime stability
        let metrics = detector.get_metrics();
        println!("    Regime Changes: {}", metrics.regime_changes_detected);
        println!();
    }

    println!("✅ Multi-stock regime analysis completed\n");
    Ok(())
}

/// Example 3: Performance Validation with High-Frequency Data
fn performance_validation_example() -> Result<()> {
    println!("⚡ Example 3: Performance Validation");
    println!("------------------------------------");

    // Load minute-level data for performance testing
    let minute_data = load_simple_csv("../examples/csv/minute_data.csv")?;
    println!(
        "📊 Loaded {} minute-level data points",
        minute_data.values.len()
    );

    let config = AdaptiveConfig::default();
    let mut detector = RegimeDetector::new(config)?;

    // Fit on subset of data
    detector.fit(&minute_data)?;

    println!("🔧 Running high-frequency detection simulation...");

    // Simulate real-time detection on streaming data
    let test_values: Vec<f64> = minute_data.values[minute_data.values.len() - 20..]
        .iter()
        .copied()
        .collect();

    let mut total_latency = 0u64;
    let mut max_latency = 0u64;
    let mut regime_changes = 0;
    let mut previous_regime = None;

    for (i, &value) in test_values.iter().enumerate() {
        let start = Instant::now();
        let result = detector.detect_regime(value)?;
        let latency = start.elapsed().as_millis() as u64;

        total_latency += latency;
        max_latency = max_latency.max(latency);

        // Track regime changes
        if let Some(prev) = previous_regime {
            if prev != result.regime_index {
                regime_changes += 1;
                println!(
                    "  🔄 Regime change detected at step {}: {:?} → {:?}",
                    i + 1,
                    MarketRegime::from_index(prev, 2),
                    result.current_regime
                );
            }
        }
        previous_regime = Some(result.regime_index);

        // Verify latency on each detection
        assert!(latency < 50, "Detection latency too high: {}ms", latency);
    }

    let avg_latency = total_latency / test_values.len() as u64;

    println!("\n📊 High-Frequency Performance Results:");
    println!("  Total Detections: {}", test_values.len());
    println!("  Average Latency: {}ms", avg_latency);
    println!("  Maximum Latency: {}ms", max_latency);
    println!("  Regime Changes: {}", regime_changes);
    println!(
        "  Throughput: {:.1} detections/second",
        1000.0 / avg_latency as f64
    );

    // Verify performance requirements
    assert!(avg_latency < 50, "Average latency requirement not met");
    assert!(max_latency < 100, "Maximum latency too high");

    println!("✅ Performance requirements validated:");
    println!("  ✓ Average latency < 50ms ({} ms)", avg_latency);
    println!("  ✓ Maximum latency < 100ms ({} ms)", max_latency);
    println!("  ✓ High-frequency detection capability confirmed");

    println!();
    Ok(())
}

/// Load OHLCV CSV data and return TimeSeriesData using Close prices
fn load_ohlcv_csv(file_path: &str) -> Result<TimeSeriesData> {
    let file = File::open(file_path).map_err(|e| {
        oxidiviner::core::OxiError::InvalidParameter(format!("Failed to open {}: {}", file_path, e))
    })?;

    let reader = BufReader::new(file);
    let mut timestamps = Vec::new();
    let mut values = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| {
            oxidiviner::core::OxiError::InvalidParameter(format!(
                "Failed to read line {}: {}",
                line_num, e
            ))
        })?;

        if line_num == 0 {
            // Skip header
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 5 {
            continue; // Skip invalid lines
        }

        // Parse timestamp
        let timestamp_str = parts[0];
        let timestamp = timestamp_str.parse::<DateTime<Utc>>().map_err(|e| {
            oxidiviner::core::OxiError::InvalidParameter(format!(
                "Invalid timestamp '{}': {}",
                timestamp_str, e
            ))
        })?;

        // Parse close price (column 4, index 4)
        let close_price: f64 = parts[4].parse().map_err(|e| {
            oxidiviner::core::OxiError::InvalidParameter(format!(
                "Invalid close price '{}': {}",
                parts[4], e
            ))
        })?;

        timestamps.push(timestamp);
        values.push(close_price);
    }

    if values.is_empty() {
        return Err(oxidiviner::core::OxiError::InvalidParameter(format!(
            "No valid data found in {}",
            file_path
        )));
    }

    let name = file_path
        .split('/')
        .last()
        .unwrap_or("csv_data")
        .replace(".csv", "");
    TimeSeriesData::new(timestamps, values, &name)
}

/// Load simple CSV data (timestamp, value format)
fn load_simple_csv(file_path: &str) -> Result<TimeSeriesData> {
    let file = File::open(file_path).map_err(|e| {
        oxidiviner::core::OxiError::InvalidParameter(format!("Failed to open {}: {}", file_path, e))
    })?;

    let reader = BufReader::new(file);
    let mut timestamps = Vec::new();
    let mut values = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| {
            oxidiviner::core::OxiError::InvalidParameter(format!(
                "Failed to read line {}: {}",
                line_num, e
            ))
        })?;

        if line_num == 0 {
            // Skip header
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue; // Skip invalid lines
        }

        // Parse timestamp
        let timestamp_str = parts[0];
        let timestamp = timestamp_str.parse::<DateTime<Utc>>().map_err(|e| {
            oxidiviner::core::OxiError::InvalidParameter(format!(
                "Invalid timestamp '{}': {}",
                timestamp_str, e
            ))
        })?;

        // Parse close price (last column or column index based on file)
        let close_idx = if parts.len() > 5 { 4 } else { parts.len() - 1 }; // Close price column
        let close_price: f64 = parts[close_idx].parse().map_err(|e| {
            oxidiviner::core::OxiError::InvalidParameter(format!(
                "Invalid price '{}': {}",
                parts[close_idx], e
            ))
        })?;

        timestamps.push(timestamp);
        values.push(close_price);
    }

    if values.is_empty() {
        return Err(oxidiviner::core::OxiError::InvalidParameter(format!(
            "No valid data found in {}",
            file_path
        )));
    }

    let name = file_path
        .split('/')
        .last()
        .unwrap_or("csv_data")
        .replace(".csv", "");
    TimeSeriesData::new(timestamps, values, &name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_loading() {
        let result = load_ohlcv_csv("../examples/csv/AAPL_daily_ohlcv.csv");
        assert!(result.is_ok());

        let data = result.unwrap();
        assert!(data.values.len() > 100); // Should have substantial data
        assert_eq!(data.timestamps.len(), data.values.len());
    }

    #[test]
    fn test_basic_regime_detection_with_real_data() {
        let result = basic_regime_detection_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_multi_stock_analysis() {
        let result = multi_stock_regime_analysis();
        assert!(result.is_ok());
    }
}
