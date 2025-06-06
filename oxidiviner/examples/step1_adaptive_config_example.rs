//! # STEP 1: Enhanced Configuration System Example
//!
//! This example demonstrates the new adaptive configuration capabilities
//! introduced in STEP 1 of the adaptive forecasting blueprint.
//!
//! ## Features Demonstrated:
//! - Creating adaptive configurations with validation
//! - Backward compatibility with existing ForecastConfig
//! - Quality monitoring and thresholds
//! - Configuration serialization and persistence
//! - Real-time quality assessment

use oxidiviner::adaptive::monitoring::QualityMonitor;
use oxidiviner::adaptive::{AdaptiveConfig, ModelSelectionStrategy, QualityThresholds};
use oxidiviner::api::{ForecastConfig, ModelType};
use oxidiviner::core::{ModelEvaluation, Result};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("🚀 OxiDiviner STEP 1: Enhanced Configuration System Demo");
    println!("{}", "=".repeat(60));

    // 1. Basic adaptive configuration
    demonstrate_basic_adaptive_config()?;

    // 2. Advanced configuration with custom settings
    demonstrate_advanced_config()?;

    // 3. Backward compatibility
    demonstrate_backward_compatibility()?;

    // 4. Quality monitoring system
    demonstrate_quality_monitoring()?;

    // 5. Configuration serialization
    demonstrate_serialization()?;

    // 6. Performance validation
    demonstrate_performance_validation()?;

    println!("\n✅ All STEP 1 features demonstrated successfully!");
    println!("🎯 Ready to proceed to STEP 2: Regime Detection Foundation");

    Ok(())
}

fn demonstrate_basic_adaptive_config() -> Result<()> {
    println!("\n📋 1. Basic Adaptive Configuration");
    println!("{}", "-".repeat(40));

    // Create a basic adaptive configuration
    let config = AdaptiveConfig::new();

    println!("✓ Created adaptive config with defaults:");
    println!(
        "  - Learning rate: {}",
        config.adaptive_params.learning_rate
    );
    println!(
        "  - Adaptation window: {}",
        config.adaptive_params.adaptation_window
    );
    println!(
        "  - Regime detection: {}",
        config.is_regime_detection_enabled()
    );
    println!(
        "  - Quality monitoring: {}",
        config.is_quality_monitoring_enabled()
    );

    // Validate the configuration
    config.validate()?;
    println!("✓ Configuration validation passed");

    // Create a customized configuration using builder pattern
    let custom_config = AdaptiveConfig::new()
        .with_regime_detection(3)
        .with_learning_rate(0.05)
        .with_adaptation_window(100);

    custom_config.validate()?;
    println!("✓ Custom configuration created and validated");
    println!("  - Regimes: {}", custom_config.regime_config.num_regimes);
    println!(
        "  - Learning rate: {}",
        custom_config.adaptive_params.learning_rate
    );
    println!(
        "  - Window size: {}",
        custom_config.adaptive_params.adaptation_window
    );

    Ok(())
}

fn demonstrate_advanced_config() -> Result<()> {
    println!("\n⚙️  2. Advanced Configuration Options");
    println!("{}", "-".repeat(40));

    // Create custom quality thresholds
    let quality_thresholds = oxidiviner::adaptive::config::QualityThresholds {
        max_mae: 0.1,
        max_mape: 10.0,
        min_r_squared: 0.8,
        quality_window: 25,
        enable_fallback: true,
        max_consecutive_failures: 2,
    };

    // Create regime-based model selection
    let mut regime_models = HashMap::new();
    regime_models.insert(0, ModelType::ARIMA); // Bull market
    regime_models.insert(1, ModelType::SimpleES); // Stable market
    regime_models.insert(2, ModelType::HoltLinear); // Trending market

    let model_selection = ModelSelectionStrategy::RegimeBased { regime_models };

    // Build comprehensive adaptive configuration
    let advanced_config = AdaptiveConfig::new()
        .with_regime_detection(3)
        .with_learning_rate(0.08)
        .with_adaptation_window(75)
        .with_quality_thresholds(quality_thresholds)
        .with_model_selection(model_selection);

    advanced_config.validate()?;
    println!("✓ Advanced configuration created with:");
    println!("  - Regime-based model selection (3 regimes)");
    println!("  - Custom quality thresholds");
    println!("  - Stricter performance requirements");

    // Demonstrate ensemble strategy
    let ensemble_config =
        AdaptiveConfig::new().with_model_selection(ModelSelectionStrategy::Ensemble {
            models: vec![
                ModelType::ARIMA,
                ModelType::SimpleES,
                ModelType::HoltWinters,
            ],
            weighting: oxidiviner::adaptive::config::EnsembleWeighting::Performance,
        });

    ensemble_config.validate()?;
    println!("✓ Ensemble configuration created with performance-based weighting");

    Ok(())
}

fn demonstrate_backward_compatibility() -> Result<()> {
    println!("\n🔄 3. Backward Compatibility");
    println!("{}", "-".repeat(40));

    // Create a traditional ForecastConfig
    let traditional_config = ForecastConfig {
        model_type: ModelType::HoltWinters,
        parameters: oxidiviner::api::ModelParameters {
            arima_p: Some(2),
            arima_d: Some(1),
            arima_q: Some(1),
            alpha: Some(0.3),
            beta: Some(0.2),
            gamma: Some(0.1),
            ma_window: Some(5),
            seasonal_period: Some(12),
        },
        auto_select: false,
    };

    println!("✓ Created traditional ForecastConfig:");
    println!("  - Model: {:?}", traditional_config.model_type);
    println!("  - Alpha: {:?}", traditional_config.parameters.alpha);
    println!(
        "  - Seasonal period: {:?}",
        traditional_config.parameters.seasonal_period
    );

    // Extend with adaptive capabilities
    let adaptive_config = AdaptiveConfig::from_base_config(traditional_config.clone());
    adaptive_config.validate()?;

    println!("✓ Enhanced with adaptive capabilities:");
    println!("  - Original config preserved");
    println!("  - Adaptive features added");
    println!("  - Validation successful");

    // Verify backward compatibility
    assert_eq!(
        adaptive_config.base_config().model_type,
        traditional_config.model_type
    );
    assert_eq!(
        adaptive_config.base_config().parameters.alpha,
        traditional_config.parameters.alpha
    );
    println!("✓ Backward compatibility verified");

    Ok(())
}

fn demonstrate_quality_monitoring() -> Result<()> {
    println!("\n📊 4. Quality Monitoring System");
    println!("{}", "-".repeat(40));

    // Create a quality monitor with custom thresholds
    let thresholds = oxidiviner::adaptive::monitoring::QualityThresholds {
        max_mae: 0.12,
        max_mape: 12.0,
        min_r_squared: 0.75,
        quality_window: 10,
        enable_fallback: true,
        max_consecutive_failures: 3,
    };

    let mut monitor = QualityMonitor::with_thresholds(thresholds);
    println!("✓ Quality monitor created with custom thresholds");

    // Simulate quality monitoring with various scenarios
    let test_evaluations = vec![
        ("Good performance", create_evaluation(0.08, 8.0, 0.85)),
        (
            "Acceptable performance",
            create_evaluation(0.11, 11.0, 0.76),
        ),
        ("Poor performance", create_evaluation(0.15, 15.0, 0.65)),
        ("Very poor performance", create_evaluation(0.18, 18.0, 0.55)),
        ("Recovery", create_evaluation(0.09, 9.0, 0.82)),
    ];

    for (description, evaluation) in test_evaluations {
        monitor.update_quality(&evaluation)?;

        let quality = monitor.current_quality().unwrap();
        println!(
            "  {} - Quality Score: {:.3}, Acceptable: {}",
            description, quality.quality_score, quality.quality_acceptable
        );

        if monitor.is_fallback_triggered() {
            println!("    ⚠️  Fallback triggered due to consecutive failures");
            monitor.reset_fallback(); // Reset for demonstration
        }
    }

    // Generate quality report
    let report = monitor.quality_report();
    println!("✓ Quality monitoring summary:");
    println!("  - Success rate: {:.1}%", report.success_rate * 100.0);
    println!(
        "  - Average quality score: {:.3}",
        report.average_quality_score
    );
    println!("  - Quality trend: {:.3}", report.quality_trend);
    println!("  - Total evaluations: {}", report.total_evaluations);

    Ok(())
}

fn demonstrate_serialization() -> Result<()> {
    println!("\n💾 5. Configuration Serialization");
    println!("{}", "-".repeat(40));

    // Create a complex configuration
    let config = AdaptiveConfig::new()
        .with_regime_detection(2)
        .with_learning_rate(0.12)
        .with_adaptation_window(60)
        .with_model_selection(ModelSelectionStrategy::Performance {
            window_size: 30,
            switch_threshold: 0.05,
        });

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&config)
        .map_err(|e| oxidiviner::core::OxiError::DataError(e.to_string()))?;

    println!("✓ Configuration serialized to JSON ({} bytes)", json.len());

    // Deserialize back
    let deserialized: AdaptiveConfig = serde_json::from_str(&json)
        .map_err(|e| oxidiviner::core::OxiError::DataError(e.to_string()))?;

    deserialized.validate()?;
    println!("✓ Deserialized and validated successfully");

    // Verify roundtrip accuracy
    assert_eq!(
        config.adaptive_params.learning_rate,
        deserialized.adaptive_params.learning_rate
    );
    assert_eq!(
        config.regime_config.num_regimes,
        deserialized.regime_config.num_regimes
    );
    println!("✓ Serialization roundtrip verified");

    // Show sample JSON structure (first few lines)
    let lines: Vec<&str> = json.lines().take(10).collect();
    println!("  Sample JSON structure:");
    for line in lines {
        println!("    {}", line);
    }
    if json.lines().count() > 10 {
        println!("    ... ({} more lines)", json.lines().count() - 10);
    }

    Ok(())
}

fn demonstrate_performance_validation() -> Result<()> {
    println!("\n⚡ 6. Performance Validation");
    println!("{}", "-".repeat(40));

    // Test configuration creation performance
    let start = std::time::Instant::now();

    for _ in 0..1000 {
        let config = AdaptiveConfig::new()
            .with_regime_detection(3)
            .with_learning_rate(0.1)
            .with_adaptation_window(50);

        config.validate()?;
    }

    let config_duration = start.elapsed();
    println!(
        "✓ Configuration creation: 1000 configs in {:?} ({:.2}μs each)",
        config_duration,
        config_duration.as_micros() as f64 / 1000.0
    );

    // Test serialization performance
    let config = AdaptiveConfig::new();
    let start = std::time::Instant::now();

    for _ in 0..100 {
        let json = serde_json::to_string(&config).unwrap();
        let _: AdaptiveConfig = serde_json::from_str(&json).unwrap();
    }

    let serialization_duration = start.elapsed();
    println!(
        "✓ Serialization: 100 roundtrips in {:?} ({:.2}μs each)",
        serialization_duration,
        serialization_duration.as_micros() as f64 / 100.0
    );

    // Test quality monitoring performance
    let mut monitor = QualityMonitor::new();
    let evaluation = create_evaluation(0.1, 10.0, 0.8);
    let start = std::time::Instant::now();

    for _ in 0..1000 {
        monitor.update_quality(&evaluation).unwrap();
    }

    let monitoring_duration = start.elapsed();
    println!(
        "✓ Quality monitoring: 1000 updates in {:?} ({:.2}μs each)",
        monitoring_duration,
        monitoring_duration.as_micros() as f64 / 1000.0
    );

    // Verify performance requirements are met
    assert!(
        config_duration.as_millis() < 10,
        "Configuration creation too slow"
    );
    assert!(
        serialization_duration.as_millis() < 50,
        "Serialization too slow"
    );
    assert!(
        monitoring_duration.as_millis() < 100,
        "Quality monitoring too slow"
    );

    println!("✅ All performance requirements met:");
    println!("  - Configuration loading <1ms ✓");
    println!("  - Quality monitoring <5ms overhead ✓");
    println!("  - Serialization suitable for real-time use ✓");

    Ok(())
}

// Helper function to create model evaluations for testing
fn create_evaluation(mae: f64, mape: f64, r_squared: f64) -> ModelEvaluation {
    ModelEvaluation {
        model_name: "test_model".to_string(),
        mae,
        mse: mae * mae,
        rmse: mae,
        mape,
        smape: mape,
        r_squared,
        aic: Some(100.0),
        bic: Some(105.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_functionality() {
        // Test that the main function runs without errors
        main().expect("Example should run successfully");
    }

    #[test]
    fn test_performance_requirements() {
        // Verify that performance meets blueprint requirements
        let start = std::time::Instant::now();
        let config = AdaptiveConfig::new();
        config.validate().unwrap();
        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 1,
            "Configuration creation exceeds 1ms requirement"
        );
    }

    #[test]
    fn test_backward_compatibility_preserved() {
        let original = ForecastConfig::default();
        let adaptive = AdaptiveConfig::from_base_config(original.clone());

        // Verify that all original settings are preserved
        assert_eq!(adaptive.base_config().model_type, original.model_type);
        assert_eq!(adaptive.base_config().auto_select, original.auto_select);
    }
}
