//! Comprehensive tests for STEP 1: Enhanced Configuration System
//!
//! This test suite validates the adaptive configuration system with
//! comprehensive coverage as required by the blueprint.

use oxidiviner::adaptive::monitoring::QualityMonitor;
use oxidiviner::adaptive::{AdaptiveConfig, AdaptiveParameters, ModelSelectionStrategy};
use oxidiviner::api::{ForecastConfig, ModelType};
use oxidiviner::core::ModelEvaluation;
use serde_json;
use std::collections::HashMap;
use std::time::SystemTime;

/// Test helper to create a basic model evaluation
fn create_test_evaluation(mae: f64, mape: f64, r_squared: f64) -> ModelEvaluation {
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
mod adaptive_config_tests {
    use super::*;

    #[test]
    fn test_adaptive_config_default_creation() {
        let config = AdaptiveConfig::default();

        // Verify default values
        assert!(config.adaptive_params.enable_adaptation);
        assert_eq!(config.adaptive_params.learning_rate, 0.1);
        assert_eq!(config.adaptive_params.adaptation_window, 50);
        assert_eq!(config.adaptive_params.confidence_threshold, 0.7);
        assert_eq!(config.adaptive_params.max_adaptation_frequency, 4);
        assert!(config.adaptive_params.regime_aware);
        assert!(config.adaptive_params.quality_monitoring);

        // Regime config defaults
        assert!(config.regime_config.enabled);
        assert_eq!(config.regime_config.num_regimes, 2);
        assert_eq!(config.regime_config.sensitivity, 0.5);
        assert_eq!(config.regime_config.min_regime_duration, 5);
        assert_eq!(config.regime_config.switching_penalty, 0.1);

        // Quality thresholds defaults
        assert_eq!(config.quality_thresholds.max_mae, 0.15);
        assert_eq!(config.quality_thresholds.max_mape, 15.0);
        assert_eq!(config.quality_thresholds.min_r_squared, 0.6);
        assert_eq!(config.quality_thresholds.quality_window, 20);
        assert!(config.quality_thresholds.enable_fallback);

        // Should validate successfully
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_adaptive_config_new() {
        let config = AdaptiveConfig::new();
        assert!(config.validate().is_ok());
        assert!(config.is_adaptation_enabled());
        assert!(config.is_regime_detection_enabled());
        assert!(config.is_quality_monitoring_enabled());
    }

    #[test]
    fn test_adaptive_config_from_base_config() {
        let mut base_config = ForecastConfig::default();
        base_config.model_type = ModelType::ARIMA;
        base_config.auto_select = false;

        let adaptive_config = AdaptiveConfig::from_base_config(base_config.clone());

        assert_eq!(adaptive_config.base_config.model_type, ModelType::ARIMA);
        assert_eq!(adaptive_config.base_config.auto_select, false);
        assert!(adaptive_config.validate().is_ok());
        assert_eq!(adaptive_config.base_config(), &base_config);
    }

    #[test]
    fn test_adaptive_config_builder_methods() {
        let config = AdaptiveConfig::new()
            .with_regime_detection(3)
            .with_learning_rate(0.05)
            .with_adaptation_window(100);

        assert_eq!(config.regime_config.num_regimes, 3);
        assert_eq!(config.adaptive_params.learning_rate, 0.05);
        assert_eq!(config.adaptive_params.adaptation_window, 100);
        assert!(config.regime_config.enabled);
        assert!(config.adaptive_params.regime_aware);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_adaptive_config_with_quality_thresholds() {
        let custom_thresholds = oxidiviner::adaptive::config::QualityThresholds {
            max_mae: 0.1,
            max_mape: 10.0,
            min_r_squared: 0.8,
            quality_window: 30,
            enable_fallback: false,
            max_consecutive_failures: 5,
        };

        let config = AdaptiveConfig::new().with_quality_thresholds(custom_thresholds.clone());

        assert_eq!(config.quality_thresholds.max_mae, 0.1);
        assert_eq!(config.quality_thresholds.max_mape, 10.0);
        assert_eq!(config.quality_thresholds.min_r_squared, 0.8);
        assert_eq!(config.quality_thresholds.quality_window, 30);
        assert!(!config.quality_thresholds.enable_fallback);
        assert_eq!(config.quality_thresholds.max_consecutive_failures, 5);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_adaptive_config_with_model_selection_strategies() {
        // Test Fixed strategy
        let config = AdaptiveConfig::new().with_model_selection(ModelSelectionStrategy::Fixed);
        assert!(matches!(
            config.model_selection,
            ModelSelectionStrategy::Fixed
        ));

        // Test Performance strategy
        let config =
            AdaptiveConfig::new().with_model_selection(ModelSelectionStrategy::Performance {
                window_size: 50,
                switch_threshold: 0.1,
            });
        if let ModelSelectionStrategy::Performance {
            window_size,
            switch_threshold,
        } = config.model_selection
        {
            assert_eq!(window_size, 50);
            assert_eq!(switch_threshold, 0.1);
        } else {
            panic!("Expected Performance strategy");
        }

        // Test RegimeBased strategy
        let mut regime_models = HashMap::new();
        regime_models.insert(0, ModelType::ARIMA);
        regime_models.insert(1, ModelType::SimpleES);

        let config =
            AdaptiveConfig::new().with_model_selection(ModelSelectionStrategy::RegimeBased {
                regime_models: regime_models.clone(),
            });
        if let ModelSelectionStrategy::RegimeBased {
            regime_models: models,
        } = config.model_selection
        {
            assert_eq!(models.len(), 2);
            assert_eq!(models[&0], ModelType::ARIMA);
            assert_eq!(models[&1], ModelType::SimpleES);
        } else {
            panic!("Expected RegimeBased strategy");
        }

        // Test Ensemble strategy
        let models = vec![ModelType::ARIMA, ModelType::SimpleES, ModelType::HoltLinear];
        let config = AdaptiveConfig::new().with_model_selection(ModelSelectionStrategy::Ensemble {
            models: models.clone(),
            weighting: oxidiviner::adaptive::config::EnsembleWeighting::Performance,
        });
        if let ModelSelectionStrategy::Ensemble {
            models: ensemble_models,
            weighting,
        } = config.model_selection
        {
            assert_eq!(ensemble_models.len(), 3);
            assert!(matches!(
                weighting,
                oxidiviner::adaptive::config::EnsembleWeighting::Performance
            ));
        } else {
            panic!("Expected Ensemble strategy");
        }
    }
}

#[cfg(test)]
mod adaptive_config_validation_tests {
    use super::*;

    #[test]
    fn test_learning_rate_validation() {
        let mut config = AdaptiveConfig::default();

        // Valid learning rates
        config.adaptive_params.learning_rate = 0.01;
        assert!(config.validate().is_ok());

        config.adaptive_params.learning_rate = 0.5;
        assert!(config.validate().is_ok());

        config.adaptive_params.learning_rate = 1.0;
        assert!(config.validate().is_ok());

        // Invalid learning rates
        config.adaptive_params.learning_rate = 0.0;
        assert!(config.validate().is_err());

        config.adaptive_params.learning_rate = -0.1;
        assert!(config.validate().is_err());

        config.adaptive_params.learning_rate = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_adaptation_window_validation() {
        let mut config = AdaptiveConfig::default();

        // Valid windows
        config.adaptive_params.adaptation_window = 10;
        assert!(config.validate().is_ok());

        config.adaptive_params.adaptation_window = 100;
        assert!(config.validate().is_ok());

        // Invalid windows
        config.adaptive_params.adaptation_window = 9;
        assert!(config.validate().is_err());

        config.adaptive_params.adaptation_window = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_confidence_threshold_validation() {
        let mut config = AdaptiveConfig::default();

        // Valid thresholds
        config.adaptive_params.confidence_threshold = 0.0;
        assert!(config.validate().is_ok());

        config.adaptive_params.confidence_threshold = 0.5;
        assert!(config.validate().is_ok());

        config.adaptive_params.confidence_threshold = 1.0;
        assert!(config.validate().is_ok());

        // Invalid thresholds
        config.adaptive_params.confidence_threshold = -0.1;
        assert!(config.validate().is_err());

        config.adaptive_params.confidence_threshold = 1.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_regime_config_validation() {
        let mut config = AdaptiveConfig::default();

        // Valid regime configurations
        config.regime_config.num_regimes = 2;
        assert!(config.validate().is_ok());

        config.regime_config.num_regimes = 5;
        assert!(config.validate().is_ok());

        // Invalid num_regimes
        config.regime_config.num_regimes = 1;
        assert!(config.validate().is_err());

        config.regime_config.num_regimes = 6;
        assert!(config.validate().is_err());

        // Reset to valid
        config.regime_config.num_regimes = 3;

        // Valid sensitivity
        config.regime_config.sensitivity = 0.1;
        assert!(config.validate().is_ok());

        config.regime_config.sensitivity = 0.9;
        assert!(config.validate().is_ok());

        // Invalid sensitivity
        config.regime_config.sensitivity = 0.0;
        assert!(config.validate().is_err());

        config.regime_config.sensitivity = 1.0;
        assert!(config.validate().is_err());

        // Reset to valid
        config.regime_config.sensitivity = 0.5;

        // Valid min_regime_duration
        config.regime_config.min_regime_duration = 2;
        assert!(config.validate().is_ok());

        config.regime_config.min_regime_duration = 10;
        assert!(config.validate().is_ok());

        // Invalid min_regime_duration
        config.regime_config.min_regime_duration = 1;
        assert!(config.validate().is_err());

        config.regime_config.min_regime_duration = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_quality_thresholds_validation() {
        let mut config = AdaptiveConfig::default();

        // Valid max_mae
        config.quality_thresholds.max_mae = 0.001;
        assert!(config.validate().is_ok());

        config.quality_thresholds.max_mae = 1.0;
        assert!(config.validate().is_ok());

        // Invalid max_mae
        config.quality_thresholds.max_mae = 0.0;
        assert!(config.validate().is_err());

        config.quality_thresholds.max_mae = -0.1;
        assert!(config.validate().is_err());

        // Reset to valid
        config.quality_thresholds.max_mae = 0.1;

        // Valid max_mape
        config.quality_thresholds.max_mape = 0.1;
        assert!(config.validate().is_ok());

        config.quality_thresholds.max_mape = 100.0;
        assert!(config.validate().is_ok());

        // Invalid max_mape
        config.quality_thresholds.max_mape = 0.0;
        assert!(config.validate().is_err());

        config.quality_thresholds.max_mape = -5.0;
        assert!(config.validate().is_err());

        config.quality_thresholds.max_mape = 150.0;
        assert!(config.validate().is_err());

        // Reset to valid
        config.quality_thresholds.max_mape = 15.0;

        // Valid min_r_squared
        config.quality_thresholds.min_r_squared = 0.0;
        assert!(config.validate().is_ok());

        config.quality_thresholds.min_r_squared = 1.0;
        assert!(config.validate().is_ok());

        // Invalid min_r_squared
        config.quality_thresholds.min_r_squared = -0.1;
        assert!(config.validate().is_err());

        config.quality_thresholds.min_r_squared = 1.1;
        assert!(config.validate().is_err());

        // Reset to valid
        config.quality_thresholds.min_r_squared = 0.7;

        // Valid quality_window
        config.quality_thresholds.quality_window = 5;
        assert!(config.validate().is_ok());

        config.quality_thresholds.quality_window = 100;
        assert!(config.validate().is_ok());

        // Invalid quality_window
        config.quality_thresholds.quality_window = 4;
        assert!(config.validate().is_err());

        config.quality_thresholds.quality_window = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_disabled_regime_detection_validation() {
        let mut config = AdaptiveConfig::default();
        config.regime_config.enabled = false;

        // Should pass validation even with invalid regime params when disabled
        config.regime_config.num_regimes = 1;
        config.regime_config.sensitivity = 0.0;
        config.regime_config.min_regime_duration = 1;

        assert!(config.validate().is_ok());
        assert!(!config.is_regime_detection_enabled());
    }
}

#[cfg(test)]
mod serialization_tests {
    use super::*;

    #[test]
    fn test_adaptive_config_serialization_roundtrip() {
        let original = AdaptiveConfig::default();

        // Serialize to JSON
        let json = serde_json::to_string(&original).expect("Serialization failed");
        assert!(!json.is_empty());

        // Deserialize back
        let deserialized: AdaptiveConfig =
            serde_json::from_str(&json).expect("Deserialization failed");

        // Verify key fields match
        assert_eq!(
            original.adaptive_params.learning_rate,
            deserialized.adaptive_params.learning_rate
        );
        assert_eq!(
            original.adaptive_params.adaptation_window,
            deserialized.adaptive_params.adaptation_window
        );
        assert_eq!(
            original.regime_config.num_regimes,
            deserialized.regime_config.num_regimes
        );
        assert_eq!(
            original.quality_thresholds.max_mae,
            deserialized.quality_thresholds.max_mae
        );

        // Both should validate successfully
        assert!(original.validate().is_ok());
        assert!(deserialized.validate().is_ok());
    }

    #[test]
    fn test_adaptive_parameters_serialization() {
        let params = AdaptiveParameters {
            enable_adaptation: false,
            learning_rate: 0.25,
            adaptation_window: 75,
            confidence_threshold: 0.85,
            max_adaptation_frequency: 8,
            regime_aware: false,
            quality_monitoring: false,
        };

        let json = serde_json::to_string(&params).unwrap();
        let deserialized: AdaptiveParameters = serde_json::from_str(&json).unwrap();

        assert_eq!(params.enable_adaptation, deserialized.enable_adaptation);
        assert_eq!(params.learning_rate, deserialized.learning_rate);
        assert_eq!(params.adaptation_window, deserialized.adaptation_window);
        assert_eq!(
            params.confidence_threshold,
            deserialized.confidence_threshold
        );
        assert_eq!(
            params.max_adaptation_frequency,
            deserialized.max_adaptation_frequency
        );
        assert_eq!(params.regime_aware, deserialized.regime_aware);
        assert_eq!(params.quality_monitoring, deserialized.quality_monitoring);
    }

    #[test]
    fn test_model_selection_strategy_serialization() {
        // Test Fixed strategy
        let strategy = ModelSelectionStrategy::Fixed;
        let json = serde_json::to_string(&strategy).unwrap();
        let deserialized: ModelSelectionStrategy = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, ModelSelectionStrategy::Fixed));

        // Test Performance strategy
        let strategy = ModelSelectionStrategy::Performance {
            window_size: 25,
            switch_threshold: 0.03,
        };
        let json = serde_json::to_string(&strategy).unwrap();
        let deserialized: ModelSelectionStrategy = serde_json::from_str(&json).unwrap();
        if let ModelSelectionStrategy::Performance {
            window_size,
            switch_threshold,
        } = deserialized
        {
            assert_eq!(window_size, 25);
            assert_eq!(switch_threshold, 0.03);
        } else {
            panic!("Expected Performance strategy");
        }
    }

    #[test]
    fn test_complex_config_serialization() {
        let mut regime_models = HashMap::new();
        regime_models.insert(0, ModelType::ARIMA);
        regime_models.insert(1, ModelType::SimpleES);

        let config = AdaptiveConfig::new()
            .with_regime_detection(3)
            .with_learning_rate(0.15)
            .with_adaptation_window(80)
            .with_model_selection(ModelSelectionStrategy::RegimeBased { regime_models });

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AdaptiveConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(
            config.regime_config.num_regimes,
            deserialized.regime_config.num_regimes
        );
        assert_eq!(
            config.adaptive_params.learning_rate,
            deserialized.adaptive_params.learning_rate
        );
        assert_eq!(
            config.adaptive_params.adaptation_window,
            deserialized.adaptive_params.adaptation_window
        );

        // Both should validate
        assert!(config.validate().is_ok());
        assert!(deserialized.validate().is_ok());
    }
}

#[cfg(test)]
mod quality_monitor_tests {
    use super::*;

    #[test]
    fn test_quality_monitor_creation() {
        let monitor = QualityMonitor::new();
        assert!(!monitor.is_quality_acceptable());
        assert!(!monitor.is_fallback_triggered());
        assert_eq!(monitor.success_rate(), 0.0);
        assert_eq!(monitor.average_quality_score(), 0.0);
        assert_eq!(monitor.quality_trend(), 0.0);
    }

    #[test]
    fn test_quality_monitor_with_custom_thresholds() {
        let thresholds = oxidiviner::adaptive::monitoring::QualityThresholds {
            max_mae: 0.05,
            max_mape: 5.0,
            min_r_squared: 0.9,
            quality_window: 15,
            enable_fallback: false,
            max_consecutive_failures: 1,
        };

        let monitor = QualityMonitor::with_thresholds(thresholds.clone());

        // Verify thresholds are set
        let report = monitor.quality_report();
        assert_eq!(report.thresholds.max_mae, 0.05);
        assert_eq!(report.thresholds.max_mape, 5.0);
        assert_eq!(report.thresholds.min_r_squared, 0.9);
        assert_eq!(report.thresholds.quality_window, 15);
        assert!(!report.thresholds.enable_fallback);
    }

    #[test]
    fn test_quality_update_acceptable() {
        let mut monitor = QualityMonitor::new();
        let evaluation = create_test_evaluation(0.1, 10.0, 0.8);

        monitor.update_quality(&evaluation).unwrap();

        assert!(monitor.is_quality_acceptable());
        assert!(!monitor.is_fallback_triggered());
        assert_eq!(monitor.success_rate(), 1.0);

        let metrics = monitor.current_quality().unwrap();
        assert_eq!(metrics.current_mae, 0.1);
        assert_eq!(metrics.current_mape, 10.0);
        assert_eq!(metrics.current_r_squared, 0.8);
        assert!(metrics.quality_acceptable);
        assert_eq!(metrics.consecutive_failures, 0);
        assert!(metrics.quality_score > 0.5);
    }

    #[test]
    fn test_quality_update_unacceptable() {
        let mut monitor = QualityMonitor::new();
        let evaluation = create_test_evaluation(0.3, 25.0, 0.3);

        monitor.update_quality(&evaluation).unwrap();

        assert!(!monitor.is_quality_acceptable());
        assert!(!monitor.is_fallback_triggered()); // Only one failure
        assert_eq!(monitor.success_rate(), 0.0);

        let metrics = monitor.current_quality().unwrap();
        assert!(!metrics.quality_acceptable);
        assert_eq!(metrics.consecutive_failures, 1);
        assert!(metrics.quality_score < 0.5);
    }

    #[test]
    fn test_fallback_trigger_mechanism() {
        let mut monitor = QualityMonitor::new();
        let bad_evaluation = create_test_evaluation(0.3, 25.0, 0.3);

        // Add consecutive failures
        for i in 0..3 {
            monitor.update_quality(&bad_evaluation).unwrap();
            if i < 2 {
                assert!(!monitor.is_fallback_triggered());
            } else {
                assert!(monitor.is_fallback_triggered());
            }
        }

        // Reset fallback
        monitor.reset_fallback();
        assert!(!monitor.is_fallback_triggered());

        // Add a good evaluation - should reset consecutive failures
        let good_evaluation = create_test_evaluation(0.1, 10.0, 0.8);
        monitor.update_quality(&good_evaluation).unwrap();
        let metrics = monitor.current_quality().unwrap();
        assert_eq!(metrics.consecutive_failures, 0);
    }

    #[test]
    fn test_rolling_averages_calculation() {
        let mut monitor = QualityMonitor::new();

        let evaluations = vec![
            create_test_evaluation(0.1, 10.0, 0.8),
            create_test_evaluation(0.12, 12.0, 0.75),
            create_test_evaluation(0.08, 8.0, 0.85),
        ];

        for eval in evaluations {
            monitor.update_quality(&eval).unwrap();
        }

        let metrics = monitor.current_quality().unwrap();

        // Check rolling averages
        let expected_mae = (0.1 + 0.12 + 0.08) / 3.0;
        let expected_mape = (10.0 + 12.0 + 8.0) / 3.0;
        let expected_rsq = (0.8 + 0.75 + 0.85) / 3.0;

        assert!((metrics.rolling_mae - expected_mae).abs() < 0.001);
        assert!((metrics.rolling_mape - expected_mape).abs() < 0.001);
        assert!((metrics.rolling_r_squared - expected_rsq).abs() < 0.001);
    }

    #[test]
    fn test_quality_score_calculation() {
        let mut monitor = QualityMonitor::new();

        // Perfect quality
        let perfect_eval = create_test_evaluation(0.0, 0.0, 1.0);
        monitor.update_quality(&perfect_eval).unwrap();
        let metrics = monitor.current_quality().unwrap();
        assert!(metrics.quality_score > 0.9);

        // Reset monitor for poor quality test
        let mut monitor2 = QualityMonitor::new();
        let poor_eval = create_test_evaluation(0.2, 20.0, 0.2);
        monitor2.update_quality(&poor_eval).unwrap();
        let metrics2 = monitor2.current_quality().unwrap();
        assert!(metrics2.quality_score < 0.5);
    }

    #[test]
    fn test_quality_trend_calculation() {
        let mut monitor = QualityMonitor::new();

        // Add improving quality trend
        let evaluations = vec![
            create_test_evaluation(0.2, 20.0, 0.5),  // Poor
            create_test_evaluation(0.15, 15.0, 0.6), // Better
            create_test_evaluation(0.1, 10.0, 0.7),  // Better
            create_test_evaluation(0.08, 8.0, 0.8),  // Better
            create_test_evaluation(0.05, 5.0, 0.85), // Best
        ];

        for eval in evaluations {
            monitor.update_quality(&eval).unwrap();
        }

        let trend = monitor.quality_trend();
        let avg_score = monitor.average_quality_score();
        assert!(
            trend > 0.0,
            "Quality trend should be positive (improving), got: {}",
            trend
        );
        assert!(
            avg_score > 0.4,
            "Average quality score should be reasonable, got: {}",
            avg_score
        );
    }

    #[test]
    fn test_quality_report_generation() {
        let mut monitor = QualityMonitor::new();

        // Add mixed quality evaluations
        let evaluations = vec![
            create_test_evaluation(0.1, 10.0, 0.8),   // Good
            create_test_evaluation(0.12, 12.0, 0.75), // Good
            create_test_evaluation(0.3, 25.0, 0.3),   // Bad
            create_test_evaluation(0.09, 9.0, 0.82),  // Good
        ];

        for eval in evaluations {
            monitor.update_quality(&eval).unwrap();
        }

        let report = monitor.quality_report();

        assert_eq!(report.total_evaluations, 4);
        assert_eq!(report.successful_evaluations, 3);
        assert!((report.success_rate - 0.75).abs() < 0.01);
        assert!(report.current_metrics.is_some());
        assert!(!report.fallback_triggered);
        assert!(report.average_quality_score > 0.0);
    }

    #[test]
    fn test_quality_monitor_update_thresholds() {
        let mut monitor = QualityMonitor::new();

        let new_thresholds = oxidiviner::adaptive::monitoring::QualityThresholds {
            max_mae: 0.05,
            max_mape: 5.0,
            min_r_squared: 0.95,
            quality_window: 10,
            enable_fallback: false,
            max_consecutive_failures: 1,
        };

        // Add some history first
        for _ in 0..5 {
            let eval = create_test_evaluation(0.1, 10.0, 0.8);
            monitor.update_quality(&eval).unwrap();
        }

        // Update thresholds
        monitor.update_thresholds(new_thresholds.clone());

        let report = monitor.quality_report();
        assert_eq!(report.thresholds.max_mae, 0.05);
        assert_eq!(report.thresholds.quality_window, 10);
        assert!(!report.thresholds.enable_fallback);
    }

    #[test]
    fn test_quality_metrics_timestamp() {
        let mut monitor = QualityMonitor::new();
        let eval = create_test_evaluation(0.1, 10.0, 0.8);

        let before = SystemTime::now();
        monitor.update_quality(&eval).unwrap();
        let after = SystemTime::now();

        let metrics = monitor.current_quality().unwrap();
        assert!(metrics.last_updated >= before);
        assert!(metrics.last_updated <= after);
    }

    #[test]
    fn test_quality_monitor_window_size_management() {
        let thresholds = oxidiviner::adaptive::monitoring::QualityThresholds {
            quality_window: 3,
            ..Default::default()
        };
        let mut monitor = QualityMonitor::with_thresholds(thresholds);

        // Add more evaluations than window size
        let evaluations = vec![
            create_test_evaluation(0.1, 10.0, 0.8),
            create_test_evaluation(0.11, 11.0, 0.79),
            create_test_evaluation(0.12, 12.0, 0.78),
            create_test_evaluation(0.13, 13.0, 0.77),
            create_test_evaluation(0.14, 14.0, 0.76),
        ];

        for eval in evaluations {
            monitor.update_quality(&eval).unwrap();
        }

        // Window should only contain last 3 evaluations
        let report = monitor.quality_report();
        assert_eq!(report.total_evaluations, 5);

        // Rolling average should be based on window size, not total evaluations
        let metrics = monitor.current_quality().unwrap();
        let expected_mae = (0.12 + 0.13 + 0.14) / 3.0;
        assert!((metrics.rolling_mae - expected_mae).abs() < 0.001);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_adaptive_system_integration() {
        // Create a complex adaptive configuration
        let mut regime_models = HashMap::new();
        regime_models.insert(0, ModelType::ARIMA);
        regime_models.insert(1, ModelType::SimpleES);

        let custom_thresholds = oxidiviner::adaptive::config::QualityThresholds {
            max_mae: 0.12,
            max_mape: 12.0,
            min_r_squared: 0.7,
            quality_window: 15,
            enable_fallback: true,
            max_consecutive_failures: 2,
        };

        let config = AdaptiveConfig::new()
            .with_regime_detection(2)
            .with_learning_rate(0.08)
            .with_adaptation_window(75)
            .with_quality_thresholds(custom_thresholds.clone())
            .with_model_selection(ModelSelectionStrategy::RegimeBased { regime_models });

        // Validate the entire configuration
        assert!(config.validate().is_ok());

        // Test serialization of the complex config
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AdaptiveConfig = serde_json::from_str(&json).unwrap();
        assert!(deserialized.validate().is_ok());

        // Create a quality monitor with compatible thresholds
        let monitor_thresholds = oxidiviner::adaptive::monitoring::QualityThresholds {
            max_mae: custom_thresholds.max_mae,
            max_mape: custom_thresholds.max_mape,
            min_r_squared: custom_thresholds.min_r_squared,
            quality_window: custom_thresholds.quality_window,
            enable_fallback: custom_thresholds.enable_fallback,
            max_consecutive_failures: custom_thresholds.max_consecutive_failures,
        };
        let mut monitor = QualityMonitor::with_thresholds(monitor_thresholds);

        // Simulate a quality monitoring session
        let evaluations = vec![
            create_test_evaluation(0.1, 10.0, 0.8),   // Good
            create_test_evaluation(0.11, 11.0, 0.78), // Good
            create_test_evaluation(0.15, 15.0, 0.6),  // Bad (exceeds thresholds)
            create_test_evaluation(0.16, 16.0, 0.55), // Bad (consecutive failure)
            create_test_evaluation(0.09, 9.0, 0.82),  // Good (recovery)
        ];

        for (i, eval) in evaluations.iter().enumerate() {
            monitor.update_quality(eval).unwrap();

            if i == 3 {
                // After 2 consecutive failures
                assert!(monitor.is_fallback_triggered());
            }
        }

        // Final state should show recovery
        assert!(monitor.is_quality_acceptable());
        let report = monitor.quality_report();
        assert_eq!(report.total_evaluations, 5);
        assert_eq!(report.successful_evaluations, 3);
        assert!((report.success_rate - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_backward_compatibility() {
        // Ensure that AdaptiveConfig works with existing ForecastConfig
        let original_config = ForecastConfig {
            model_type: ModelType::HoltWinters,
            parameters: oxidiviner::api::ModelParameters {
                arima_p: Some(2),
                arima_d: Some(1),
                arima_q: Some(2),
                alpha: Some(0.3),
                beta: Some(0.2),
                gamma: Some(0.1),
                ma_window: Some(10),
                seasonal_period: Some(12),
            },
            auto_select: false,
        };

        let adaptive_config = AdaptiveConfig::from_base_config(original_config.clone());

        // Verify that the base config is preserved exactly
        assert_eq!(
            adaptive_config.base_config.model_type,
            ModelType::HoltWinters
        );
        assert_eq!(adaptive_config.base_config.auto_select, false);
        assert_eq!(adaptive_config.base_config.parameters.arima_p, Some(2));
        assert_eq!(adaptive_config.base_config.parameters.alpha, Some(0.3));
        assert_eq!(
            adaptive_config.base_config.parameters.seasonal_period,
            Some(12)
        );

        // Verify adaptive features are available
        assert!(adaptive_config.is_adaptation_enabled());
        assert!(adaptive_config.is_regime_detection_enabled());
        assert!(adaptive_config.is_quality_monitoring_enabled());

        // Should validate successfully
        assert!(adaptive_config.validate().is_ok());
    }

    #[test]
    fn test_configuration_performance_requirements() {
        // Test that configuration loading is fast (<1ms requirement)
        let start = std::time::Instant::now();

        // Create multiple configs to simulate realistic usage
        for _ in 0..100 {
            let config = AdaptiveConfig::new()
                .with_regime_detection(3)
                .with_learning_rate(0.1)
                .with_adaptation_window(50);

            config.validate().unwrap();
        }

        let duration = start.elapsed();

        // Should complete well under 1ms per configuration
        assert!(
            duration.as_millis() < 10,
            "Configuration creation too slow: {:?}",
            duration
        );
    }

    #[test]
    fn test_serialization_performance() {
        // Test serialization roundtrip performance
        let config = AdaptiveConfig::new()
            .with_regime_detection(4)
            .with_learning_rate(0.15)
            .with_adaptation_window(100);

        let start = std::time::Instant::now();

        // Perform multiple serialization roundtrips
        for _ in 0..50 {
            let json = serde_json::to_string(&config).unwrap();
            let _deserialized: AdaptiveConfig = serde_json::from_str(&json).unwrap();
        }

        let duration = start.elapsed();

        // Should be fast enough for real-time usage
        assert!(
            duration.as_millis() < 100,
            "Serialization too slow: {:?}",
            duration
        );
    }
}
