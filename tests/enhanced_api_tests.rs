/*!
# Enhanced API Integration Tests

This module contains comprehensive integration tests for the enhanced API
features including Quick API, Builder Pattern, Smart Selection, and
Validation Utilities.
*/

use chrono::{Duration, Utc};
use oxidiviner::{quick, ModelBuilder, AutoSelector, ModelValidator};
use oxidiviner_core::{
    validation::ValidationUtils, TimeSeriesData, SelectionCriteria, ModelConfig,
    QuickForecaster, ConfidenceForecaster, ForecastResult,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Create test data for integration testing
fn create_test_data(n: usize) -> TimeSeriesData {
    let now = Utc::now();
    let mut rng = StdRng::seed_from_u64(42);
    
    let timestamps = (0..n).map(|i| now + Duration::days(i as i64)).collect();
    let values: Vec<f64> = (0..n)
        .map(|i| i as f64 + rng.gen_range(-2.0..2.0))
        .collect();
    
    TimeSeriesData::new(timestamps, values, "test_series")
        .expect("Failed to create test data")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_api_arima() {
        let data = create_test_data(50);
        let (train, test) = ValidationUtils::time_split(&data, 0.8).unwrap();
        
        // Test basic ARIMA forecast
        let forecast = quick::arima(train.clone(), test.values.len()).unwrap();
        assert_eq!(forecast.len(), test.values.len());
        
        // Test ARIMA with custom config
        let forecast_custom = quick::arima_with_config(train, test.values.len(), Some((2, 1, 1))).unwrap();
        assert_eq!(forecast_custom.len(), test.values.len());
    }

    #[test]
    fn test_quick_api_ar() {
        let data = create_test_data(50);
        let (train, test) = ValidationUtils::time_split(&data, 0.8).unwrap();
        
        // Test AR with different orders
        for order in [1, 2, 3] {
            let forecast = quick::ar(train.clone(), test.values.len(), Some(order)).unwrap();
            assert_eq!(forecast.len(), test.values.len());
        }
    }

    #[test]
    fn test_quick_api_moving_average() {
        let data = create_test_data(50);
        let (train, test) = ValidationUtils::time_split(&data, 0.8).unwrap();
        
        // Test MA with different windows
        for window in [3, 5, 7] {
            let forecast = quick::moving_average(train.clone(), test.values.len(), Some(window)).unwrap();
            assert_eq!(forecast.len(), test.values.len());
        }
        
        // Test default window
        let forecast_default = quick::moving_average(train, test.values.len(), None).unwrap();
        assert_eq!(forecast_default.len(), test.values.len());
    }

    #[test]
    fn test_quick_api_exponential_smoothing() {
        let data = create_test_data(50);
        let (train, test) = ValidationUtils::time_split(&data, 0.8).unwrap();
        
        // Test ES with different alpha values
        for alpha in [0.1, 0.3, 0.5] {
            let forecast = quick::exponential_smoothing(train.clone(), test.values.len(), Some(alpha)).unwrap();
            assert_eq!(forecast.len(), test.values.len());
        }
    }

    #[test]
    fn test_builder_pattern() {
        // Test ARIMA builder
        let arima_config = ModelBuilder::arima()
            .with_ar(2)
            .with_differencing(1)
            .with_ma(1)
            .build_config();
        
        assert_eq!(arima_config.model_type, "ARIMA");
        assert_eq!(arima_config.parameters.get("p"), Some(&2.0));
        assert_eq!(arima_config.parameters.get("d"), Some(&1.0));
        assert_eq!(arima_config.parameters.get("q"), Some(&1.0));

        // Test AR builder
        let ar_config = ModelBuilder::ar()
            .with_ar(3)
            .build_config();
        
        assert_eq!(ar_config.model_type, "AR");
        assert_eq!(ar_config.parameters.get("p"), Some(&3.0));

        // Test MA builder
        let ma_config = ModelBuilder::moving_average()
            .with_window(7)
            .build_config();
        
        assert_eq!(ma_config.model_type, "MA");
        assert_eq!(ma_config.parameters.get("window"), Some(&7.0));

        // Test ES builder
        let es_config = ModelBuilder::exponential_smoothing()
            .with_alpha(0.3)
            .with_beta(0.2)
            .build_config();
        
        assert_eq!(es_config.model_type, "ES");
        assert_eq!(es_config.parameters.get("alpha"), Some(&0.3));
        assert_eq!(es_config.parameters.get("beta"), Some(&0.2));
    }

    #[test]
    fn test_builder_with_quick_api() {
        let data = create_test_data(50);
        let (train, test) = ValidationUtils::time_split(&data, 0.8).unwrap();
        
        let config = ModelBuilder::arima()
            .with_ar(1)
            .with_differencing(1)
            .with_ma(1)
            .build_config();
        
        let forecast = quick::forecast_with_config(train, test.values.len(), config).unwrap();
        assert_eq!(forecast.len(), test.values.len());
    }

    #[test]
    fn test_model_validator() {
        // Valid ARIMA parameters
        assert!(ModelValidator::validate_arima_params(2, 1, 1).is_ok());
        assert!(ModelValidator::validate_arima_params(1, 0, 1).is_ok());
        
        // Invalid ARIMA parameters
        assert!(ModelValidator::validate_arima_params(15, 3, 10).is_err());
        assert!(ModelValidator::validate_arima_params(0, 0, 0).is_err());

        // Valid AR parameters
        assert!(ModelValidator::validate_ar_params(1).is_ok());
        assert!(ModelValidator::validate_ar_params(5).is_ok());
        
        // Invalid AR parameters
        assert!(ModelValidator::validate_ar_params(0).is_err());
        assert!(ModelValidator::validate_ar_params(15).is_err());

        // Valid MA parameters
        assert!(ModelValidator::validate_ma_params(1).is_ok());
        assert!(ModelValidator::validate_ma_params(10).is_ok());
        
        // Invalid MA parameters
        assert!(ModelValidator::validate_ma_params(0).is_err());

        // Valid ES parameters
        assert!(ModelValidator::validate_exponential_smoothing_params(0.3, None, None).is_ok());
        assert!(ModelValidator::validate_exponential_smoothing_params(0.5, Some(0.2), Some(0.1)).is_ok());
        
        // Invalid ES parameters
        assert!(ModelValidator::validate_exponential_smoothing_params(-0.1, None, None).is_err());
        assert!(ModelValidator::validate_exponential_smoothing_params(1.5, None, None).is_err());
    }

    #[test]
    fn test_validation_utilities() {
        let data = create_test_data(100);
        
        // Test time split
        let (train, test) = ValidationUtils::time_split(&data, 0.8).unwrap();
        assert_eq!(train.values.len(), 80);
        assert_eq!(test.values.len(), 20);
        
        // Test edge cases
        assert!(ValidationUtils::time_split(&data, 0.0).is_err());
        assert!(ValidationUtils::time_split(&data, 1.0).is_err());
        
        // Test time series CV
        let splits = ValidationUtils::time_series_cv(&data, 3, Some(30)).unwrap();
        assert_eq!(splits.len(), 3);
        
        for (train, test) in splits {
            assert!(train.values.len() >= 30);
            assert!(!test.values.is_empty());
        }
    }

    #[test]
    fn test_accuracy_metrics() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![1.1, 2.1, 2.9, 4.1, 4.9];
        
        let metrics = ValidationUtils::accuracy_metrics(&actual, &predicted, None).unwrap();
        
        assert!(metrics.mae > 0.0);
        assert!(metrics.rmse > 0.0);
        assert!(metrics.mape > 0.0);
        assert!(metrics.smape > 0.0);
        assert!(metrics.r_squared > 0.0);
        assert_eq!(metrics.n_observations, 5);
        
        // Test with baseline for MASE
        let baseline = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let metrics_with_baseline = ValidationUtils::accuracy_metrics(&actual, &predicted, Some(&baseline)).unwrap();
        assert!(metrics_with_baseline.mase.is_some());
    }

    #[test]
    fn test_auto_selector() {
        // Test creating auto selectors with different criteria
        let aic_selector = AutoSelector::with_aic();
        matches!(aic_selector.criteria(), SelectionCriteria::AIC);
        
        let bic_selector = AutoSelector::with_bic();
        matches!(bic_selector.criteria(), SelectionCriteria::BIC);
        
        let cv_selector = AutoSelector::with_cross_validation(3);
        matches!(cv_selector.criteria(), SelectionCriteria::CrossValidation { folds: 3 });
        
        let holdout_selector = AutoSelector::with_hold_out(0.2);
        matches!(holdout_selector.criteria(), SelectionCriteria::HoldOut { test_ratio: 0.2 });
        
        // Test adding custom candidates
        let custom_config = ModelBuilder::ar().with_ar(5).build_config();
        let selector_with_custom = AutoSelector::with_aic().add_candidate(custom_config);
        
        assert!(selector_with_custom.candidates().len() > 0);
    }

    #[test]
    fn test_auto_select_integration() {
        let data = create_test_data(50);
        
        // Test auto select
        let (forecast, best_model) = quick::auto_select(data, 10).unwrap();
        assert_eq!(forecast.len(), 10);
        assert!(!best_model.is_empty());
        
        println!("Auto-selected model: {}", best_model);
        println!("Forecast length: {}", forecast.len());
    }

    #[test]
    fn test_parameter_validation_edge_cases() {
        let data = create_test_data(5); // Very small dataset
        
        // Should fail with insufficient data
        assert!(quick::arima(data.clone(), 2).is_err());
        
        // Test minimum data validation
        assert!(ModelValidator::validate_minimum_data(5, 10, "ARIMA").is_err());
        assert!(ModelValidator::validate_minimum_data(15, 10, "ARIMA").is_ok());
        
        // Test forecast horizon validation
        assert!(ModelValidator::validate_forecast_horizon(20, 5).is_err());
        assert!(ModelValidator::validate_forecast_horizon(5, 20).is_ok());
    }

    #[test]
    fn test_error_handling() {
        let empty_data = TimeSeriesData::new(vec![], vec![], "empty").unwrap();
        
        // Should handle empty data gracefully
        assert!(quick::arima(empty_data.clone(), 5).is_err());
        assert!(quick::moving_average(empty_data.clone(), 5, Some(3)).is_err());
        
        // Test invalid configurations
        let invalid_config = ModelConfig {
            model_type: "INVALID".to_string(),
            parameters: std::collections::HashMap::new(),
        };
        
        let data = create_test_data(20);
        assert!(quick::forecast_with_config(data, 5, invalid_config).is_err());
    }

    #[test]
    fn test_comprehensive_workflow() {
        // Create realistic test scenario
        let data = create_test_data(100);
        
        // 1. Split data for validation
        let (train, test) = ValidationUtils::time_split(&data, 0.8).unwrap();
        
        // 2. Use auto selection to find best model
        let (auto_forecast, best_model) = quick::auto_select(train.clone(), test.values.len()).unwrap();
        
        // 3. Validate the forecast
        let metrics = ValidationUtils::accuracy_metrics(&test.values, &auto_forecast, None).unwrap();
        
        // 4. Use builder pattern to create custom models
        let custom_config = ModelBuilder::arima()
            .with_ar(2)
            .with_differencing(1)
            .with_ma(1)
            .build_config();
        
        let custom_forecast = quick::forecast_with_config(train, test.values.len(), custom_config).unwrap();
        
        // 5. Compare results
        let custom_metrics = ValidationUtils::accuracy_metrics(&test.values, &custom_forecast, None).unwrap();
        
        println!("Auto-selected model: {}", best_model);
        println!("Auto model MAE: {:.3}", metrics.mae);
        println!("Custom model MAE: {:.3}", custom_metrics.mae);
        
        // Both should produce valid forecasts
        assert_eq!(auto_forecast.len(), test.values.len());
        assert_eq!(custom_forecast.len(), test.values.len());
    }
} 