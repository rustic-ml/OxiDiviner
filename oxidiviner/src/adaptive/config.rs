//! Enhanced Configuration System for Adaptive Forecasting
//!
//! This module extends the existing `ForecastConfig` with adaptive capabilities
//! while maintaining full backward compatibility.

use crate::api::{ForecastConfig, ModelType};
use crate::core::{OxiError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Enhanced configuration for adaptive forecasting that extends ForecastConfig
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdaptiveConfig {
    /// Base forecast configuration (maintains backward compatibility)
    pub base_config: ForecastConfig,

    /// Adaptive-specific parameters
    pub adaptive_params: AdaptiveParameters,

    /// Regime detection configuration
    pub regime_config: RegimeConfig,

    /// Quality monitoring thresholds
    pub quality_thresholds: QualityThresholds,

    /// Model selection strategy
    pub model_selection: ModelSelectionStrategy,
}

/// Adaptive forecasting parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveParameters {
    /// Enable real-time parameter adaptation
    pub enable_adaptation: bool,

    /// Adaptation learning rate (0.01 to 0.5)
    pub learning_rate: f64,

    /// Window size for adaptation (minimum 10 observations)
    pub adaptation_window: usize,

    /// Minimum confidence threshold for predictions
    pub confidence_threshold: f64,

    /// Maximum adaptation frequency (adaptations per day)
    pub max_adaptation_frequency: usize,

    /// Enable regime-aware forecasting
    pub regime_aware: bool,

    /// Enable quality monitoring
    pub quality_monitoring: bool,
}

/// Regime detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    /// Enable regime detection
    pub enabled: bool,

    /// Number of regimes to detect (2-5)
    pub num_regimes: usize,

    /// Regime detection sensitivity (0.1 to 0.9)
    pub sensitivity: f64,

    /// Minimum regime duration in observations
    pub min_regime_duration: usize,

    /// Regime switching penalty
    pub switching_penalty: f64,
}

/// Quality monitoring thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Maximum acceptable MAE for quality check
    pub max_mae: f64,

    /// Maximum acceptable MAPE percentage
    pub max_mape: f64,

    /// Minimum R-squared for acceptable forecasts
    pub min_r_squared: f64,

    /// Quality check window size
    pub quality_window: usize,

    /// Enable automatic fallback when quality drops
    pub enable_fallback: bool,

    /// Consecutive failures before triggering fallback
    pub max_consecutive_failures: usize,
}

/// Model selection strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSelectionStrategy {
    /// Use fixed model from base configuration
    Fixed,

    /// Automatic selection based on recent performance
    Performance {
        /// Evaluation window size
        window_size: usize,
        /// Minimum performance difference for switching
        switch_threshold: f64,
    },

    /// Regime-based model selection
    RegimeBased {
        /// Model mapping per regime
        regime_models: HashMap<usize, ModelType>,
    },

    /// Ensemble of multiple models
    Ensemble {
        /// Models to include in ensemble
        models: Vec<ModelType>,
        /// Weighting strategy
        weighting: EnsembleWeighting,
    },
}

/// Ensemble weighting strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleWeighting {
    /// Equal weights for all models
    Equal,

    /// Performance-based weights
    Performance,

    /// Regime-dependent weights
    RegimeDependent,
}



impl Default for AdaptiveParameters {
    fn default() -> Self {
        Self {
            enable_adaptation: true,
            learning_rate: 0.1,
            adaptation_window: 50,
            confidence_threshold: 0.7,
            max_adaptation_frequency: 4,
            regime_aware: true,
            quality_monitoring: true,
        }
    }
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            num_regimes: 2,
            sensitivity: 0.5,
            min_regime_duration: 5,
            switching_penalty: 0.1,
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            max_mae: 0.15,
            max_mape: 15.0,
            min_r_squared: 0.6,
            quality_window: 20,
            enable_fallback: true,
            max_consecutive_failures: 3,
        }
    }
}

impl Default for ModelSelectionStrategy {
    fn default() -> Self {
        Self::Performance {
            window_size: 30,
            switch_threshold: 0.05,
        }
    }
}

impl AdaptiveConfig {
    /// Create a new AdaptiveConfig with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create AdaptiveConfig from existing ForecastConfig
    pub fn from_base_config(base_config: ForecastConfig) -> Self {
        Self {
            base_config,
            ..Self::default()
        }
    }

    /// Enable regime-aware forecasting
    pub fn with_regime_detection(mut self, num_regimes: usize) -> Self {
        self.regime_config.enabled = true;
        self.regime_config.num_regimes = num_regimes;
        self.adaptive_params.regime_aware = true;
        self
    }

    /// Set learning rate for adaptation
    pub fn with_learning_rate(mut self, rate: f64) -> Self {
        self.adaptive_params.learning_rate = rate;
        self
    }

    /// Set adaptation window size
    pub fn with_adaptation_window(mut self, window: usize) -> Self {
        self.adaptive_params.adaptation_window = window;
        self
    }

    /// Set quality thresholds
    pub fn with_quality_thresholds(mut self, thresholds: QualityThresholds) -> Self {
        self.quality_thresholds = thresholds;
        self
    }

    /// Set model selection strategy
    pub fn with_model_selection(mut self, strategy: ModelSelectionStrategy) -> Self {
        self.model_selection = strategy;
        self
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        // Validate adaptive parameters
        if self.adaptive_params.learning_rate <= 0.0 || self.adaptive_params.learning_rate > 1.0 {
            return Err(OxiError::ConfigError(
                "Learning rate must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.adaptive_params.adaptation_window < 10 {
            return Err(OxiError::ConfigError(
                "Adaptation window must be at least 10 observations".to_string(),
            ));
        }

        if self.adaptive_params.confidence_threshold < 0.0
            || self.adaptive_params.confidence_threshold > 1.0
        {
            return Err(OxiError::ConfigError(
                "Confidence threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Validate regime configuration
        if self.regime_config.enabled {
            if self.regime_config.num_regimes < 2 || self.regime_config.num_regimes > 5 {
                return Err(OxiError::ConfigError(
                    "Number of regimes must be between 2 and 5".to_string(),
                ));
            }

            if self.regime_config.sensitivity <= 0.0 || self.regime_config.sensitivity >= 1.0 {
                return Err(OxiError::ConfigError(
                    "Regime sensitivity must be between 0.0 and 1.0".to_string(),
                ));
            }

            if self.regime_config.min_regime_duration < 2 {
                return Err(OxiError::ConfigError(
                    "Minimum regime duration must be at least 2 observations".to_string(),
                ));
            }
        }

        // Validate quality thresholds
        if self.quality_thresholds.max_mae <= 0.0 {
            return Err(OxiError::ConfigError(
                "Maximum MAE must be positive".to_string(),
            ));
        }

        if self.quality_thresholds.max_mape <= 0.0 || self.quality_thresholds.max_mape > 100.0 {
            return Err(OxiError::ConfigError(
                "Maximum MAPE must be between 0.0 and 100.0".to_string(),
            ));
        }

        if self.quality_thresholds.min_r_squared < 0.0
            || self.quality_thresholds.min_r_squared > 1.0
        {
            return Err(OxiError::ConfigError(
                "Minimum R-squared must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.quality_thresholds.quality_window < 5 {
            return Err(OxiError::ConfigError(
                "Quality window must be at least 5 observations".to_string(),
            ));
        }

        Ok(())
    }

    /// Get the base ForecastConfig for backward compatibility
    pub fn base_config(&self) -> &ForecastConfig {
        &self.base_config
    }

    /// Check if adaptation is enabled
    pub fn is_adaptation_enabled(&self) -> bool {
        self.adaptive_params.enable_adaptation
    }

    /// Check if regime detection is enabled
    pub fn is_regime_detection_enabled(&self) -> bool {
        self.regime_config.enabled && self.adaptive_params.regime_aware
    }

    /// Check if quality monitoring is enabled
    pub fn is_quality_monitoring_enabled(&self) -> bool {
        self.adaptive_params.quality_monitoring
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ModelType;

    #[test]
    fn test_adaptive_config_default() {
        let config = AdaptiveConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.is_adaptation_enabled());
        assert!(config.is_regime_detection_enabled());
        assert!(config.is_quality_monitoring_enabled());
    }

    #[test]
    fn test_adaptive_config_from_base() {
        let mut base_config = ForecastConfig::default();
        base_config.model_type = ModelType::ARIMA;

        let config = AdaptiveConfig::from_base_config(base_config);
        assert_eq!(config.base_config.model_type, ModelType::ARIMA);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_learning_rate() {
        let mut config = AdaptiveConfig::default();
        config.adaptive_params.learning_rate = 1.5;
        assert!(config.validate().is_err());

        config.adaptive_params.learning_rate = -0.1;
        assert!(config.validate().is_err());

        config.adaptive_params.learning_rate = 0.1;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_adaptation_window() {
        let mut config = AdaptiveConfig::default();
        config.adaptive_params.adaptation_window = 5;
        assert!(config.validate().is_err());

        config.adaptive_params.adaptation_window = 20;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_regime_params() {
        let mut config = AdaptiveConfig::default();
        config.regime_config.num_regimes = 1;
        assert!(config.validate().is_err());

        config.regime_config.num_regimes = 6;
        assert!(config.validate().is_err());

        config.regime_config.num_regimes = 3;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder_methods() {
        let config = AdaptiveConfig::new()
            .with_regime_detection(3)
            .with_learning_rate(0.2)
            .with_adaptation_window(100);

        assert_eq!(config.regime_config.num_regimes, 3);
        assert_eq!(config.adaptive_params.learning_rate, 0.2);
        assert_eq!(config.adaptive_params.adaptation_window, 100);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let original = AdaptiveConfig::default();
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: AdaptiveConfig = serde_json::from_str(&json).unwrap();

        // Compare key fields since we can't derive PartialEq easily
        assert_eq!(
            original.adaptive_params.learning_rate,
            deserialized.adaptive_params.learning_rate
        );
        assert_eq!(
            original.regime_config.num_regimes,
            deserialized.regime_config.num_regimes
        );
        assert_eq!(
            original.quality_thresholds.max_mae,
            deserialized.quality_thresholds.max_mae
        );
    }

    #[test]
    fn test_quality_thresholds_validation() {
        let mut config = AdaptiveConfig::default();

        config.quality_thresholds.max_mae = -1.0;
        assert!(config.validate().is_err());

        config.quality_thresholds.max_mae = 0.1;
        config.quality_thresholds.max_mape = -5.0;
        assert!(config.validate().is_err());

        config.quality_thresholds.max_mape = 150.0;
        assert!(config.validate().is_err());

        config.quality_thresholds.max_mape = 15.0;
        config.quality_thresholds.min_r_squared = -0.5;
        assert!(config.validate().is_err());

        config.quality_thresholds.min_r_squared = 1.5;
        assert!(config.validate().is_err());

        config.quality_thresholds.min_r_squared = 0.7;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_model_selection_strategies() {
        let config = AdaptiveConfig::new().with_model_selection(ModelSelectionStrategy::Fixed);
        assert!(matches!(
            config.model_selection,
            ModelSelectionStrategy::Fixed
        ));

        let mut regime_models = HashMap::new();
        regime_models.insert(0, ModelType::ARIMA);
        regime_models.insert(1, ModelType::SimpleES);

        let config = AdaptiveConfig::new()
            .with_model_selection(ModelSelectionStrategy::RegimeBased { regime_models });
        assert!(matches!(
            config.model_selection,
            ModelSelectionStrategy::RegimeBased { .. }
        ));
    }
}
