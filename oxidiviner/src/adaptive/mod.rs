//! Adaptive Forecasting System
//!
//! This module provides adaptive forecasting capabilities that can automatically
//! adjust to changing market conditions and optimize forecast performance in real-time.
//!
//! ## Core Components
//!
//! - [`AdaptiveConfig`] - Enhanced configuration for adaptive forecasting
//! - [`AdaptiveForecaster`] - Main adaptive forecasting interface
//! - [`RegimeDetector`] - Market regime detection capabilities
//! - [`QualityMonitor`] - Real-time quality monitoring and fallback management
//!
//! ## Features
//!
//! - **Automatic Model Selection**: Selects the best model based on current conditions
//! - **Regime-Aware Forecasting**: Adapts to different market regimes
//! - **Quality Monitoring**: Continuous monitoring with automatic fallbacks
//! - **Parameter Optimization**: Real-time parameter tuning
//! - **Performance Tracking**: Comprehensive metrics and diagnostics

pub mod config;
// pub mod forecaster; // Temporarily disabled due to compilation issues
pub mod monitoring;
pub mod quality_system;
pub mod regime_detection;

pub use config::{AdaptiveConfig, AdaptiveParameters, ModelSelectionStrategy, RegimeConfig};
// pub use forecaster::AdaptiveForecaster; // Temporarily disabled
pub use monitoring::{QualityMetrics, QualityMonitor, QualityThresholds};
pub use quality_system::{
    PerformanceMetrics, QualityEvaluationResult, QualitySystemConfig, RealTimeQualitySystem,
    TrendDirection,
};
pub use regime_detection::{
    MarketRegime, RegimeDetectionMetrics, RegimeDetectionResult, RegimeDetector,
    RegimeDetectorBuilder,
};

// Integration with existing systems
use crate::api::ForecastConfig;
use crate::core::Result;

/// Builder for creating adaptive forecasting systems that integrate
/// with existing OxiDiviner components
pub struct AdaptiveBuilder {
    adaptive_config: AdaptiveConfig,
}

impl AdaptiveBuilder {
    /// Create a new adaptive builder with default configuration
    pub fn new() -> Self {
        Self {
            adaptive_config: AdaptiveConfig::default(),
        }
    }

    /// Set the base forecasting configuration
    pub fn with_base_config(mut self, config: ForecastConfig) -> Self {
        self.adaptive_config = AdaptiveConfig::from_base_config(config);
        self
    }

    /// Enable regime-aware adaptation
    pub fn enable_regime_detection(mut self, num_regimes: usize) -> Self {
        self.adaptive_config = self.adaptive_config.with_regime_detection(num_regimes);
        self
    }

    /// Set learning rate for real-time adaptation
    pub fn with_learning_rate(mut self, rate: f64) -> Self {
        self.adaptive_config = self.adaptive_config.with_learning_rate(rate);
        self
    }

    /// Set adaptation window size
    pub fn with_adaptation_window(mut self, window: usize) -> Self {
        self.adaptive_config = self.adaptive_config.with_adaptation_window(window);
        self
    }

    /// Build the adaptive forecaster
    pub fn build(self) -> Result<()> {
        self.adaptive_config.validate()?;
        // Ok(AdaptiveForecaster::new()) // Temporarily disabled
        Ok(())
    }
}

impl Default for AdaptiveBuilder {
    fn default() -> Self {
        Self::new()
    }
}
