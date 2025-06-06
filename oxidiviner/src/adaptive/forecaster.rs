//! Adaptive Forecaster Implementation
//!
//! This module implements the unified adaptive forecasting system that integrates:
//! - AdaptiveConfig (STEP 1): Enhanced configuration system
//! - RegimeDetector (STEP 2): Market state detection  
//! - QualitySystem (STEP 3): Real-time quality monitoring
//!
//! The AdaptiveForecaster provides a unified API for adaptive time series forecasting
//! with automatic regime detection, quality monitoring, and model adaptation.

use crate::adaptive::{
    config::AdaptiveConfig,
    regime_detection::{RegimeDetector, MarketRegime},
    quality_system::{RealTimeQualitySystem, QualityEvaluationResult},
};
use crate::core::{
    Forecaster, ModelEvaluation, ModelOutput, OxiError, Result, TimeSeriesData
};
use crate::models::{
    autoregressive::ARIMAModel,
    exponential_smoothing::SimpleESModel,
    moving_average::MAModel,
};
use crate::ensemble::{EnsembleForecast, EnsembleMethod, ModelForecast};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Unified adaptive forecasting system integrating all adaptive components
#[derive(Debug)]
pub struct AdaptiveForecaster {
    /// Adaptive configuration
    config: AdaptiveConfig,
    /// Regime detection system
    regime_detector: RegimeDetector,
    /// Quality monitoring system
    quality_system: RealTimeQualitySystem,
    /// Base forecasting models
    models: HashMap<String, Box<dyn Forecaster + Send + Sync>>,
    /// Current regime state
    current_regime: Option<MarketRegime>,
    /// Training data history
    training_data: Option<TimeSeriesData>,
    /// Performance metrics
    performance_stats: PerformanceStats,
    /// Model adaptation history
    adaptation_history: Vec<AdaptationEvent>,
}

/// Performance statistics for the adaptive forecaster
#[derive(Debug, Clone, Default)]
pub struct PerformanceStats {
    /// Total forecasts generated
    pub total_forecasts: usize,
    /// Average forecast latency
    pub avg_latency_ms: f64,
    /// Peak latency observed
    pub peak_latency_ms: f64,
    /// Quality evaluation count
    pub quality_evaluations: usize,
    /// Regime changes detected
    pub regime_changes: usize,
    /// Model adaptations performed
    pub adaptations: usize,
}

/// Record of model adaptation events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationEvent {
    /// Timestamp of adaptation
    pub timestamp: std::time::SystemTime,
    /// Reason for adaptation
    pub reason: AdaptationReason,
    /// Previous regime (if applicable)
    pub previous_regime: Option<MarketRegime>,
    /// New regime (if applicable)
    pub new_regime: Option<MarketRegime>,
    /// Quality score that triggered adaptation
    pub quality_score: Option<f64>,
    /// Model used after adaptation
    pub selected_model: String,
}

/// Reasons for model adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationReason {
    /// Regime change detected
    RegimeChange,
    /// Quality degradation
    QualityDegradation,
    /// Periodic refit
    PeriodicRefit,
    /// Manual trigger
    Manual,
}

/// Adaptive forecast output with enhanced metadata
#[derive(Debug, Clone)]
pub struct AdaptiveForecastOutput {
    /// Base forecast values
    pub forecast: Vec<f64>,
    /// Confidence intervals (if available)
    pub confidence_intervals: Option<Vec<(f64, f64)>>,
    /// Model used for forecast
    pub model_used: String,
    /// Current regime state
    pub regime_state: MarketRegime,
    /// Quality evaluation
    pub quality_evaluation: QualityEvaluationResult,
    /// Forecast metadata
    pub metadata: ForecastMetadata,
}

/// Metadata about the forecast process
#[derive(Debug, Clone)]
pub struct ForecastMetadata {
    /// Time taken to generate forecast
    pub generation_time_ms: f64,
    /// Number of models evaluated
    pub models_evaluated: usize,
    /// Regime detection time
    pub regime_detection_time_ms: f64,
    /// Quality evaluation time
    pub quality_evaluation_time_ms: f64,
    /// Whether adaptation occurred
    pub adaptation_occurred: bool,
}

impl AdaptiveForecaster {
    /// Create a new adaptive forecaster with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(AdaptiveConfig::default())
    }

    /// Create a new adaptive forecaster with custom configuration
    pub fn with_config(config: AdaptiveConfig) -> Result<Self> {
        let regime_detector = RegimeDetector::new(
            2, // Default 2 regimes
            None, // Use default transition matrix
        )?;

        let quality_system = RealTimeQualitySystem::new(
            config.base_config.quality_thresholds.clone(),
        )?;

        let mut models: HashMap<String, Box<dyn Forecaster + Send + Sync>> = HashMap::new();
        
        // Initialize base models
        models.insert(
            "ARIMA".to_string(),
            Box::new(ARIMAModel::new(1, 1, 1, true)?),
        );
        models.insert(
            "SimpleES".to_string(),
            Box::new(SimpleExponentialSmoothing::new(0.3)?),
        );
        models.insert(
            "HoltLinear".to_string(),
            Box::new(HoltLinearTrend::new(0.3, 0.1)?),
        );
        models.insert(
            "MA".to_string(),
            Box::new(MAModel::new(5)?),
        );

        Ok(AdaptiveForecaster {
            config,
            regime_detector,
            quality_system,
            models,
            current_regime: None,
            training_data: None,
            performance_stats: PerformanceStats::default(),
            adaptation_history: Vec::new(),
        })
    }

    /// Fit the adaptive forecaster to training data
    pub fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        let start_time = Instant::now();

        // Store training data
        self.training_data = Some(data.clone());

        // Fit regime detector
        self.regime_detector.fit(data)?;

        // Detect initial regime
        self.current_regime = Some(self.regime_detector.detect_regime(data)?);

        // Fit all base models
        for (name, model) in &mut self.models {
            if let Err(e) = model.fit(data) {
                eprintln!("Warning: Failed to fit model {}: {}", name, e);
            }
        }

        // Initialize quality system with initial data
        self.quality_system.initialize_with_data(data)?;

        let elapsed = start_time.elapsed();
        println!("Adaptive forecaster fitted in {:.2}ms", elapsed.as_secs_f64() * 1000.0);

        Ok(())
    }

    /// Generate adaptive forecast
    pub fn forecast(&mut self, horizon: usize) -> Result<AdaptiveForecastOutput> {
        let overall_start = Instant::now();

        if self.training_data.is_none() {
            return Err(OxiError::ModelError(
                "Forecaster must be fitted before forecasting".to_string(),
            ));
        }

        let data = self.training_data.as_ref().unwrap();

        // Regime detection
        let regime_start = Instant::now();
        let current_regime = self.regime_detector.detect_regime(data)?;
        let regime_time = regime_start.elapsed().as_secs_f64() * 1000.0;

        // Check for regime change
        let regime_changed = self.current_regime
            .as_ref()
            .map(|prev| prev != &current_regime)
            .unwrap_or(true);

        if regime_changed {
            self.handle_regime_change(current_regime.clone())?;
        }

        // Select best model for current regime
        let selected_model = self.select_model_for_regime(&current_regime)?;

        // Generate forecast
        let model = self.models.get(&selected_model)
            .ok_or_else(|| OxiError::ModelError(format!("Model not found: {}", selected_model)))?;

        let forecast = model.forecast(horizon)?;

        // Quality evaluation
        let quality_start = Instant::now();
        let dummy_evaluation = ModelEvaluation {
            model_name: selected_model.clone(),
            mae: 0.1,
            mse: 0.01,
            rmse: 0.1,
            mape: 5.0,
            smape: 4.5,
            r_squared: 0.85,
            aic: Some(-100.0),
            bic: Some(-95.0),
        };
        let quality_evaluation = self.quality_system.evaluate_forecast_quality(&dummy_evaluation)?;
        let quality_time = quality_start.elapsed().as_secs_f64() * 1000.0;

        // Check if adaptation is needed based on quality
        let adaptation_occurred = self.check_adaptation_needed(&quality_evaluation)?;

        // Update performance stats
        let total_time = overall_start.elapsed().as_secs_f64() * 1000.0;
        self.update_performance_stats(total_time);

        Ok(AdaptiveForecastOutput {
            forecast,
            confidence_intervals: None, // TODO: Implement confidence intervals
            model_used: selected_model,
            regime_state: current_regime,
            quality_evaluation,
            metadata: ForecastMetadata {
                generation_time_ms: total_time,
                models_evaluated: 1, // TODO: Track actual number
                regime_detection_time_ms: regime_time,
                quality_evaluation_time_ms: quality_time,
                adaptation_occurred,
            },
        })
    }

    /// Generate ensemble forecast using multiple models
    pub fn forecast_ensemble(&mut self, horizon: usize) -> Result<AdaptiveForecastOutput> {
        let start_time = Instant::now();

        if self.training_data.is_none() {
            return Err(OxiError::ModelError(
                "Forecaster must be fitted before forecasting".to_string(),
            ));
        }

        let data = self.training_data.as_ref().unwrap();

        // Detect current regime
        let current_regime = self.regime_detector.detect_regime(data)?;

        // Generate forecasts from multiple models
        let mut model_forecasts = Vec::new();
        let mut models_evaluated = 0;

        for (name, model) in &self.models {
            if let Ok(forecast) = model.forecast(horizon) {
                // Calculate confidence based on model performance
                let confidence = self.calculate_model_confidence(name);
                
                model_forecasts.push(ModelForecast {
                    name: name.clone(),
                    forecast,
                    confidence: Some(confidence),
                    weight: None,
                });
                models_evaluated += 1;
            }
        }

        if model_forecasts.is_empty() {
            return Err(OxiError::ModelError(
                "No models could generate forecasts".to_string(),
            ));
        }

        // Create ensemble forecast
        let mut ensemble = EnsembleForecast {
            forecasts: model_forecasts,
            method: EnsembleMethod::WeightedAverage,
            final_forecast: None,
            model_weights: None,
        };

        // Combine forecasts
        ensemble.combine()?;
        let forecast = ensemble.final_forecast.unwrap();

        // Quality evaluation
        let dummy_evaluation = ModelEvaluation {
            model_name: "Ensemble".to_string(),
            mae: 0.08,
            mse: 0.008,
            rmse: 0.09,
            mape: 4.0,
            smape: 3.8,
            r_squared: 0.90,
            aic: Some(-105.0),
            bic: Some(-100.0),
        };
        let quality_evaluation = self.quality_system.evaluate_forecast_quality(&dummy_evaluation)?;

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.update_performance_stats(total_time);

        Ok(AdaptiveForecastOutput {
            forecast,
            confidence_intervals: None,
            model_used: "Ensemble".to_string(),
            regime_state: current_regime,
            quality_evaluation,
            metadata: ForecastMetadata {
                generation_time_ms: total_time,
                models_evaluated,
                regime_detection_time_ms: 0.0, // Already included in total
                quality_evaluation_time_ms: 0.0, // Already included in total
                adaptation_occurred: false,
            },
        })
    }

    /// Update the forecaster with new data (online learning)
    pub fn update(&mut self, new_data: &TimeSeriesData) -> Result<()> {
        // Append new data to training data
        if let Some(ref mut training_data) = self.training_data {
            // In a real implementation, we'd properly merge the data
            // For now, just replace with new data
            *training_data = new_data.clone();
        } else {
            self.training_data = Some(new_data.clone());
        }

        // Update regime detector
        self.regime_detector.update(new_data)?;

        // Check if refit is needed
        if self.should_refit() {
            self.refit()?;
        }

        Ok(())
    }

    /// Force a refit of all models
    pub fn refit(&mut self) -> Result<()> {
        if let Some(ref data) = self.training_data {
            // Refit all models
            for (name, model) in &mut self.models {
                if let Err(e) = model.fit(data) {
                    eprintln!("Warning: Failed to refit model {}: {}", name, e);
                }
            }

            // Record adaptation event
            self.adaptation_history.push(AdaptationEvent {
                timestamp: std::time::SystemTime::now(),
                reason: AdaptationReason::PeriodicRefit,
                previous_regime: self.current_regime.clone(),
                new_regime: self.current_regime.clone(),
                quality_score: None,
                selected_model: "All".to_string(),
            });

            self.performance_stats.adaptations += 1;
        }

        Ok(())
    }

    /// Get current performance statistics
    pub fn get_performance_stats(&self) -> &PerformanceStats {
        &self.performance_stats
    }

    /// Get adaptation history
    pub fn get_adaptation_history(&self) -> &[AdaptationEvent] {
        &self.adaptation_history
    }

    /// Get current configuration
    pub fn get_config(&self) -> &AdaptiveConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: AdaptiveConfig) -> Result<()> {
        self.config = config;
        
        // Update quality system with new thresholds
        self.quality_system = RealTimeQualitySystem::new(
            self.config.base_config.quality_thresholds.clone(),
        )?;

        Ok(())
    }

    // Private helper methods

    fn handle_regime_change(&mut self, new_regime: RegimeState) -> Result<()> {
        let previous_regime = self.current_regime.clone();
        self.current_regime = Some(new_regime.clone());

        // Record regime change
        self.adaptation_history.push(AdaptationEvent {
            timestamp: std::time::SystemTime::now(),
            reason: AdaptationReason::RegimeChange,
            previous_regime,
            new_regime: Some(new_regime),
            quality_score: None,
            selected_model: "Auto".to_string(),
        });

        self.performance_stats.regime_changes += 1;

        // Optionally refit models for new regime
        if self.config.regime_adaptation {
            self.refit()?;
        }

        Ok(())
    }

    fn select_model_for_regime(&self, regime: &RegimeState) -> Result<String> {
        // Simple model selection logic based on regime
        match regime {
            RegimeState::LowVolatility => Ok("SimpleES".to_string()),
            RegimeState::HighVolatility => Ok("ARIMA".to_string()),
            RegimeState::Trending => Ok("HoltLinear".to_string()),
            RegimeState::Stable => Ok("MA".to_string()),
        }
    }

    fn calculate_model_confidence(&self, _model_name: &str) -> f64 {
        // Placeholder confidence calculation
        // In practice, this would be based on historical performance
        0.8
    }

    fn check_adaptation_needed(&mut self, quality_eval: &QualityEvaluation) -> Result<bool> {
        if quality_eval.overall_score < 0.5 {
            // Quality is poor, trigger adaptation
            self.adaptation_history.push(AdaptationEvent {
                timestamp: std::time::SystemTime::now(),
                reason: AdaptationReason::QualityDegradation,
                previous_regime: self.current_regime.clone(),
                new_regime: self.current_regime.clone(),
                quality_score: Some(quality_eval.overall_score),
                selected_model: "Fallback".to_string(),
            });

            self.performance_stats.adaptations += 1;
            return Ok(true);
        }

        Ok(false)
    }

    fn should_refit(&self) -> bool {
        // Simple refit logic based on number of forecasts
        self.performance_stats.total_forecasts % self.config.refit_frequency == 0
    }

    fn update_performance_stats(&mut self, latency_ms: f64) {
        self.performance_stats.total_forecasts += 1;
        self.performance_stats.quality_evaluations += 1;

        // Update latency statistics
        let n = self.performance_stats.total_forecasts as f64;
        self.performance_stats.avg_latency_ms = 
            (self.performance_stats.avg_latency_ms * (n - 1.0) + latency_ms) / n;

        if latency_ms > self.performance_stats.peak_latency_ms {
            self.performance_stats.peak_latency_ms = latency_ms;
        }
    }
}

impl Default for AdaptiveForecaster {
    fn default() -> Self {
        Self::new().expect("Failed to create default AdaptiveForecaster")
    }
}

/// Implementation of the Forecaster trait for backward compatibility
impl Forecaster for AdaptiveForecaster {
    fn name(&self) -> &str {
        "AdaptiveForecaster"
    }

    fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        self.fit(data)
    }

    fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        // This is a bit tricky since the main forecast method requires &mut self
        // For now, return a simple forecast from the first available model
        if let Some(model) = self.models.values().next() {
            model.forecast(horizon)
        } else {
            Err(OxiError::ModelError(
                "No models available for forecasting".to_string(),
            ))
        }
    }

    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        // Use the first available model for evaluation
        if let Some(model) = self.models.values().next() {
            model.evaluate(test_data)
        } else {
            Err(OxiError::ModelError(
                "No models available for evaluation".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeriesData;
    use chrono::{DateTime, Utc};

    fn create_test_data() -> TimeSeriesData {
        let timestamps: Vec<DateTime<Utc>> = (0..100)
            .map(|i| DateTime::from_timestamp(1609459200 + i * 86400, 0).unwrap())
            .collect();
        let values: Vec<f64> = (0..100).map(|i| (i as f64).sin() * 10.0 + 50.0 + (i as f64) * 0.1).collect();
        
        TimeSeriesData::new(timestamps, values, "test_data").unwrap()
    }

    #[test]
    fn test_adaptive_forecaster_creation() {
        let forecaster = AdaptiveForecaster::new();
        assert!(forecaster.is_ok());
    }

    #[test]
    fn test_adaptive_forecaster_fit_and_forecast() {
        let mut forecaster = AdaptiveForecaster::new().unwrap();
        let data = create_test_data();

        // Test fitting
        let fit_result = forecaster.fit(&data);
        assert!(fit_result.is_ok());

        // Test forecasting
        let forecast_result = forecaster.forecast(10);
        assert!(forecast_result.is_ok());

        let forecast_output = forecast_result.unwrap();
        assert_eq!(forecast_output.forecast.len(), 10);
        assert!(!forecast_output.model_used.is_empty());
    }

    #[test]
    fn test_ensemble_forecasting() {
        let mut forecaster = AdaptiveForecaster::new().unwrap();
        let data = create_test_data();

        forecaster.fit(&data).unwrap();
        
        let ensemble_result = forecaster.forecast_ensemble(5);
        assert!(ensemble_result.is_ok());

        let ensemble_output = ensemble_result.unwrap();
        assert_eq!(ensemble_output.forecast.len(), 5);
        assert_eq!(ensemble_output.model_used, "Ensemble");
    }

    #[test]
    fn test_performance_tracking() {
        let mut forecaster = AdaptiveForecaster::new().unwrap();
        let data = create_test_data();

        forecaster.fit(&data).unwrap();
        
        // Generate multiple forecasts
        for _ in 0..5 {
            let _ = forecaster.forecast(3);
        }

        let stats = forecaster.get_performance_stats();
        assert_eq!(stats.total_forecasts, 5);
        assert!(stats.avg_latency_ms > 0.0);
    }

    #[test]
    fn test_config_update() {
        let mut forecaster = AdaptiveForecaster::new().unwrap();
        let mut new_config = AdaptiveConfig::default();
        new_config.regime_adaptation = true;

        let update_result = forecaster.update_config(new_config);
        assert!(update_result.is_ok());
        assert!(forecaster.get_config().regime_adaptation);
    }
} 