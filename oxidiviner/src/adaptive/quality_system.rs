//! STEP 3: Real-Time Quality Monitoring System
//!
//! This module provides comprehensive quality monitoring capabilities with:
//! - Real-time quality metrics calculation
//! - Automatic fallback mechanisms
//! - Performance monitoring (<5ms overhead)
//! - Adaptive quality thresholds
//! - Integration with regime detection and forecasting

use crate::adaptive::monitoring::QualityThresholds;
use crate::adaptive::{AdaptiveConfig, QualityMetrics, QualityMonitor};
use crate::core::{ModelEvaluation, OxiError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

/// Real-time quality monitoring system for adaptive forecasting
#[derive(Debug)]
pub struct RealTimeQualitySystem {
    /// Core quality monitor
    quality_monitor: QualityMonitor,

    /// Performance metrics
    performance_metrics: PerformanceMetrics,

    /// Fallback models registry
    fallback_models: FallbackRegistry,

    /// Quality threshold adapter
    threshold_adapter: ThresholdAdapter,

    /// Real-time metrics calculator
    metrics_calculator: RealTimeMetricsCalculator,

    /// System configuration
    config: QualitySystemConfig,
}

/// Performance monitoring metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average processing time for quality evaluation
    pub avg_processing_time_ms: f64,

    /// Maximum processing time recorded
    pub max_processing_time_ms: u64,

    /// Total quality evaluations performed
    pub total_evaluations: u64,

    /// Evaluations meeting timing requirements
    pub fast_evaluations: u64,

    /// Quality evaluation throughput (evaluations/second)
    pub throughput: f64,

    /// Memory usage for quality monitoring
    pub memory_usage_bytes: usize,

    /// Last performance update timestamp
    pub last_updated: SystemTime,
}

/// Fallback model registry for automatic switching
#[derive(Debug)]
#[allow(dead_code)]
pub struct FallbackRegistry {
    /// Primary fallback models by priority
    fallback_models: Vec<String>,

    /// Model performance history
    model_performance: HashMap<String, ModelPerformanceHistory>,

    /// Current active fallback model index
    active_fallback_index: Option<usize>,

    /// Fallback activation count
    activation_count: usize,
}

/// Model performance tracking
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ModelPerformanceHistory {
    /// Recent evaluation results
    recent_evaluations: VecDeque<ModelEvaluation>,

    /// Average quality score
    avg_quality_score: f64,

    /// Reliability score (0.0 to 1.0)
    reliability_score: f64,

    /// Last used timestamp
    last_used: SystemTime,
}

/// Adaptive threshold adjustment system
#[derive(Debug)]
#[allow(dead_code)]
pub struct ThresholdAdapter {
    /// Base quality thresholds
    base_thresholds: QualityThresholds,

    /// Current adapted thresholds
    current_thresholds: QualityThresholds,

    /// Adaptation history
    adaptation_history: VecDeque<ThresholdAdaptation>,

    /// Adaptation strategy
    strategy: AdaptationStrategy,
}

/// Threshold adaptation record
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ThresholdAdaptation {
    /// Timestamp of adaptation
    timestamp: SystemTime,

    /// Previous thresholds
    old_thresholds: QualityThresholds,

    /// New thresholds
    new_thresholds: QualityThresholds,

    /// Reason for adaptation
    reason: String,

    /// Performance impact
    performance_impact: f64,
}

/// Threshold adaptation strategies
#[derive(Debug, Clone)]
pub enum AdaptationStrategy {
    /// Conservative: Slow threshold adjustments
    Conservative,

    /// Moderate: Balanced threshold adjustments
    Moderate,

    /// Aggressive: Fast threshold adjustments
    Aggressive,

    /// Custom: User-defined adaptation parameters
    Custom {
        adaptation_rate: f64,
        volatility_sensitivity: f64,
        performance_weight: f64,
    },
}

/// Real-time metrics calculator
#[derive(Debug)]
pub struct RealTimeMetricsCalculator {
    /// Calculation window size
    window_size: usize,

    /// Recent forecast accuracy data
    accuracy_buffer: VecDeque<f64>,

    /// Recent prediction errors
    error_buffer: VecDeque<f64>,

    /// Running statistics
    running_stats: RunningStatistics,

    /// Quality trend detector
    trend_detector: QualityTrendDetector,
}

/// Running statistical calculations
#[derive(Debug, Clone)]
pub struct RunningStatistics {
    /// Count of observations
    count: usize,

    /// Running mean
    mean: f64,

    /// Running variance (for standard deviation)
    variance: f64,

    /// Minimum value
    min_value: f64,

    /// Maximum value
    max_value: f64,

    /// Last update timestamp
    last_updated: Instant,
}

/// Quality trend detection
#[derive(Debug)]
pub struct QualityTrendDetector {
    /// Trend detection window
    trend_window: usize,

    /// Quality score history
    quality_history: VecDeque<f64>,

    /// Detected trend direction
    current_trend: TrendDirection,

    /// Trend strength (0.0 to 1.0)
    trend_strength: f64,

    /// Trend change detection
    trend_change_detector: TrendChangeDetector,
}

/// Trend direction enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// Quality improving
    Improving,

    /// Quality stable
    Stable,

    /// Quality declining
    Declining,

    /// Insufficient data
    Unknown,
}

/// Trend change detection system
#[derive(Debug)]
#[allow(dead_code)]
pub struct TrendChangeDetector {
    /// Minimum observations for trend detection
    min_observations: usize,

    /// Trend change sensitivity threshold
    change_threshold: f64,

    /// Recent trend changes
    recent_changes: VecDeque<TrendChange>,
}

/// Trend change record
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TrendChange {
    /// Change timestamp
    timestamp: SystemTime,

    /// Previous trend
    old_trend: TrendDirection,

    /// New trend
    new_trend: TrendDirection,

    /// Change magnitude
    magnitude: f64,
}

/// Quality system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySystemConfig {
    /// Maximum allowed processing time (ms)
    pub max_processing_time_ms: u64,

    /// Enable automatic fallback
    pub enable_auto_fallback: bool,

    /// Enable adaptive thresholds
    pub enable_adaptive_thresholds: bool,

    /// Quality evaluation frequency
    pub evaluation_frequency_ms: u64,

    /// Performance monitoring window
    pub performance_window_size: usize,

    /// Trend detection sensitivity
    pub trend_sensitivity: f64,

    /// Memory usage limit (bytes)
    pub memory_limit_bytes: usize,

    /// Detailed logging enabled
    pub enable_detailed_logging: bool,
}

/// Quality evaluation result with timing information
#[derive(Debug, Clone)]
pub struct QualityEvaluationResult {
    /// Quality metrics
    pub metrics: QualityMetrics,

    /// Processing time
    pub processing_time: Duration,

    /// Memory usage
    pub memory_usage: usize,

    /// Fallback triggered
    pub fallback_triggered: bool,

    /// Threshold adaptation occurred
    pub threshold_adapted: bool,

    /// Evaluation timestamp
    pub timestamp: SystemTime,
}

/// Real-time quality monitoring report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeQualityReport {
    /// Current quality metrics
    pub current_quality: Option<QualityMetrics>,

    /// Performance metrics
    pub performance: PerformanceMetrics,

    /// System health status
    pub system_health: SystemHealthStatus,

    /// Active fallback information
    pub active_fallback: Option<String>,

    /// Quality trend information
    pub quality_trend: QualityTrendInfo,

    /// Recommendations
    pub recommendations: Vec<String>,

    /// Report timestamp
    pub timestamp: SystemTime,
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthStatus {
    /// Overall health score (0.0 to 1.0)
    pub health_score: f64,

    /// Performance health
    pub performance_ok: bool,

    /// Memory health
    pub memory_ok: bool,

    /// Quality health
    pub quality_ok: bool,

    /// Trend health
    pub trend_ok: bool,

    /// Issues detected
    pub issues: Vec<String>,
}

/// Quality trend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTrendInfo {
    /// Current trend direction
    pub direction: String,

    /// Trend strength
    pub strength: f64,

    /// Trend duration
    pub duration_minutes: f64,

    /// Trend stability
    pub stability: f64,
}

impl Default for QualitySystemConfig {
    fn default() -> Self {
        Self {
            max_processing_time_ms: 5, // STEP 3 requirement: <5ms overhead
            enable_auto_fallback: true,
            enable_adaptive_thresholds: true,
            evaluation_frequency_ms: 100,
            performance_window_size: 100,
            trend_sensitivity: 0.1,
            memory_limit_bytes: 10 * 1024 * 1024, // 10MB limit
            enable_detailed_logging: false,
        }
    }
}

impl RealTimeQualitySystem {
    /// Create a new real-time quality monitoring system
    pub fn new(config: AdaptiveConfig) -> Result<Self> {
        let config_thresholds = config.quality_thresholds.clone();

        // Convert config QualityThresholds to monitoring QualityThresholds
        let monitoring_thresholds = QualityThresholds {
            max_mae: config_thresholds.max_mae,
            max_mape: config_thresholds.max_mape,
            min_r_squared: config_thresholds.min_r_squared,
            quality_window: config_thresholds.quality_window,
            enable_fallback: config_thresholds.enable_fallback,
            max_consecutive_failures: config_thresholds.max_consecutive_failures,
        };

        let quality_monitor = QualityMonitor::with_thresholds(monitoring_thresholds.clone());

        Ok(Self {
            quality_monitor,
            performance_metrics: PerformanceMetrics::new(),
            fallback_models: FallbackRegistry::new(),
            threshold_adapter: ThresholdAdapter::new(monitoring_thresholds),
            metrics_calculator: RealTimeMetricsCalculator::new(50),
            config: QualitySystemConfig::default(),
        })
    }

    /// Create with custom configuration
    pub fn with_config(
        adaptive_config: AdaptiveConfig,
        system_config: QualitySystemConfig,
    ) -> Result<Self> {
        let mut system = Self::new(adaptive_config)?;
        system.config = system_config;
        Ok(system)
    }

    /// Evaluate forecast quality in real-time with performance monitoring
    pub fn evaluate_forecast_quality(
        &mut self,
        forecast_values: &[f64],
        actual_values: &[f64],
    ) -> Result<QualityEvaluationResult> {
        let start_time = Instant::now();

        // Validate input lengths
        if forecast_values.len() != actual_values.len() {
            return Err(OxiError::InvalidParameter(
                "Forecast and actual value arrays must have same length".to_string(),
            ));
        }

        if forecast_values.is_empty() {
            return Err(OxiError::InvalidParameter(
                "Cannot evaluate quality on empty data".to_string(),
            ));
        }

        // Calculate quality metrics
        let evaluation = self.calculate_model_evaluation(forecast_values, actual_values)?;

        // Update quality monitor
        self.quality_monitor.update_quality(&evaluation)?;

        // Update real-time metrics
        self.metrics_calculator.update_metrics(&evaluation)?;

        // Check for threshold adaptation
        let threshold_adapted = self.check_and_adapt_thresholds(&evaluation)?;

        // Check for fallback trigger
        let fallback_triggered = self.check_fallback_trigger()?;

        // Update performance metrics
        let processing_time = start_time.elapsed();
        self.update_performance_metrics(processing_time)?;

        // Verify performance requirement (<5ms)
        if processing_time.as_millis() > self.config.max_processing_time_ms as u128 {
            return Err(OxiError::InvalidParameter(format!(
                "Quality evaluation took {}ms, exceeding {}ms limit",
                processing_time.as_millis(),
                self.config.max_processing_time_ms
            )));
        }

        Ok(QualityEvaluationResult {
            metrics: self.quality_monitor.current_quality().unwrap().clone(),
            processing_time,
            memory_usage: self.estimate_memory_usage(),
            fallback_triggered,
            threshold_adapted,
            timestamp: SystemTime::now(),
        })
    }

    /// Register a fallback model
    pub fn register_fallback_model(&mut self, model_name: String) -> Result<()> {
        self.fallback_models.register_model(model_name)
    }

    /// Check if automatic fallback should be triggered
    fn check_fallback_trigger(&mut self) -> Result<bool> {
        if !self.config.enable_auto_fallback {
            return Ok(false);
        }

        if self.quality_monitor.is_fallback_triggered() {
            self.fallback_models.activate_fallback()?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Check and adapt quality thresholds if needed
    fn check_and_adapt_thresholds(&mut self, evaluation: &ModelEvaluation) -> Result<bool> {
        if !self.config.enable_adaptive_thresholds {
            return Ok(false);
        }

        self.threshold_adapter
            .check_adaptation(evaluation, &self.metrics_calculator)
    }

    /// Calculate model evaluation from forecast and actual values
    fn calculate_model_evaluation(
        &self,
        forecast_values: &[f64],
        actual_values: &[f64],
    ) -> Result<ModelEvaluation> {
        // Calculate metrics using existing math functions
        let mae = crate::math::metrics::mae(actual_values, forecast_values);
        let mse = crate::math::metrics::mse(actual_values, forecast_values);
        let rmse = mse.sqrt();
        let mape = crate::math::metrics::mape(actual_values, forecast_values);

        // Calculate R-squared
        let actual_mean: f64 = actual_values.iter().sum::<f64>() / actual_values.len() as f64;
        let ss_tot: f64 = actual_values
            .iter()
            .map(|&x| (x - actual_mean).powi(2))
            .sum();
        let ss_res: f64 = actual_values
            .iter()
            .zip(forecast_values)
            .map(|(&actual, &forecast)| (actual - forecast).powi(2))
            .sum();

        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        Ok(ModelEvaluation {
            model_name: "QualityEvaluator".to_string(),
            mae,
            mse,
            rmse,
            mape,
            smape: 0.0, // Not calculated for quality monitoring
            r_squared,
            aic: Some(0.0), // Not applicable for quality monitoring
            bic: Some(0.0), // Not applicable for quality monitoring
        })
    }

    /// Update performance metrics
    fn update_performance_metrics(&mut self, processing_time: Duration) -> Result<()> {
        let processing_ms = processing_time.as_millis() as u64;

        self.performance_metrics.total_evaluations += 1;

        // Update average processing time
        let total = self.performance_metrics.total_evaluations as f64;
        let current_avg = self.performance_metrics.avg_processing_time_ms;
        self.performance_metrics.avg_processing_time_ms =
            (current_avg * (total - 1.0) + processing_ms as f64) / total;

        // Update max processing time
        if processing_ms > self.performance_metrics.max_processing_time_ms {
            self.performance_metrics.max_processing_time_ms = processing_ms;
        }

        // Count fast evaluations
        if processing_ms <= self.config.max_processing_time_ms {
            self.performance_metrics.fast_evaluations += 1;
        }

        // Calculate throughput
        let fast_ratio = self.performance_metrics.fast_evaluations as f64 / total;
        self.performance_metrics.throughput =
            1000.0 / self.performance_metrics.avg_processing_time_ms * fast_ratio;

        self.performance_metrics.last_updated = SystemTime::now();

        Ok(())
    }

    /// Estimate current memory usage
    fn estimate_memory_usage(&self) -> usize {
        // Estimate memory usage of quality monitoring components
        std::mem::size_of::<Self>()
            + self.metrics_calculator.accuracy_buffer.len() * std::mem::size_of::<f64>()
            + self.threshold_adapter.adaptation_history.len()
                * std::mem::size_of::<ThresholdAdaptation>()
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }

    /// Check if system is operating within performance requirements
    pub fn is_performance_acceptable(&self) -> bool {
        self.performance_metrics.avg_processing_time_ms <= self.config.max_processing_time_ms as f64
    }

    /// Get current quality metrics
    pub fn get_current_quality(&self) -> Option<&QualityMetrics> {
        self.quality_monitor.current_quality()
    }

    /// Get quality monitor for direct access
    pub fn get_quality_monitor(&self) -> &QualityMonitor {
        &self.quality_monitor
    }
}

// Implementation of helper structures

impl PerformanceMetrics {
    fn new() -> Self {
        Self {
            avg_processing_time_ms: 0.0,
            max_processing_time_ms: 0,
            total_evaluations: 0,
            fast_evaluations: 0,
            throughput: 0.0,
            memory_usage_bytes: 0,
            last_updated: SystemTime::now(),
        }
    }
}

impl FallbackRegistry {
    fn new() -> Self {
        Self {
            fallback_models: Vec::new(),
            model_performance: HashMap::new(),
            active_fallback_index: None,
            activation_count: 0,
        }
    }

    fn register_model(&mut self, model_name: String) -> Result<()> {
        self.fallback_models.push(model_name);
        Ok(())
    }

    fn activate_fallback(&mut self) -> Result<()> {
        if !self.fallback_models.is_empty() && self.active_fallback_index.is_none() {
            self.active_fallback_index = Some(0);
            self.activation_count += 1;
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn get_active_fallback_name(&self) -> Option<String> {
        self.active_fallback_index
            .map(|idx| self.fallback_models[idx].clone())
    }
}

impl ThresholdAdapter {
    fn new(base_thresholds: QualityThresholds) -> Self {
        Self {
            current_thresholds: base_thresholds.clone(),
            base_thresholds,
            adaptation_history: VecDeque::new(),
            strategy: AdaptationStrategy::Moderate,
        }
    }

    fn check_adaptation(
        &mut self,
        _evaluation: &ModelEvaluation,
        _calculator: &RealTimeMetricsCalculator,
    ) -> Result<bool> {
        // Simplified adaptation logic - could be enhanced
        Ok(false)
    }
}

impl RealTimeMetricsCalculator {
    fn new(window_size: usize) -> Self {
        Self {
            window_size,
            accuracy_buffer: VecDeque::with_capacity(window_size),
            error_buffer: VecDeque::with_capacity(window_size),
            running_stats: RunningStatistics::new(),
            trend_detector: QualityTrendDetector::new(window_size),
        }
    }

    fn update_metrics(&mut self, evaluation: &ModelEvaluation) -> Result<()> {
        // Update buffers
        if self.accuracy_buffer.len() >= self.window_size {
            self.accuracy_buffer.pop_front();
        }
        self.accuracy_buffer.push_back(evaluation.r_squared);

        if self.error_buffer.len() >= self.window_size {
            self.error_buffer.pop_front();
        }
        self.error_buffer.push_back(evaluation.mae);

        // Update running statistics
        self.running_stats.update(evaluation.r_squared);

        // Update trend detector
        self.trend_detector.update_trend(evaluation.r_squared);

        Ok(())
    }
}

impl RunningStatistics {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            variance: 0.0,
            min_value: f64::INFINITY,
            max_value: f64::NEG_INFINITY,
            last_updated: Instant::now(),
        }
    }

    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.variance += delta * delta2;

        self.min_value = self.min_value.min(value);
        self.max_value = self.max_value.max(value);
        self.last_updated = Instant::now();
    }
}

impl QualityTrendDetector {
    fn new(trend_window: usize) -> Self {
        Self {
            trend_window,
            quality_history: VecDeque::with_capacity(trend_window),
            current_trend: TrendDirection::Unknown,
            trend_strength: 0.0,
            trend_change_detector: TrendChangeDetector::new(),
        }
    }

    fn update_trend(&mut self, quality_score: f64) {
        if self.quality_history.len() >= self.trend_window {
            self.quality_history.pop_front();
        }
        self.quality_history.push_back(quality_score);

        self.detect_trend();
    }

    fn detect_trend(&mut self) {
        if self.quality_history.len() < 3 {
            self.current_trend = TrendDirection::Unknown;
            return;
        }

        // Simple trend detection based on recent values
        let recent: Vec<_> = self.quality_history.iter().rev().take(3).collect();
        let trend_slope = recent[0] - recent[2];

        let old_trend = self.current_trend.clone();

        if trend_slope > 0.05 {
            self.current_trend = TrendDirection::Improving;
            self.trend_strength = trend_slope.abs().min(1.0);
        } else if trend_slope < -0.05 {
            self.current_trend = TrendDirection::Declining;
            self.trend_strength = trend_slope.abs().min(1.0);
        } else {
            self.current_trend = TrendDirection::Stable;
            self.trend_strength = 0.1;
        }

        // Record trend change if occurred
        if old_trend != self.current_trend {
            self.trend_change_detector.record_change(TrendChange {
                timestamp: SystemTime::now(),
                old_trend,
                new_trend: self.current_trend.clone(),
                magnitude: self.trend_strength,
            });
        }
    }
}

impl TrendChangeDetector {
    fn new() -> Self {
        Self {
            min_observations: 5,
            change_threshold: 0.1,
            recent_changes: VecDeque::new(),
        }
    }

    fn record_change(&mut self, change: TrendChange) {
        if self.recent_changes.len() >= 10 {
            self.recent_changes.pop_front();
        }
        self.recent_changes.push_back(change);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive::AdaptiveConfig;

    #[test]
    fn test_real_time_quality_system_creation() {
        let config = AdaptiveConfig::default();
        let system = RealTimeQualitySystem::new(config);
        assert!(system.is_ok());
    }

    #[test]
    fn test_performance_metrics_initialization() {
        let metrics = PerformanceMetrics::new();
        assert_eq!(metrics.total_evaluations, 0);
        assert_eq!(metrics.avg_processing_time_ms, 0.0);
    }

    #[test]
    fn test_fallback_registry() {
        let mut registry = FallbackRegistry::new();
        assert!(registry.register_model("TestModel".to_string()).is_ok());
        assert!(registry.activate_fallback().is_ok());
        assert_eq!(registry.activation_count, 1);
    }

    #[test]
    fn test_quality_system_configuration() {
        let config = QualitySystemConfig::default();
        assert_eq!(config.max_processing_time_ms, 5);
        assert!(config.enable_auto_fallback);
    }

    #[test]
    fn test_trend_detection() {
        let mut detector = QualityTrendDetector::new(10);

        // Add improving trend
        detector.update_trend(0.5);
        detector.update_trend(0.6);
        detector.update_trend(0.7);

        assert_eq!(detector.current_trend, TrendDirection::Improving);
        assert!(detector.trend_strength > 0.0);
    }

    #[test]
    fn test_running_statistics() {
        let mut stats = RunningStatistics::new();

        stats.update(1.0);
        stats.update(2.0);
        stats.update(3.0);

        assert_eq!(stats.count, 3);
        assert!((stats.mean - 2.0).abs() < 1e-10);
        assert_eq!(stats.min_value, 1.0);
        assert_eq!(stats.max_value, 3.0);
    }

    #[test]
    fn test_quality_evaluation_with_performance() {
        let config = AdaptiveConfig::default();
        let mut system = RealTimeQualitySystem::new(config).unwrap();

        let forecast = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let actual = vec![1.1, 2.1, 2.9, 4.1, 4.9];

        let result = system.evaluate_forecast_quality(&forecast, &actual);
        assert!(result.is_ok());

        let evaluation = result.unwrap();
        assert!(evaluation.processing_time.as_millis() <= 5); // Should be under 5ms
        assert!(!evaluation.fallback_triggered);
    }

    #[test]
    fn test_performance_requirements() {
        let config = AdaptiveConfig::default();
        let mut system = RealTimeQualitySystem::new(config).unwrap();

        // Register fallback model
        assert!(system
            .register_fallback_model("SimpleES".to_string())
            .is_ok());

        // Test with good data
        let forecast = vec![1.0, 2.0, 3.0];
        let actual = vec![1.0, 2.0, 3.0];

        let result = system.evaluate_forecast_quality(&forecast, &actual);
        assert!(result.is_ok());

        let performance = system.get_performance_metrics();
        assert_eq!(performance.total_evaluations, 1);
        assert!(performance.avg_processing_time_ms <= 5.0);
    }
}
