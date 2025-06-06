//! Quality Monitoring System for Adaptive Forecasting
//!
//! This module provides real-time quality monitoring capabilities with automatic
//! fallback mechanisms when forecast quality degrades.

use crate::core::{ModelEvaluation, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Quality thresholds for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Maximum acceptable Mean Absolute Error
    pub max_mae: f64,

    /// Maximum acceptable Mean Absolute Percentage Error
    pub max_mape: f64,

    /// Minimum acceptable R-squared value
    pub min_r_squared: f64,

    /// Window size for quality evaluation
    pub quality_window: usize,

    /// Enable automatic fallback when quality drops
    pub enable_fallback: bool,

    /// Consecutive failures before triggering fallback
    pub max_consecutive_failures: usize,
}

/// Real-time quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Current Mean Absolute Error
    pub current_mae: f64,

    /// Current Mean Absolute Percentage Error
    pub current_mape: f64,

    /// Current R-squared value
    pub current_r_squared: f64,

    /// Rolling average MAE
    pub rolling_mae: f64,

    /// Rolling average MAPE
    pub rolling_mape: f64,

    /// Rolling average R-squared
    pub rolling_r_squared: f64,

    /// Quality score (0.0 to 1.0, higher is better)
    pub quality_score: f64,

    /// Is current quality acceptable
    pub quality_acceptable: bool,

    /// Number of consecutive quality failures
    pub consecutive_failures: usize,

    /// Timestamp of last update
    pub last_updated: std::time::SystemTime,
}

/// Quality monitoring system
#[derive(Debug)]
pub struct QualityMonitor {
    /// Quality thresholds
    thresholds: QualityThresholds,

    /// Recent quality metrics history
    metrics_history: VecDeque<QualityMetrics>,

    /// Current quality metrics
    current_metrics: Option<QualityMetrics>,

    /// Fallback triggered flag
    fallback_triggered: bool,

    /// Total evaluations performed
    total_evaluations: usize,

    /// Successful evaluations count
    successful_evaluations: usize,
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

impl QualityMonitor {
    /// Create a new quality monitor with default thresholds
    pub fn new() -> Self {
        Self::with_thresholds(QualityThresholds::default())
    }

    /// Create a new quality monitor with custom thresholds
    pub fn with_thresholds(thresholds: QualityThresholds) -> Self {
        let capacity = thresholds.quality_window;
        Self {
            thresholds,
            metrics_history: VecDeque::with_capacity(capacity),
            current_metrics: None,
            fallback_triggered: false,
            total_evaluations: 0,
            successful_evaluations: 0,
        }
    }

    /// Update quality metrics with new evaluation
    pub fn update_quality(&mut self, evaluation: &ModelEvaluation) -> Result<()> {
        let metrics = self.calculate_quality_metrics(evaluation)?;

        // Update history
        if self.metrics_history.len() >= self.thresholds.quality_window {
            self.metrics_history.pop_front();
        }
        self.metrics_history.push_back(metrics.clone());

        // Update current metrics
        self.current_metrics = Some(metrics);

        // Update counters
        self.total_evaluations += 1;
        if self.current_metrics.as_ref().unwrap().quality_acceptable {
            self.successful_evaluations += 1;
        }

        // Check for fallback trigger
        self.check_fallback_trigger();

        Ok(())
    }

    /// Calculate quality metrics from model evaluation
    fn calculate_quality_metrics(&self, evaluation: &ModelEvaluation) -> Result<QualityMetrics> {
        let current_mae = evaluation.mae;
        let current_mape = evaluation.mape;
        let current_r_squared = evaluation.r_squared;

        // Calculate rolling averages based on the window size
        let (rolling_mae, rolling_mape, rolling_r_squared) = if self.metrics_history.is_empty() {
            (current_mae, current_mape, current_r_squared)
        } else {
            // Take only the most recent values up to window size
            let window_size = self.thresholds.quality_window;
            let recent_metrics: Vec<_> = self
                .metrics_history
                .iter()
                .rev()
                .take(window_size.saturating_sub(1)) // Leave room for current value
                .collect();

            let count = recent_metrics.len() as f64;
            let sum_mae: f64 = recent_metrics.iter().map(|m| m.current_mae).sum();
            let sum_mape: f64 = recent_metrics.iter().map(|m| m.current_mape).sum();
            let sum_rsq: f64 = recent_metrics.iter().map(|m| m.current_r_squared).sum();

            (
                (sum_mae + current_mae) / (count + 1.0),
                (sum_mape + current_mape) / (count + 1.0),
                (sum_rsq + current_r_squared) / (count + 1.0),
            )
        };

        // Check quality acceptability
        let quality_acceptable = current_mae <= self.thresholds.max_mae
            && current_mape <= self.thresholds.max_mape
            && current_r_squared >= self.thresholds.min_r_squared;

        // Calculate quality score (weighted combination of metrics)
        let mae_score = (self.thresholds.max_mae - current_mae.min(self.thresholds.max_mae))
            / self.thresholds.max_mae;
        let mape_score = (self.thresholds.max_mape - current_mape.min(self.thresholds.max_mape))
            / self.thresholds.max_mape;
        let rsq_score = current_r_squared.clamp(0.0, 1.0);

        let quality_score = (mae_score * 0.3 + mape_score * 0.3 + rsq_score * 0.4)
            .clamp(0.0, 1.0);

        // Update consecutive failures
        let consecutive_failures = if quality_acceptable {
            0
        } else {
            self.current_metrics
                .as_ref()
                .map(|m| m.consecutive_failures + 1)
                .unwrap_or(1)
        };

        Ok(QualityMetrics {
            current_mae,
            current_mape,
            current_r_squared,
            rolling_mae,
            rolling_mape,
            rolling_r_squared,
            quality_score,
            quality_acceptable,
            consecutive_failures,
            last_updated: std::time::SystemTime::now(),
        })
    }

    /// Check if fallback should be triggered
    fn check_fallback_trigger(&mut self) {
        if !self.thresholds.enable_fallback {
            return;
        }

        if let Some(metrics) = &self.current_metrics {
            if metrics.consecutive_failures >= self.thresholds.max_consecutive_failures {
                self.fallback_triggered = true;
            }
        }
    }

    /// Get current quality metrics
    pub fn current_quality(&self) -> Option<&QualityMetrics> {
        self.current_metrics.as_ref()
    }

    /// Check if quality is currently acceptable
    pub fn is_quality_acceptable(&self) -> bool {
        self.current_metrics
            .as_ref()
            .map(|m| m.quality_acceptable)
            .unwrap_or(false)
    }

    /// Check if fallback has been triggered
    pub fn is_fallback_triggered(&self) -> bool {
        self.fallback_triggered
    }

    /// Reset fallback trigger (call after taking corrective action)
    pub fn reset_fallback(&mut self) {
        self.fallback_triggered = false;
    }

    /// Get quality success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_evaluations == 0 {
            0.0
        } else {
            self.successful_evaluations as f64 / self.total_evaluations as f64
        }
    }

    /// Get average quality score over the monitoring window
    pub fn average_quality_score(&self) -> f64 {
        if self.metrics_history.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.metrics_history.iter().map(|m| m.quality_score).sum();
        sum / self.metrics_history.len() as f64
    }

    /// Get quality trend (positive means improving, negative means degrading)
    pub fn quality_trend(&self) -> f64 {
        if self.metrics_history.len() < 2 {
            return 0.0;
        }

        // Get recent scores in chronological order (oldest to newest)
        let recent_scores: Vec<f64> = self
            .metrics_history
            .iter()
            .rev()
            .take(5)
            .map(|m| m.quality_score)
            .collect::<Vec<_>>()
            .into_iter()
            .rev() // Reverse again to get chronological order
            .collect();

        if recent_scores.len() < 2 {
            return 0.0;
        }

        // Simple linear trend calculation
        let n = recent_scores.len() as f64;
        let sum_x: f64 = (0..recent_scores.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent_scores.iter().sum();
        let sum_xy: f64 = recent_scores
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum();
        let sum_x2: f64 = (0..recent_scores.len()).map(|i| (i as f64).powi(2)).sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < 1e-10 {
            return 0.0; // Avoid division by zero
        }

        // Calculate slope (trend)
        (n * sum_xy - sum_x * sum_y) / denominator
    }

    /// Get detailed quality report
    pub fn quality_report(&self) -> QualityReport {
        QualityReport {
            current_metrics: self.current_metrics.clone(),
            success_rate: self.success_rate(),
            average_quality_score: self.average_quality_score(),
            quality_trend: self.quality_trend(),
            fallback_triggered: self.fallback_triggered,
            total_evaluations: self.total_evaluations,
            successful_evaluations: self.successful_evaluations,
            thresholds: self.thresholds.clone(),
        }
    }

    /// Update thresholds
    pub fn update_thresholds(&mut self, thresholds: QualityThresholds) {
        self.thresholds = thresholds;
        // Resize history buffer if needed
        self.metrics_history.reserve(self.thresholds.quality_window);
        while self.metrics_history.len() > self.thresholds.quality_window {
            self.metrics_history.pop_front();
        }
    }
}

/// Comprehensive quality report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    /// Current quality metrics
    pub current_metrics: Option<QualityMetrics>,

    /// Overall success rate
    pub success_rate: f64,

    /// Average quality score
    pub average_quality_score: f64,

    /// Quality trend
    pub quality_trend: f64,

    /// Fallback triggered status
    pub fallback_triggered: bool,

    /// Total evaluations performed
    pub total_evaluations: usize,

    /// Successful evaluations count
    pub successful_evaluations: usize,

    /// Current thresholds
    pub thresholds: QualityThresholds,
}

impl Default for QualityMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_evaluation(mae: f64, mape: f64, r_squared: f64) -> ModelEvaluation {
        ModelEvaluation {
            model_name: "test".to_string(),
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

    #[test]
    fn test_quality_monitor_creation() {
        let monitor = QualityMonitor::new();
        assert!(!monitor.is_quality_acceptable());
        assert!(!monitor.is_fallback_triggered());
        assert_eq!(monitor.success_rate(), 0.0);
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
    }

    #[test]
    fn test_fallback_trigger() {
        let mut monitor = QualityMonitor::new();
        let bad_evaluation = create_test_evaluation(0.3, 25.0, 0.3);

        // Trigger multiple consecutive failures
        for _ in 0..3 {
            monitor.update_quality(&bad_evaluation).unwrap();
        }

        assert!(monitor.is_fallback_triggered());

        // Reset fallback
        monitor.reset_fallback();
        assert!(!monitor.is_fallback_triggered());
    }

    #[test]
    fn test_rolling_averages() {
        let mut monitor = QualityMonitor::new();

        // Add multiple evaluations
        let evaluations = vec![
            create_test_evaluation(0.1, 10.0, 0.8),
            create_test_evaluation(0.12, 12.0, 0.75),
            create_test_evaluation(0.08, 8.0, 0.85),
        ];

        for eval in evaluations {
            monitor.update_quality(&eval).unwrap();
        }

        let metrics = monitor.current_quality().unwrap();
        assert!((metrics.rolling_mae - 0.1).abs() < 0.01);
        assert!((metrics.rolling_mape - 10.0).abs() < 0.01);
        assert!((metrics.rolling_r_squared - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_quality_score_calculation() {
        let mut monitor = QualityMonitor::new();

        // Perfect quality
        let perfect_eval = create_test_evaluation(0.0, 0.0, 1.0);
        monitor.update_quality(&perfect_eval).unwrap();
        let metrics = monitor.current_quality().unwrap();
        assert!(metrics.quality_score > 0.9);

        // Poor quality
        let mut monitor2 = QualityMonitor::new();
        let poor_eval = create_test_evaluation(0.2, 20.0, 0.2);
        monitor2.update_quality(&poor_eval).unwrap();
        let metrics2 = monitor2.current_quality().unwrap();
        assert!(metrics2.quality_score < 0.5);
    }

    #[test]
    fn test_quality_trend() {
        let mut monitor = QualityMonitor::new();

        // Add improving quality
        let evaluations = vec![
            create_test_evaluation(0.2, 20.0, 0.5),
            create_test_evaluation(0.15, 15.0, 0.6),
            create_test_evaluation(0.1, 10.0, 0.7),
            create_test_evaluation(0.08, 8.0, 0.8),
            create_test_evaluation(0.05, 5.0, 0.85),
        ];

        for eval in evaluations {
            monitor.update_quality(&eval).unwrap();
        }

        let trend = monitor.quality_trend();
        assert!(trend > 0.0, "Quality trend should be positive (improving)");
    }

    #[test]
    fn test_custom_thresholds() {
        let custom_thresholds = QualityThresholds {
            max_mae: 0.05,
            max_mape: 5.0,
            min_r_squared: 0.9,
            quality_window: 10,
            enable_fallback: true,
            max_consecutive_failures: 2,
        };

        let mut monitor = QualityMonitor::with_thresholds(custom_thresholds);
        let evaluation = create_test_evaluation(0.1, 10.0, 0.8);

        monitor.update_quality(&evaluation).unwrap();

        // Should be unacceptable with stricter thresholds
        assert!(!monitor.is_quality_acceptable());
    }

    #[test]
    fn test_quality_report() {
        let mut monitor = QualityMonitor::new();

        // Add some evaluations
        let evaluations = vec![
            create_test_evaluation(0.1, 10.0, 0.8),
            create_test_evaluation(0.12, 12.0, 0.75),
            create_test_evaluation(0.3, 25.0, 0.3), // Bad one
        ];

        for eval in evaluations {
            monitor.update_quality(&eval).unwrap();
        }

        let report = monitor.quality_report();
        assert_eq!(report.total_evaluations, 3);
        assert_eq!(report.successful_evaluations, 2);
        assert!((report.success_rate - 2.0 / 3.0).abs() < 0.01);
        assert!(report.current_metrics.is_some());
    }
}
