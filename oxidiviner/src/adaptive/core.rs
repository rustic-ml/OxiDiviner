//! Core adaptive forecasting traits and structures
//!
//! This module defines the fundamental types and traits for adaptive forecasting
//! that build upon the existing OxiDiviner infrastructure.

use crate::api::{ForecastConfig, ModelType};
use crate::core::{TimeSeriesData, ForecastResult, Result, OxiError};
use crate::ensemble::EnsembleMethod;
use crate::models::regime_switching::MarkovSwitchingModel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for adaptive forecasting that extends the existing ForecastConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Base forecasting configuration from existing API
    pub base_config: ForecastConfig,
    
    /// Enable regime-aware parameter adaptation
    pub regime_adaptation: bool,
    
    /// Enable real-time parameter updates
    pub real_time_updates: bool,
    
    /// Enable adaptive ensemble weighting
    pub adaptive_ensemble: bool,
    
    /// Dynamic threshold configuration
    pub dynamic_threshold: bool,
    pub volatility_lookback: usize,
    pub volatility_multiplier: f64,
    pub min_threshold: f64,
    pub max_threshold: f64,
    
    /// Flexible model selection ranges
    pub model_selection: bool,
    pub p_range: (usize, usize),
    pub d_range: (usize, usize),
    pub q_range: (usize, usize),
    pub ic_criterion: InformationCriterion,
    
    /// Data quality control
    pub outlier_detection: bool,
    pub outlier_threshold: f64,
    pub seasonal_detection: bool,
    pub max_seasonal_period: usize,
    pub ensemble_models: usize,
    
    /// Real-time adaptation parameters
    pub refit_frequency: usize,
    pub rolling_window_size: usize,
    pub parameter_decay_rate: f64,
    pub learning_rate: f64,
    pub adaptation_threshold: f64,
}

/// Information criteria for model selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InformationCriterion {
    AIC,    // Akaike Information Criterion
    BIC,    // Bayesian Information Criterion
    HQIC,   // Hannan-Quinn Information Criterion
}

/// Regime state enum for market conditions
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum RegimeState {
    Bull,
    Bear,
    HighVolatility,
    LowVolatility,
    Consolidation,
    Transitional,
}

/// Model parameter set for different regimes
#[derive(Debug, Clone)]
pub struct ModelParamSet {
    pub arima_p: (usize, usize),
    pub arima_d: (usize, usize),
    pub arima_q: (usize, usize),
    pub es_alpha: (f64, f64),
    pub learning_rate: f64,
    pub ensemble_weights: HashMap<String, f64>,
}

/// Enhanced forecast result with adaptive information
#[derive(Debug)]
pub struct AdaptiveForecast {
    /// Base forecast result
    pub forecast_result: ForecastResult,
    
    /// Regime probabilities at forecast time
    pub regime_probability: HashMap<RegimeState, f64>,
    
    /// Model weights used in ensemble
    pub model_weights: HashMap<String, f64>,
    
    /// Quality score of the forecast
    pub quality_score: f64,
    
    /// Current regime state
    pub current_regime: RegimeState,
    
    /// Parameters used for this forecast
    pub parameters_used: HashMap<String, f64>,
    
    /// Adaptation metadata
    pub adaptation_info: AdaptationInfo,
}

/// Information about the adaptation process
#[derive(Debug)]
pub struct AdaptationInfo {
    pub last_refit_time: Option<chrono::DateTime<chrono::Utc>>,
    pub regime_changes: usize,
    pub parameter_updates: usize,
    pub model_switches: usize,
    pub performance_trend: f64,
}

/// Trait for models that support adaptive behavior
/// 
/// This trait extends the existing Forecaster trait with adaptive capabilities
pub trait AdaptiveForecaster: Send + Sync {
    /// Fit the model with adaptive configuration
    fn fit_adaptive(&mut self, data: &TimeSeriesData, config: &AdaptiveConfig) -> Result<()>;
    
    /// Generate adaptive forecast
    fn forecast_adaptive(&self, horizon: usize, config: &AdaptiveConfig) -> Result<AdaptiveForecast>;
    
    /// Update parameters with new data (real-time adaptation)
    fn update_parameters(&mut self, new_data: &[f64], regime: &RegimeState) -> Result<()>;
    
    /// Get parameters for specific regime
    fn get_regime_params(&self, regime: &RegimeState) -> ModelParamSet;
    
    /// Validate forecast quality
    fn validate_quality(&self, forecast: &AdaptiveForecast) -> Result<bool>;
    
    /// Get current adaptation state
    fn get_adaptation_info(&self) -> AdaptationInfo;
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            base_config: ForecastConfig::default(),
            regime_adaptation: false,
            real_time_updates: false,
            adaptive_ensemble: false,
            dynamic_threshold: false,
            volatility_lookback: 30,
            volatility_multiplier: 2.0,
            min_threshold: 0.01,
            max_threshold: 0.10,
            model_selection: false,
            p_range: (1, 3),
            d_range: (0, 2),
            q_range: (1, 3),
            ic_criterion: InformationCriterion::AIC,
            outlier_detection: false,
            outlier_threshold: 2.0,
            seasonal_detection: false,
            max_seasonal_period: 12,
            ensemble_models: 3,
            refit_frequency: 50,
            rolling_window_size: 100,
            parameter_decay_rate: 0.95,
            learning_rate: 0.01,
            adaptation_threshold: 0.05,
        }
    }
}

impl Default for ModelParamSet {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("ARIMA".to_string(), 0.4);
        weights.insert("ES".to_string(), 0.3);
        weights.insert("MA".to_string(), 0.3);
        
        Self {
            arima_p: (1, 3),
            arima_d: (1, 1),
            arima_q: (1, 3),
            es_alpha: (0.1, 0.5),
            learning_rate: 0.01,
            ensemble_weights: weights,
        }
    }
}

/// Utility functions for adaptive forecasting
pub struct AdaptiveUtils;

impl AdaptiveUtils {
    /// Calculate regime transition probability
    pub fn calculate_regime_transition_prob(
        current_regime: &RegimeState,
        data_window: &[f64],
        config: &AdaptiveConfig,
    ) -> HashMap<RegimeState, f64> {
        let mut probabilities = HashMap::new();
        
        // Simple volatility-based regime detection
        let volatility = Self::calculate_volatility(data_window);
        
        match current_regime {
            RegimeState::LowVolatility => {
                if volatility > config.max_threshold {
                    probabilities.insert(RegimeState::HighVolatility, 0.7);
                    probabilities.insert(RegimeState::LowVolatility, 0.3);
                } else {
                    probabilities.insert(RegimeState::LowVolatility, 0.8);
                    probabilities.insert(RegimeState::Consolidation, 0.2);
                }
            }
            RegimeState::HighVolatility => {
                if volatility < config.min_threshold {
                    probabilities.insert(RegimeState::LowVolatility, 0.6);
                    probabilities.insert(RegimeState::HighVolatility, 0.4);
                } else {
                    probabilities.insert(RegimeState::HighVolatility, 0.7);
                    probabilities.insert(RegimeState::Transitional, 0.3);
                }
            }
            _ => {
                // Default equal probabilities for other regimes
                probabilities.insert(RegimeState::Consolidation, 0.5);
                probabilities.insert(RegimeState::Transitional, 0.5);
            }
        }
        
        probabilities
    }
    
    /// Calculate volatility for regime detection
    pub fn calculate_volatility(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        
        let returns: Vec<f64> = data.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        
        variance.sqrt()
    }
    
    /// Optimize parameters for current regime
    pub fn optimize_for_regime(
        data: &TimeSeriesData,
        regime: &RegimeState,
        config: &AdaptiveConfig,
    ) -> Result<ModelParamSet> {
        // Use existing optimization framework with regime-specific constraints
        use crate::optimization::{OptimizerBuilder, OptimizationMethod, OptimizationMetric};
        
        let optimizer = OptimizerBuilder::new()
            .method(OptimizationMethod::GridSearch)
            .metric(OptimizationMetric::MAE)
            .max_evaluations(20)
            .build();
        
        // Get regime-specific parameter ranges
        let param_set = match regime {
            RegimeState::HighVolatility => {
                // Favor more responsive models in high volatility
                let mut params = ModelParamSet::default();
                params.es_alpha = (0.3, 0.7);
                params.arima_p = (1, 2);
                params
            }
            RegimeState::LowVolatility => {
                // Favor more stable models in low volatility
                let mut params = ModelParamSet::default();
                params.es_alpha = (0.1, 0.3);
                params.arima_p = (2, 4);
                params
            }
            _ => ModelParamSet::default(),
        };
        
        Ok(param_set)
    }
} 