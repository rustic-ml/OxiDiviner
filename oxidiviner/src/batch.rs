//! Batch processing module
//!
//! This module provides functionality for processing multiple time series
//! simultaneously, enabling efficient parallel forecasting and batch operations.

use crate::core::{OxiError, Result, TimeSeriesData};
use crate::models::autoregressive::ARIMAModel;
use crate::models::exponential_smoothing::SimpleESModel;
use crate::models::moving_average::MAModel;
use chrono::{DateTime, Utc};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A collection of time series for batch processing
pub struct BatchTimeSeries {
    /// The time series data indexed by name
    series: HashMap<String, TimeSeriesData>,
}

/// Result of batch forecasting
#[derive(Debug, Clone)]
pub struct BatchForecastResult {
    /// Forecasts indexed by series name
    pub forecasts: HashMap<String, Vec<f64>>,
    /// Models used for each series
    pub models_used: HashMap<String, String>,
    /// Any errors that occurred during processing
    pub errors: HashMap<String, String>,
}

/// Configuration for batch processing
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of periods to forecast for each series
    pub forecast_periods: usize,
    /// Whether to use parallel processing
    pub parallel: bool,
    /// Model to use for all series (None for auto-selection)
    pub model_type: Option<BatchModelType>,
    /// Whether to continue processing if some series fail
    pub continue_on_error: bool,
}

/// Available models for batch processing
#[derive(Debug, Clone)]
pub enum BatchModelType {
    /// ARIMA with specified parameters
    ARIMA { p: usize, d: usize, q: usize },
    /// Simple Exponential Smoothing with alpha
    SimpleES { alpha: f64 },
    /// Moving Average with window size
    MovingAverage { window: usize },
    /// Automatic model selection
    Auto,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            forecast_periods: 5,
            parallel: true,
            model_type: Some(BatchModelType::Auto),
            continue_on_error: true,
        }
    }
}

impl BatchTimeSeries {
    /// Create a new empty batch
    pub fn new() -> Self {
        Self {
            series: HashMap::new(),
        }
    }

    /// Add a time series to the batch
    pub fn add_series(&mut self, name: String, data: TimeSeriesData) {
        self.series.insert(name, data);
    }

    /// Add multiple series from timestamps and values
    pub fn add_from_data(
        &mut self,
        name: String,
        timestamps: Vec<DateTime<Utc>>,
        values: Vec<f64>,
    ) -> Result<()> {
        let data = TimeSeriesData::new(timestamps, values, &name)?;
        self.add_series(name, data);
        Ok(())
    }

    /// Get the number of series in the batch
    pub fn len(&self) -> usize {
        self.series.len()
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }

    /// Get the names of all series
    pub fn series_names(&self) -> Vec<&String> {
        self.series.keys().collect()
    }

    /// Get a specific series by name
    pub fn get_series(&self, name: &str) -> Option<&TimeSeriesData> {
        self.series.get(name)
    }

    /// Remove a series from the batch
    pub fn remove_series(&mut self, name: &str) -> Option<TimeSeriesData> {
        self.series.remove(name)
    }

    /// Perform batch forecasting with default configuration
    pub fn forecast(&self, periods: usize) -> Result<BatchForecastResult> {
        let config = BatchConfig {
            forecast_periods: periods,
            ..Default::default()
        };
        self.forecast_with_config(&config)
    }

    /// Perform batch forecasting with custom configuration
    pub fn forecast_with_config(&self, config: &BatchConfig) -> Result<BatchForecastResult> {
        let mut forecasts = HashMap::new();
        let mut models_used = HashMap::new();
        let mut errors = HashMap::new();

        if config.parallel {
            self.forecast_parallel(config, &mut forecasts, &mut models_used, &mut errors)?;
        } else {
            self.forecast_sequential(config, &mut forecasts, &mut models_used, &mut errors)?;
        }

        Ok(BatchForecastResult {
            forecasts,
            models_used,
            errors,
        })
    }

    /// Sequential forecasting implementation
    fn forecast_sequential(
        &self,
        config: &BatchConfig,
        forecasts: &mut HashMap<String, Vec<f64>>,
        models_used: &mut HashMap<String, String>,
        errors: &mut HashMap<String, String>,
    ) -> Result<()> {
        for (name, data) in &self.series {
            match self.forecast_single(data, config) {
                Ok((forecast, model_name)) => {
                    forecasts.insert(name.clone(), forecast);
                    models_used.insert(name.clone(), model_name);
                }
                Err(e) => {
                    errors.insert(name.clone(), e.to_string());
                    if !config.continue_on_error {
                        return Err(e);
                    }
                }
            }
        }
        Ok(())
    }

    /// Parallel forecasting implementation using rayon
    fn forecast_parallel(
        &self,
        config: &BatchConfig,
        forecasts: &mut HashMap<String, Vec<f64>>,
        models_used: &mut HashMap<String, String>,
        errors: &mut HashMap<String, String>,
    ) -> Result<()> {
        // Convert series to a vector for parallel processing
        let series_vec: Vec<(String, &TimeSeriesData)> = self
            .series
            .iter()
            .map(|(name, data)| (name.clone(), data))
            .collect();

        // Use Arc<Mutex<>> for thread-safe accumulation of results
        let forecasts_mutex = Arc::new(Mutex::new(HashMap::new()));
        let models_used_mutex = Arc::new(Mutex::new(HashMap::new()));
        let errors_mutex = Arc::new(Mutex::new(HashMap::new()));
        let first_error = Arc::new(Mutex::new(None));

        // Process series in parallel
        series_vec
            .par_iter()
            .for_each(|(name, data)| match self.forecast_single(data, config) {
                Ok((forecast, model_name)) => {
                    forecasts_mutex
                        .lock()
                        .unwrap()
                        .insert(name.clone(), forecast);
                    models_used_mutex
                        .lock()
                        .unwrap()
                        .insert(name.clone(), model_name);
                }
                Err(e) => {
                    errors_mutex
                        .lock()
                        .unwrap()
                        .insert(name.clone(), e.to_string());
                    if !config.continue_on_error {
                        let mut first_err = first_error.lock().unwrap();
                        if first_err.is_none() {
                            *first_err = Some(e);
                        }
                    }
                }
            });

        // Check if we should return early due to error
        if let Some(err) = first_error.lock().unwrap().take() {
            return Err(err);
        }

        // Move results back to the output HashMaps
        let forecasts_result = forecasts_mutex.lock().unwrap();
        let models_used_result = models_used_mutex.lock().unwrap();
        let errors_result = errors_mutex.lock().unwrap();

        forecasts.extend(forecasts_result.clone());
        models_used.extend(models_used_result.clone());
        errors.extend(errors_result.clone());

        Ok(())
    }

    /// Forecast a single time series
    fn forecast_single(
        &self,
        data: &TimeSeriesData,
        config: &BatchConfig,
    ) -> Result<(Vec<f64>, String)> {
        match &config.model_type {
            Some(BatchModelType::ARIMA { p, d, q }) => {
                let mut model = ARIMAModel::new(*p, *d, *q, true)?;
                model.fit(data)?;
                let forecast = model.forecast(config.forecast_periods)?;
                Ok((forecast, format!("ARIMA({},{},{})", p, d, q)))
            }
            Some(BatchModelType::SimpleES { alpha }) => {
                let mut model = SimpleESModel::new(*alpha)?;
                model.fit(data)?;
                let forecast = model.forecast(config.forecast_periods)?;
                Ok((forecast, format!("SimpleES(α={})", alpha)))
            }
            Some(BatchModelType::MovingAverage { window }) => {
                let mut model = MAModel::new(*window)?;
                model.fit(data)?;
                let forecast = model.forecast(config.forecast_periods)?;
                Ok((forecast, format!("MA({})", window)))
            }
            Some(BatchModelType::Auto) | None => {
                self.auto_forecast_single(data, config.forecast_periods)
            }
        }
    }

    /// Automatic model selection for a single series
    fn auto_forecast_single(
        &self,
        data: &TimeSeriesData,
        periods: usize,
    ) -> Result<(Vec<f64>, String)> {
        // Try different models in order of preference
        // Use individual function calls instead of closures to avoid type issues

        // Try ARIMA first
        if let Ok(forecast) = self.try_arima_single(data, periods) {
            return Ok((forecast, "ARIMA(1,1,1)".to_string()));
        }

        // Try Simple ES next
        if let Ok(forecast) = self.try_es_single(data, periods) {
            return Ok((forecast, "SimpleES(α=0.3)".to_string()));
        }

        // Try Moving Average last
        if let Ok(forecast) = self.try_ma_single(data, periods) {
            return Ok((forecast, "MA(5)".to_string()));
        }

        Err(OxiError::ModelError(
            "All models failed for series".to_string(),
        )) // Use existing error type
    }

    // Helper methods for individual model attempts
    fn try_arima_single(&self, data: &TimeSeriesData, periods: usize) -> Result<Vec<f64>> {
        let mut model = ARIMAModel::new(1, 1, 1, true)?;
        model.fit(data)?;
        model.forecast(periods)
    }

    fn try_es_single(&self, data: &TimeSeriesData, periods: usize) -> Result<Vec<f64>> {
        let mut model = SimpleESModel::new(0.3)?;
        model.fit(data)?;
        model.forecast(periods)
    }

    fn try_ma_single(&self, data: &TimeSeriesData, periods: usize) -> Result<Vec<f64>> {
        let mut model = MAModel::new(5)?;
        model.fit(data)?;
        model.forecast(periods)
    }

    /// Create a batch from a vector of named time series
    pub fn from_series(series: Vec<(String, TimeSeriesData)>) -> Self {
        let mut batch = Self::new();
        for (name, data) in series {
            batch.add_series(name, data);
        }
        batch
    }

    /// Create a batch from multiple data arrays
    pub fn from_data_arrays(data: Vec<(String, Vec<DateTime<Utc>>, Vec<f64>)>) -> Result<Self> {
        let mut batch = Self::new();
        for (name, timestamps, values) in data {
            batch.add_from_data(name, timestamps, values)?;
        }
        Ok(batch)
    }

    /// Export batch results to a simple format
    pub fn export_results(
        &self,
        results: &BatchForecastResult,
    ) -> HashMap<String, BatchSeriesResult> {
        let mut exported = HashMap::new();

        for name in self.series_names() {
            let forecast = results.forecasts.get(name).cloned();
            let model_used = results.models_used.get(name).cloned();
            let error = results.errors.get(name).cloned();
            let success = forecast.is_some();

            exported.insert(
                name.clone(),
                BatchSeriesResult {
                    forecast,
                    model_used,
                    error,
                    success,
                },
            );
        }

        exported
    }

    /// Get summary statistics for the batch
    pub fn summary(&self) -> BatchSummary {
        let total_series = self.len();
        let series_lengths: Vec<usize> = self.series.values().map(|s| s.len()).collect();
        let avg_length = if !series_lengths.is_empty() {
            series_lengths.iter().sum::<usize>() as f64 / series_lengths.len() as f64
        } else {
            0.0
        };
        let min_length = series_lengths.iter().min().copied().unwrap_or(0);
        let max_length = series_lengths.iter().max().copied().unwrap_or(0);

        BatchSummary {
            total_series,
            avg_length,
            min_length,
            max_length,
            series_names: self.series_names().into_iter().cloned().collect(),
        }
    }
}

/// Result for a single series in batch processing
#[derive(Debug, Clone)]
pub struct BatchSeriesResult {
    /// The forecast (if successful)
    pub forecast: Option<Vec<f64>>,
    /// The model used (if successful)
    pub model_used: Option<String>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Whether the forecasting was successful
    pub success: bool,
}

/// Summary statistics for a batch
#[derive(Debug, Clone)]
pub struct BatchSummary {
    /// Total number of series
    pub total_series: usize,
    /// Average length of series
    pub avg_length: f64,
    /// Minimum series length
    pub min_length: usize,
    /// Maximum series length
    pub max_length: usize,
    /// Names of all series
    pub series_names: Vec<String>,
}

impl Default for BatchTimeSeries {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    fn create_test_data(name: &str, length: usize) -> (String, Vec<DateTime<Utc>>, Vec<f64>) {
        let start = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..length)
            .map(|i| start + Duration::days(i as i64))
            .collect();
        let values: Vec<f64> = (0..length)
            .map(|i| 100.0 + (i as f64) * 2.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();
        (name.to_string(), timestamps, values)
    }

    #[test]
    fn test_batch_creation() {
        let mut batch = BatchTimeSeries::new();
        assert!(batch.is_empty());

        let (name1, timestamps1, values1) = create_test_data("series1", 20);
        batch
            .add_from_data(name1.clone(), timestamps1, values1)
            .unwrap();

        let (name2, timestamps2, values2) = create_test_data("series2", 15);
        batch
            .add_from_data(name2.clone(), timestamps2, values2)
            .unwrap();

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
        assert!(batch.series_names().contains(&&name1));
        assert!(batch.series_names().contains(&&name2));
    }

    #[test]
    fn test_batch_forecasting() {
        let data = vec![
            create_test_data("series1", 20),
            create_test_data("series2", 15),
            create_test_data("series3", 25),
        ];

        let batch = BatchTimeSeries::from_data_arrays(data).unwrap();
        let results = batch.forecast(5).unwrap();

        assert_eq!(results.forecasts.len(), 3);
        assert_eq!(results.models_used.len(), 3);

        for forecast in results.forecasts.values() {
            assert_eq!(forecast.len(), 5);
        }
    }

    #[test]
    fn test_batch_summary() {
        let data = vec![
            create_test_data("short", 10),
            create_test_data("medium", 20),
            create_test_data("long", 30),
        ];

        let batch = BatchTimeSeries::from_data_arrays(data).unwrap();
        let summary = batch.summary();

        assert_eq!(summary.total_series, 3);
        assert_eq!(summary.min_length, 10);
        assert_eq!(summary.max_length, 30);
        assert_eq!(summary.avg_length, 20.0);
        assert_eq!(summary.series_names.len(), 3);
    }
}
