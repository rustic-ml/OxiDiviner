//! Model persistence functionality
//!
//! This module provides serialization and deserialization capabilities for trained models,
//! allowing them to be saved to and loaded from disk.

use crate::core::{OxiError, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

/// Serializable model state for ARIMA models
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ARIMAModelState {
    pub name: String,
    pub p: usize,
    pub d: usize,
    pub q: usize,
    pub include_intercept: bool,
    pub ar_coefficients: Option<Vec<f64>>,
    pub ma_coefficients: Option<Vec<f64>>,
    pub intercept: Option<f64>,
    pub last_values: Option<Vec<f64>>,
}

/// Serializable model state for AR models
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ARModelState {
    pub name: String,
    pub p: usize,
    pub include_intercept: bool,
    pub intercept: Option<f64>,
    pub coefficients: Option<Vec<f64>>,
    pub last_values: Option<Vec<f64>>,
    pub mean: Option<f64>,
}

/// Serializable model state for Simple Exponential Smoothing models
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SimpleESModelState {
    pub name: String,
    pub alpha: f64,
    pub level: Option<f64>,
}

/// Serializable model state for ETS models
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ETSModelState {
    pub name: String,
    pub error_type: String,
    pub trend_type: String,
    pub seasonal_type: String,
    pub alpha: f64,
    pub beta: Option<f64>,
    pub phi: Option<f64>,
    pub gamma: Option<f64>,
    pub period: Option<usize>,
    pub level: Option<f64>,
    pub trend: Option<f64>,
    pub seasonal: Option<Vec<f64>>,
}

/// Enum representing different model types that can be persisted
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ModelState {
    ARIMA(ARIMAModelState),
    AR(ARModelState),
    SimpleES(SimpleESModelState),
    ETS(ETSModelState),
}

/// Container for model persistence with metadata
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PersistedModel {
    /// Model state
    pub model: ModelState,
    /// Timestamp when the model was saved
    pub saved_at: String,
    /// Version of the library
    pub version: String,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// Trait for models that can be persisted
pub trait Persistable {
    /// Save the model state to a JSON string
    fn to_json(&self) -> Result<String>;

    /// Save the model to a file
    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()>;

    /// Get the model state for serialization
    fn get_model_state(&self) -> ModelState;
}

/// Utility functions for model persistence
pub struct ModelPersistence;

impl ModelPersistence {
    /// Save a model state to a file
    pub fn save_model<P: AsRef<Path>>(model_state: ModelState, path: P) -> Result<()> {
        let persisted = PersistedModel {
            model: model_state,
            saved_at: chrono::Utc::now().to_rfc3339(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            metadata: std::collections::HashMap::new(),
        };

        let file = File::create(path)
            .map_err(|e| OxiError::ModelError(format!("Failed to create file: {}", e)))?;

        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &persisted)
            .map_err(|e| OxiError::ModelError(format!("Failed to serialize model: {}", e)))?;

        writer
            .flush()
            .map_err(|e| OxiError::ModelError(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Load a model state from a file
    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<PersistedModel> {
        let file = File::open(path)
            .map_err(|e| OxiError::ModelError(format!("Failed to open file: {}", e)))?;

        let reader = BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(|e| OxiError::ModelError(format!("Failed to deserialize model: {}", e)))
    }

    /// Save a model state to a JSON string
    pub fn to_json(model_state: ModelState) -> Result<String> {
        let persisted = PersistedModel {
            model: model_state,
            saved_at: chrono::Utc::now().to_rfc3339(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            metadata: std::collections::HashMap::new(),
        };

        serde_json::to_string_pretty(&persisted)
            .map_err(|e| OxiError::ModelError(format!("Failed to serialize to JSON: {}", e)))
    }

    /// Load a model state from a JSON string
    pub fn from_json(json: &str) -> Result<PersistedModel> {
        serde_json::from_str(json)
            .map_err(|e| OxiError::ModelError(format!("Failed to deserialize from JSON: {}", e)))
    }

    /// Create a model metadata entry
    pub fn create_metadata() -> std::collections::HashMap<String, String> {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "library_name".to_string(),
            env!("CARGO_PKG_NAME").to_string(),
        );
        metadata.insert(
            "library_version".to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
        );
        metadata.insert("saved_by".to_string(), "oxidiviner".to_string());
        metadata
    }

    /// Validate that a persisted model is compatible with current library version
    pub fn validate_compatibility(persisted: &PersistedModel) -> Result<()> {
        let current_version = env!("CARGO_PKG_VERSION");

        // For now, just check that the version exists
        if persisted.version.is_empty() {
            return Err(OxiError::ModelError(
                "Model was saved with unknown library version".to_string(),
            ));
        }

        // In a production system, you'd implement proper version compatibility checking
        if persisted.version != current_version {
            eprintln!(
                "Warning: Model was saved with version {}, current version is {}",
                persisted.version, current_version
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_state_serialization() {
        let ar_state = ARModelState {
            name: "AR(2)".to_string(),
            p: 2,
            include_intercept: true,
            intercept: Some(0.5),
            coefficients: Some(vec![0.3, 0.2]),
            last_values: Some(vec![1.0, 2.0]),
            mean: Some(1.5),
        };

        let model_state = ModelState::AR(ar_state);
        let json = ModelPersistence::to_json(model_state).unwrap();

        assert!(json.contains("AR"));
        assert!(json.contains("0.3"));
        assert!(json.contains("0.2"));
    }

    #[test]
    fn test_model_roundtrip() {
        let arima_state = ARIMAModelState {
            name: "ARIMA(1,1,1)".to_string(),
            p: 1,
            d: 1,
            q: 1,
            include_intercept: true,
            ar_coefficients: Some(vec![0.7]),
            ma_coefficients: Some(vec![0.3]),
            intercept: Some(0.1),
            last_values: Some(vec![5.0, 6.0]),
        };

        let model_state = ModelState::ARIMA(arima_state.clone());

        // Serialize to JSON
        let json = ModelPersistence::to_json(model_state).unwrap();

        // Deserialize back
        let loaded = ModelPersistence::from_json(&json).unwrap();

        // Verify compatibility
        ModelPersistence::validate_compatibility(&loaded).unwrap();

        // Check that the core data is preserved
        if let ModelState::ARIMA(loaded_arima) = loaded.model {
            assert_eq!(loaded_arima.name, arima_state.name);
            assert_eq!(loaded_arima.p, arima_state.p);
            assert_eq!(loaded_arima.d, arima_state.d);
            assert_eq!(loaded_arima.q, arima_state.q);
            assert_eq!(loaded_arima.ar_coefficients, arima_state.ar_coefficients);
            assert_eq!(loaded_arima.ma_coefficients, arima_state.ma_coefficients);
        } else {
            panic!("Model type mismatch after deserialization");
        }
    }

    #[test]
    fn test_persistence_metadata() {
        let metadata = ModelPersistence::create_metadata();
        assert!(metadata.contains_key("library_name"));
        assert!(metadata.contains_key("library_version"));
        assert!(metadata.contains_key("saved_by"));
        assert_eq!(metadata.get("library_name").unwrap(), "oxidiviner");
    }
}
