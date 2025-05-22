use chrono::{DateTime, Utc};
#[cfg(feature = "ndarray_support")]
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::OxiError;
use crate::Result;

/// Represents financial OHLCV (Open, High, Low, Close, Volume) data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVData {
    pub symbol: String,
    pub timestamps: Vec<DateTime<Utc>>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub adjusted_close: Option<Vec<f64>>,
}

impl OHLCVData {
    /// Create a new empty OHLCV dataset for a given symbol
    pub fn new(symbol: &str) -> Self {
        OHLCVData {
            symbol: symbol.to_string(),
            timestamps: Vec::new(),
            open: Vec::new(),
            high: Vec::new(),
            low: Vec::new(),
            close: Vec::new(),
            volume: Vec::new(),
            adjusted_close: None,
        }
    }

    /// Get the length of the data
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Check if the data is empty
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// Load OHLCV data from a CSV file
    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        timestamp_format: &str,
        has_header: bool,
    ) -> Result<Self> {
        let file = File::open(&path).map_err(OxiError::IoError)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Skip header if needed
        if has_header {
            if let Some(Ok(_)) = lines.next() {
                // Header skipped
            }
        }

        let mut symbol = path
            .as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let mut timestamps = Vec::new();
        let mut open = Vec::new();
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();
        let mut volume = Vec::new();

        for line in lines {
            let line = line.map_err(OxiError::IoError)?;
            let fields: Vec<&str> = line.split(',').collect();

            if fields.len() < 6 {
                return Err(OxiError::DataError(
                    "CSV file must have at least 6 columns: date,open,high,low,close,volume"
                        .to_string(),
                ));
            }

            // Parse timestamp
            let timestamp = match chrono::NaiveDateTime::parse_from_str(fields[0], timestamp_format)
            {
                Ok(dt) => DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc),
                Err(e) => {
                    return Err(OxiError::DataError(format!(
                        "Error parsing date '{}': {}",
                        fields[0], e
                    )))
                }
            };

            // Parse OHLCV values
            let open_val = fields[1]
                .parse::<f64>()
                .map_err(|e| OxiError::DataError(format!("Error parsing open: {}", e)))?;
            let high_val = fields[2]
                .parse::<f64>()
                .map_err(|e| OxiError::DataError(format!("Error parsing high: {}", e)))?;
            let low_val = fields[3]
                .parse::<f64>()
                .map_err(|e| OxiError::DataError(format!("Error parsing low: {}", e)))?;
            let close_val = fields[4]
                .parse::<f64>()
                .map_err(|e| OxiError::DataError(format!("Error parsing close: {}", e)))?;
            let volume_val = fields[5]
                .parse::<f64>()
                .map_err(|e| OxiError::DataError(format!("Error parsing volume: {}", e)))?;

            // If we have a 7th column, it might be the symbol or adjusted close
            if fields.len() > 6 && symbol == "unknown" {
                // Try to use the 7th field as the symbol
                symbol = fields[6].trim().to_string();
            }

            timestamps.push(timestamp);
            open.push(open_val);
            high.push(high_val);
            low.push(low_val);
            close.push(close_val);
            volume.push(volume_val);
        }

        Ok(OHLCVData {
            symbol,
            timestamps,
            open,
            high,
            low,
            close,
            volume,
            adjusted_close: None,
        })
    }

    /// Convert to a TimeSeriesData for forecasting
    pub fn to_time_series(&self, use_adjusted_close: bool) -> TimeSeriesData {
        let values = if use_adjusted_close && self.adjusted_close.is_some() {
            self.adjusted_close.as_ref().unwrap().clone()
        } else {
            self.close.clone()
        };

        TimeSeriesData {
            timestamps: self.timestamps.clone(),
            values,
            name: format!("{} close", self.symbol),
        }
    }
}

/// A general time series data structure used by forecasting models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesData {
    pub timestamps: Vec<DateTime<Utc>>,
    pub values: Vec<f64>,
    pub name: String,
}

impl TimeSeriesData {
    /// Create a new time series with given data
    pub fn new(timestamps: Vec<DateTime<Utc>>, values: Vec<f64>, name: &str) -> Result<Self> {
        if timestamps.len() != values.len() {
            return Err(OxiError::DataError(
                "Timestamps and values must have the same length".to_string(),
            ));
        }

        Ok(TimeSeriesData {
            timestamps,
            values,
            name: name.to_string(),
        })
    }

    /// Get the length of the time series
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the time series is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get the values as an ndarray Array1
    #[cfg(feature = "ndarray_support")]
    pub fn values_array(&self) -> Array1<f64> {
        Array1::from(self.values.clone())
    }

    /// Get the values as an ndarray ArrayView1
    #[cfg(feature = "ndarray_support")]
    pub fn values_view(&self) -> ArrayView1<f64> {
        ArrayView1::from(&self.values)
    }

    /// Split the time series into training and testing sets
    pub fn train_test_split(&self, train_ratio: f64) -> Result<(Self, Self)> {
        if train_ratio <= 0.0 || train_ratio >= 1.0 {
            return Err(OxiError::InvalidParameter(
                "Train ratio must be between 0 and 1".to_string(),
            ));
        }

        let split_idx = (self.len() as f64 * train_ratio).round() as usize;
        if split_idx == 0 || split_idx >= self.len() {
            return Err(OxiError::InvalidParameter(
                "Invalid split index".to_string(),
            ));
        }

        let train = TimeSeriesData {
            timestamps: self.timestamps[0..split_idx].to_vec(),
            values: self.values[0..split_idx].to_vec(),
            name: format!("{}_train", self.name),
        };

        let test = TimeSeriesData {
            timestamps: self.timestamps[split_idx..].to_vec(),
            values: self.values[split_idx..].to_vec(),
            name: format!("{}_test", self.name),
        };

        Ok((train, test))
    }

    /// Attempt to convert TimeSeriesData to OHLCVData
    /// Returns None if the data doesn't originate from OHLCV data
    pub fn as_ohlcv(&self) -> Option<&OHLCVData> {
        // This is a simplified implementation. In real code, you would
        // need to track whether this TimeSeriesData was created from OHLCVData
        // and store a reference to the original data.
        None
    }

    /// Create a TimeSeriesData from OHLCVData
    pub fn from_ohlcv(ohlcv: &OHLCVData, use_adjusted_close: bool) -> Self {
        ohlcv.to_time_series(use_adjusted_close)
    }
}
