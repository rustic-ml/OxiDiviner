use chrono::{DateTime, Utc};
#[cfg(feature = "ndarray_support")]
use ndarray::{Array1, ArrayView1};
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
        if timestamps.is_empty() || values.is_empty() {
            return Err(OxiError::DataError(
                "Timestamps and values cannot be empty".to_string(),
            ));
        }

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

        let mut train = self.slice(0, split_idx)?;
        let mut test = self.slice(split_idx, self.len())?;

        // Update names to follow the convention
        train.name = format!("{}_train", self.name);
        test.name = format!("{}_test", self.name);

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

    /// Create a new TimeSeriesData with a subset of the data from start_idx to end_idx
    pub fn slice(&self, start_idx: usize, end_idx: usize) -> Result<Self> {
        if start_idx >= end_idx || end_idx > self.len() {
            return Err(OxiError::InvalidParameter(
                "Invalid slice indices".to_string(),
            ));
        }

        Ok(TimeSeriesData {
            timestamps: self.timestamps[start_idx..end_idx].to_vec(),
            values: self.values[start_idx..end_idx].to_vec(),
            name: format!("{}_slice", self.name),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_time_series_data_creation() {
        let now = Utc::now();
        let timestamps = vec![
            now,
            now + chrono::Duration::days(1),
            now + chrono::Duration::days(2),
        ];
        let values = vec![1.0, 2.0, 3.0];

        // Test successful creation
        let ts = TimeSeriesData::new(timestamps.clone(), values.clone(), "test_series").unwrap();

        assert_eq!(ts.len(), 3);
        assert_eq!(ts.timestamps, timestamps);
        assert_eq!(ts.values, values);
        assert_eq!(ts.name, "test_series");

        // Test empty data - this should fail with an error
        let empty_result = TimeSeriesData::new(vec![], vec![], "empty");
        assert!(empty_result.is_err(), "Empty data should return an error");

        // Test mismatched lengths
        let mismatched_result = TimeSeriesData::new(timestamps, vec![1.0, 2.0], "mismatched");
        assert!(
            mismatched_result.is_err(),
            "Mismatched lengths should return an error"
        );
    }

    #[test]
    fn test_ohlcv_data_creation() {
        let now = Utc::now();
        let timestamps = vec![
            now,
            now + chrono::Duration::days(1),
            now + chrono::Duration::days(2),
        ];
        let open = vec![100.0, 101.0, 102.0];
        let high = vec![105.0, 106.0, 107.0];
        let low = vec![95.0, 96.0, 97.0];
        let close = vec![102.0, 103.0, 104.0];
        let volume = vec![1000.0, 1100.0, 1200.0];

        // Create a new OHLCV data object
        let mut ohlcv = OHLCVData::new("AAPL");

        // Manually populate it
        ohlcv.timestamps = timestamps.clone();
        ohlcv.open = open.clone();
        ohlcv.high = high.clone();
        ohlcv.low = low.clone();
        ohlcv.close = close.clone();
        ohlcv.volume = volume.clone();

        // Test the data is stored correctly
        assert_eq!(ohlcv.len(), 3);
        assert_eq!(ohlcv.symbol, "AAPL");
        assert_eq!(ohlcv.timestamps, timestamps);
        assert_eq!(ohlcv.open, open);
        assert_eq!(ohlcv.high, high);
        assert_eq!(ohlcv.low, low);
        assert_eq!(ohlcv.close, close);
        assert_eq!(ohlcv.volume, volume);
        assert!(ohlcv.adjusted_close.is_none());

        // Test with adjusted close
        let adjusted = vec![101.5, 102.5, 103.5];
        ohlcv.adjusted_close = Some(adjusted.clone());

        assert!(ohlcv.adjusted_close.is_some());
        assert_eq!(ohlcv.adjusted_close.unwrap(), adjusted);
    }

    #[test]
    fn test_ohlcv_to_time_series() {
        let now = Utc::now();
        let timestamps = vec![
            now,
            now + chrono::Duration::days(1),
            now + chrono::Duration::days(2),
        ];
        let open = vec![100.0, 101.0, 102.0];
        let high = vec![105.0, 106.0, 107.0];
        let low = vec![95.0, 96.0, 97.0];
        let close = vec![102.0, 103.0, 104.0];
        let volume = vec![1000.0, 1100.0, 1200.0];

        // Create and populate OHLCV data
        let mut ohlcv = OHLCVData::new("AAPL");
        ohlcv.timestamps = timestamps;
        ohlcv.open = open;
        ohlcv.high = high;
        ohlcv.low = low;
        ohlcv.close = close;
        ohlcv.volume = volume;

        // Test default to_time_series (using close prices)
        let ts = ohlcv.to_time_series(false);
        assert_eq!(ts.name, "AAPL close");
        assert_eq!(ts.values, ohlcv.close);

        // Add adjusted close and test with it
        let adjusted = vec![101.5, 102.5, 103.5];
        ohlcv.adjusted_close = Some(adjusted.clone());

        let ts_adjusted = ohlcv.to_time_series(true);
        assert_eq!(ts_adjusted.name, "AAPL close");
        assert_eq!(ts_adjusted.values, adjusted);
    }

    #[test]
    fn test_time_series_operations() {
        let now = Utc::now();
        let timestamps = vec![
            now,
            now + chrono::Duration::days(1),
            now + chrono::Duration::days(2),
            now + chrono::Duration::days(3),
            now + chrono::Duration::days(4),
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let ts = TimeSeriesData::new(timestamps, values, "test_series").unwrap();

        // Test train_test_split
        let (train, test) = ts.train_test_split(0.6).unwrap();
        assert_eq!(train.len(), 3); // 60% of 5 = 3
        assert_eq!(test.len(), 2);
        assert_eq!(train.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(test.values, vec![4.0, 5.0]);

        // Test invalid ratio
        let result = ts.train_test_split(0.0);
        assert!(result.is_err());

        let result = ts.train_test_split(1.0);
        assert!(result.is_err());
    }
}
