use chrono::{DateTime, Utc};
#[cfg(feature = "ndarray_support")]
use ndarray::{Array1, ArrayView1};
#[cfg(feature = "polars_integration")]
use polars::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(feature = "polars_integration")]
use std::path::Path;
#[cfg(feature = "polars_integration")]
use std::fs::File;

use crate::error::{OxiError, Result};

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
    #[cfg(feature = "polars_integration")]
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        
        let df = CsvReader::new(File::open(path.as_ref())
            .map_err(|e| OxiError::data_error(format!("Failed to open CSV {}: {}", path_str, e)))?)
            .has_header(true)
            .finish()
            .map_err(|e| OxiError::data_error(format!("Failed to parse CSV {}: {}", path_str, e)))?;

        Self::from_dataframe(df)
    }

    /// Load OHLCV data from a Parquet file
    #[cfg(feature = "polars_integration")]
    pub fn from_parquet<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file = File::open(path.as_ref())
            .map_err(|e| OxiError::data_error(format!("Failed to open parquet file {}: {}", path_str, e)))?;
        
        let df = ParquetReader::new(file)
            .finish()
            .map_err(|e| OxiError::data_error(format!("Failed to parse Parquet {}: {}", path_str, e)))?;

        Self::from_dataframe(df)
    }

    /// Convert a DataFrame to OHLCVData
    #[cfg(feature = "polars_integration")]
    fn from_dataframe(df: DataFrame) -> Result<Self> {
        // Check required columns
        let column_names = df.get_column_names();
        let required_cols = ["symbol", "open", "high", "low", "close", "volume"];
        for col in required_cols.iter() {
            if !column_names.iter().any(|&name| name == *col) {
                return Err(OxiError::data_error(format!("Missing required column: {}", col)));
            }
        }

        // Extract symbol (assuming all rows have the same symbol)
        let symbol_col = df.column("symbol").unwrap();
        let symbol = symbol_col.get(0).unwrap().to_string();

        // Extract timestamp column - assuming it's the second column after symbol
        let timestamp_col_name = df.get_column_names()[1];
        let timestamp_col = df.column(timestamp_col_name)
            .map_err(|e| OxiError::data_error(format!("Failed to get timestamp column: {}", e)))?;

        // Parse timestamps based on column type
        let timestamps = match timestamp_col.dtype() {
            DataType::String => {
                let str_series = timestamp_col.str()
                    .map_err(|e| OxiError::data_error(format!("Failed to get timestamp strings: {}", e)))?;
                
                str_series.into_iter()
                    .filter_map(|opt_s| {
                        opt_s.and_then(|s| {
                            NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S %Z")
                                .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
                                .map(|ndt| Utc.from_utc_datetime(&ndt))
                                .ok()
                        })
                    })
                    .collect()
            },
            DataType::Datetime(_, _) => {
                timestamp_col.datetime()
                    .map_err(|e| OxiError::data_error(format!("Failed to get datetime values: {}", e)))?
                    .into_iter()
                    .filter_map(|opt_i64| {
                        opt_i64.and_then(|i64_val| {
                            let seconds = i64_val / 1_000_000_000;
                            let nanos = i64_val % 1_000_000_000;
                            Utc.timestamp_opt(seconds, nanos as u32).single()
                        })
                    })
                    .collect()
            },
            dt => return Err(OxiError::data_error(format!("Unsupported timestamp data type: {:?}", dt))),
        };

        // Extract other numeric columns
        let extract_float_vec = |col_name: &str| -> Result<Vec<f64>> {
            df.column(col_name)
                .map_err(|e| OxiError::data_error(format!("Failed to get column {}: {}", col_name, e)))?
                .f64()
                .map_err(|e| OxiError::data_error(format!("Failed to convert {} to f64: {}", col_name, e)))?
                .into_iter()
                .map(|opt_val| opt_val.ok_or_else(|| OxiError::data_error(format!("Missing value in {} column", col_name))))
                .collect()
        };

        let open = extract_float_vec("open")?;
        let high = extract_float_vec("high")?;
        let low = extract_float_vec("low")?;
        let close = extract_float_vec("close")?;
        let volume = extract_float_vec("volume")?;
        
        // Adjusted close might not be present in all datasets
        let column_names = df.get_column_names();
        let adjusted_close = if column_names.iter().any(|name| name == "adjusted_close") {
            Some(extract_float_vec("adjusted_close")?)
        } else {
            None
        };

        Ok(OHLCVData {
            symbol,
            timestamps,
            open,
            high,
            low,
            close,
            volume,
            adjusted_close,
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
            return Err(OxiError::data_error("Timestamps and values must have the same length"));
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
            return Err(OxiError::invalid_params("Train ratio must be between 0 and 1"));
        }

        let split_idx = (self.len() as f64 * train_ratio).round() as usize;
        if split_idx == 0 || split_idx >= self.len() {
            return Err(OxiError::invalid_params("Invalid split index"));
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
} 