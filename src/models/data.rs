use chrono::{DateTime, Utc};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

// OHLCV time series data structure
pub struct OHLCVData {
    pub timestamps: Vec<DateTime<Utc>>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>, 
    pub volume: Vec<f64>,
    pub name: String,
}

impl OHLCVData {
    pub fn new(
        timestamps: Vec<DateTime<Utc>>,
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
        name: &str
    ) -> Result<Self, String> {
        let len = timestamps.len();
        
        // Check that all arrays have the same length
        if open.len() != len || high.len() != len || low.len() != len || 
           close.len() != len || volume.len() != len {
            return Err("All data arrays must have the same length".to_string());
        }
        
        Ok(OHLCVData {
            timestamps,
            open,
            high,
            low,
            close,
            volume,
            name: name.to_string(),
        })
    }
    
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }
    
    pub fn train_test_split(&self, train_ratio: f64) -> Result<(Self, Self), String> {
        if train_ratio <= 0.0 || train_ratio >= 1.0 {
            return Err("Train ratio must be between 0 and 1".to_string());
        }

        let split_idx = (self.len() as f64 * train_ratio).round() as usize;
        if split_idx == 0 || split_idx >= self.len() {
            return Err("Invalid split index".to_string());
        }

        let train = OHLCVData {
            timestamps: self.timestamps[0..split_idx].to_vec(),
            open: self.open[0..split_idx].to_vec(),
            high: self.high[0..split_idx].to_vec(),
            low: self.low[0..split_idx].to_vec(),
            close: self.close[0..split_idx].to_vec(),
            volume: self.volume[0..split_idx].to_vec(),
            name: format!("{}_train", self.name),
        };

        let test = OHLCVData {
            timestamps: self.timestamps[split_idx..].to_vec(),
            open: self.open[split_idx..].to_vec(),
            high: self.high[split_idx..].to_vec(),
            low: self.low[split_idx..].to_vec(),
            close: self.close[split_idx..].to_vec(),
            volume: self.volume[split_idx..].to_vec(),
            name: format!("{}_test", self.name),
        };

        Ok((train, test))
    }
    
    // Read OHLCV data from a CSV file
    pub fn from_csv<P: AsRef<Path>>(
        path: P, 
        name: &str,
        timestamp_format: &str,
        has_header: bool
    ) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        let mut timestamps = Vec::new();
        let mut open = Vec::new();
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();
        let mut volume = Vec::new();
        
        // Skip the header if needed
        let mut lines = reader.lines();
        if has_header {
            if let Some(Ok(_)) = lines.next() {
                // Header skipped
            }
        }
        
        for line in lines {
            let line = line?;
            let fields: Vec<&str> = line.split(',').collect();
            
            if fields.len() < 6 {
                return Err("CSV file must have at least 6 columns: date,open,high,low,close,volume".into());
            }
            
            // Parse timestamp
            let timestamp = match chrono::NaiveDateTime::parse_from_str(fields[0], timestamp_format) {
                Ok(dt) => DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc),
                Err(e) => return Err(format!("Error parsing date '{}': {}", fields[0], e).into()),
            };
            
            // Parse OHLCV values
            let open_val = fields[1].parse::<f64>()?;
            let high_val = fields[2].parse::<f64>()?;
            let low_val = fields[3].parse::<f64>()?;
            let close_val = fields[4].parse::<f64>()?;
            let volume_val = fields[5].parse::<f64>()?;
            
            timestamps.push(timestamp);
            open.push(open_val);
            high.push(high_val);
            low.push(low_val);
            close.push(close_val);
            volume.push(volume_val);
        }
        
        Ok(OHLCVData {
            timestamps,
            open,
            high,
            low,
            close,
            volume,
            name: name.to_string(),
        })
    }
} 