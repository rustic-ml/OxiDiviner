//! Streaming data processing for large time series
//!
//! This module provides memory-efficient streaming capabilities for processing
//! large time series datasets that don't fit in memory.

use crate::core::{OxiError, Result, TimeSeriesData};
use chrono::{DateTime, Utc};
use std::collections::VecDeque;

/// Streaming buffer for time series data
pub struct StreamingBuffer {
    /// Maximum buffer size
    max_size: usize,
    /// Current buffer of values
    values: VecDeque<f64>,
    /// Current buffer of timestamps
    timestamps: VecDeque<DateTime<Utc>>,
    /// Total number of points processed
    total_processed: usize,
    /// Running statistics
    running_stats: RunningStats,
}

/// Running statistics for streaming data
#[derive(Debug, Clone)]
pub struct RunningStats {
    /// Count of processed points
    pub count: usize,
    /// Running mean
    pub mean: f64,
    /// Running variance (using Welford's algorithm)
    pub variance: f64,
    /// Minimum value seen
    pub min: f64,
    /// Maximum value seen
    pub max: f64,
    /// Sum of values
    pub sum: f64,
    /// Sum of squared values
    pub sum_squares: f64,
}

impl Default for RunningStats {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            variance: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            sum_squares: 0.0,
        }
    }
}

impl RunningStats {
    /// Update statistics with a new value
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_squares += value * value;

        // Update min/max
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }

        // Welford's algorithm for running mean and variance
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.variance += delta * delta2;
    }

    /// Get the sample variance
    pub fn sample_variance(&self) -> f64 {
        if self.count > 1 {
            self.variance / (self.count - 1) as f64
        } else {
            0.0
        }
    }

    /// Get the standard deviation
    pub fn std_dev(&self) -> f64 {
        self.sample_variance().sqrt()
    }
}

impl StreamingBuffer {
    /// Create a new streaming buffer
    pub fn new(max_size: usize) -> Result<Self> {
        if max_size == 0 {
            return Err(OxiError::InvalidParameter(
                "Buffer size must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            max_size,
            values: VecDeque::with_capacity(max_size),
            timestamps: VecDeque::with_capacity(max_size),
            total_processed: 0,
            running_stats: RunningStats::default(),
        })
    }

    /// Add a new data point to the buffer
    pub fn push(&mut self, timestamp: DateTime<Utc>, value: f64) -> Result<()> {
        // Validate the value
        if !value.is_finite() {
            return Err(OxiError::InvalidParameter(
                "Value must be finite".to_string(),
            ));
        }

        // If buffer is at max capacity, remove oldest
        if self.values.len() >= self.max_size {
            self.values.pop_front();
            self.timestamps.pop_front();
        }

        // Add new value
        self.values.push_back(value);
        self.timestamps.push_back(timestamp);
        self.total_processed += 1;

        // Update running statistics
        self.running_stats.update(value);

        Ok(())
    }

    /// Get the current buffer as a TimeSeriesData
    pub fn get_current_data(&self, name: &str) -> Result<TimeSeriesData> {
        if self.values.is_empty() {
            return Err(OxiError::InvalidParameter("Buffer is empty".to_string()));
        }

        let timestamps: Vec<DateTime<Utc>> = self.timestamps.iter().cloned().collect();
        let values: Vec<f64> = self.values.iter().cloned().collect();

        TimeSeriesData::new(timestamps, values, name)
    }

    /// Get the current buffer size
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get total number of points processed
    pub fn total_processed(&self) -> usize {
        self.total_processed
    }

    /// Get running statistics
    pub fn stats(&self) -> &RunningStats {
        &self.running_stats
    }

    /// Get the most recent values (up to n)
    pub fn get_recent_values(&self, n: usize) -> Vec<f64> {
        let start = if n >= self.values.len() {
            0
        } else {
            self.values.len() - n
        };

        self.values.iter().skip(start).cloned().collect()
    }

    /// Clear the buffer but keep statistics
    pub fn clear_buffer(&mut self) {
        self.values.clear();
        self.timestamps.clear();
    }

    /// Reset everything including statistics
    pub fn reset(&mut self) {
        self.values.clear();
        self.timestamps.clear();
        self.total_processed = 0;
        self.running_stats = RunningStats::default();
    }
}

/// Streaming processor for time series models
pub struct StreamingProcessor {
    /// Buffer for incoming data
    buffer: StreamingBuffer,
    /// Window size for model fitting
    window_size: usize,
    /// Update frequency (fit model every N points)
    update_frequency: usize,
    /// Points since last model update
    points_since_update: usize,
}

impl StreamingProcessor {
    /// Create a new streaming processor
    pub fn new(buffer_size: usize, window_size: usize, update_frequency: usize) -> Result<Self> {
        if window_size > buffer_size {
            return Err(OxiError::InvalidParameter(
                "Window size cannot be larger than buffer size".to_string(),
            ));
        }

        if update_frequency == 0 {
            return Err(OxiError::InvalidParameter(
                "Update frequency must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            buffer: StreamingBuffer::new(buffer_size)?,
            window_size,
            update_frequency,
            points_since_update: 0,
        })
    }

    /// Process a new data point
    pub fn process_point(
        &mut self,
        timestamp: DateTime<Utc>,
        value: f64,
    ) -> Result<Option<TimeSeriesData>> {
        // Add to buffer
        self.buffer.push(timestamp, value)?;
        self.points_since_update += 1;

        // Check if we should trigger model update
        if self.points_since_update >= self.update_frequency
            && self.buffer.len() >= self.window_size
        {
            self.points_since_update = 0;

            // Get windowed data for model fitting
            let recent_values = self.buffer.get_recent_values(self.window_size);
            let recent_timestamps: Vec<DateTime<Utc>> = self
                .buffer
                .timestamps
                .iter()
                .rev()
                .take(self.window_size)
                .rev()
                .cloned()
                .collect();

            let windowed_data =
                TimeSeriesData::new(recent_timestamps, recent_values, "streaming_window")?;

            return Ok(Some(windowed_data));
        }

        Ok(None)
    }

    /// Get current buffer statistics
    pub fn get_stats(&self) -> &RunningStats {
        self.buffer.stats()
    }

    /// Get current buffer length
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Get total processed count
    pub fn total_processed(&self) -> usize {
        self.buffer.total_processed()
    }
}

/// Batch iterator for processing large datasets in chunks
pub struct BatchIterator<I>
where
    I: Iterator<Item = (DateTime<Utc>, f64)>,
{
    source: I,
    batch_size: usize,
    buffer: Vec<(DateTime<Utc>, f64)>,
}

impl<I> BatchIterator<I>
where
    I: Iterator<Item = (DateTime<Utc>, f64)>,
{
    /// Create a new batch iterator
    pub fn new(source: I, batch_size: usize) -> Self {
        Self {
            source,
            batch_size,
            buffer: Vec::with_capacity(batch_size),
        }
    }
}

impl<I> Iterator for BatchIterator<I>
where
    I: Iterator<Item = (DateTime<Utc>, f64)>,
{
    type Item = Result<TimeSeriesData>;

    fn next(&mut self) -> Option<Self::Item> {
        self.buffer.clear();

        // Fill buffer with next batch
        for _ in 0..self.batch_size {
            if let Some((timestamp, value)) = self.source.next() {
                self.buffer.push((timestamp, value));
            } else {
                break;
            }
        }

        if self.buffer.is_empty() {
            return None;
        }

        // Convert buffer to TimeSeriesData
        let (timestamps, values): (Vec<_>, Vec<_>) = self.buffer.iter().cloned().unzip();

        match TimeSeriesData::new(timestamps, values, "batch") {
            Ok(data) => Some(Ok(data)),
            Err(e) => Some(Err(e)),
        }
    }
}

/// Create a batch iterator from any iterator of timestamp-value pairs
pub fn batch_process<I>(source: I, batch_size: usize) -> BatchIterator<I>
where
    I: Iterator<Item = (DateTime<Utc>, f64)>,
{
    BatchIterator::new(source, batch_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_streaming_buffer() {
        let mut buffer = StreamingBuffer::new(3).unwrap();
        let start_time = Utc::now();

        // Add some data points
        for i in 0..5 {
            let timestamp = start_time + Duration::seconds(i);
            let value = i as f64;
            buffer.push(timestamp, value).unwrap();
        }

        // Buffer should only contain last 3 points
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.total_processed(), 5);

        // Check recent values
        let recent = buffer.get_recent_values(2);
        assert_eq!(recent, vec![3.0, 4.0]);
    }

    #[test]
    fn test_running_stats() {
        let mut buffer = StreamingBuffer::new(10).unwrap();
        let start_time = Utc::now();

        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        for (i, &value) in values.iter().enumerate() {
            let timestamp = start_time + Duration::seconds(i as i64);
            buffer.push(timestamp, value).unwrap();
        }

        let stats = buffer.stats();
        assert_eq!(stats.count, 5);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.sum, 15.0);
    }

    #[test]
    fn test_streaming_processor() {
        let mut processor = StreamingProcessor::new(10, 5, 3).unwrap();
        let start_time = Utc::now();

        let mut update_count = 0;

        // Process 10 data points
        for i in 0..10 {
            let timestamp = start_time + Duration::seconds(i);
            let value = i as f64;

            if let Some(_windowed_data) = processor.process_point(timestamp, value).unwrap() {
                update_count += 1;
            }
        }

        assert!(update_count > 0);
        assert_eq!(processor.total_processed(), 10);
    }

    #[test]
    fn test_batch_iterator() {
        let start_time = Utc::now();
        let data: Vec<(DateTime<Utc>, f64)> = (0..10)
            .map(|i| (start_time + Duration::seconds(i), i as f64))
            .collect();

        let batches: Vec<_> = batch_process(data.into_iter(), 3)
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(batches.len(), 4); // 10 items in batches of 3 = 4 batches
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 3);
        assert_eq!(batches[2].len(), 3);
        assert_eq!(batches[3].len(), 1); // Last batch has remainder
    }
}
