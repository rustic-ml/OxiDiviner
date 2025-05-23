# OxiDiviner Enhanced API - Separate Modules Summary

## Overview

This document summarizes the separate modules created to enhance OxiDiviner's accessibility as an API dependency. The modular design allows users to import and use specific functionality based on their needs.

## Module Structure

```
oxidiviner/src/
├── lib.rs              # Main library with module exports
├── api.rs              # High-level unified interface
├── financial.rs        # Financial time series specialization  
├── quick.rs            # One-line utility functions
├── batch.rs            # Batch processing for multiple series
├── core/               # Core library (existing)
├── models/             # Model implementations (existing)
└── math/               # Mathematical utilities (existing)
```

## 1. Financial Module (`financial.rs`)

### Purpose
Specialized functionality for financial time series analysis with domain-specific methods.

### Key Components

#### `FinancialTimeSeries` Struct
```rust
use oxidiviner::financial::FinancialTimeSeries;

// Create from price data
let financial_ts = FinancialTimeSeries::from_prices(
    timestamps,
    prices, 
    "AAPL"
)?;

// Calculate returns
let simple_returns = financial_ts.simple_returns()?;
let log_returns = financial_ts.log_returns()?;

// Automatic forecasting with financial defaults
let (forecast, model_used) = financial_ts.auto_forecast(5)?;

// Compare multiple models
let comparison = financial_ts.compare_models(5)?;
if let Some(best) = comparison.best() {
    println!("Best model: {}", best.name);
}
```

#### Features
- Price-to-returns calculations (simple and log returns)
- Financial-appropriate model selection priorities
- Model comparison with error metrics
- Symbol/asset identification

### Usage Patterns
```rust
// Basic usage
let ts = FinancialTimeSeries::from_prices(dates, prices, "STOCK")?;
let (forecast, model) = ts.auto_forecast(5)?;

// Advanced analysis
let returns = ts.simple_returns()?;
let comparison = ts.compare_models(10)?;
```

## 2. API Module (`api.rs`)

### Purpose
High-level unified interface providing consistent access to all forecasting models with configuration.

### Key Components

#### `Forecaster` and `ForecastBuilder`
```rust
use oxidiviner::api::{Forecaster, ForecastBuilder, ModelType};

// Builder pattern
let forecaster = ForecastBuilder::new()
    .arima(2, 1, 2)
    .build();

// Direct configuration
let forecaster = Forecaster::new()
    .model(ModelType::SimpleES)
    .alpha(0.4);

// Forecasting
let result = forecaster.forecast(&data, 5)?;
println!("Used {}: {:?}", result.model_used, result.forecast);
```

#### Configuration Types
- `ForecastConfig`: Complete forecasting configuration
- `ModelType`: Available model types (ARIMA, SimpleES, MovingAverage, Auto)
- `ModelParameters`: Model-specific parameters
- `ForecastOutput`: Comprehensive results with metadata

#### Features
- Unified interface for all models
- Builder pattern for easy configuration
- Automatic model selection
- Extensible configuration system
- Serializable configurations (JSON/YAML)

### Usage Patterns
```rust
// Quick auto-selection
let forecaster = ForecastBuilder::new().auto().build();
let result = forecaster.forecast(&data, 5)?;

// Specific model configuration
let forecaster = ForecastBuilder::new()
    .arima(1, 1, 1)
    .build();

// Custom parameters
let forecaster = Forecaster::new()
    .model(ModelType::SimpleES)
    .alpha(0.2);
```

## 3. Quick Module (`quick.rs`)

### Purpose
One-line utility functions for rapid prototyping and simple forecasting tasks.

### Key Functions

#### Basic Forecasting
```rust
use oxidiviner::quick;

// ARIMA with defaults (1,1,1)
let forecast = quick::arima_forecast(timestamps, values, 5)?;

// Exponential smoothing with default alpha (0.3)
let forecast = quick::es_forecast(timestamps, values, 5)?;

// Moving average with default window (5)
let forecast = quick::ma_forecast(timestamps, values, 5)?;

// Automatic model selection
let (forecast, model_used) = quick::auto_forecast(timestamps, values, 5)?;
```

#### Custom Parameters
```rust
// Custom ARIMA parameters
let forecast = quick::arima_forecast_custom(timestamps, values, 5, 2, 1, 2)?;

// Custom exponential smoothing alpha
let forecast = quick::es_forecast_custom(timestamps, values, 5, 0.2)?;

// Custom moving average window
let forecast = quick::ma_forecast_custom(timestamps, values, 5, 10)?;
```

#### Convenience Functions
```rust
// From values only (auto-generates timestamps)
let (forecast, model) = quick::values_only_forecast(values, 5)?;

// For daily financial data
let (forecast, model) = quick::daily_price_forecast(prices, 5)?;

// Model comparison
let comparisons = quick::compare_models(timestamps, values, 5)?;
for (model_name, forecast) in comparisons {
    println!("{}: {:?}", model_name, forecast);
}
```

### Usage Patterns
```rust
// Minimal usage
let (forecast, model) = quick::auto_forecast(timestamps, values, 5)?;

// Quick comparison
let models = quick::compare_models(timestamps, values, 5)?;

// Values-only forecasting
let (forecast, model) = quick::values_only_forecast(vec![1.0, 2.0, 3.0], 3)?;
```

## 4. Batch Module (`batch.rs`)

### Purpose
Batch processing for multiple time series simultaneously with parallel processing support.

### Key Components

#### `BatchTimeSeries`
```rust
use oxidiviner::batch::{BatchTimeSeries, BatchConfig, BatchModelType};

// Create batch
let mut batch = BatchTimeSeries::new();
batch.add_from_data("series1".to_string(), timestamps1, values1)?;
batch.add_from_data("series2".to_string(), timestamps2, values2)?;

// Or from arrays
let batch = BatchTimeSeries::from_data_arrays(vec![
    ("series1".to_string(), timestamps1, values1),
    ("series2".to_string(), timestamps2, values2),
])?;
```

#### Batch Processing
```rust
// Default configuration (auto-selection)
let results = batch.forecast(5)?;

// Custom configuration
let config = BatchConfig {
    forecast_periods: 10,
    parallel: true,
    model_type: Some(BatchModelType::ARIMA { p: 1, d: 1, q: 1 }),
    continue_on_error: true,
};
let results = batch.forecast_with_config(&config)?;

// Process results
for (name, forecast) in &results.forecasts {
    let model = results.models_used.get(name).unwrap();
    println!("{}: {} -> {:?}", name, model, forecast);
}
```

#### Batch Analysis
```rust
// Get summary statistics
let summary = batch.summary();
println!("Processing {} series", summary.total_series);
println!("Average length: {:.1}", summary.avg_length);

// Export results for further processing
let exported = batch.export_results(&results);
for (name, result) in exported {
    if result.success {
        println!("{}: Success with {}", name, result.model_used.unwrap());
    } else {
        println!("{}: Failed - {}", name, result.error.unwrap());
    }
}
```

### Usage Patterns
```rust
// Simple batch processing
let batch = BatchTimeSeries::from_data_arrays(data_arrays)?;
let results = batch.forecast(5)?;

// Custom batch configuration
let config = BatchConfig::default();
let results = batch.forecast_with_config(&config)?;

// Batch with specific model
let config = BatchConfig {
    model_type: Some(BatchModelType::SimpleES { alpha: 0.3 }),
    ..Default::default()
};
```

## Integration Examples

### 1. Progressive Workflow
```rust
use oxidiviner::{quick, financial, api, batch};

// Start with quick exploration
let (initial_forecast, model) = quick::auto_forecast(timestamps.clone(), values.clone(), 5)?;

// Move to financial analysis
let financial_ts = financial::FinancialTimeSeries::from_prices(timestamps, values, "STOCK")?;
let returns = financial_ts.simple_returns()?;

// Use API for precise configuration
let forecaster = api::ForecastBuilder::new().arima(2, 1, 2).build();
let precise_result = forecaster.forecast(&data, 5)?;

// Scale to batch processing
let mut batch = batch::BatchTimeSeries::new();
batch.add_from_data("original".to_string(), timestamps.clone(), values.clone())?;
// Add more series...
let batch_results = batch.forecast(5)?;
```

### 2. Model Comparison Across Modules
```rust
// Quick comparison
let quick_models = quick::compare_models(timestamps.clone(), values.clone(), 5)?;

// Financial comparison
let financial_ts = financial::FinancialTimeSeries::from_prices(timestamps, values, "TEST")?;
let financial_comparison = financial_ts.compare_models(5)?;

// API comparison
let forecasters = vec![
    api::ForecastBuilder::new().arima(1, 1, 1).build(),
    api::ForecastBuilder::new().simple_es(0.3).build(),
    api::ForecastBuilder::new().moving_average(5).build(),
];
// Compare each forecaster...
```

## Benefits of Modular Design

### 1. **Selective Import**
```rust
// Import only what you need
use oxidiviner::quick;                    // For rapid prototyping
use oxidiviner::financial;               // For financial analysis
use oxidiviner::api::{Forecaster};       // For production systems
use oxidiviner::batch::BatchTimeSeries;  // For bulk processing
```

### 2. **Clear Use Cases**
- **quick**: Rapid prototyping, exploration, one-off forecasts
- **financial**: Financial analysis, trading systems, portfolio management
- **api**: Production applications, consistent interfaces, configuration
- **batch**: Data processing pipelines, bulk forecasting, parallel processing

### 3. **Progressive Enhancement**
- Start with `quick` for exploration
- Move to `financial` for domain-specific analysis
- Use `api` for production applications
- Scale with `batch` for multiple series

### 4. **Consistent Error Handling**
All modules use the unified `oxidiviner::Result<T>` type for consistent error handling across the entire API.

### 5. **Documentation and Examples**
Each module includes comprehensive documentation, usage examples, and test cases demonstrating proper usage patterns.

## Next Steps

1. **Error Handling Improvements**: Unified error types across all modules
2. **Performance Optimization**: Parallel processing in batch module
3. **Extended Financial Features**: More financial-specific indicators and metrics
4. **Configuration Persistence**: Save/load forecaster configurations
5. **Streaming Interface**: Real-time data processing capabilities

## Conclusion

The modular design significantly improves OxiDiviner's accessibility as an API dependency by providing:

- **Multiple abstraction levels** for different use cases
- **Clear separation of concerns** between modules
- **Easy-to-use interfaces** for common tasks
- **Scalable processing capabilities** for production use
- **Domain-specific optimizations** for financial data

This structure makes OxiDiviner much more suitable for integration into larger projects like NyxsOwl while maintaining the flexibility to use only the required functionality. 