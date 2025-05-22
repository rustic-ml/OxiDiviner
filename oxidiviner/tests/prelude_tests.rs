// This file tests that the prelude module exports all necessary types
// and that they can be used correctly.

// Just importing from the prelude should be sufficient for basic usage
use oxidiviner::prelude::*;

#[test]
fn test_prelude_contains_core_types() {
    // Verify we can create TimeSeriesData
    let _time_series = TimeSeriesData {
        timestamps: vec![chrono::Utc::now()],
        values: vec![1.0],
        name: "test".to_string(),
    };
    
    // Verify we can create OHLCVData
    let _ohlcv_data = OHLCVData {
        symbol: "AAPL".to_string(),
        timestamps: vec![chrono::Utc::now()],
        open: vec![100.0],
        high: vec![110.0],
        low: vec![90.0],
        close: vec![105.0],
        volume: vec![1000.0],
        adjusted_close: None,
    };
    
    // Verify we can create Result and OxiError
    let _result: Result<()> = Ok(());
    let _error = OxiError::InvalidParameter("test".to_string());
    
    // Verify model types are available
    let _model_output = ModelOutput {
        model_name: "TestModel".to_string(),
        forecasts: vec![1.0, 2.0, 3.0],
        evaluation: Some(ModelEvaluation {
            model_name: "TestModel".to_string(),
            mae: 1.0,
            rmse: 2.0,
            mape: 3.0,
            mse: 4.0,
            smape: 5.0,
        }),
    };
}

#[test]
fn test_prelude_contains_model_implementations() {
    // Verify we can use various model implementations from the prelude
    let _ma_model = MAModel::new(5);
    let _ses_model = SESModel::new(0.3);
    let _hw_model = HoltWintersModel::new(0.2, 0.1, 0.1, 7);
    let _ar_model = ARModel::new(3, true);
    
    // Additional models if needed
    // let _arma_model = ARMAModel::new(2, 1);
    // let _garch_model = GARCHModel::new(1, 1);
} 