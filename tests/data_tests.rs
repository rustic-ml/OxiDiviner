use chrono::{Duration, Utc};
use oxidiviner::OHLCVData;

#[test]
fn test_ohlcv_data_creation() {
    let now = Utc::now();
    let timestamps = vec![now, now + Duration::days(1), now + Duration::days(2)];
    let open = vec![100.0, 101.0, 102.0];
    let high = vec![105.0, 106.0, 107.0]; 
    let low = vec![98.0, 99.0, 100.0];
    let close = vec![103.0, 104.0, 105.0];
    let volume = vec![1000.0, 1100.0, 1200.0];
    
    let data = OHLCVData::new(
        timestamps,
        open,
        high,
        low,
        close,
        volume,
        Some("TEST".to_string())
    ).unwrap();
    
    assert_eq!(data.name(), Some("TEST"));
    assert!(!data.is_empty());
    assert_eq!(data.len(), 3);
}

#[test]
fn test_ohlcv_train_test_split() {
    let now = Utc::now();
    let timestamps = vec![
        now,
        now + Duration::days(1),
        now + Duration::days(2),
        now + Duration::days(3),
        now + Duration::days(4),
    ];
    
    let open = vec![100.0, 101.0, 102.0, 103.0, 104.0];
    let high = vec![105.0, 106.0, 107.0, 108.0, 109.0];
    let low = vec![98.0, 99.0, 100.0, 101.0, 102.0];
    let close = vec![103.0, 104.0, 105.0, 106.0, 107.0];
    let volume = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0];
    
    let data = OHLCVData::new(
        timestamps.clone(),
        open.clone(),
        high.clone(),
        low.clone(),
        close.clone(),
        volume.clone(),
        Some("TEST".to_string())
    ).unwrap();
    
    // Split with 60% training data
    let (train, test) = data.train_test_split(0.6).unwrap();
    
    // Check train data
    assert_eq!(train.len(), 3);
    assert_eq!(train.name(), Some("TEST_train"));
    // Check that close values are correct
    assert_eq!(train.close()[0], 103.0);
    assert_eq!(train.close()[1], 104.0);
    assert_eq!(train.close()[2], 105.0);
    
    // Check test data
    assert_eq!(test.len(), 2);
    assert_eq!(test.name(), Some("TEST_test"));
    // Check that close values are correct
    assert_eq!(test.close()[0], 106.0);
    assert_eq!(test.close()[1], 107.0);
}

#[test]
fn test_ohlcv_data_validation() {
    let now = Utc::now();
    let timestamps = vec![now, now + Duration::days(1)];
    let open = vec![100.0];  // Length mismatch
    let high = vec![105.0, 106.0];
    let low = vec![98.0, 99.0];
    let close = vec![103.0, 104.0];
    let volume = vec![1000.0, 1100.0];
    
    // This should fail because lengths don't match
    let result = OHLCVData::new(
        timestamps,
        open,
        high,
        low,
        close,
        volume,
        Some("TEST".to_string())
    );
    assert!(result.is_err());
    
    // Test invalid train ratio
    let good_data = OHLCVData::new(
        vec![now, now + Duration::days(1), now + Duration::days(2)],
        vec![100.0, 101.0, 102.0],
        vec![105.0, 106.0, 107.0],
        vec![98.0, 99.0, 100.0],
        vec![103.0, 104.0, 105.0],
        vec![1000.0, 1100.0, 1200.0],
        Some("TEST".to_string())
    ).unwrap();
    
    assert!(good_data.train_test_split(0.0).is_err());
    assert!(good_data.train_test_split(1.0).is_err());
} 