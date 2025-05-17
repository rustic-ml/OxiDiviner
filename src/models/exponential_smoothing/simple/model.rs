use crate::models::data::OHLCVData;
use crate::models::ModelEvaluation;

/// Simple Exponential Smoothing (SES) model for forecasting.
///
/// This model only has a level component and is suitable for forecasting time series 
/// without clear trend or seasonality. It's equivalent to ETS(A,N,N) in the ETS framework.
///
/// # Model Equation:
/// - Level: l_t = α * y_t + (1 - α) * l_{t-1}
/// - Forecast: ŷ_{t+h|t} = l_t
///
/// where:
/// - l_t is the level at time t
/// - y_t is the observed value at time t
/// - α is the smoothing parameter (0 < α < 1)
/// - ŷ_{t+h|t} is the h-step ahead forecast from time t
pub struct SESModel {
    /// Model name
    name: String,
    /// Smoothing parameter for level (0 < α < 1)
    alpha: f64,
    /// Current level value (after fitting)
    level: Option<f64>,
    /// Fitted values over the training period
    fitted_values: Option<Vec<f64>>,
    /// Which column from OHLCV data to use for prediction (defaults to close)
    target_column: TargetColumn,
}

/// Defines which price column to use for the SES model
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TargetColumn {
    /// Use the Open price
    Open,
    /// Use the High price 
    High,
    /// Use the Low price
    Low,
    /// Use the Close price (default)
    Close,
    /// Use the Volume
    Volume,
}

impl SESModel {
    /// Creates a new Simple Exponential Smoothing model.
    ///
    /// # Arguments
    /// * `alpha` - Smoothing parameter for level (0 < α < 1)
    /// * `target_column` - Which column of OHLCV data to use (defaults to Close)
    ///
    /// # Returns
    /// * `Result<Self, String>` - A new SES model if parameters are valid, or an error message if invalid
    ///
    /// # Examples
    /// ```
    /// use oxidiviner::models::exponential_smoothing::simple::SESModel;
    ///
    /// // Create a Simple Exponential Smoothing model with alpha = 0.3
    /// let model = SESModel::new(0.3, None).unwrap();
    /// ```
    pub fn new(
        alpha: f64,
        target_column: Option<TargetColumn>,
    ) -> Result<Self, String> {
        // Validate parameters
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err("Alpha must be between 0 and 1 exclusive".to_string());
        }
        
        // Default to close price if target column not specified
        let target_column = target_column.unwrap_or(TargetColumn::Close);
        
        // Create model name
        let column_name = match target_column {
            TargetColumn::Open => "Open",
            TargetColumn::High => "High",
            TargetColumn::Low => "Low",
            TargetColumn::Close => "Close",
            TargetColumn::Volume => "Volume",
        };
        
        let name = format!("SES({} | α={:.3})", column_name, alpha);
        
        Ok(SESModel {
            name,
            alpha,
            level: None,
            fitted_values: None,
            target_column,
        })
    }
    
    /// Extract the target data from OHLCV based on the selected column
    fn extract_target_data(&self, data: &OHLCVData) -> Vec<f64> {
        match self.target_column {
            TargetColumn::Open => data.open.clone(),
            TargetColumn::High => data.high.clone(),
            TargetColumn::Low => data.low.clone(),
            TargetColumn::Close => data.close.clone(),
            TargetColumn::Volume => data.volume.clone(),
        }
    }
    
    /// Fit the model to the provided OHLCV data.
    ///
    /// # Arguments
    /// * `data` - The OHLCV data to fit the model to
    ///
    /// # Returns
    /// * `Result<(), String>` - Ok if fitting is successful, or an error message if not
    pub fn fit(&mut self, data: &OHLCVData) -> Result<(), String> {
        if data.is_empty() {
            return Err("Cannot fit model on empty data".to_string());
        }
        
        let values = self.extract_target_data(data);
        let n = values.len();
        
        // Initialize the level with the first observation
        let mut level = values[0];
        
        // Prepare to store fitted values
        let mut fitted_values = Vec::with_capacity(n);
        fitted_values.push(level); // First fitted value is the initial level
        
        // Apply the SES model recursively
        for i in 1..n {
            // Calculate forecast for this step (which is just the previous level)
            let forecast = level;
            
            // Update the level based on the observed value
            level = self.alpha * values[i] + (1.0 - self.alpha) * level;
            
            // Store the forecast
            fitted_values.push(forecast);
        }
        
        // Store the final level and fitted values
        self.level = Some(level);
        self.fitted_values = Some(fitted_values);
        
        Ok(())
    }
    
    /// Generate forecasts for future periods.
    ///
    /// # Arguments
    /// * `horizon` - The number of periods to forecast ahead
    ///
    /// # Returns
    /// * `Result<Vec<f64>, String>` - Vector of forecasted values, or an error message if forecasting fails
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>, String> {
        if let Some(level) = self.level {
            // For SES, all future forecasts are just the last level
            Ok(vec![level; horizon])
        } else {
            Err("Model has not been fitted yet".to_string())
        }
    }
    
    /// Evaluate the model on test data.
    ///
    /// # Arguments
    /// * `test_data` - The test OHLCV data to evaluate against
    ///
    /// # Returns
    /// * `Result<ModelEvaluation, String>` - Evaluation metrics, or an error message if evaluation fails
    pub fn evaluate(&self, test_data: &OHLCVData) -> Result<ModelEvaluation, String> {
        if self.level.is_none() {
            return Err("Model has not been fitted yet".to_string());
        }
        
        let actual = self.extract_target_data(test_data);
        let forecast = self.forecast(actual.len())?;
        
        // Calculate error metrics
        let mae = mean_absolute_error(&actual, &forecast);
        let mse = mean_squared_error(&actual, &forecast);
        let rmse = mse.sqrt();
        let mape = mean_absolute_percentage_error(&actual, &forecast);
        let smape = symmetric_mean_absolute_percentage_error(&actual, &forecast);
        
        Ok(ModelEvaluation {
            model_name: self.name.clone(),
            mae,
            mse,
            rmse,
            mape,
            smape,
        })
    }
    
    /// Get the model name.
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the fitted values if available.
    pub fn fitted_values(&self) -> Option<&Vec<f64>> {
        self.fitted_values.as_ref()
    }
}

// Error metrics functions
fn mean_absolute_error(actual: &[f64], forecast: &[f64]) -> f64 {
    let n = actual.len().min(forecast.len());
    if n == 0 {
        return 0.0;
    }
    
    let mut sum = 0.0;
    for i in 0..n {
        sum += (actual[i] - forecast[i]).abs();
    }
    
    sum / n as f64
}

fn mean_squared_error(actual: &[f64], forecast: &[f64]) -> f64 {
    let n = actual.len().min(forecast.len());
    if n == 0 {
        return 0.0;
    }
    
    let mut sum = 0.0;
    for i in 0..n {
        let error = actual[i] - forecast[i];
        sum += error * error;
    }
    
    sum / n as f64
}

fn mean_absolute_percentage_error(actual: &[f64], forecast: &[f64]) -> f64 {
    let n = actual.len().min(forecast.len());
    if n == 0 {
        return 0.0;
    }
    
    let mut sum = 0.0;
    let mut count = 0;
    
    for i in 0..n {
        if actual[i] != 0.0 {
            sum += ((actual[i] - forecast[i]).abs() / actual[i].abs()) * 100.0;
            count += 1;
        }
    }
    
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

fn symmetric_mean_absolute_percentage_error(actual: &[f64], forecast: &[f64]) -> f64 {
    let n = actual.len().min(forecast.len());
    if n == 0 {
        return 0.0;
    }
    
    let mut sum = 0.0;
    let mut count = 0;
    
    for i in 0..n {
        let abs_actual = actual[i].abs();
        let abs_forecast = forecast[i].abs();
        if abs_actual + abs_forecast > 0.0 {
            sum += 200.0 * (abs_actual - abs_forecast).abs() / (abs_actual + abs_forecast);
            count += 1;
        }
    }
    
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
} 