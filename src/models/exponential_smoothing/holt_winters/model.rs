use crate::models::data::OHLCVData;
use crate::models::ModelEvaluation;

/// Holt-Winters Seasonal model for forecasting.
///
/// This model has level, trend, and seasonal components and is suitable for forecasting 
/// time series with both trend and seasonality.
/// It's equivalent to ETS(A,A,A) or ETS(A,A,M) in the ETS framework, depending on the seasonal type.
///
/// # Model Equations (Additive Seasonality):
/// - Level: l_t = α * (y_t - s_{t-m}) + (1 - α) * (l_{t-1} + b_{t-1})
/// - Trend: b_t = β * (l_t - l_{t-1}) + (1 - β) * b_{t-1}
/// - Seasonal: s_t = γ * (y_t - l_t) + (1 - γ) * s_{t-m}
/// - Forecast: ŷ_{t+h|t} = l_t + h * b_t + s_{t-m+h_m}
///
/// # Model Equations (Multiplicative Seasonality):
/// - Level: l_t = α * (y_t / s_{t-m}) + (1 - α) * (l_{t-1} + b_{t-1})
/// - Trend: b_t = β * (l_t - l_{t-1}) + (1 - β) * b_{t-1}
/// - Seasonal: s_t = γ * (y_t / l_t) + (1 - γ) * s_{t-m}
/// - Forecast: ŷ_{t+h|t} = (l_t + h * b_t) * s_{t-m+h_m}
///
/// where:
/// - l_t is the level at time t
/// - b_t is the trend at time t
/// - s_t is the seasonal component at time t
/// - y_t is the observed value at time t
/// - m is the seasonal period
/// - h_m is h modulo m (keeps the seasonal index within the correct range)
/// - α is the smoothing parameter for level (0 < α < 1)
/// - β is the smoothing parameter for trend (0 < β < 1)
/// - γ is the smoothing parameter for seasonality (0 < γ < 1)
/// - ŷ_{t+h|t} is the h-step ahead forecast from time t
pub struct HoltWintersModel {
    /// Model name
    name: String,
    /// Smoothing parameter for level (0 < α < 1)
    alpha: f64,
    /// Smoothing parameter for trend (0 < β < 1)
    beta: f64,
    /// Smoothing parameter for seasonality (0 < γ < 1)
    gamma: f64,
    /// Damping parameter (0 < φ < 1), if damped trend is used
    phi: Option<f64>,
    /// Number of periods in a seasonal cycle
    seasonal_period: usize,
    /// Whether to use multiplicative seasonality (false = additive)
    multiplicative_seasonality: bool,
    /// Current level value (after fitting)
    level: Option<f64>,
    /// Current trend value (after fitting)
    trend: Option<f64>,
    /// Current seasonal components (after fitting)
    seasonal: Option<Vec<f64>>,
    /// Fitted values over the training period
    fitted_values: Option<Vec<f64>>,
    /// Which column from OHLCV data to use for prediction (defaults to close)
    target_column: TargetColumn,
}

/// Defines which price column to use for the Holt-Winters model
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

/// Seasonal component type for Holt-Winters model
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeasonalType {
    /// Additive seasonality
    Additive,
    /// Multiplicative seasonality
    Multiplicative,
}

impl HoltWintersModel {
    /// Creates a new Holt-Winters Seasonal model.
    ///
    /// # Arguments
    /// * `alpha` - Smoothing parameter for level (0 < α < 1)
    /// * `beta` - Smoothing parameter for trend (0 < β < 1)
    /// * `gamma` - Smoothing parameter for seasonality (0 < γ < 1)
    /// * `phi` - Optional damping parameter (0 < φ < 1)
    /// * `seasonal_period` - Number of periods in a seasonal cycle (e.g., 7 for weekly)
    /// * `seasonal_type` - Type of seasonality (Additive or Multiplicative)
    /// * `target_column` - Which column of OHLCV data to use (defaults to Close)
    ///
    /// # Returns
    /// * `Result<Self, String>` - A new Holt-Winters model if parameters are valid, or an error message if invalid
    ///
    /// # Examples
    /// ```
    /// use oxidiviner::models::exponential_smoothing::holt_winters::{HoltWintersModel, SeasonalType};
    ///
    /// // Create a Holt-Winters Seasonal model with weekly seasonality
    /// let model = HoltWintersModel::new(
    ///     0.3,                      // alpha = 0.3
    ///     0.1,                      // beta = 0.1
    ///     0.1,                      // gamma = 0.1
    ///     None,                     // No damping
    ///     7,                        // Weekly seasonality
    ///     SeasonalType::Additive,   // Additive seasonality
    ///     None,                     // Default to Close price
    /// ).unwrap();
    /// ```
    pub fn new(
        alpha: f64,
        beta: f64,
        gamma: f64,
        phi: Option<f64>,
        seasonal_period: usize,
        seasonal_type: SeasonalType,
        target_column: Option<TargetColumn>,
    ) -> Result<Self, String> {
        // Validate parameters
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err("Alpha must be between 0 and 1 exclusive".to_string());
        }
        
        if beta <= 0.0 || beta >= 1.0 {
            return Err("Beta must be between 0 and 1 exclusive".to_string());
        }
        
        if gamma <= 0.0 || gamma >= 1.0 {
            return Err("Gamma must be between 0 and 1 exclusive".to_string());
        }
        
        if let Some(phi_val) = phi {
            if phi_val <= 0.0 || phi_val >= 1.0 {
                return Err("Phi must be between 0 and 1 exclusive".to_string());
            }
        }
        
        if seasonal_period < 2 {
            return Err("Seasonal period must be at least 2".to_string());
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
        
        let seasonal_str = match seasonal_type {
            SeasonalType::Additive => "Additive",
            SeasonalType::Multiplicative => "Multiplicative",
        };
        
        let model_type = if phi.is_some() { "Damped" } else { "" };
        let name = format!(
            "Holt-Winters {} {}({} | m={}, α={:.3}, β={:.3}, γ={:.3}{})",
            model_type,
            seasonal_str,
            column_name,
            seasonal_period,
            alpha,
            beta,
            gamma,
            phi.map_or(String::new(), |p| format!(", φ={:.3}", p)),
        );
        
        Ok(HoltWintersModel {
            name,
            alpha,
            beta,
            gamma,
            phi,
            seasonal_period,
            multiplicative_seasonality: seasonal_type == SeasonalType::Multiplicative,
            level: None,
            trend: None,
            seasonal: None,
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
        
        // Ensure enough data points for initialization
        if n < 2 * self.seasonal_period {
            return Err(format!(
                "Need at least {} data points to fit Holt-Winters model with period {}",
                2 * self.seasonal_period,
                self.seasonal_period
            ));
        }
        
        // Initialize level, trend, and seasonal components
        // For initialization, we use the first two complete seasons
        
        // 1. Initialize seasonal components
        let mut seasonal = vec![0.0; self.seasonal_period];
        
        if self.multiplicative_seasonality {
            // For multiplicative seasonality, find average values for each position in the cycle
            for i in 0..self.seasonal_period {
                let mut sum = 0.0;
                let mut count = 0;
                
                for j in 0..(2 * self.seasonal_period) {
                    let idx = i + j * self.seasonal_period;
                    if idx < n {
                        sum += values[idx];
                        count += 1;
                    }
                }
                
                if count > 0 {
                    seasonal[i] = sum / count as f64;
                }
            }
            
            // Normalize seasonal factors to average 1.0
            let seasonal_avg = seasonal.iter().sum::<f64>() / self.seasonal_period as f64;
            for i in 0..self.seasonal_period {
                seasonal[i] /= seasonal_avg;
            }
        } else {
            // For additive seasonality, subtract overall average from cycle averages
            let global_avg = values.iter().sum::<f64>() / n as f64;
            
            for i in 0..self.seasonal_period {
                let mut sum = 0.0;
                let mut count = 0;
                
                for j in 0..(2 * self.seasonal_period) {
                    let idx = i + j * self.seasonal_period;
                    if idx < n {
                        sum += values[idx];
                        count += 1;
                    }
                }
                
                if count > 0 {
                    seasonal[i] = sum / count as f64 - global_avg;
                }
            }
        }
        
        // 2. Initialize level and trend
        let mut level;
        let mut trend;
        
        if self.multiplicative_seasonality {
            // For multiplicative seasonality, divide by seasonal factor
            let deseasonalized_first = values[0] / seasonal[0];
            let deseasonalized_second = values[self.seasonal_period] / seasonal[0];
            
            level = deseasonalized_first;
            trend = (deseasonalized_second - deseasonalized_first) / self.seasonal_period as f64;
        } else {
            // For additive seasonality, subtract seasonal component
            let deseasonalized_first = values[0] - seasonal[0];
            let deseasonalized_second = values[self.seasonal_period] - seasonal[0];
            
            level = deseasonalized_first;
            trend = (deseasonalized_second - deseasonalized_first) / self.seasonal_period as f64;
        }
        
        // Prepare to store fitted values
        let mut fitted_values = Vec::with_capacity(n);
        fitted_values.push(values[0]); // First fitted value matches the first observation
        
        // Apply the Holt-Winters model recursively
        for i in 1..n {
            // Get the appropriate seasonal index
            let s_idx = (i % self.seasonal_period) as usize;
            let s_idx_prev = ((i - 1) % self.seasonal_period) as usize;
            
            // Calculate forecast for this step
            let forecast = if self.multiplicative_seasonality {
                // Multiplicative seasonality
                let trend_component = if let Some(phi) = self.phi {
                    phi * trend
                } else {
                    trend
                };
                (level + trend_component) * seasonal[s_idx_prev]
            } else {
                // Additive seasonality
                let trend_component = if let Some(phi) = self.phi {
                    phi * trend
                } else {
                    trend
                };
                level + trend_component + seasonal[s_idx_prev]
            };
            
            // Update components based on the observed value
            let level_prev = level;
            if self.multiplicative_seasonality {
                // Multiplicative seasonality updates
                level = self.alpha * (values[i] / seasonal[s_idx_prev]) + 
                       (1.0 - self.alpha) * (level_prev + trend);
                
                seasonal[s_idx] = self.gamma * (values[i] / level) + 
                                 (1.0 - self.gamma) * seasonal[s_idx_prev];
            } else {
                // Additive seasonality updates
                level = self.alpha * (values[i] - seasonal[s_idx_prev]) + 
                       (1.0 - self.alpha) * (level_prev + trend);
                
                seasonal[s_idx] = self.gamma * (values[i] - level) + 
                                 (1.0 - self.gamma) * seasonal[s_idx_prev];
            }
            
            // Update trend, with damping if specified
            if let Some(phi) = self.phi {
                trend = self.beta * (level - level_prev) + (1.0 - self.beta) * phi * trend;
            } else {
                trend = self.beta * (level - level_prev) + (1.0 - self.beta) * trend;
            }
            
            // Store the forecast
            fitted_values.push(forecast);
        }
        
        // Store the final parameters and fitted values
        self.level = Some(level);
        self.trend = Some(trend);
        self.seasonal = Some(seasonal);
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
        if let (Some(level), Some(trend), Some(seasonal)) = (self.level, self.trend, self.seasonal.as_ref()) {
            let mut forecasts = Vec::with_capacity(horizon);
            
            for h in 1..=horizon {
                // Calculate the trend component with damping if applicable
                let trend_component = if let Some(phi) = self.phi {
                    trend * (1.0 - phi.powi(h as i32)) / (1.0 - phi)
                } else {
                    trend * h as f64
                };
                
                // Get the appropriate seasonal index
                let s_idx = ((h - 1) % self.seasonal_period) as usize;
                
                // Calculate forecast based on seasonality type
                let forecast = if self.multiplicative_seasonality {
                    // Multiplicative seasonality
                    (level + trend_component) * seasonal[s_idx]
                } else {
                    // Additive seasonality
                    level + trend_component + seasonal[s_idx]
                };
                
                forecasts.push(forecast);
            }
            
            Ok(forecasts)
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
        if self.level.is_none() || self.trend.is_none() || self.seasonal.is_none() {
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
    
    /// Get the current level value if available.
    pub fn level(&self) -> Option<f64> {
        self.level
    }
    
    /// Get the current trend value if available.
    pub fn trend(&self) -> Option<f64> {
        self.trend
    }
    
    /// Get the current seasonal components if available.
    pub fn seasonal(&self) -> Option<&Vec<f64>> {
        self.seasonal.as_ref()
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