use crate::models::data::OHLCVData;
use super::{ETSComponent, component_to_char, ModelEvaluation, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error};

/// ETS model implementation for daily OHLCV data.
///
/// This implements Exponential Smoothing State Space models for daily stock price data.
/// The ETS model framework supports various combinations of Error, Trend, and Seasonal components.
///
/// # Error Component Types:
/// - Additive (A): Errors are added to the forecast
/// - Multiplicative (M): Errors are multiplied by the forecast
///
/// # Trend Component Types:
/// - None (N): No trend component
/// - Additive (A): Linear trend is added to the level
/// - Multiplicative (M): Trend is multiplied with the level (exponential growth)
/// - Damped (D): Trend effect diminishes over time
///
/// # Seasonal Component Types:
/// - None (N): No seasonality
/// - Additive (A): Seasonal effects are added to the trend and level
/// - Multiplicative (M): Seasonal effects are multiplied with the trend and level
///
/// # Model Parameters:
/// - Alpha (α): Level smoothing parameter (0 < α < 1)
/// - Beta (β): Trend smoothing parameter (0 < β < 1), used when trend is present
/// - Gamma (γ): Seasonal smoothing parameter (0 < γ < 1), used when seasonality is present
/// - Phi (φ): Damping parameter (0 < φ < 1), used with damped trend
pub struct ETSModel {
    name: String,
    error_type: ETSComponent,
    trend_type: ETSComponent,
    seasonal_type: ETSComponent,
    alpha: f64,
    beta: Option<f64>,
    gamma: Option<f64>,
    phi: Option<f64>,
    seasonal_period: Option<usize>,
    level: Option<f64>,
    trend: Option<f64>,
    seasonal: Option<Vec<f64>>,
    fitted_values: Option<Vec<f64>>,
    /// Which column from OHLCV data to use for prediction (defaults to close)
    target_column: TargetColumn,
}

/// Defines which price column to use for the ETS model
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

impl ETSModel {
    /// Creates a new ETS model with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `error_type` - The type of error component (Additive or Multiplicative)
    /// * `trend_type` - The type of trend component (None, Additive, Multiplicative, or Damped)
    /// * `seasonal_type` - The type of seasonal component (None, Additive, or Multiplicative)
    /// * `alpha` - Level smoothing parameter (0 < α < 1)
    /// * `beta` - Trend smoothing parameter (0 < β < 1), required if trend_type is not None
    /// * `gamma` - Seasonal smoothing parameter (0 < γ < 1), required if seasonal_type is not None
    /// * `phi` - Damping parameter (0 < φ < 1), required if trend_type is Damped
    /// * `seasonal_period` - The number of periods in a seasonal cycle, required if seasonal_type is not None
    /// * `target_column` - Which column of OHLCV data to use (defaults to Close)
    ///
    /// # Returns
    ///
    /// * `Result<Self, String>` - A new ETS model if parameters are valid, or an error message if invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use oxdiviner::models::ets::{ETSComponent, DailyETSModel};
    ///
    /// // Create a simple exponential smoothing model (ETS(A,N,N))
    /// let model = DailyETSModel::new(
    ///     ETSComponent::Additive,  // Error type
    ///     ETSComponent::None,      // No trend
    ///     ETSComponent::None,      // No seasonality
    ///     0.3,                     // alpha = 0.3
    ///     None,                    // No beta (no trend)
    ///     None,                    // No gamma (no seasonality)
    ///     None,                    // No phi (no damping)
    ///     None,                    // No seasonal period
    ///     None,                    // Default to Close price
    /// ).unwrap();
    /// ```
    pub fn new(
        error_type: ETSComponent,
        trend_type: ETSComponent,
        seasonal_type: ETSComponent,
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
        seasonal_period: Option<usize>,
        target_column: Option<TargetColumn>,
    ) -> Result<Self, String> {
        // Validate parameters
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err("Alpha must be between 0 and 1 exclusive".to_string());
        }
        
        if let Some(beta) = beta {
            if beta <= 0.0 || beta >= 1.0 {
                return Err("Beta must be between 0 and 1 exclusive".to_string());
            }
        }
        
        if let Some(gamma) = gamma {
            if gamma <= 0.0 || gamma >= 1.0 {
                return Err("Gamma must be between 0 and 1 exclusive".to_string());
            }
        }
        
        if let Some(phi) = phi {
            if phi <= 0.0 || phi >= 1.0 {
                return Err("Phi must be between 0 and 1 exclusive".to_string());
            }
        }
        
        // Check compatibility of parameters with model specification
        if trend_type == ETSComponent::None && beta.is_some() {
            return Err("Beta should not be specified for models without trend".to_string());
        }
        
        if seasonal_type == ETSComponent::None && gamma.is_some() {
            return Err("Gamma should not be specified for models without seasonality".to_string());
        }
        
        if trend_type != ETSComponent::Damped && phi.is_some() {
            return Err("Phi should only be specified for models with damped trend".to_string());
        }
        
        if seasonal_type != ETSComponent::None && seasonal_period.is_none() {
            return Err("Seasonal period must be specified for seasonal models".to_string());
        }
        
        if let Some(period) = seasonal_period {
            if period < 2 {
                return Err("Seasonal period must be at least 2".to_string());
            }
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
        
        let name = format!(
            "Daily ETS({},{},{}, {}| α={:.3}{}{}{})",
            component_to_char(error_type),
            component_to_char(trend_type),
            component_to_char(seasonal_type),
            column_name,
            alpha,
            beta.map_or(String::new(), |b| format!(", β={:.3}", b)),
            gamma.map_or(String::new(), |g| format!(", γ={:.3}", g)),
            phi.map_or(String::new(), |p| format!(", φ={:.3}", p)),
        );
        
        Ok(ETSModel {
            name,
            error_type,
            trend_type,
            seasonal_type,
            alpha,
            beta,
            gamma,
            phi,
            seasonal_period,
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
    
    /// Fits the model to the given OHLCV data.
    ///
    /// This function initializes the model state variables (level, trend, seasonal components)
    /// and then performs iterative one-step forecasts and state updates to learn the
    /// underlying patterns in the data.
    ///
    /// # Arguments
    ///
    /// * `data` - The OHLCV data to fit the model to
    ///
    /// # Returns
    ///
    /// * `Result<(), String>` - Success or an error message
    ///
    /// # Examples
    ///
    /// ```
    /// use oxdiviner::models::ets::{ETSComponent, DailyETSModel};
    /// use oxdiviner::ModelsOHLCVData;
    ///
    /// // Assume we have some data
    /// let data = ModelsOHLCVData::new(...);
    ///
    /// // Create and fit a model
    /// let mut model = DailyETSModel::new(
    ///     ETSComponent::Additive, ETSComponent::None, ETSComponent::None,
    ///     0.3, None, None, None, None, None
    /// ).unwrap();
    ///
    /// model.fit(&data).unwrap();
    /// ```
    pub fn fit(&mut self, data: &OHLCVData) -> Result<(), String> {
        if data.is_empty() {
            return Err("Cannot fit model on empty data".to_string());
        }
        
        let values = self.extract_target_data(data);
        let n = values.len();
        
        // Initialize components
        self.initialize(&values)?;
        
        // Storage for fitted values
        let mut fitted = Vec::with_capacity(n);
        
        let m = self.seasonal_period.unwrap_or(1);
        
        // First m values are used for initialization
        let mut level = self.level.unwrap();
        let mut trend = self.trend;
        let mut seasonal = self.seasonal.clone();
        
        for i in 0..n {
            // Calculate forecast for this time step
            let forecast = self.forecast_one_step(level, trend, 
                seasonal.as_ref().map(|s| s[i % m]))?;
            
            // Store fitted value
            fitted.push(forecast);
            
            // Update state based on current observation
            let error = match self.error_type {
                ETSComponent::Additive => values[i] - forecast,
                ETSComponent::Multiplicative => if forecast != 0.0 { values[i] / forecast - 1.0 } else { 0.0 },
                _ => return Err("Invalid error type".to_string()),
            };
            
            // Update level
            level = match self.trend_type {
                ETSComponent::None => level + self.alpha * error,
                ETSComponent::Additive => level + trend.unwrap() + self.alpha * error,
                ETSComponent::Multiplicative => level * trend.unwrap() + self.alpha * error,
                ETSComponent::Damped => level + self.phi.unwrap() * trend.unwrap() + self.alpha * error,
            };
            
            // Update trend if present
            if self.trend_type != ETSComponent::None {
                let beta = self.beta.unwrap();
                let trend_val = trend.unwrap();
                
                trend = Some(match self.trend_type {
                    ETSComponent::Additive => trend_val + beta * self.alpha * error,
                    ETSComponent::Multiplicative => trend_val * (1.0 + beta * self.alpha * error),
                    ETSComponent::Damped => self.phi.unwrap() * trend_val + beta * self.alpha * error,
                    ETSComponent::None => 0.0, // This shouldn't happen based on the earlier check
                });
            }
            
            // Update seasonal component if present
            if self.seasonal_type != ETSComponent::None {
                let gamma = self.gamma.unwrap();
                let mut seas = seasonal.unwrap();
                let old_season = seas[i % m];
                
                match self.seasonal_type {
                    ETSComponent::Additive => {
                        seas[i % m] = old_season + gamma * self.alpha * error;
                    },
                    ETSComponent::Multiplicative => {
                        seas[i % m] = old_season * (1.0 + gamma * self.alpha * error);
                    },
                    ETSComponent::None => (), // No action needed
                    ETSComponent::Damped => (), // Not applicable to seasonal components
                }
                
                seasonal = Some(seas);
            }
        }
        
        // Store final state
        self.level = Some(level);
        self.trend = trend;
        self.seasonal = seasonal;
        self.fitted_values = Some(fitted);
        
        Ok(())
    }
    
    /// Generates forecasts for the specified horizon.
    ///
    /// # Arguments
    ///
    /// * `horizon` - The number of periods to forecast
    ///
    /// # Returns
    ///
    /// * `Result<Vec<f64>, String>` - The forecasted values or an error message
    ///
    /// # Examples
    ///
    /// ```
    /// // Continuing from the fit example above:
    /// let forecasts = model.forecast(30).unwrap(); // 30-day forecast
    /// ```
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>, String> {
        if horizon == 0 {
            return Ok(Vec::new());
        }
        
        // Check if model has been fit
        let _level = self.level.ok_or_else(|| "Model has not been fit to data yet".to_string())?;
        
        let mut forecasts = Vec::with_capacity(horizon);
        
        for h in 1..=horizon {
            forecasts.push(self.forecast_h_step(h)?);
        }
        
        Ok(forecasts)
    }
    
    /// Generate a forecast h steps ahead
    fn forecast_h_step(&self, h: usize) -> Result<f64, String> {
        let _level = self.level.ok_or_else(|| "Model has not been fit to data yet".to_string())?;
        
        if self.trend_type == ETSComponent::None && self.seasonal_type == ETSComponent::None {
            return Ok(_level);
        }
        
        // Calculate trend component
        let trend_component = if self.trend_type != ETSComponent::None {
            let trend = self.trend.unwrap();
            
            match self.trend_type {
                ETSComponent::Additive => h as f64 * trend,
                ETSComponent::Multiplicative => trend.powi(h as i32) - 1.0,
                ETSComponent::Damped => {
                    let phi = self.phi.unwrap();
                    let mut sum = 0.0;
                    let mut phi_pow = 1.0;
                    for _ in 0..h {
                        sum += phi_pow;
                        phi_pow *= phi;
                    }
                    trend * sum
                },
                ETSComponent::None => 0.0,
            }
        } else {
            0.0
        };
        
        // Calculate seasonal component
        let seasonal_component = if self.seasonal_type != ETSComponent::None {
            let seasonal = self.seasonal.as_ref().unwrap();
            let m = self.seasonal_period.unwrap();
            let season_idx = ((h - 1) % m) as usize;
            seasonal[season_idx]
        } else {
            match self.seasonal_type {
                ETSComponent::Additive => 0.0,
                ETSComponent::Multiplicative => 1.0,
                ETSComponent::None => 0.0,
                _ => 0.0,
            }
        };
        
        // Combine components
        let forecast = match (self.trend_type, self.seasonal_type) {
            (ETSComponent::None, ETSComponent::None) => _level,
            (ETSComponent::None, ETSComponent::Additive) => _level + seasonal_component,
            (ETSComponent::None, ETSComponent::Multiplicative) => _level * seasonal_component,
            (ETSComponent::Additive, ETSComponent::None) => _level + trend_component,
            (ETSComponent::Multiplicative, ETSComponent::None) => _level * (1.0 + trend_component),
            (ETSComponent::Damped, ETSComponent::None) => _level + trend_component,
            (ETSComponent::Additive, ETSComponent::Additive) => _level + trend_component + seasonal_component,
            (ETSComponent::Additive, ETSComponent::Multiplicative) => (_level + trend_component) * seasonal_component,
            (ETSComponent::Multiplicative, ETSComponent::Additive) => _level * (1.0 + trend_component) + seasonal_component,
            (ETSComponent::Multiplicative, ETSComponent::Multiplicative) => _level * (1.0 + trend_component) * seasonal_component,
            (ETSComponent::Damped, ETSComponent::Additive) => _level + trend_component + seasonal_component,
            (ETSComponent::Damped, ETSComponent::Multiplicative) => (_level + trend_component) * seasonal_component,
            _ => unreachable!(),
        };
        
        Ok(forecast)
    }
    
    /// Generate one-step forecast from given state values
    fn forecast_one_step(&self, state_level: f64, state_trend: Option<f64>, state_season: Option<f64>) -> Result<f64, String> {
        match (self.trend_type, self.seasonal_type) {
            (ETSComponent::None, ETSComponent::None) => {
                // Simple Exponential Smoothing
                Ok(state_level)
            },
            (ETSComponent::None, _) => {
                // Seasonal model without trend
                match self.seasonal_type {
                    ETSComponent::Additive => Ok(state_level + state_season.unwrap()),
                    ETSComponent::Multiplicative => Ok(state_level * state_season.unwrap()),
                    _ => Err("Invalid seasonal type".to_string()),
                }
            },
            (_, ETSComponent::None) => {
                // Trend model without seasonality
                let trend = state_trend.unwrap();
                match self.trend_type {
                    ETSComponent::Additive => Ok(state_level + trend),
                    ETSComponent::Multiplicative => Ok(state_level * trend),
                    ETSComponent::Damped => Ok(state_level + self.phi.unwrap() * trend),
                    _ => Err("Invalid trend type".to_string()),
                }
            },
            (_, _) => {
                // Model with both trend and seasonality
                let trend = state_trend.unwrap();
                let season = state_season.unwrap();
                
                match (self.trend_type, self.seasonal_type) {
                    (ETSComponent::Additive, ETSComponent::Additive) => {
                        Ok(state_level + trend + season)
                    },
                    (ETSComponent::Additive, ETSComponent::Multiplicative) => {
                        Ok((state_level + trend) * season)
                    },
                    (ETSComponent::Multiplicative, ETSComponent::Additive) => {
                        Ok(state_level * trend + season)
                    },
                    (ETSComponent::Multiplicative, ETSComponent::Multiplicative) => {
                        Ok(state_level * trend * season)
                    },
                    (ETSComponent::Damped, ETSComponent::Additive) => {
                        Ok(state_level + self.phi.unwrap() * trend + season)
                    },
                    (ETSComponent::Damped, ETSComponent::Multiplicative) => {
                        Ok((state_level + self.phi.unwrap() * trend) * season)
                    },
                    _ => Err("Invalid combination of trend and seasonal types".to_string()),
                }
            }
        }
    }
    
    /// Initialize the model components from data
    fn initialize(&mut self, values: &[f64]) -> Result<(), String> {
        let n = values.len();
        let m = self.seasonal_period.unwrap_or(1);
        
        // Make sure we have enough data for seasonal models
        if self.seasonal_type != ETSComponent::None && n < 2 * m {
            return Err(format!("Need at least {} observations for seasonal model", 2 * m));
        }
        
        // Initialize level
        let initial_level = match self.seasonal_type {
            ETSComponent::None => values[0],
            _ => {
                // Average of first m observations
                let mut sum = 0.0;
                for i in 0..m {
                    sum += values[i];
                }
                sum / m as f64
            },
        };
        self.level = Some(initial_level);
        
        // Initialize trend
        if self.trend_type != ETSComponent::None {
            let initial_trend = match self.trend_type {
                ETSComponent::Additive | ETSComponent::Damped => {
                    // Average trend over first m periods
                    let mut sum = 0.0;
                    for i in 0..m {
                        if i + m < n {
                            sum += (values[i + m] - values[i]) / m as f64;
                        }
                    }
                    sum / m as f64
                },
                ETSComponent::Multiplicative => {
                    // Average growth rate
                    let mut sum = 0.0;
                    for i in 0..m {
                        if i + m < n && values[i] > 0.0 {
                            sum += (values[i + m] / values[i]).powf(1.0 / m as f64) - 1.0;
                        }
                    }
                    1.0 + sum / m as f64
                },
                ETSComponent::None => 0.0, // This shouldn't happen based on the check in initialize
            };
            self.trend = Some(initial_trend);
        }
        
        // Initialize seasonal components
        if self.seasonal_type != ETSComponent::None {
            let mut initial_seasonal = vec![0.0; m];
            
            match self.seasonal_type {
                ETSComponent::Additive => {
                    // Calculate average value for each season
                    for season in 0..m {
                        let mut season_sum = 0.0;
                        let mut count = 0;
                        
                        for i in (season..n).step_by(m) {
                            // Detrended observation
                            let detrended = values[i] - initial_level;
                            season_sum += detrended;
                            count += 1;
                        }
                        
                        if count > 0 {
                            initial_seasonal[season] = season_sum / count as f64;
                        }
                    }
                    
                    // Normalize to ensure they sum to zero
                    let avg = initial_seasonal.iter().sum::<f64>() / m as f64;
                    for i in 0..m {
                        initial_seasonal[i] -= avg;
                    }
                },
                ETSComponent::Multiplicative => {
                    // Calculate average value for each season
                    for season in 0..m {
                        let mut season_sum = 0.0;
                        let mut count = 0;
                        
                        for i in (season..n).step_by(m) {
                            // Detrended observation
                            let detrended = values[i] / initial_level;
                            season_sum += detrended;
                            count += 1;
                        }
                        
                        if count > 0 {
                            initial_seasonal[season] = season_sum / count as f64;
                        }
                    }
                    
                    // Normalize to ensure they multiply to 1
                    let avg = initial_seasonal.iter().fold(1.0, |acc, &x| acc * x.powf(1.0 / m as f64));
                    for i in 0..m {
                        initial_seasonal[i] /= avg;
                    }
                },
                ETSComponent::None => (), // No action needed
                ETSComponent::Damped => (), // Not applicable to seasonal components
            }
            
            self.seasonal = Some(initial_seasonal);
        }
        
        Ok(())
    }
    
    /// Evaluates the model on test data, computing various error metrics.
    ///
    /// # Arguments
    ///
    /// * `test_data` - The OHLCV data to evaluate the model against
    ///
    /// # Returns
    ///
    /// * `Result<ModelEvaluation, String>` - Evaluation metrics or an error message
    ///
    /// # Examples
    ///
    /// ```
    /// // Continuing from previous examples:
    /// let (train_data, test_data) = data.train_test_split(0.8).unwrap();
    /// 
    /// model.fit(&train_data).unwrap();
    /// let eval = model.evaluate(&test_data).unwrap();
    /// 
    /// println!("MAE: {:.4}", eval.mae);
    /// println!("RMSE: {:.4}", eval.rmse);
    /// println!("MAPE: {:.2}%", eval.mape);
    /// ```
    pub fn evaluate(&self, test_data: &OHLCVData) -> Result<ModelEvaluation, String> {
        if test_data.is_empty() {
            return Err("Test data is empty".to_string());
        }
        
        // Extract the target column from test data
        let test_values = self.extract_target_data(test_data);
        
        // Generate forecasts
        let forecasts = self.forecast(test_values.len())?;
        
        // Calculate error metrics
        let mae = mean_absolute_error(&test_values, &forecasts);
        let mse = mean_squared_error(&test_values, &forecasts);
        let rmse = mse.sqrt();
        let mape = mean_absolute_percentage_error(&test_values, &forecasts);
        
        Ok(ModelEvaluation {
            model_name: self.name.clone(),
            mae,
            mse,
            rmse,
            mape,
        })
    }
    
    /// Get the model name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get fitted values (forecasts on training data)
    pub fn fitted_values(&self) -> Option<&Vec<f64>> {
        self.fitted_values.as_ref()
    }
} 