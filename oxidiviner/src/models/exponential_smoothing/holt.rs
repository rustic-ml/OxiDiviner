use crate::error::ESError;
use oxidiviner_core::{Forecaster, ModelEvaluation, ModelOutput, OxiError, Result, TimeSeriesData};
use oxidiviner_math::metrics::{mae, mape, mse, rmse, smape};

/// Holt Linear (Double Exponential Smoothing) model for forecasting time series with trend.
///
/// This model has level and trend components, making it suitable for forecasting time series
/// with trend but no seasonality. It's equivalent to ETS(A,A,N) in the ETS framework.
///
/// # Model Equations:
/// - Level: l_t = α * y_t + (1 - α) * (l_{t-1} + b_{t-1})
/// - Trend: b_t = β * (l_t - l_{t-1}) + (1 - β) * b_{t-1}
/// - Forecast: ŷ_{t+h|t} = l_t + h * b_t
///
/// where:
/// - l_t is the level at time t
/// - b_t is the trend at time t
/// - y_t is the observed value at time t
/// - α is the smoothing parameter for level (0 < α < 1)
/// - β is the smoothing parameter for trend (0 < β < 1)
/// - ŷ_{t+h|t} is the h-step ahead forecast from time t
pub struct HoltLinearModel {
    /// Model name
    name: String,
    /// Smoothing parameter for level (0 < α < 1)
    alpha: f64,
    /// Smoothing parameter for trend (0 < β < 1)
    beta: f64,
    /// Current level value (after fitting)
    level: Option<f64>,
    /// Current trend value (after fitting)
    trend: Option<f64>,
    /// Fitted values over the training period
    fitted_values: Option<Vec<f64>>,
}

impl HoltLinearModel {
    /// Creates a new Holt Linear (double exponential smoothing) model.
    ///
    /// # Arguments
    /// * `alpha` - Smoothing parameter for level (0 < α < 1)
    /// * `beta` - Smoothing parameter for trend (0 < β < 1)
    ///
    /// # Returns
    /// * `Result<Self>` - A new Holt Linear model if parameters are valid
    pub fn new(alpha: f64, beta: f64) -> std::result::Result<Self, ESError> {
        // Validate parameters
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(ESError::InvalidAlpha(alpha));
        }

        if beta <= 0.0 || beta >= 1.0 {
            return Err(ESError::InvalidBeta(beta));
        }

        let name = format!("Holt(α={:.3}, β={:.3})", alpha, beta);

        Ok(HoltLinearModel {
            name,
            alpha,
            beta,
            level: None,
            trend: None,
            fitted_values: None,
        })
    }

    /// Fit the model to the provided time series data.
    /// This is a convenience method that calls the trait method directly.
    pub fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        <Self as Forecaster>::fit(self, data)
    }

    /// Forecast future values.
    /// This is a convenience method that calls the trait method directly.
    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        <Self as Forecaster>::forecast(self, horizon)
    }

    /// Evaluate the model on test data.
    /// This is a convenience method that calls the trait method directly.
    pub fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        <Self as Forecaster>::evaluate(self, test_data)
    }

    /// Generate forecasts and evaluation in a standardized format.
    /// This is a convenience method that calls the trait method directly.
    pub fn predict(
        &self,
        horizon: usize,
        test_data: Option<&TimeSeriesData>,
    ) -> Result<ModelOutput> {
        <Self as Forecaster>::predict(self, horizon, test_data)
    }

    /// Get the fitted values if available.
    pub fn fitted_values(&self) -> Option<&Vec<f64>> {
        self.fitted_values.as_ref()
    }
}

impl Forecaster for HoltLinearModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        if data.is_empty() {
            return Err(OxiError::from(ESError::EmptyData));
        }

        let n = data.values.len();

        if n < 2 {
            return Err(OxiError::from(ESError::InsufficientData {
                actual: n,
                expected: 2,
            }));
        }

        // Initialize level with the first observation
        let mut level = data.values[0];

        // Initialize trend with simple difference
        let mut trend = data.values[1] - data.values[0];

        // Prepare to store fitted values
        let mut fitted_values = Vec::with_capacity(n);
        fitted_values.push(level); // First fitted value is just the initial level

        // Apply the Holt model recursively
        for i in 1..n {
            // Calculate forecast for this step
            let forecast = level + trend;

            // Store the forecast
            fitted_values.push(forecast);

            // Update the level and trend based on the observed value
            let old_level = level;
            level = self.alpha * data.values[i] + (1.0 - self.alpha) * (level + trend);
            trend = self.beta * (level - old_level) + (1.0 - self.beta) * trend;
        }

        // Store the final level, trend, and fitted values
        self.level = Some(level);
        self.trend = Some(trend);
        self.fitted_values = Some(fitted_values);

        Ok(())
    }

    fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        if let (Some(level), Some(trend)) = (self.level, self.trend) {
            if horizon == 0 {
                return Err(OxiError::from(ESError::InvalidHorizon(horizon)));
            }

            // For Holt model, forecasts increase by the trend value each step
            let mut forecasts = Vec::with_capacity(horizon);
            for h in 1..=horizon {
                forecasts.push(level + h as f64 * trend);
            }

            Ok(forecasts)
        } else {
            Err(OxiError::from(ESError::NotFitted))
        }
    }

    fn evaluate(&self, test_data: &TimeSeriesData) -> Result<ModelEvaluation> {
        if self.level.is_none() || self.trend.is_none() {
            return Err(OxiError::from(ESError::NotFitted));
        }

        let forecast = self.forecast(test_data.values.len())?;

        // Calculate error metrics
        let mae = mae(&test_data.values, &forecast);
        let mse = mse(&test_data.values, &forecast);
        let rmse = rmse(&test_data.values, &forecast);
        let mape = mape(&test_data.values, &forecast);
        let smape = smape(&test_data.values, &forecast);

        Ok(ModelEvaluation {
            model_name: self.name.clone(),
            mae,
            mse,
            rmse,
            mape,
            smape,
        })
    }

    // Using the default predict implementation from the trait
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, TimeZone, Utc};

    #[test]
    fn test_holt_forecast() {
        // Create a simple time series with linear trend
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();

        // Linear trend data: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        let values: Vec<f64> = (1..=10).map(|i| i as f64).collect();

        let time_series = TimeSeriesData::new(timestamps, values, "test_series").unwrap();

        // Create and fit the Holt model with alpha = 0.8, beta = 0.2
        let mut model = HoltLinearModel::new(0.8, 0.2).unwrap();
        model.fit(&time_series).unwrap();

        // Forecast 5 periods ahead
        let forecast_horizon = 5;
        let forecasts = model.forecast(forecast_horizon).unwrap();

        // Check that the number of forecasts matches the horizon
        assert_eq!(forecasts.len(), forecast_horizon);

        // For a perfect linear trend, the model should predict a continuation of that trend
        // The forecasts should be approximately 11, 12, 13, 14, 15
        for (i, forecast) in forecasts.iter().enumerate() {
            let expected = 11.0 + i as f64;
            assert!(
                (forecast - expected).abs() < 0.5,
                "Forecast {} expected to be close to {}, but was {}",
                i + 1,
                expected,
                forecast
            );
        }

        // Test the standardized ModelOutput from predict()
        let output = model.predict(forecast_horizon, None).unwrap();

        // Check basic properties of the ModelOutput
        assert_eq!(output.model_name, model.name());
        assert_eq!(output.forecasts.len(), forecast_horizon);
        assert_eq!(output.forecasts, forecasts);

        // Test with evaluation
        let output_with_eval = model.predict(forecast_horizon, Some(&time_series)).unwrap();

        // Should have evaluation metrics
        assert!(output_with_eval.evaluation.is_some());
        let eval = output_with_eval.evaluation.unwrap();
        assert_eq!(eval.model_name, model.name());
    }
}
