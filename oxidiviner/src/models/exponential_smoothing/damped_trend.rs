use crate::error::ESError;
use oxidiviner_core::{Forecaster, ModelEvaluation, ModelOutput, OxiError, Result, TimeSeriesData};
use oxidiviner_math::metrics::{mae, mape, mse, rmse, smape};

/// Damped Trend Model, an extension of Holt's Linear method.
///
/// This model has level and damped trend components, making it suitable for forecasting time series
/// with trend. The damping factor reduces the trend over time, which often results in better long-term forecasts
/// compared to Holt's Linear method.
///
/// # Model Equations:
/// - Level: l_t = α * y_t + (1 - α) * (l_{t-1} + φ * b_{t-1})
/// - Trend: b_t = β * (l_t - l_{t-1}) + (1 - β) * φ * b_{t-1}
/// - Forecast: ŷ_{t+h|t} = l_t + (φ + φ² + ... + φʰ) * b_t
///
/// where:
/// - l_t is the level at time t
/// - b_t is the trend at time t
/// - y_t is the observed value at time t
/// - α is the smoothing parameter for level (0 < α < 1)
/// - β is the smoothing parameter for trend (0 < β < 1)
/// - φ is the damping parameter (0 < φ < 1)
/// - ŷ_{t+h|t} is the h-step ahead forecast from time t
pub struct DampedTrendModel {
    /// Model name
    name: String,
    /// Smoothing parameter for level (0 < α < 1)
    alpha: f64,
    /// Smoothing parameter for trend (0 < β < 1)
    beta: f64,
    /// Damping parameter (0 < φ < 1)
    phi: f64,
    /// Current level value (after fitting)
    level: Option<f64>,
    /// Current trend value (after fitting)
    trend: Option<f64>,
    /// Fitted values over the training period
    fitted_values: Option<Vec<f64>>,
}

impl DampedTrendModel {
    /// Creates a new Damped Trend model.
    ///
    /// # Arguments
    /// * `alpha` - Smoothing parameter for level (0 < α < 1)
    /// * `beta` - Smoothing parameter for trend (0 < β < 1)
    /// * `phi` - Damping parameter (0 < φ < 1)
    ///
    /// # Returns
    /// * `Result<Self>` - A new Damped Trend model if parameters are valid
    pub fn new(alpha: f64, beta: f64, phi: f64) -> std::result::Result<Self, ESError> {
        // Validate parameters
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(ESError::InvalidAlpha(alpha));
        }

        if beta <= 0.0 || beta >= 1.0 {
            return Err(ESError::InvalidBeta(beta));
        }

        if phi <= 0.0 || phi >= 1.0 {
            return Err(ESError::InvalidDampingFactor(phi));
        }

        let name = format!("DampedTrend(α={:.3}, β={:.3}, φ={:.3})", alpha, beta, phi);

        Ok(DampedTrendModel {
            name,
            alpha,
            beta,
            phi,
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

    /// Get the final level value.
    pub fn level(&self) -> Option<f64> {
        self.level
    }

    /// Get the final trend value.
    pub fn trend(&self) -> Option<f64> {
        self.trend
    }
}

impl Forecaster for DampedTrendModel {
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

        // Apply the Damped Trend model recursively
        for i in 1..n {
            // Calculate forecast for this step
            let forecast = level + self.phi * trend;

            // Store the forecast
            fitted_values.push(forecast);

            // Update the level and trend based on the observed value
            let old_level = level;
            level = self.alpha * data.values[i] + (1.0 - self.alpha) * (level + self.phi * trend);
            trend = self.beta * (level - old_level) + (1.0 - self.beta) * self.phi * trend;
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

            // For Damped Trend model, forecasts increase by the damped trend
            let mut forecasts = Vec::with_capacity(horizon);

            // Calculate cumulative damping factors for each horizon
            for h in 1..=horizon {
                // Calculate sum of damping factors: φ + φ² + ... + φʰ
                let mut damping_sum = 0.0;
                let mut phi_power = self.phi;

                for _ in 1..=h {
                    damping_sum += phi_power;
                    phi_power *= self.phi;
                }

                forecasts.push(level + damping_sum * trend);
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
    fn test_damped_trend_forecast() {
        // Create a simple time series with linear trend
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();

        // Linear trend data: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        let values: Vec<f64> = (1..=10).map(|i| i as f64).collect();

        let time_series = TimeSeriesData::new(timestamps, values, "test_series").unwrap();

        // Create and fit the Damped Trend model with alpha = 0.8, beta = 0.2, phi = 0.9
        let mut model = DampedTrendModel::new(0.8, 0.2, 0.9).unwrap();
        model.fit(&time_series).unwrap();

        // Forecast 10 periods ahead
        let forecast_horizon = 10;
        let forecasts = model.forecast(forecast_horizon).unwrap();

        // Check that the number of forecasts matches the horizon
        assert_eq!(forecasts.len(), forecast_horizon);

        // For a damped trend model, the forecasts should eventually flatten out
        // Check that the slope decreases over time
        for i in 1..forecast_horizon {
            let slope_current = forecasts[i] - forecasts[i - 1];

            if i < forecast_horizon - 1 {
                let slope_next = forecasts[i + 1] - forecasts[i];
                assert!(
                    slope_next <= slope_current + 1e-6,
                    "Slope should decrease or stay the same, but got {} vs {}",
                    slope_next,
                    slope_current
                );
            }
        }

        // The damped forecast should be lower than a linear forecast for longer horizons
        let undamped_10step = 10.0 + 10.0; // Linear projection for 10 steps ahead
        assert!(
            forecasts[forecast_horizon - 1] < undamped_10step,
            "The damped forecast should be lower than the undamped linear projection"
        );

        // Test the standardized ModelOutput from predict()
        let output = model.predict(forecast_horizon, None).unwrap();

        // Check basic properties of the ModelOutput
        assert_eq!(output.model_name, model.name());
        assert_eq!(output.forecasts.len(), forecast_horizon);
        assert_eq!(output.forecasts, forecasts);
    }

    #[test]
    fn test_damped_trend_vs_linear() {
        // Create a simple time series with linear trend
        let now = Utc::now();
        let timestamps: Vec<DateTime<Utc>> = (0..10)
            .map(|i| Utc.timestamp_opt(now.timestamp() + i * 86400, 0).unwrap())
            .collect();

        // Linear trend data: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        let values: Vec<f64> = (1..=10).map(|i| i as f64).collect();

        let time_series = TimeSeriesData::new(timestamps, values, "test_series").unwrap();

        // Create a Holt Linear model with alpha = 0.8, beta = 0.2
        let mut linear_model = crate::holt::HoltLinearModel::new(0.8, 0.2).unwrap();
        linear_model.fit(&time_series).unwrap();

        // Create a Damped Trend model with the same alpha, beta, and phi = 0.9
        let mut damped_model = DampedTrendModel::new(0.8, 0.2, 0.9).unwrap();
        damped_model.fit(&time_series).unwrap();

        // Forecast 20 periods ahead with both models
        let forecast_horizon = 20;
        let linear_forecasts = linear_model.forecast(forecast_horizon).unwrap();
        let damped_forecasts = damped_model.forecast(forecast_horizon).unwrap();

        // Linear model's forecasts should be higher than damped model's for the far-out horizons
        for i in 10..forecast_horizon {
            assert!(
                linear_forecasts[i] > damped_forecasts[i],
                "Linear forecast ({}) should exceed damped forecast ({}) at horizon {}",
                linear_forecasts[i],
                damped_forecasts[i],
                i + 1
            );
        }
    }
}
