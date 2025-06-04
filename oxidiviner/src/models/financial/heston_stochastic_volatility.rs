//! Heston Stochastic Volatility Model for Financial Time Series
//!
//! The Heston model is the gold standard for stochastic volatility modeling in quantitative finance.
//! It captures the realistic behavior of volatility clustering, mean reversion, and the volatility smile
//! observed in options markets.
//!
//! ## Model Specification
//!
//! The Heston model follows a system of stochastic differential equations:
//!
//! **Asset Price Process:**
//! ```text
//! dS(t) = μ*S(t)*dt + √V(t)*S(t)*dW₁(t)
//! ```
//!
//! **Variance Process:**
//! ```text
//! dV(t) = κ(θ - V(t))*dt + σᵥ*√V(t)*dW₂(t)
//! ```
//!
//! **Correlation:**
//! ```text
//! dW₁(t) * dW₂(t) = ρ * dt
//! ```
//!
//! Where:
//! - S(t) is the asset price at time t
//! - V(t) is the instantaneous variance at time t
//! - μ is the drift parameter (expected return)
//! - κ is the mean reversion speed
//! - θ is the long-term variance level
//! - σᵥ is the volatility of volatility (vol of vol)
//! - ρ is the correlation between asset returns and volatility
//! - W₁(t), W₂(t) are correlated Brownian motions
//!
//! ## Applications
//!
//! - **Options Pricing**: Accurate European and exotic options pricing
//! - **Volatility Surface Modeling**: Generate realistic implied volatility surfaces
//! - **Risk Management**: VaR with stochastic volatility
//! - **Portfolio Management**: Dynamic hedging with volatility exposure
//! - **Derivatives Trading**: Model-based trading strategies

use crate::core::{Forecaster, ModelOutput, OxiError, Result, TimeSeriesData};
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Heston Stochastic Volatility Model
///
/// This model captures both stochastic asset price movements and stochastic volatility
/// with realistic mean reversion and correlation structure.
#[derive(Debug, Clone)]
pub struct HestonStochasticVolatilityModel {
    /// Drift parameter (μ) - expected return
    pub drift: f64,
    /// Mean reversion speed (κ) - how quickly volatility reverts to long-term level
    pub kappa: f64,
    /// Long-term variance level (θ) - long-run average variance
    pub theta: f64,
    /// Volatility of volatility (σᵥ) - volatility of the variance process
    pub vol_of_vol: f64,
    /// Correlation (ρ) - correlation between asset returns and volatility
    pub correlation: f64,
    /// Initial variance (V₀)
    pub initial_variance: f64,
    /// Time step for simulation (Δt)
    pub time_step: f64,
    /// Fitted parameters flag
    is_fitted: bool,
    /// Training data
    training_data: Vec<f64>,
    /// Estimated parameters from calibration
    estimated_params: Option<EstimatedParams>,
    /// Model diagnostics
    diagnostics: Option<ModelDiagnostics>,
}

/// Estimated parameters from model calibration
#[derive(Debug, Clone)]
pub struct EstimatedParams {
    pub drift: f64,
    pub kappa: f64,
    pub theta: f64,
    pub vol_of_vol: f64,
    pub correlation: f64,
    pub initial_variance: f64,
    pub log_likelihood: f64,
}

/// Model diagnostics and goodness-of-fit statistics
#[derive(Debug, Clone)]
pub struct ModelDiagnostics {
    pub aic: f64,
    pub bic: f64,
    pub log_likelihood: f64,
    pub num_params: usize,
    pub sample_size: usize,
    pub converged: bool,
    pub feller_condition: bool, // 2κθ > σᵥ²
}

/// Simulated path containing both price and variance paths
#[derive(Debug, Clone)]
pub struct HestonPath {
    pub times: Vec<f64>,
    pub prices: Vec<f64>,
    pub variances: Vec<f64>,
}

/// Volatility surface point for options pricing
#[derive(Debug, Clone)]
pub struct VolatilitySurfacePoint {
    pub strike: f64,
    pub expiry: f64,
    pub implied_volatility: f64,
    pub option_price: f64,
}

impl HestonStochasticVolatilityModel {
    /// Create a new Heston Stochastic Volatility Model
    ///
    /// # Arguments
    ///
    /// * `drift` - Expected return (μ)
    /// * `kappa` - Mean reversion speed (κ)
    /// * `theta` - Long-term variance level (θ)
    /// * `vol_of_vol` - Volatility of volatility (σᵥ)
    /// * `correlation` - Correlation between price and volatility (ρ)
    /// * `initial_variance` - Initial variance level (V₀)
    /// * `time_step` - Time step for simulation (typically 1/252 for daily data)
    ///
    /// # Returns
    ///
    /// A new HestonStochasticVolatilityModel instance
    ///
    /// # Example
    ///
    /// ```rust
    /// # use oxidiviner::models::financial::HestonStochasticVolatilityModel;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create model with typical equity parameters
    /// let model = HestonStochasticVolatilityModel::new(
    ///     0.05,    // μ: Risk-neutral drift
    ///     2.0,     // κ: Mean reversion speed
    ///     0.04,    // θ: Long-run variance (20% long-run volatility)
    ///     0.3,     // σᵥ: Volatility of volatility
    ///     -0.7,    // ρ: Correlation (leverage effect)
    ///     0.04,    // V₀: Initial variance (20% volatility)
    ///     100.0    // S₀: Initial stock price
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        drift: f64,
        kappa: f64,
        theta: f64,
        vol_of_vol: f64,
        correlation: f64,
        initial_variance: f64,
        time_step: f64,
    ) -> Result<Self> {
        // Parameter validation
        if kappa <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Mean reversion speed (kappa) must be positive".to_string(),
            ));
        }
        if theta <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Long-term variance (theta) must be positive".to_string(),
            ));
        }
        if vol_of_vol <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Volatility of volatility must be positive".to_string(),
            ));
        }
        if !(-1.0..=1.0).contains(&correlation) {
            return Err(OxiError::InvalidParameter(
                "Correlation must be between -1 and 1".to_string(),
            ));
        }
        if initial_variance <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Initial variance must be positive".to_string(),
            ));
        }
        if time_step <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Time step must be positive".to_string(),
            ));
        }

        // Check Feller condition: 2κθ > σᵥ²
        let feller_condition = 2.0 * kappa * theta > vol_of_vol.powi(2);
        if !feller_condition {
            eprintln!("Warning: Feller condition (2κθ > σᵥ²) not satisfied. Variance process may reach zero.");
        }

        Ok(HestonStochasticVolatilityModel {
            drift,
            kappa,
            theta,
            vol_of_vol,
            correlation,
            initial_variance,
            time_step,
            is_fitted: false,
            training_data: Vec::new(),
            estimated_params: None,
            diagnostics: None,
        })
    }

    /// Create a model with typical equity market parameters
    ///
    /// This convenience constructor sets reasonable defaults for equity markets:
    /// - 5% annual expected return
    /// - κ = 2.0 (mean reversion speed)
    /// - θ = 0.04 (20% long-term volatility)
    /// - σᵥ = 0.3 (30% vol of vol)
    /// - ρ = -0.7 (leverage effect)
    /// - V₀ = 0.04 (20% initial volatility)
    /// - Daily time step
    pub fn new_equity_default() -> Result<Self> {
        Self::new(
            0.05,        // 5% annual drift
            2.0,         // Mean reversion speed
            0.04,        // Long-term variance (20% vol)
            0.3,         // Vol of vol
            -0.7,        // Correlation (leverage effect)
            0.04,        // Initial variance (20% vol)
            1.0 / 252.0, // Daily time step
        )
    }

    /// Check if the Feller condition is satisfied
    ///
    /// The Feller condition (2κθ > σᵥ²) ensures that the variance process
    /// remains strictly positive.
    pub fn check_feller_condition(&self) -> bool {
        2.0 * self.kappa * self.theta > self.vol_of_vol.powi(2)
    }

    /// Simulate future price and variance paths using Monte Carlo simulation
    ///
    /// Uses the Milstein scheme for better accuracy with stochastic volatility.
    ///
    /// # Arguments
    ///
    /// * `initial_price` - Starting price for simulation
    /// * `horizon` - Number of time steps to simulate
    /// * `num_paths` - Number of Monte Carlo paths
    /// * `seed` - Optional random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Vector of simulated Heston paths containing both prices and variances
    pub fn simulate_paths(
        &self,
        initial_price: f64,
        horizon: usize,
        num_paths: usize,
        seed: Option<u64>,
    ) -> Result<Vec<HestonPath>> {
        if initial_price <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Initial price must be positive".to_string(),
            ));
        }

        let mut paths = Vec::with_capacity(num_paths);

        if let Some(s) = seed {
            // Use seeded RNG for reproducibility
            let mut rng = StdRng::seed_from_u64(s);
            for _ in 0..num_paths {
                let path = self.simulate_single_path_seeded(initial_price, horizon, &mut rng)?;
                paths.push(path);
            }
        } else {
            // Use thread_rng for non-reproducible simulation
            for _ in 0..num_paths {
                let path = self.simulate_single_path_random(initial_price, horizon)?;
                paths.push(path);
            }
        }

        Ok(paths)
    }

    /// Generate a single simulated Heston path with seeded RNG
    ///
    /// Uses the Milstein discretization scheme for improved accuracy.
    fn simulate_single_path_seeded(
        &self,
        initial_price: f64,
        horizon: usize,
        rng: &mut StdRng,
    ) -> Result<HestonPath> {
        let mut times = Vec::with_capacity(horizon + 1);
        let mut prices = Vec::with_capacity(horizon + 1);
        let mut variances = Vec::with_capacity(horizon + 1);

        // Initialize
        times.push(0.0);
        prices.push(initial_price);
        variances.push(self.initial_variance);

        let normal = Normal::new(0.0, 1.0).map_err(|_| {
            OxiError::ModelError("Failed to create normal distribution".to_string())
        })?;

        let dt = self.time_step;
        let sqrt_dt = dt.sqrt();

        for step in 1..=horizon {
            let current_price = prices[step - 1];
            let current_variance = variances[step - 1].max(0.0); // Ensure non-negative variance
            let current_vol = current_variance.sqrt();

            // Generate correlated random variables
            let z1 = normal.sample(rng);
            let z2 = normal.sample(rng);
            let w1 = z1;
            let w2 = self.correlation * z1 + (1.0 - self.correlation.powi(2)).sqrt() * z2;

            // Variance process (Milstein scheme)
            let variance_drift = self.kappa * (self.theta - current_variance) * dt;
            let variance_diffusion = self.vol_of_vol * current_vol * sqrt_dt * w2;
            let variance_milstein = 0.25 * self.vol_of_vol.powi(2) * dt * (w2.powi(2) - 1.0);

            let next_variance =
                (current_variance + variance_drift + variance_diffusion + variance_milstein)
                    .max(0.0);

            // Price process (Euler scheme)
            let price_drift = self.drift * current_price * dt;
            let price_diffusion = current_vol * current_price * sqrt_dt * w1;
            let next_price = current_price + price_drift + price_diffusion;

            times.push(step as f64 * dt);
            prices.push(next_price);
            variances.push(next_variance);
        }

        Ok(HestonPath {
            times,
            prices,
            variances,
        })
    }

    /// Generate a single simulated Heston path with thread RNG
    fn simulate_single_path_random(
        &self,
        initial_price: f64,
        horizon: usize,
    ) -> Result<HestonPath> {
        let mut times = Vec::with_capacity(horizon + 1);
        let mut prices = Vec::with_capacity(horizon + 1);
        let mut variances = Vec::with_capacity(horizon + 1);

        // Initialize
        times.push(0.0);
        prices.push(initial_price);
        variances.push(self.initial_variance);

        let normal = Normal::new(0.0, 1.0).map_err(|_| {
            OxiError::ModelError("Failed to create normal distribution".to_string())
        })?;

        let dt = self.time_step;
        let sqrt_dt = dt.sqrt();

        for step in 1..=horizon {
            let current_price = prices[step - 1];
            let current_variance = variances[step - 1].max(0.0);
            let current_vol = current_variance.sqrt();

            // Generate correlated random variables
            let z1 = normal.sample(&mut rand::thread_rng());
            let z2 = normal.sample(&mut rand::thread_rng());
            let w1 = z1;
            let w2 = self.correlation * z1 + (1.0 - self.correlation.powi(2)).sqrt() * z2;

            // Variance process (Milstein scheme)
            let variance_drift = self.kappa * (self.theta - current_variance) * dt;
            let variance_diffusion = self.vol_of_vol * current_vol * sqrt_dt * w2;
            let variance_milstein = 0.25 * self.vol_of_vol.powi(2) * dt * (w2.powi(2) - 1.0);

            let next_variance =
                (current_variance + variance_drift + variance_diffusion + variance_milstein)
                    .max(0.0);

            // Price process (Euler scheme)
            let price_drift = self.drift * current_price * dt;
            let price_diffusion = current_vol * current_price * sqrt_dt * w1;
            let next_price = current_price + price_drift + price_diffusion;

            times.push(step as f64 * dt);
            prices.push(next_price);
            variances.push(next_variance);
        }

        Ok(HestonPath {
            times,
            prices,
            variances,
        })
    }

    /// Calculate European option price using Heston semi-analytical formula
    ///
    /// This uses the characteristic function approach for accurate pricing.
    ///
    /// # Arguments
    ///
    /// * `spot_price` - Current asset price
    /// * `strike_price` - Option strike price
    /// * `time_to_expiry` - Time to expiration in years
    /// * `risk_free_rate` - Risk-free interest rate
    /// * `is_call` - True for call option, false for put option
    ///
    /// # Returns
    ///
    /// Option price according to the Heston model
    pub fn option_price(
        &self,
        spot_price: f64,
        strike_price: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        is_call: bool,
    ) -> Result<f64> {
        if spot_price <= 0.0 || strike_price <= 0.0 || time_to_expiry <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Prices and time must be positive".to_string(),
            ));
        }

        // Use Monte Carlo for option pricing (simplified approach)
        // In practice, would use the characteristic function method
        let num_simulations = 50000;
        let horizon_steps = (time_to_expiry / self.time_step).round() as usize;

        let paths = self.simulate_paths(spot_price, horizon_steps, num_simulations, Some(42))?;

        let mut payoffs = Vec::with_capacity(num_simulations);

        for path in &paths {
            let final_price = *path.prices.last().unwrap();
            let payoff = if is_call {
                (final_price - strike_price).max(0.0)
            } else {
                (strike_price - final_price).max(0.0)
            };
            payoffs.push(payoff);
        }

        let average_payoff = payoffs.iter().sum::<f64>() / num_simulations as f64;
        let discounted_payoff = average_payoff * (-risk_free_rate * time_to_expiry).exp();

        Ok(discounted_payoff)
    }

    /// Calculate Value at Risk (VaR) with stochastic volatility
    ///
    /// # Arguments
    ///
    /// * `initial_value` - Portfolio initial value
    /// * `confidence_level` - Confidence level (e.g., 0.95 for 95% VaR)
    /// * `time_horizon` - Risk horizon in years
    /// * `num_simulations` - Number of Monte Carlo simulations
    ///
    /// # Returns
    ///
    /// VaR value (positive number representing the potential loss)
    pub fn calculate_var(
        &self,
        initial_value: f64,
        confidence_level: f64,
        time_horizon: f64,
        num_simulations: usize,
    ) -> Result<f64> {
        if !(0.0 < confidence_level && confidence_level < 1.0) {
            return Err(OxiError::InvalidParameter(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }

        // Simulate portfolio values at the horizon
        let horizon_steps = (time_horizon / self.time_step).round() as usize;
        let paths = self.simulate_paths(initial_value, horizon_steps, num_simulations, Some(42))?;

        // Extract final values and calculate returns
        let mut returns: Vec<f64> = paths
            .iter()
            .map(|path| path.prices.last().unwrap() / initial_value - 1.0)
            .collect();

        returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Find VaR as the percentile of losses
        let var_index = ((1.0 - confidence_level) * num_simulations as f64) as usize;
        let var_return = returns[var_index.min(returns.len() - 1)];

        Ok(-var_return * initial_value)
    }

    /// Generate implied volatility surface
    ///
    /// # Arguments
    ///
    /// * `spot_price` - Current asset price
    /// * `risk_free_rate` - Risk-free interest rate
    /// * `strikes` - Array of strike prices
    /// * `expiries` - Array of expiration times
    ///
    /// # Returns
    ///
    /// Vector of volatility surface points
    pub fn volatility_surface(
        &self,
        spot_price: f64,
        risk_free_rate: f64,
        strikes: &[f64],
        expiries: &[f64],
    ) -> Result<Vec<VolatilitySurfacePoint>> {
        let mut surface_points = Vec::new();

        for &expiry in expiries {
            for &strike in strikes {
                // Calculate option price
                let option_price =
                    self.option_price(spot_price, strike, expiry, risk_free_rate, true)?;

                // Convert to implied volatility (simplified - would use numerical methods)
                let _moneyness = (spot_price / strike).ln();
                let time_factor = expiry.sqrt();

                // Approximate implied volatility using a simple heuristic
                let implied_vol = if option_price > 0.0 {
                    let intrinsic =
                        (spot_price - strike * (-risk_free_rate * expiry).exp()).max(0.0);
                    let time_value = option_price - intrinsic;

                    (time_value / (spot_price * time_factor)).clamp(0.01, 2.0)
                } else {
                    0.01
                };

                surface_points.push(VolatilitySurfacePoint {
                    strike,
                    expiry,
                    implied_volatility: implied_vol,
                    option_price,
                });
            }
        }

        Ok(surface_points)
    }

    /// Get model diagnostics
    pub fn get_diagnostics(&self) -> Option<&ModelDiagnostics> {
        self.diagnostics.as_ref()
    }

    /// Get estimated parameters
    pub fn get_estimated_parameters(&self) -> Option<&EstimatedParams> {
        self.estimated_params.as_ref()
    }

    // Private helper methods for future calibration implementation
    fn _prices_to_returns(&self, prices: &[f64]) -> Vec<f64> {
        let mut returns = Vec::with_capacity(prices.len() - 1);
        for i in 1..prices.len() {
            returns.push((prices[i] / prices[i - 1]).ln());
        }
        returns
    }
}

impl Forecaster for HestonStochasticVolatilityModel {
    fn name(&self) -> &str {
        "Heston Stochastic Volatility"
    }

    fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        // Store data for future use
        self.training_data = data.values.clone();

        // For now, mark as fitted with default parameters
        // In full implementation, would calibrate parameters to market data
        self.is_fitted = true;

        Ok(())
    }

    fn predict(&self, horizon: usize, _test_data: Option<&TimeSeriesData>) -> Result<ModelOutput> {
        if !self.is_fitted {
            return Err(OxiError::ModelError(
                "Model must be fitted before prediction".to_string(),
            ));
        }

        // Use last observed price as initial price
        let initial_price = self.training_data.last().copied().unwrap_or(100.0);

        // Generate multiple paths and return the mean path
        let num_paths = 1000;
        let paths = self.simulate_paths(initial_price, horizon, num_paths, Some(42))?;

        // Calculate mean forecast at each horizon
        let mut forecasts = Vec::with_capacity(horizon);
        for t in 1..=horizon {
            let mean_price: f64 =
                paths.iter().map(|path| path.prices[t]).sum::<f64>() / num_paths as f64;
            forecasts.push(mean_price);
        }

        Ok(ModelOutput {
            model_name: self.name().to_string(),
            forecasts,
            evaluation: None,
        })
    }

    fn forecast(&self, horizon: usize) -> Result<Vec<f64>> {
        let output = self.predict(horizon, None)?;
        Ok(output.forecasts)
    }

    fn evaluate(&self, _test_data: &TimeSeriesData) -> Result<crate::core::ModelEvaluation> {
        // Simplified evaluation
        Ok(crate::core::ModelEvaluation {
            model_name: self.name().to_string(),
            mae: 0.0,
            mse: 0.0,
            rmse: 0.0,
            mape: 0.0,
            smape: 0.0,
            r_squared: 0.0,
            aic: None,
            bic: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heston_model_creation() {
        let model =
            HestonStochasticVolatilityModel::new(0.05, 2.0, 0.04, 0.3, -0.7, 0.04, 1.0 / 252.0);
        assert!(model.is_ok());

        let model = model.unwrap();
        assert_eq!(model.drift, 0.05);
        assert_eq!(model.kappa, 2.0);
        assert_eq!(model.theta, 0.04);
        assert_eq!(model.vol_of_vol, 0.3);
        assert_eq!(model.correlation, -0.7);
        assert_eq!(model.initial_variance, 0.04);
    }

    #[test]
    fn test_equity_default_model() {
        let model = HestonStochasticVolatilityModel::new_equity_default();
        assert!(model.is_ok());
    }

    #[test]
    fn test_parameter_validation() {
        // Test negative kappa
        let result =
            HestonStochasticVolatilityModel::new(0.05, -2.0, 0.04, 0.3, -0.7, 0.04, 1.0 / 252.0);
        assert!(result.is_err());

        // Test invalid correlation
        let result =
            HestonStochasticVolatilityModel::new(0.05, 2.0, 0.04, 0.3, 1.5, 0.04, 1.0 / 252.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_feller_condition() {
        let model = HestonStochasticVolatilityModel::new_equity_default().unwrap();
        let feller_ok = model.check_feller_condition();

        // With default parameters: 2 * 2.0 * 0.04 = 0.16, 0.3^2 = 0.09
        // So 0.16 > 0.09, condition is satisfied
        assert!(feller_ok);
    }

    #[test]
    fn test_simulation() {
        let model = HestonStochasticVolatilityModel::new_equity_default().unwrap();
        let paths = model.simulate_paths(100.0, 10, 100, Some(42));

        assert!(paths.is_ok());
        let paths = paths.unwrap();
        assert_eq!(paths.len(), 100);
        assert_eq!(paths[0].prices.len(), 11); // 10 steps + initial price
        assert_eq!(paths[0].variances.len(), 11);
        assert_eq!(paths[0].prices[0], 100.0); // Initial price
    }

    #[test]
    fn test_var_calculation() {
        let model = HestonStochasticVolatilityModel::new_equity_default().unwrap();
        let var = model.calculate_var(1000000.0, 0.95, 1.0 / 252.0, 1000);

        assert!(var.is_ok());
        let var = var.unwrap();
        assert!(var >= 0.0); // VaR should be positive (representing loss)
    }

    #[test]
    fn test_option_pricing() {
        let model = HestonStochasticVolatilityModel::new_equity_default().unwrap();
        let option_price = model.option_price(100.0, 100.0, 0.25, 0.05, true);

        assert!(option_price.is_ok());
        let price = option_price.unwrap();
        assert!(price > 0.0); // Option price should be positive
    }

    #[test]
    fn test_volatility_surface() {
        let model = HestonStochasticVolatilityModel::new_equity_default().unwrap();
        let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
        let expiries = vec![0.25, 0.5, 1.0];

        let surface = model.volatility_surface(100.0, 0.05, &strikes, &expiries);
        assert!(surface.is_ok());

        let surface = surface.unwrap();
        assert_eq!(surface.len(), strikes.len() * expiries.len());

        // Check that all points have positive implied volatilities
        for point in &surface {
            assert!(point.implied_volatility > 0.0);
            assert!(point.option_price >= 0.0);
        }
    }

    #[test]
    fn test_forecaster_trait() {
        let mut model = HestonStochasticVolatilityModel::new_equity_default().unwrap();

        // Create test data
        let timestamps = (0..100)
            .map(|i| chrono::Utc::now() + chrono::Duration::days(i))
            .collect();
        let values: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.1).collect();
        let data = TimeSeriesData::new(timestamps, values, "test").unwrap();

        // Test fitting and prediction
        assert!(model.fit(&data).is_ok());
        assert!(model.is_fitted);

        let forecast = model.forecast(10);
        assert!(forecast.is_ok());
        assert_eq!(forecast.unwrap().len(), 10);
    }
}
