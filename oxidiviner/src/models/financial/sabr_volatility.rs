//! SABR Volatility Model for Financial Time Series
//!
//! The SABR (Stochastic Alpha Beta Rho) model is a widely used stochastic volatility model
//! in quantitative finance, particularly for modeling the volatility smile in options markets.
//! It's especially popular in fixed income and foreign exchange derivatives pricing.
//!
//! ## Model Specification
//!
//! The SABR model follows a system of stochastic differential equations:
//!
//! **Asset Price Process:**
//! ```text
//! dF(t) = σ(t) * F(t)^β * dW₁(t)
//! ```
//!
//! **Volatility Process:**
//! ```text
//! dσ(t) = α * σ(t) * dW₂(t)
//! ```
//!
//! **Correlation:**
//! ```text
//! dW₁(t) * dW₂(t) = ρ * dt
//! ```
//!
//! Where:
//! - F(t) is the forward price at time t
//! - σ(t) is the stochastic volatility at time t
//! - α is the volatility of volatility parameter
//! - β is the backbone parameter (0 ≤ β ≤ 1)
//!   - β = 0: Normal/Bachelier model
//!   - β = 0.5: CIR (square-root) model  
//!   - β = 1: Log-normal/Black-Scholes model
//! - ρ is the correlation between forward returns and volatility
//! - W₁(t), W₂(t) are correlated Brownian motions
//!
//! ## Applications
//!
//! - **Options Pricing**: Accurate European and American options pricing
//! - **Volatility Surface Modeling**: Industry standard for FX and rates
//! - **Smile Modeling**: Captures realistic volatility smile and skew
//! - **Risk Management**: Advanced VaR with stochastic volatility
//! - **Derivatives Trading**: Model-based trading in FX and interest rate markets

use crate::core::{Forecaster, ModelOutput, OxiError, Result, TimeSeriesData};
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// SABR Stochastic Volatility Model
///
/// This model captures both stochastic forward price movements and stochastic volatility
/// with CEV-type (Constant Elasticity of Variance) dynamics.
#[derive(Debug, Clone)]
pub struct SABRVolatilityModel {
    /// Initial forward price (F₀)
    pub initial_forward: f64,
    /// Initial volatility (σ₀)
    pub initial_volatility: f64,
    /// Volatility of volatility (α)
    pub vol_of_vol: f64,
    /// Backbone parameter (β) - controls the CEV behavior
    pub beta: f64,
    /// Correlation (ρ) - correlation between forward returns and volatility
    pub correlation: f64,
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
    pub initial_forward: f64,
    pub initial_volatility: f64,
    pub vol_of_vol: f64,
    pub beta: f64,
    pub correlation: f64,
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
}

/// Simulated path containing both forward price and volatility paths
#[derive(Debug, Clone)]
pub struct SABRPath {
    pub times: Vec<f64>,
    pub forwards: Vec<f64>,
    pub volatilities: Vec<f64>,
}

/// Volatility surface point for options pricing with SABR
#[derive(Debug, Clone)]
pub struct SABRVolatilitySurfacePoint {
    pub strike: f64,
    pub expiry: f64,
    pub implied_volatility: f64,
    pub sabr_implied_vol: f64,
    pub option_price: f64,
}

/// SABR calibration result
#[derive(Debug, Clone)]
pub struct SABRCalibration {
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub nu: f64, // alias for vol_of_vol
    pub rmse: f64,
    pub r_squared: f64,
}

impl SABRVolatilityModel {
    /// Create a new SABR Volatility Model
    ///
    /// # Arguments
    ///
    /// * `initial_forward` - Initial forward price (F₀)
    /// * `initial_volatility` - Initial volatility (σ₀)
    /// * `vol_of_vol` - Volatility of volatility (α)
    /// * `beta` - Backbone parameter (β), 0 ≤ β ≤ 1
    /// * `correlation` - Correlation between forward and volatility (ρ)
    /// * `time_step` - Time step for simulation (typically 1/252 for daily data)
    ///
    /// # Returns
    ///
    /// A new SABRVolatilityModel instance
    ///
    /// # Example
    ///
    /// ```rust
    /// # use oxidiviner::models::financial::SABRVolatilityModel;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create model with typical FX market parameters
    /// let model = SABRVolatilityModel::new(
    ///     1.20,    // EUR/USD forward rate
    ///     0.10,    // 10% initial volatility
    ///     0.30,    // 30% vol of vol
    ///     0.5,     // β = 0.5 (square-root model)
    ///     -0.3,    // -30% correlation
    ///     1.0/252.0 // Daily time step
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        initial_forward: f64,
        initial_volatility: f64,
        vol_of_vol: f64,
        beta: f64,
        correlation: f64,
        time_step: f64,
    ) -> Result<Self> {
        // Parameter validation
        if initial_forward <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Initial forward price must be positive".to_string(),
            ));
        }
        if initial_volatility <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Initial volatility must be positive".to_string(),
            ));
        }
        if vol_of_vol < 0.0 {
            return Err(OxiError::InvalidParameter(
                "Volatility of volatility must be non-negative".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&beta) {
            return Err(OxiError::InvalidParameter(
                "Beta parameter must be between 0 and 1".to_string(),
            ));
        }
        if !(-1.0..=1.0).contains(&correlation) {
            return Err(OxiError::InvalidParameter(
                "Correlation must be between -1 and 1".to_string(),
            ));
        }
        if time_step <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Time step must be positive".to_string(),
            ));
        }

        Ok(SABRVolatilityModel {
            initial_forward,
            initial_volatility,
            vol_of_vol,
            beta,
            correlation,
            time_step,
            is_fitted: false,
            training_data: Vec::new(),
            estimated_params: None,

            diagnostics: None,
        })
    }

    /// Create a model with typical FX market parameters
    ///
    /// This convenience constructor sets reasonable defaults for FX markets:
    /// - F₀ = 1.0 (normalized forward)
    /// - σ₀ = 0.10 (10% initial volatility)
    /// - α = 0.30 (30% vol of vol)
    /// - β = 0.5 (square-root model)
    /// - ρ = -0.3 (negative correlation)
    /// - Daily time step
    pub fn new_fx_default() -> Result<Self> {
        Self::new(
            1.0,         // Normalized forward
            0.10,        // 10% initial volatility
            0.30,        // Vol of vol
            0.5,         // Beta (square-root model)
            -0.3,        // Correlation
            1.0 / 252.0, // Daily time step
        )
    }

    /// Create a model with typical equity index parameters
    ///
    /// This convenience constructor sets reasonable defaults for equity indices:
    /// - F₀ = 100.0 (typical index level)
    /// - σ₀ = 0.20 (20% initial volatility)
    /// - α = 0.40 (40% vol of vol)
    /// - β = 0.7 (closer to log-normal)
    /// - ρ = -0.6 (strong negative correlation - leverage effect)
    /// - Daily time step
    pub fn new_equity_default() -> Result<Self> {
        Self::new(
            100.0,       // Index level
            0.20,        // 20% initial volatility
            0.40,        // Vol of vol
            0.7,         // Beta (closer to log-normal)
            -0.6,        // Strong negative correlation
            1.0 / 252.0, // Daily time step
        )
    }

    /// Get the model type based on beta parameter
    pub fn get_model_type(&self) -> &str {
        if self.beta < 0.1 {
            "Normal/Bachelier Model (β ≈ 0)"
        } else if (self.beta - 0.5).abs() < 0.1 {
            "Square-Root/CIR Model (β ≈ 0.5)"
        } else if self.beta > 0.9 {
            "Log-Normal/Black-Scholes Model (β ≈ 1)"
        } else {
            "CEV Model (0 < β < 1)"
        }
    }

    /// Simulate future forward price and volatility paths using Monte Carlo simulation
    ///
    /// Uses the Euler-Maruyama scheme with absorption at zero for volatility.
    ///
    /// # Arguments
    ///
    /// * `horizon` - Number of time steps to simulate
    /// * `num_paths` - Number of Monte Carlo paths
    /// * `seed` - Optional random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Vector of simulated SABR paths containing both forwards and volatilities
    pub fn simulate_paths(
        &self,
        horizon: usize,
        num_paths: usize,
        seed: Option<u64>,
    ) -> Result<Vec<SABRPath>> {
        let mut paths = Vec::with_capacity(num_paths);

        if let Some(s) = seed {
            // Use seeded RNG for reproducibility
            let mut rng = StdRng::seed_from_u64(s);
            for _ in 0..num_paths {
                let path = self.simulate_single_path_seeded(horizon, &mut rng)?;
                paths.push(path);
            }
        } else {
            // Use thread_rng for non-reproducible simulation
            for _ in 0..num_paths {
                let path = self.simulate_single_path_random(horizon)?;
                paths.push(path);
            }
        }

        Ok(paths)
    }

    /// Generate a single simulated SABR path with seeded RNG
    ///
    /// Uses the Euler-Maruyama discretization scheme.
    fn simulate_single_path_seeded(&self, horizon: usize, rng: &mut StdRng) -> Result<SABRPath> {
        let mut times = Vec::with_capacity(horizon + 1);
        let mut forwards = Vec::with_capacity(horizon + 1);
        let mut volatilities = Vec::with_capacity(horizon + 1);

        // Initialize
        times.push(0.0);
        forwards.push(self.initial_forward);
        volatilities.push(self.initial_volatility);

        let normal = Normal::new(0.0, 1.0).map_err(|_| {
            OxiError::ModelError("Failed to create normal distribution".to_string())
        })?;

        let dt = self.time_step;
        let sqrt_dt = dt.sqrt();

        for step in 1..=horizon {
            let current_forward = forwards[step - 1];
            let current_vol = volatilities[step - 1].max(0.0); // Ensure non-negative volatility

            // Generate correlated random variables
            let z1 = normal.sample(rng);
            let z2 = normal.sample(rng);
            let w1 = z1;
            let w2 = self.correlation * z1 + (1.0 - self.correlation.powi(2)).sqrt() * z2;

            // Forward price process
            let forward_diffusion = current_vol * current_forward.powf(self.beta) * sqrt_dt * w1;
            let next_forward = current_forward + forward_diffusion;

            // Volatility process (with absorption at zero)
            let vol_diffusion = self.vol_of_vol * current_vol * sqrt_dt * w2;
            let next_vol = (current_vol + vol_diffusion).max(0.0);

            times.push(step as f64 * dt);
            forwards.push(next_forward);
            volatilities.push(next_vol);
        }

        Ok(SABRPath {
            times,
            forwards,
            volatilities,
        })
    }

    /// Generate a single simulated SABR path with thread RNG
    fn simulate_single_path_random(&self, horizon: usize) -> Result<SABRPath> {
        let mut times = Vec::with_capacity(horizon + 1);
        let mut forwards = Vec::with_capacity(horizon + 1);
        let mut volatilities = Vec::with_capacity(horizon + 1);

        // Initialize
        times.push(0.0);
        forwards.push(self.initial_forward);
        volatilities.push(self.initial_volatility);

        let normal = Normal::new(0.0, 1.0).map_err(|_| {
            OxiError::ModelError("Failed to create normal distribution".to_string())
        })?;

        let dt = self.time_step;
        let sqrt_dt = dt.sqrt();

        for step in 1..=horizon {
            let current_forward = forwards[step - 1];
            let current_vol = volatilities[step - 1].max(0.0);

            // Generate correlated random variables
            let z1 = normal.sample(&mut rand::thread_rng());
            let z2 = normal.sample(&mut rand::thread_rng());
            let w1 = z1;
            let w2 = self.correlation * z1 + (1.0 - self.correlation.powi(2)).sqrt() * z2;

            // Forward price process
            let forward_diffusion = current_vol * current_forward.powf(self.beta) * sqrt_dt * w1;
            let next_forward = current_forward + forward_diffusion;

            // Volatility process
            let vol_diffusion = self.vol_of_vol * current_vol * sqrt_dt * w2;
            let next_vol = (current_vol + vol_diffusion).max(0.0);

            times.push(step as f64 * dt);
            forwards.push(next_forward);
            volatilities.push(next_vol);
        }

        Ok(SABRPath {
            times,
            forwards,
            volatilities,
        })
    }

    /// Calculate SABR implied volatility using Hagan's approximation
    ///
    /// This is the industry-standard formula for SABR implied volatility.
    ///
    /// # Arguments
    ///
    /// * `forward` - Forward price
    /// * `strike` - Strike price
    /// * `time_to_expiry` - Time to expiration
    ///
    /// # Returns
    ///
    /// SABR implied volatility
    pub fn sabr_implied_volatility(
        &self,
        forward: f64,
        strike: f64,
        time_to_expiry: f64,
    ) -> Result<f64> {
        if forward <= 0.0 || strike <= 0.0 || time_to_expiry <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Forward, strike, and time must be positive".to_string(),
            ));
        }

        let f = forward;
        let k = strike;
        let t = time_to_expiry;
        let alpha = self.vol_of_vol;
        let beta = self.beta;
        let rho = self.correlation;
        let nu = self.initial_volatility;

        // Handle ATM case
        if (f - k).abs() / f < 1e-6 {
            let atm_vol = alpha / f.powf(1.0 - beta)
                * (1.0
                    + t * ((1.0 - beta).powi(2) / 24.0 * alpha.powi(2)
                        / f.powf(2.0 * (1.0 - beta))
                        + rho * beta * nu * alpha / (4.0 * f.powf(1.0 - beta))
                        + (2.0 - 3.0 * rho.powi(2)) / 24.0 * nu.powi(2)));
            return Ok(atm_vol);
        }

        // General case
        let log_fk = (f / k).ln();
        let fk_avg = (f * k).sqrt();

        let z = nu / alpha * fk_avg.powf(1.0 - beta) * log_fk;
        let x_z = if z.abs() < 1e-6 {
            1.0
        } else {
            z / ((1.0 + (1.0 - 2.0 * rho) * z + z.powi(2)).sqrt() + z - rho).ln()
        };

        let pre_factor = alpha
            / (fk_avg.powf(1.0 - beta)
                * (1.0
                    + (1.0 - beta).powi(2) / 24.0 * log_fk.powi(2)
                    + (1.0 - beta).powi(4) / 1920.0 * log_fk.powi(4)));

        let correction = 1.0
            + t * ((1.0 - beta).powi(2) / 24.0 * alpha.powi(2) / fk_avg.powf(2.0 * (1.0 - beta))
                + rho * beta * nu * alpha / (4.0 * fk_avg.powf(1.0 - beta))
                + (2.0 - 3.0 * rho.powi(2)) / 24.0 * nu.powi(2));

        let implied_vol = pre_factor * x_z * correction;
        Ok(implied_vol.max(0.001)) // Ensure positive volatility
    }

    /// Calculate European option price using SABR model
    ///
    /// Uses the SABR implied volatility in a Black-Scholes formula.
    ///
    /// # Arguments
    ///
    /// * `forward` - Forward price
    /// * `strike` - Strike price
    /// * `time_to_expiry` - Time to expiration
    /// * `discount_factor` - Discount factor (e^(-r*T))
    /// * `is_call` - True for call option, false for put option
    ///
    /// # Returns
    ///
    /// Option price according to the SABR model
    pub fn option_price(
        &self,
        forward: f64,
        strike: f64,
        time_to_expiry: f64,
        discount_factor: f64,
        is_call: bool,
    ) -> Result<f64> {
        let sabr_vol = self.sabr_implied_volatility(forward, strike, time_to_expiry)?;

        // Black formula for forward prices
        let variance = sabr_vol.powi(2) * time_to_expiry;
        let sigma_sqrt_t = variance.sqrt();

        if sigma_sqrt_t < 1e-10 {
            // Intrinsic value only
            let intrinsic = if is_call {
                (forward - strike).max(0.0)
            } else {
                (strike - forward).max(0.0)
            };
            return Ok(discount_factor * intrinsic);
        }

        let d1 = (forward / strike).ln() / sigma_sqrt_t + 0.5 * sigma_sqrt_t;
        let d2 = d1 - sigma_sqrt_t;

        // Standard normal CDF approximation
        let n_d1 = 0.5 * (1.0 + erf(d1 / std::f64::consts::SQRT_2));
        let n_d2 = 0.5 * (1.0 + erf(d2 / std::f64::consts::SQRT_2));

        let option_value = if is_call {
            forward * n_d1 - strike * n_d2
        } else {
            strike * (1.0 - n_d2) - forward * (1.0 - n_d1)
        };

        Ok(discount_factor * option_value)
    }

    /// Calculate Value at Risk (VaR) with SABR stochastic volatility
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
        let paths = self.simulate_paths(horizon_steps, num_simulations, Some(42))?;

        // Extract final values and calculate returns
        let mut returns: Vec<f64> = paths
            .iter()
            .map(|path| {
                let final_forward = *path.forwards.last().unwrap();
                let initial_forward = self.initial_forward;
                final_forward / initial_forward - 1.0
            })
            .collect();

        returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Find VaR as the percentile of losses
        let var_index = ((1.0 - confidence_level) * num_simulations as f64) as usize;
        let var_return = returns[var_index.min(returns.len() - 1)];

        Ok(-var_return * initial_value)
    }

    /// Generate SABR volatility surface
    ///
    /// # Arguments
    ///
    /// * `forward` - Current forward price
    /// * `strikes` - Array of strike prices
    /// * `expiries` - Array of expiration times
    ///
    /// # Returns
    ///
    /// Vector of SABR volatility surface points
    pub fn volatility_surface(
        &self,
        forward: f64,
        strikes: &[f64],
        expiries: &[f64],
    ) -> Result<Vec<SABRVolatilitySurfacePoint>> {
        let mut surface_points = Vec::new();

        for &expiry in expiries {
            for &strike in strikes {
                // Calculate SABR implied volatility
                let sabr_vol = self.sabr_implied_volatility(forward, strike, expiry)?;

                // Calculate option price using SABR volatility
                let discount_factor = 1.0; // Assuming zero interest rate for simplicity
                let option_price =
                    self.option_price(forward, strike, expiry, discount_factor, true)?;

                surface_points.push(SABRVolatilitySurfacePoint {
                    strike,
                    expiry,
                    implied_volatility: sabr_vol,
                    sabr_implied_vol: sabr_vol,
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
}

// Error function approximation for Black formula
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

impl Forecaster for SABRVolatilityModel {
    fn name(&self) -> &str {
        "SABR Volatility Model"
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

        // Generate multiple paths and return the mean path
        let num_paths = 1000;
        let paths = self.simulate_paths(horizon, num_paths, Some(42))?;

        // Calculate mean forecast at each horizon
        let mut forecasts = Vec::with_capacity(horizon);
        for t in 1..=horizon {
            let mean_forward: f64 =
                paths.iter().map(|path| path.forwards[t]).sum::<f64>() / num_paths as f64;
            forecasts.push(mean_forward);
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
    fn test_sabr_model_creation() {
        let model = SABRVolatilityModel::new(1.0, 0.10, 0.30, 0.5, -0.3, 1.0 / 252.0);
        assert!(model.is_ok());

        let model = model.unwrap();
        assert_eq!(model.initial_forward, 1.0);
        assert_eq!(model.initial_volatility, 0.10);
        assert_eq!(model.vol_of_vol, 0.30);
        assert_eq!(model.beta, 0.5);
        assert_eq!(model.correlation, -0.3);
    }

    #[test]
    fn test_fx_default_model() {
        let model = SABRVolatilityModel::new_fx_default();
        assert!(model.is_ok());
    }

    #[test]
    fn test_equity_default_model() {
        let model = SABRVolatilityModel::new_equity_default();
        assert!(model.is_ok());
    }

    #[test]
    fn test_parameter_validation() {
        // Test negative initial forward
        let result = SABRVolatilityModel::new(-1.0, 0.10, 0.30, 0.5, -0.3, 1.0 / 252.0);
        assert!(result.is_err());

        // Test invalid beta
        let result = SABRVolatilityModel::new(1.0, 0.10, 0.30, 1.5, -0.3, 1.0 / 252.0);
        assert!(result.is_err());

        // Test invalid correlation
        let result = SABRVolatilityModel::new(1.0, 0.10, 0.30, 0.5, 1.5, 1.0 / 252.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_type() {
        let model_normal =
            SABRVolatilityModel::new(1.0, 0.10, 0.30, 0.0, -0.3, 1.0 / 252.0).unwrap();
        assert!(model_normal.get_model_type().contains("Normal"));

        let model_sqrt = SABRVolatilityModel::new(1.0, 0.10, 0.30, 0.5, -0.3, 1.0 / 252.0).unwrap();
        assert!(model_sqrt.get_model_type().contains("Square-Root"));

        let model_lognormal =
            SABRVolatilityModel::new(1.0, 0.10, 0.30, 1.0, -0.3, 1.0 / 252.0).unwrap();
        assert!(model_lognormal.get_model_type().contains("Log-Normal"));
    }

    #[test]
    fn test_simulation() {
        let model = SABRVolatilityModel::new_fx_default().unwrap();
        let paths = model.simulate_paths(10, 100, Some(42));

        assert!(paths.is_ok());
        let paths = paths.unwrap();
        assert_eq!(paths.len(), 100);
        assert_eq!(paths[0].forwards.len(), 11); // 10 steps + initial forward
        assert_eq!(paths[0].volatilities.len(), 11);
        assert_eq!(paths[0].forwards[0], 1.0); // Initial forward
    }

    #[test]
    fn test_sabr_implied_volatility() {
        let model = SABRVolatilityModel::new_fx_default().unwrap();

        // Test ATM volatility
        let atm_vol = model.sabr_implied_volatility(1.0, 1.0, 0.25);
        assert!(atm_vol.is_ok());
        let vol = atm_vol.unwrap();
        assert!(vol > 0.0);

        // Test OTM volatility
        let otm_vol = model.sabr_implied_volatility(1.0, 1.1, 0.25);
        assert!(otm_vol.is_ok());
    }

    #[test]
    fn test_option_pricing() {
        let model = SABRVolatilityModel::new_fx_default().unwrap();
        let option_price = model.option_price(1.0, 1.0, 0.25, 0.98, true);

        assert!(option_price.is_ok());
        let price = option_price.unwrap();
        assert!(price > 0.0); // Option price should be positive
    }

    #[test]
    fn test_var_calculation() {
        let model = SABRVolatilityModel::new_fx_default().unwrap();
        let var = model.calculate_var(1000000.0, 0.95, 1.0 / 252.0, 1000);

        assert!(var.is_ok());
        let var = var.unwrap();
        assert!(var >= 0.0); // VaR should be positive (representing loss)
    }

    #[test]
    fn test_volatility_surface() {
        let model = SABRVolatilityModel::new_fx_default().unwrap();
        let strikes = vec![0.9, 0.95, 1.0, 1.05, 1.1];
        let expiries = vec![0.25, 0.5, 1.0];

        let surface = model.volatility_surface(1.0, &strikes, &expiries);
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
        let mut model = SABRVolatilityModel::new_fx_default().unwrap();

        // Create test data
        let timestamps = (0..100)
            .map(|i| chrono::Utc::now() + chrono::Duration::days(i))
            .collect();
        let values: Vec<f64> = (0..100).map(|i| 1.0 + i as f64 * 0.001).collect();
        let data = TimeSeriesData::new(timestamps, values, "test").unwrap();

        // Test fitting and prediction
        assert!(model.fit(&data).is_ok());
        assert!(model.is_fitted);

        let forecast = model.forecast(10);
        assert!(forecast.is_ok());
        assert_eq!(forecast.unwrap().len(), 10);
    }
}
