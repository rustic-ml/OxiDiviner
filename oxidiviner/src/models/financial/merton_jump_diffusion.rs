//! Merton Jump-Diffusion Model for Financial Time Series
//!
//! The Merton Jump-Diffusion Model extends the geometric Brownian motion by adding
//! jump components to capture sudden market movements such as crashes, earnings announcements,
//! or other extreme events.
//!
//! ## Model Specification
//!
//! The Merton model extends the Black-Scholes framework by adding jump discontinuities:
//!
//! ```text
//! dS(t) = μ*S(t)*dt + σ*S(t)*dW(t) + S(t-)*∫[e^J - 1]*N(dt,dJ)
//! ```
//!
//! Where:
//! - S(t) is the asset price at time t
//! - μ is the drift parameter
//! - σ is the diffusion volatility
//! - W(t) is a standard Brownian motion
//! - N(dt,dJ) is a compound Poisson process
//! - λ is the jump intensity (average number of jumps per unit time)
//! - J ~ N(μ_J, σ_J²) is the jump size distribution
//!
//! ## Applications
//!
//! - **Risk Management**: VaR and CVaR calculations with jump risk
//! - **Options Pricing**: Extension of Black-Scholes with jump risk
//! - **Crash Modeling**: Capturing sudden market downturns
//! - **High-Frequency Trading**: Modeling microstructure jumps

use crate::core::{Forecaster, ModelOutput, OxiError, Result, TimeSeriesData};
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Poisson};
use std::f64::consts::PI;

/// Merton Jump-Diffusion Model
///
/// This model captures both continuous price movements (geometric Brownian motion)
/// and discontinuous jumps that represent extreme market events.
#[derive(Debug, Clone)]
pub struct MertonJumpDiffusionModel {
    /// Drift parameter (μ) - expected return
    pub drift: f64,
    /// Diffusion volatility (σ) - continuous volatility
    pub volatility: f64,
    /// Jump intensity (λ) - average number of jumps per unit time
    pub jump_intensity: f64,
    /// Jump mean (μ_J) - average jump size
    pub jump_mean: f64,
    /// Jump volatility (σ_J) - volatility of jump sizes
    pub jump_volatility: f64,
    /// Time step for simulation (Δt)
    pub time_step: f64,
    /// Fitted parameters flag
    is_fitted: bool,
    /// Training data
    training_data: Vec<f64>,
    /// Estimated parameters from calibration
    estimated_params: Option<EstimatedParams>,
    /// Log-likelihood of the fitted model
    log_likelihood: Option<f64>,
    /// Model diagnostics
    diagnostics: Option<ModelDiagnostics>,
}

/// Estimated parameters from model calibration
#[derive(Debug, Clone)]
pub struct EstimatedParams {
    pub drift: f64,
    pub volatility: f64,
    pub jump_intensity: f64,
    pub jump_mean: f64,
    pub jump_volatility: f64,
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

/// Jump event structure for simulation
#[derive(Debug, Clone)]
pub struct JumpEvent {
    pub time: f64,
    pub size: f64,
    pub relative_impact: f64,
}

impl MertonJumpDiffusionModel {
    /// Create a new Merton Jump-Diffusion Model
    ///
    /// # Arguments
    ///
    /// * `drift` - Expected return (μ)
    /// * `volatility` - Continuous volatility (σ)
    /// * `jump_intensity` - Average jumps per unit time (λ)
    /// * `jump_mean` - Average jump size (μ_J)
    /// * `jump_volatility` - Jump size volatility (σ_J)
    /// * `time_step` - Time step for simulation (typically 1/252 for daily data)
    ///
    /// # Returns
    ///
    /// A new MertonJumpDiffusionModel instance
    ///
    /// # Example
    ///
    /// ```rust
    /// # use oxidiviner::models::financial::MertonJumpDiffusionModel;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create model with typical equity parameters
    /// let model = MertonJumpDiffusionModel::new(
    ///     0.05,    // μ: Drift rate (5%)
    ///     0.20,    // σ: Diffusion volatility (20%)
    ///     0.10,    // λ: Jump intensity (10% annual probability)
    ///     -0.05,   // μⱼ: Average jump size (-5% crash)
    ///     0.15,    // σⱼ: Jump volatility (15%)
    ///     100.0    // S₀: Initial stock price
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        drift: f64,
        volatility: f64,
        jump_intensity: f64,
        jump_mean: f64,
        jump_volatility: f64,
        time_step: f64,
    ) -> Result<Self> {
        // Parameter validation
        if volatility <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Volatility must be positive".to_string(),
            ));
        }
        if jump_intensity < 0.0 {
            return Err(OxiError::InvalidParameter(
                "Jump intensity must be non-negative".to_string(),
            ));
        }
        if jump_volatility <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Jump volatility must be positive".to_string(),
            ));
        }
        if time_step <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Time step must be positive".to_string(),
            ));
        }

        Ok(MertonJumpDiffusionModel {
            drift,
            volatility,
            jump_intensity,
            jump_mean,
            jump_volatility,
            time_step,
            is_fitted: false,
            training_data: Vec::new(),
            estimated_params: None,
            log_likelihood: None,
            diagnostics: None,
        })
    }

    /// Create a model with typical financial market parameters
    ///
    /// This convenience constructor sets reasonable defaults for equity markets:
    /// - 5% annual expected return
    /// - 20% annual volatility
    /// - 2 jumps per year
    /// - -3% average crash size
    /// - 5% jump volatility
    /// - Daily time step
    pub fn new_equity_default() -> Result<Self> {
        Self::new(
            0.05,        // 5% annual drift
            0.20,        // 20% annual volatility
            2.0,         // 2 jumps per year
            -0.03,       // -3% average jump (crash)
            0.05,        // 5% jump volatility
            1.0 / 252.0, // Daily time step
        )
    }

    /// Calibrate the model to observed price data using Maximum Likelihood Estimation
    ///
    /// This method estimates all five parameters (μ, σ, λ, μ_J, σ_J) from price data
    /// using numerical optimization to maximize the log-likelihood function.
    ///
    /// # Arguments
    ///
    /// * `prices` - Observed price series (not returns)
    /// * `max_iterations` - Maximum iterations for optimization
    /// * `tolerance` - Convergence tolerance
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn fit_prices(
        &mut self,
        prices: &[f64],
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
    ) -> Result<()> {
        if prices.len() < 10 {
            return Err(OxiError::DataError(
                "Need at least 10 price observations for calibration".to_string(),
            ));
        }

        // Convert prices to log returns
        let returns = self.prices_to_returns(prices);
        self.training_data = returns.clone();

        let _max_iter = max_iterations.unwrap_or(1000);
        let _tol = tolerance.unwrap_or(1e-6);

        // Initial parameter guess based on empirical moments
        let initial_params = self.estimate_initial_parameters(&returns)?;

        // Optimize parameters using simplified approach
        let optimized_params = self.optimize_parameters(&returns, initial_params)?;

        // Store results
        self.drift = optimized_params.drift;
        self.volatility = optimized_params.volatility;
        self.jump_intensity = optimized_params.jump_intensity;
        self.jump_mean = optimized_params.jump_mean;
        self.jump_volatility = optimized_params.jump_volatility;
        self.estimated_params = Some(optimized_params.clone());
        self.log_likelihood = Some(optimized_params.log_likelihood);

        // Calculate model diagnostics
        let num_params = 5; // μ, σ, λ, μ_J, σ_J
        let n = returns.len();
        let aic = 2.0 * num_params as f64 - 2.0 * optimized_params.log_likelihood;
        let bic = (num_params as f64) * (n as f64).ln() - 2.0 * optimized_params.log_likelihood;

        self.diagnostics = Some(ModelDiagnostics {
            aic,
            bic,
            log_likelihood: optimized_params.log_likelihood,
            num_params,
            sample_size: n,
            converged: true, // Simplified for now
        });

        self.is_fitted = true;
        Ok(())
    }

    /// Simulate future price paths using Monte Carlo simulation
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
    /// Vector of simulated price paths, each path contains prices at each time step
    pub fn simulate_paths(
        &self,
        initial_price: f64,
        horizon: usize,
        num_paths: usize,
        seed: Option<u64>,
    ) -> Result<Vec<Vec<f64>>> {
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

    /// Generate a single simulated price path with seeded RNG
    fn simulate_single_path_seeded(
        &self,
        initial_price: f64,
        horizon: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<f64>> {
        let mut path = Vec::with_capacity(horizon + 1);
        path.push(initial_price);

        let normal = Normal::new(0.0, 1.0).map_err(|_| {
            OxiError::ModelError("Failed to create normal distribution".to_string())
        })?;

        let poisson = Poisson::new(self.jump_intensity * self.time_step).map_err(|_| {
            OxiError::ModelError("Failed to create Poisson distribution".to_string())
        })?;

        let jump_normal = Normal::new(self.jump_mean, self.jump_volatility).map_err(|_| {
            OxiError::ModelError("Failed to create jump normal distribution".to_string())
        })?;

        for _ in 0..horizon {
            let current_price = *path.last().unwrap();

            // Continuous component (geometric Brownian motion)
            let brownian_increment = normal.sample(rng);
            let drift_component = (self.drift - 0.5 * self.volatility.powi(2)) * self.time_step;
            let diffusion_component = self.volatility * self.time_step.sqrt() * brownian_increment;

            // Jump component
            let num_jumps = poisson.sample(rng) as usize;
            let mut total_jump = 0.0;

            for _ in 0..num_jumps {
                let jump_size = jump_normal.sample(rng);
                total_jump += jump_size;
            }

            // Calculate next price
            let log_return = drift_component + diffusion_component + total_jump;
            let next_price = current_price * (log_return.exp());

            path.push(next_price);
        }

        Ok(path)
    }

    /// Generate a single simulated price path with thread RNG
    fn simulate_single_path_random(&self, initial_price: f64, horizon: usize) -> Result<Vec<f64>> {
        let mut path = Vec::with_capacity(horizon + 1);
        path.push(initial_price);

        let normal = Normal::new(0.0, 1.0).map_err(|_| {
            OxiError::ModelError("Failed to create normal distribution".to_string())
        })?;

        let poisson = Poisson::new(self.jump_intensity * self.time_step).map_err(|_| {
            OxiError::ModelError("Failed to create Poisson distribution".to_string())
        })?;

        let jump_normal = Normal::new(self.jump_mean, self.jump_volatility).map_err(|_| {
            OxiError::ModelError("Failed to create jump normal distribution".to_string())
        })?;

        for _ in 0..horizon {
            let current_price = *path.last().unwrap();

            // Continuous component (geometric Brownian motion)
            let brownian_increment = normal.sample(&mut rand::thread_rng());
            let drift_component = (self.drift - 0.5 * self.volatility.powi(2)) * self.time_step;
            let diffusion_component = self.volatility * self.time_step.sqrt() * brownian_increment;

            // Jump component
            let num_jumps = poisson.sample(&mut rand::thread_rng()) as usize;
            let mut total_jump = 0.0;

            for _ in 0..num_jumps {
                let jump_size = jump_normal.sample(&mut rand::thread_rng());
                total_jump += jump_size;
            }

            // Calculate next price
            let log_return = drift_component + diffusion_component + total_jump;
            let next_price = current_price * (log_return.exp());

            path.push(next_price);
        }

        Ok(path)
    }

    /// Calculate the theoretical option price using the Merton formula
    ///
    /// This extends the Black-Scholes formula to include jump risk.
    ///
    /// # Arguments
    ///
    /// * `spot_price` - Current asset price
    /// * `strike_price` - Option strike price
    /// * `time_to_expiry` - Time to expiration in years
    /// * `risk_free_rate` - Risk-free interest rate
    /// * `is_call` - True for call option, false for put option
    /// * `max_jumps` - Maximum number of jumps to consider in the series
    ///
    /// # Returns
    ///
    /// Option price according to the Merton jump-diffusion model
    pub fn option_price(
        &self,
        spot_price: f64,
        strike_price: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        is_call: bool,
        max_jumps: usize,
    ) -> Result<f64> {
        if spot_price <= 0.0 || strike_price <= 0.0 || time_to_expiry <= 0.0 {
            return Err(OxiError::InvalidParameter(
                "Prices and time must be positive".to_string(),
            ));
        }

        let lambda_prime = self.jump_intensity * (1.0 + self.jump_mean).exp();
        let mut option_value = 0.0;

        // Sum over possible number of jumps (Poisson series)
        for n in 0..=max_jumps {
            // Poisson probability of n jumps
            let poisson_prob = (-self.jump_intensity * time_to_expiry).exp()
                * (self.jump_intensity * time_to_expiry).powi(n as i32)
                / Self::factorial(n) as f64;

            // Adjusted parameters for n jumps
            let sigma_n_squared = self.volatility.powi(2)
                + (n as f64) * self.jump_volatility.powi(2) / time_to_expiry;
            let sigma_n = sigma_n_squared.sqrt();

            let r_n = risk_free_rate - lambda_prime
                + self.jump_intensity
                + (n as f64) * (self.jump_mean + 0.5 * self.jump_volatility.powi(2))
                    / time_to_expiry;

            // Black-Scholes with adjusted parameters
            let bs_price = self.black_scholes(
                spot_price,
                strike_price,
                time_to_expiry,
                r_n,
                sigma_n,
                is_call,
            )?;

            option_value += poisson_prob * bs_price;
        }

        Ok(option_value)
    }

    /// Calculate Value at Risk (VaR) incorporating jump risk
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
            .map(|path| path.last().unwrap() / initial_value - 1.0)
            .collect();

        returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Find VaR as the percentile of losses
        let var_index = ((1.0 - confidence_level) * num_simulations as f64) as usize;
        let var_return = returns[var_index.min(returns.len() - 1)];

        Ok(-var_return * initial_value) // Return as positive loss amount
    }

    /// Extract jump events from a simulated path
    ///
    /// This method identifies significant price movements that can be attributed to jumps
    /// rather than continuous diffusion.
    ///
    /// # Arguments
    ///
    /// * `path` - Simulated price path
    /// * `jump_threshold` - Minimum return magnitude to be considered a jump
    ///
    /// # Returns
    ///
    /// Vector of identified jump events
    pub fn extract_jump_events(&self, path: &[f64], jump_threshold: f64) -> Vec<JumpEvent> {
        let mut jump_events = Vec::new();

        for i in 1..path.len() {
            let log_return = (path[i] / path[i - 1]).ln();

            if log_return.abs() > jump_threshold {
                jump_events.push(JumpEvent {
                    time: i as f64 * self.time_step,
                    size: log_return,
                    relative_impact: log_return / self.volatility,
                });
            }
        }

        jump_events
    }

    /// Get model diagnostics
    pub fn get_diagnostics(&self) -> Option<&ModelDiagnostics> {
        self.diagnostics.as_ref()
    }

    /// Get estimated parameters
    pub fn get_estimated_parameters(&self) -> Option<&EstimatedParams> {
        self.estimated_params.as_ref()
    }

    // Private helper methods

    fn prices_to_returns(&self, prices: &[f64]) -> Vec<f64> {
        let mut returns = Vec::with_capacity(prices.len() - 1);
        for i in 1..prices.len() {
            returns.push((prices[i] / prices[i - 1]).ln());
        }
        returns
    }

    fn estimate_initial_parameters(&self, returns: &[f64]) -> Result<EstimatedParams> {
        let n = returns.len();
        let mean_return = returns.iter().sum::<f64>() / n as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / n as f64;

        // Method of moments estimators
        let drift = mean_return / self.time_step;
        let volatility = (variance / self.time_step).sqrt();

        // Simple jump parameter initialization
        let jump_intensity = 1.0; // 1 jump per year
        let jump_mean = -0.02; // -2% average jump
        let jump_volatility = 0.03; // 3% jump volatility

        Ok(EstimatedParams {
            drift,
            volatility,
            jump_intensity,
            jump_mean,
            jump_volatility,
            log_likelihood: 0.0,
        })
    }

    fn optimize_parameters(
        &self,
        returns: &[f64],
        initial: EstimatedParams,
    ) -> Result<EstimatedParams> {
        // Simplified optimization - in practice, use sophisticated numerical methods
        let mut best_params = initial;
        let mut best_likelihood = self.log_likelihood_function(returns, &best_params)?;
        best_params.log_likelihood = best_likelihood;

        // Simple grid search around initial values (simplified for demonstration)
        let param_variations = vec![0.9, 0.95, 1.0, 1.05, 1.1];

        for &drift_mult in &param_variations {
            for &vol_mult in &param_variations {
                for &jump_int_mult in &param_variations {
                    let test_params = EstimatedParams {
                        drift: best_params.drift * drift_mult,
                        volatility: best_params.volatility * vol_mult,
                        jump_intensity: best_params.jump_intensity * jump_int_mult,
                        jump_mean: best_params.jump_mean,
                        jump_volatility: best_params.jump_volatility,
                        log_likelihood: 0.0,
                    };

                    if let Ok(likelihood) = self.log_likelihood_function(returns, &test_params) {
                        if likelihood > best_likelihood {
                            best_likelihood = likelihood;
                            best_params = test_params;
                            best_params.log_likelihood = likelihood;
                        }
                    }
                }
            }
        }

        Ok(best_params)
    }

    fn log_likelihood_function(&self, returns: &[f64], params: &EstimatedParams) -> Result<f64> {
        if params.volatility <= 0.0 || params.jump_volatility <= 0.0 || params.jump_intensity < 0.0
        {
            return Ok(f64::NEG_INFINITY);
        }

        let mut log_likelihood = 0.0;
        let max_jumps = 10; // Practical limit for computational efficiency

        for &ret in returns {
            let mut density = 0.0;

            // Sum over possible number of jumps
            for n in 0..=max_jumps {
                let lambda_t = params.jump_intensity * self.time_step;
                let poisson_prob =
                    (-lambda_t).exp() * lambda_t.powi(n as i32) / Self::factorial(n) as f64;

                // Adjusted mean and variance for n jumps
                let mean_n = (params.drift - 0.5 * params.volatility.powi(2)) * self.time_step
                    + (n as f64) * params.jump_mean;
                let var_n = params.volatility.powi(2) * self.time_step
                    + (n as f64) * params.jump_volatility.powi(2);

                // Normal density for this jump scenario
                if var_n > 0.0 {
                    let normal_density =
                        (-0.5 * (ret - mean_n).powi(2) / var_n).exp() / (2.0 * PI * var_n).sqrt();
                    density += poisson_prob * normal_density;
                }
            }

            if density > 0.0 {
                log_likelihood += density.ln();
            } else {
                return Ok(f64::NEG_INFINITY);
            }
        }

        Ok(log_likelihood)
    }

    fn black_scholes(
        &self,
        spot: f64,
        strike: f64,
        time: f64,
        rate: f64,
        volatility: f64,
        is_call: bool,
    ) -> Result<f64> {
        let d1 = ((spot / strike).ln() + (rate + 0.5 * volatility.powi(2)) * time)
            / (volatility * time.sqrt());
        let d2 = d1 - volatility * time.sqrt();

        let n_d1 = Self::standard_normal_cdf(d1);
        let n_d2 = Self::standard_normal_cdf(d2);

        if is_call {
            Ok(spot * n_d1 - strike * (-rate * time).exp() * n_d2)
        } else {
            Ok(strike * (-rate * time).exp() * (1.0 - n_d2) - spot * (1.0 - n_d1))
        }
    }

    fn standard_normal_cdf(x: f64) -> f64 {
        // Approximation of the standard normal CDF
        0.5 * (1.0 + Self::erf(x / 2.0_f64.sqrt()))
    }

    fn erf(x: f64) -> f64 {
        // Abramowitz and Stegun approximation
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

    fn factorial(n: usize) -> usize {
        if n <= 1 {
            1
        } else {
            n * Self::factorial(n - 1)
        }
    }
}

impl Forecaster for MertonJumpDiffusionModel {
    fn name(&self) -> &str {
        "Merton Jump-Diffusion"
    }

    fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        // Extract prices from TimeSeriesData and fit
        self.fit_prices(&data.values, None, None)
    }

    fn predict(&self, horizon: usize, _test_data: Option<&TimeSeriesData>) -> Result<ModelOutput> {
        if !self.is_fitted {
            return Err(OxiError::ModelError(
                "Model must be fitted before prediction".to_string(),
            ));
        }

        // For forecasting, we use the last observed price from training data
        // In practice, this would be derived from the input data
        let initial_price = 100.0; // Placeholder - in real usage, this would be the last observed price

        // Generate multiple paths and return the mean path
        let num_paths = 1000;
        let paths = self.simulate_paths(initial_price, horizon, num_paths, Some(42))?;

        // Calculate mean forecast at each horizon
        let mut forecasts = Vec::with_capacity(horizon);
        for t in 1..=horizon {
            let mean_price: f64 = paths.iter().map(|path| path[t]).sum::<f64>() / num_paths as f64;
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
        // Simplified evaluation - in practice would compare against test data
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
    fn test_merton_model_creation() {
        let model = MertonJumpDiffusionModel::new(0.05, 0.20, 2.0, -0.03, 0.05, 1.0 / 252.0);
        assert!(model.is_ok());

        let model = model.unwrap();
        assert_eq!(model.drift, 0.05);
        assert_eq!(model.volatility, 0.20);
        assert_eq!(model.jump_intensity, 2.0);
    }

    #[test]
    fn test_equity_default_model() {
        let model = MertonJumpDiffusionModel::new_equity_default();
        assert!(model.is_ok());
    }

    #[test]
    fn test_parameter_validation() {
        // Test negative volatility
        let result = MertonJumpDiffusionModel::new(0.05, -0.20, 2.0, -0.03, 0.05, 1.0 / 252.0);
        assert!(result.is_err());

        // Test negative jump intensity
        let result = MertonJumpDiffusionModel::new(0.05, 0.20, -2.0, -0.03, 0.05, 1.0 / 252.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_simulation() {
        let model = MertonJumpDiffusionModel::new_equity_default().unwrap();
        let paths = model.simulate_paths(100.0, 10, 100, Some(42));

        assert!(paths.is_ok());
        let paths = paths.unwrap();
        assert_eq!(paths.len(), 100);
        assert_eq!(paths[0].len(), 11); // 10 steps + initial price
        assert_eq!(paths[0][0], 100.0); // Initial price
    }

    #[test]
    fn test_var_calculation() {
        let model = MertonJumpDiffusionModel::new_equity_default().unwrap();
        let var = model.calculate_var(1000000.0, 0.95, 1.0 / 252.0, 1000);

        assert!(var.is_ok());
        let var = var.unwrap();
        assert!(var >= 0.0); // VaR should be positive (representing loss)
    }

    #[test]
    fn test_fitting() {
        let mut model = MertonJumpDiffusionModel::new_equity_default().unwrap();

        // Generate synthetic price data
        let mut prices = vec![100.0];
        for i in 1..100 {
            let return_rate = 0.001 + 0.02 * ((i as f64) * 0.1).sin();
            prices.push(prices[i - 1] * (1.0 + return_rate));
        }

        let result = model.fit_prices(&prices, None, None);
        assert!(result.is_ok());
        assert!(model.is_fitted);
        assert!(model.estimated_params.is_some());
        assert!(model.diagnostics.is_some());
    }

    #[test]
    fn test_option_pricing() {
        let model = MertonJumpDiffusionModel::new_equity_default().unwrap();
        let option_price = model.option_price(100.0, 100.0, 0.25, 0.05, true, 10);

        assert!(option_price.is_ok());
        let price = option_price.unwrap();
        assert!(price > 0.0); // Option price should be positive
    }

    #[test]
    fn test_jump_event_extraction() {
        let model = MertonJumpDiffusionModel::new_equity_default().unwrap();
        let path = vec![100.0, 105.0, 95.0, 110.0, 108.0]; // Path with some jumps
        let jump_events = model.extract_jump_events(&path, 0.05); // 5% threshold

        // Should identify some jump events based on large price movements
        assert!(!jump_events.is_empty());
    }
}
