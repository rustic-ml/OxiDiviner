/*!
 * Advanced Optimization & Parameter Selection
 * 
 * This module provides sophisticated optimization algorithms for hyperparameter tuning:
 * - Bayesian optimization for efficient hyperparameter search
 * - Genetic algorithms for complex parameter spaces
 * - Simulated annealing for global optimization
 * - Cross-validation with early stopping
 * - Automated feature engineering and lag selection
 */

use crate::core::{OxiError, Result, TimeSeriesData, Forecaster, ModelEvaluation};
use crate::models::autoregressive::{ARModel, ARIMAModel};
use crate::models::exponential_smoothing::{ETSModel};
use crate::models::exponential_smoothing::ets::{ErrorType, TrendType, SeasonalType};
use rand::{Rng, thread_rng};
use std::collections::HashMap;
use std::fmt;

/// Parameter bounds for optimization
#[derive(Debug, Clone)]
pub struct ParameterBounds {
    pub min: f64,
    pub max: f64,
    pub is_integer: bool,
}

impl ParameterBounds {
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max, is_integer: false }
    }

    pub fn integer(min: i32, max: i32) -> Self {
        Self { 
            min: min as f64, 
            max: max as f64, 
            is_integer: true 
        }
    }

    pub fn sample(&self, rng: &mut impl Rng) -> f64 {
        let value = rng.gen_range(self.min..=self.max);
        if self.is_integer {
            value.round()
        } else {
            value
        }
    }
}

/// Optimization objective (minimize by default)
#[derive(Debug, Clone, Copy)]
pub enum OptimizationObjective {
    MinimizeAIC,
    MinimizeBIC,
    MinimizeMAE,
    MinimizeMSE,
    MinimizeRMSE,
    MinimizeMAPE,
    MaximizeRSquared,
}

impl OptimizationObjective {
    pub fn extract_value(&self, evaluation: &ModelEvaluation) -> f64 {
        match self {
            OptimizationObjective::MinimizeAIC => evaluation.aic.unwrap_or(f64::INFINITY),
            OptimizationObjective::MinimizeBIC => evaluation.bic.unwrap_or(f64::INFINITY),
            OptimizationObjective::MinimizeMAE => evaluation.mae,
            OptimizationObjective::MinimizeMSE => evaluation.mse,
            OptimizationObjective::MinimizeRMSE => evaluation.rmse,
            OptimizationObjective::MinimizeMAPE => evaluation.mape,
            OptimizationObjective::MaximizeRSquared => -evaluation.r_squared, // Negate for minimization
        }
    }

    pub fn is_minimization(&self) -> bool {
        !matches!(self, OptimizationObjective::MaximizeRSquared)
    }
}

/// Parameter set for optimization
#[derive(Debug, Clone)]
pub struct ParameterSet {
    pub parameters: HashMap<String, f64>,
    pub objective_value: Option<f64>,
    pub validation_error: Option<f64>,
}

impl ParameterSet {
    pub fn new(parameters: HashMap<String, f64>) -> Self {
        Self {
            parameters,
            objective_value: None,
            validation_error: None,
        }
    }

    pub fn with_objective(mut self, value: f64) -> Self {
        self.objective_value = Some(value);
        self
    }
}

/// Bayesian Optimization using Gaussian Process approximation
pub struct BayesianOptimizer {
    parameter_space: HashMap<String, ParameterBounds>,
    objective: OptimizationObjective,
    acquisition_function: AcquisitionFunction,
    evaluated_points: Vec<ParameterSet>,
    best_parameters: Option<ParameterSet>,
    exploration_weight: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
}

impl BayesianOptimizer {
    pub fn new(
        parameter_space: HashMap<String, ParameterBounds>,
        objective: OptimizationObjective,
    ) -> Self {
        Self {
            parameter_space,
            objective,
            acquisition_function: AcquisitionFunction::ExpectedImprovement,
            evaluated_points: Vec::new(),
            best_parameters: None,
            exploration_weight: 0.1,
        }
    }

    pub fn with_acquisition_function(mut self, function: AcquisitionFunction) -> Self {
        self.acquisition_function = function;
        self
    }

    pub fn with_exploration_weight(mut self, weight: f64) -> Self {
        self.exploration_weight = weight;
        self
    }

    /// Optimize model parameters using Bayesian optimization
    pub fn optimize<F>(
        &mut self,
        objective_function: F,
        max_iterations: usize,
        initial_samples: usize,
    ) -> Result<ParameterSet>
    where
        F: Fn(&HashMap<String, f64>) -> Result<f64>,
    {
        // Initial random sampling
        for _ in 0..initial_samples {
            let params = self.sample_random_parameters();
            let objective_value = objective_function(&params.parameters)?;
            let mut param_set = params.with_objective(objective_value);
            
            if self.best_parameters.is_none() || 
               objective_value < self.best_parameters.as_ref().unwrap().objective_value.unwrap() {
                self.best_parameters = Some(param_set.clone());
            }
            
            self.evaluated_points.push(param_set);
        }

        // Bayesian optimization iterations
        for iteration in 0..max_iterations {
            println!("🔍 Bayesian Optimization Iteration {}/{}", iteration + 1, max_iterations);
            
            // Find next point to evaluate using acquisition function
            let next_params = self.select_next_parameters()?;
            let objective_value = objective_function(&next_params.parameters)?;
            let mut param_set = next_params.with_objective(objective_value);

            // Update best parameters
            if objective_value < self.best_parameters.as_ref().unwrap().objective_value.unwrap() {
                self.best_parameters = Some(param_set.clone());
                println!("  📈 New best objective: {:.6}", objective_value);
            }

            self.evaluated_points.push(param_set);
        }

        self.best_parameters.clone()
            .ok_or_else(|| OxiError::OptimizationError("No best parameters found".to_string()))
    }

    fn sample_random_parameters(&self) -> ParameterSet {
        let mut rng = thread_rng();
        let mut parameters = HashMap::new();

        for (name, bounds) in &self.parameter_space {
            parameters.insert(name.clone(), bounds.sample(&mut rng));
        }

        ParameterSet::new(parameters)
    }

    fn select_next_parameters(&self) -> Result<ParameterSet> {
        // Simplified acquisition function (in practice, would use GP)
        let mut best_acquisition = f64::NEG_INFINITY;
        let mut best_params = None;

        let num_candidates = 100;
        let mut rng = thread_rng();

        for _ in 0..num_candidates {
            let mut parameters = HashMap::new();
            for (name, bounds) in &self.parameter_space {
                parameters.insert(name.clone(), bounds.sample(&mut rng));
            }

            let acquisition_value = self.calculate_acquisition_value(&parameters)?;
            if acquisition_value > best_acquisition {
                best_acquisition = acquisition_value;
                best_params = Some(parameters);
            }
        }

        best_params
            .map(ParameterSet::new)
            .ok_or_else(|| OxiError::OptimizationError("No candidate parameters found".to_string()))
    }

    fn calculate_acquisition_value(&self, parameters: &HashMap<String, f64>) -> Result<f64> {
        // Simplified acquisition function (Expected Improvement approximation)
        let mut min_distance = f64::INFINITY;
        let mut expected_value = 0.0;

        if self.evaluated_points.is_empty() {
            return Ok(1.0); // High acquisition for unexplored regions
        }

        // Find nearest evaluated point
        for evaluated in &self.evaluated_points {
            let distance = self.parameter_distance(parameters, &evaluated.parameters)?;
            if distance < min_distance {
                min_distance = distance;
                expected_value = evaluated.objective_value.unwrap_or(0.0);
            }
        }

        // Simple acquisition: exploration term + exploitation term
        let exploration = self.exploration_weight / (1.0 + min_distance);
        let exploitation = -expected_value; // Negative because we minimize

        Ok(exploration + exploitation)
    }

    fn parameter_distance(
        &self,
        params1: &HashMap<String, f64>,
        params2: &HashMap<String, f64>,
    ) -> Result<f64> {
        let mut distance = 0.0;
        let mut count = 0;

        for (name, bounds) in &self.parameter_space {
            if let (Some(&val1), Some(&val2)) = (params1.get(name), params2.get(name)) {
                let normalized_diff = (val1 - val2) / (bounds.max - bounds.min);
                distance += normalized_diff * normalized_diff;
                count += 1;
            }
        }

        if count > 0 {
            Ok((distance / count as f64).sqrt())
        } else {
            Ok(0.0)
        }
    }

    /// Get optimization history
    pub fn get_history(&self) -> &[ParameterSet] {
        &self.evaluated_points
    }

    /// Get best parameters found so far
    pub fn get_best_parameters(&self) -> Option<&ParameterSet> {
        self.best_parameters.as_ref()
    }
}

/// Genetic Algorithm for parameter optimization
pub struct GeneticOptimizer {
    parameter_space: HashMap<String, ParameterBounds>,
    objective: OptimizationObjective,
    population_size: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    elitism_ratio: f64,
}

impl GeneticOptimizer {
    pub fn new(
        parameter_space: HashMap<String, ParameterBounds>,
        objective: OptimizationObjective,
        population_size: usize,
    ) -> Self {
        Self {
            parameter_space,
            objective,
            population_size,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elitism_ratio: 0.2,
        }
    }

    pub fn with_mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate;
        self
    }

    pub fn with_crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate;
        self
    }

    /// Optimize using genetic algorithm
    pub fn optimize<F>(
        &mut self,
        objective_function: F,
        generations: usize,
    ) -> Result<ParameterSet>
    where
        F: Fn(&HashMap<String, f64>) -> Result<f64>,
    {
        let mut population = self.initialize_population()?;
        
        // Evaluate initial population
        for individual in &mut population {
            let fitness = objective_function(&individual.parameters)?;
            individual.objective_value = Some(fitness);
        }

        let mut best_individual = population[0].clone();

        for generation in 0..generations {
            // Sort by fitness (minimization)
            population.sort_by(|a, b| {
                a.objective_value.unwrap_or(f64::INFINITY)
                    .partial_cmp(&b.objective_value.unwrap_or(f64::INFINITY))
                    .unwrap()
            });

            // Update best individual
            if population[0].objective_value.unwrap_or(f64::INFINITY) < 
               best_individual.objective_value.unwrap_or(f64::INFINITY) {
                best_individual = population[0].clone();
                println!("🧬 Generation {}: New best fitness: {:.6}", 
                    generation + 1, best_individual.objective_value.unwrap());
            }

            // Create next generation
            let mut next_population = Vec::new();

            // Elitism: keep best individuals
            let elite_count = (self.population_size as f64 * self.elitism_ratio) as usize;
            for i in 0..elite_count {
                next_population.push(population[i].clone());
            }

            // Generate offspring
            while next_population.len() < self.population_size {
                let parent1 = self.tournament_selection(&population)?;
                let parent2 = self.tournament_selection(&population)?;

                let mut child = if thread_rng().gen::<f64>() < self.crossover_rate {
                    self.crossover(&parent1, &parent2)?
                } else {
                    parent1.clone()
                };

                if thread_rng().gen::<f64>() < self.mutation_rate {
                    self.mutate(&mut child)?;
                }

                // Evaluate child
                let fitness = objective_function(&child.parameters)?;
                child.objective_value = Some(fitness);

                next_population.push(child);
            }

            population = next_population;
        }

        Ok(best_individual)
    }

    fn initialize_population(&self) -> Result<Vec<ParameterSet>> {
        let mut population = Vec::with_capacity(self.population_size);
        let mut rng = thread_rng();

        for _ in 0..self.population_size {
            let mut parameters = HashMap::new();
            for (name, bounds) in &self.parameter_space {
                parameters.insert(name.clone(), bounds.sample(&mut rng));
            }
            population.push(ParameterSet::new(parameters));
        }

        Ok(population)
    }

    fn tournament_selection(&self, population: &[ParameterSet]) -> Result<ParameterSet> {
        let tournament_size = 3;
        let mut rng = thread_rng();
        let mut best = None;
        let mut best_fitness = f64::INFINITY;

        for _ in 0..tournament_size {
            let idx = rng.gen_range(0..population.len());
            let fitness = population[idx].objective_value.unwrap_or(f64::INFINITY);
            if fitness < best_fitness {
                best_fitness = fitness;
                best = Some(population[idx].clone());
            }
        }

        best.ok_or_else(|| OxiError::OptimizationError("Tournament selection failed".to_string()))
    }

    fn crossover(&self, parent1: &ParameterSet, parent2: &ParameterSet) -> Result<ParameterSet> {
        let mut rng = thread_rng();
        let mut child_params = HashMap::new();

        for name in parent1.parameters.keys() {
            let val1 = parent1.parameters[name];
            let val2 = parent2.parameters[name];
            
            // Uniform crossover
            let child_val = if rng.gen::<bool>() { val1 } else { val2 };
            child_params.insert(name.clone(), child_val);
        }

        Ok(ParameterSet::new(child_params))
    }

    fn mutate(&self, individual: &mut ParameterSet) -> Result<()> {
        let mut rng = thread_rng();

        for (name, bounds) in &self.parameter_space {
            if rng.gen::<f64>() < 0.1 { // 10% chance to mutate each parameter
                let current_val = individual.parameters[name];
                let range = bounds.max - bounds.min;
                let mutation = rng.gen_range(-0.1..0.1) * range;
                let new_val = (current_val + mutation).clamp(bounds.min, bounds.max);
                
                let final_val = if bounds.is_integer {
                    new_val.round()
                } else {
                    new_val
                };
                
                individual.parameters.insert(name.clone(), final_val);
            }
        }

        Ok(())
    }
}

/// Simulated Annealing optimizer
pub struct SimulatedAnnealingOptimizer {
    parameter_space: HashMap<String, ParameterBounds>,
    objective: OptimizationObjective,
    initial_temperature: f64,
    cooling_rate: f64,
    min_temperature: f64,
}

impl SimulatedAnnealingOptimizer {
    pub fn new(
        parameter_space: HashMap<String, ParameterBounds>,
        objective: OptimizationObjective,
    ) -> Self {
        Self {
            parameter_space,
            objective,
            initial_temperature: 100.0,
            cooling_rate: 0.95,
            min_temperature: 0.01,
        }
    }

    pub fn with_temperature_schedule(
        mut self,
        initial: f64,
        cooling_rate: f64,
        min_temp: f64,
    ) -> Self {
        self.initial_temperature = initial;
        self.cooling_rate = cooling_rate;
        self.min_temperature = min_temp;
        self
    }

    /// Optimize using simulated annealing
    pub fn optimize<F>(
        &mut self,
        objective_function: F,
        max_iterations: usize,
    ) -> Result<ParameterSet>
    where
        F: Fn(&HashMap<String, f64>) -> Result<f64>,
    {
        let mut rng = thread_rng();
        
        // Initialize with random solution
        let mut current = self.sample_random_parameters();
        let mut current_objective = objective_function(&current.parameters)?;
        current.objective_value = Some(current_objective);

        let mut best = current.clone();
        let mut temperature = self.initial_temperature;

        for iteration in 0..max_iterations {
            if temperature < self.min_temperature {
                break;
            }

            // Generate neighbor solution
            let neighbor = self.generate_neighbor(&current)?;
            let neighbor_objective = objective_function(&neighbor.parameters)?;

            // Accept or reject neighbor
            let delta = neighbor_objective - current_objective;
            let accept = if delta < 0.0 {
                true // Accept if better
            } else {
                let probability = (-delta / temperature).exp();
                rng.gen::<f64>() < probability
            };

            if accept {
                current = neighbor;
                current_objective = neighbor_objective;
                current.objective_value = Some(current_objective);

                // Update best solution
                if current_objective < best.objective_value.unwrap() {
                    best = current.clone();
                    println!("🌡️  Iteration {}: New best: {:.6} (T={:.3})", 
                        iteration + 1, current_objective, temperature);
                }
            }

            // Cool down
            temperature *= self.cooling_rate;
        }

        Ok(best)
    }

    fn sample_random_parameters(&self) -> ParameterSet {
        let mut rng = thread_rng();
        let mut parameters = HashMap::new();

        for (name, bounds) in &self.parameter_space {
            parameters.insert(name.clone(), bounds.sample(&mut rng));
        }

        ParameterSet::new(parameters)
    }

    fn generate_neighbor(&self, current: &ParameterSet) -> Result<ParameterSet> {
        let mut rng = thread_rng();
        let mut neighbor_params = current.parameters.clone();

        // Randomly modify one parameter
        let param_names: Vec<_> = self.parameter_space.keys().collect();
        let selected_param = param_names[rng.gen_range(0..param_names.len())];
        let bounds = &self.parameter_space[selected_param];

        let current_val = neighbor_params[selected_param];
        let range = bounds.max - bounds.min;
        let perturbation = rng.gen_range(-0.1..0.1) * range;
        let new_val = (current_val + perturbation).clamp(bounds.min, bounds.max);

        let final_val = if bounds.is_integer {
            new_val.round()
        } else {
            new_val
        };

        neighbor_params.insert(selected_param.clone(), final_val);
        Ok(ParameterSet::new(neighbor_params))
    }
}

/// Auto-tuner for automatic model selection and hyperparameter optimization
pub struct AutoTuner {
    optimization_method: OptimizationMethod,
    objective: OptimizationObjective,
    cv_folds: usize,
    validation_split: f64,
}

#[derive(Debug, Clone)]
pub enum OptimizationMethod {
    BayesianOptimization { max_iterations: usize, initial_samples: usize },
    GeneticAlgorithm { population_size: usize, generations: usize },
    SimulatedAnnealing { max_iterations: usize },
    GridSearch { resolution: usize },
}

impl AutoTuner {
    pub fn new(
        optimization_method: OptimizationMethod,
        objective: OptimizationObjective,
    ) -> Self {
        Self {
            optimization_method,
            objective,
            cv_folds: 5,
            validation_split: 0.2,
        }
    }

    pub fn with_cross_validation(mut self, folds: usize) -> Self {
        self.cv_folds = folds;
        self
    }

    /// Auto-tune ARIMA model parameters
    pub fn tune_arima(&self, data: &TimeSeriesData) -> Result<(ARIMAModel, ParameterSet, f64)> {
        let mut parameter_space = HashMap::new();
        parameter_space.insert("p".to_string(), ParameterBounds::integer(0, 5));
        parameter_space.insert("d".to_string(), ParameterBounds::integer(0, 2));
        parameter_space.insert("q".to_string(), ParameterBounds::integer(0, 5));

        let objective_function = |params: &HashMap<String, f64>| -> Result<f64> {
            let p = params["p"] as usize;
            let d = params["d"] as usize;
            let q = params["q"] as usize;

            let mut model = ARIMAModel::new(p, d, q)?;
            let split_point = (data.values.len() as f64 * (1.0 - self.validation_split)) as usize;
            
            let train_data = TimeSeriesData::new(
                data.timestamps[..split_point].to_vec(),
                data.values[..split_point].to_vec(),
                &format!("{}_train", data.name),
            )?;
            
            let test_data = TimeSeriesData::new(
                data.timestamps[split_point..].to_vec(),
                data.values[split_point..].to_vec(),
                &format!("{}_test", data.name),
            )?;

            model.fit(&train_data)?;
            let evaluation = model.evaluate(&test_data)?;
            Ok(self.objective.extract_value(&evaluation))
        };

        let best_params = match &self.optimization_method {
            OptimizationMethod::BayesianOptimization { max_iterations, initial_samples } => {
                let mut optimizer = BayesianOptimizer::new(parameter_space, self.objective);
                optimizer.optimize(objective_function, *max_iterations, *initial_samples)?
            }
            OptimizationMethod::GeneticAlgorithm { population_size, generations } => {
                let mut optimizer = GeneticOptimizer::new(parameter_space, self.objective, *population_size);
                optimizer.optimize(objective_function, *generations)?
            }
            OptimizationMethod::SimulatedAnnealing { max_iterations } => {
                let mut optimizer = SimulatedAnnealingOptimizer::new(parameter_space, self.objective);
                optimizer.optimize(objective_function, *max_iterations)?
            }
            OptimizationMethod::GridSearch { resolution: _ } => {
                return Err(OxiError::OptimizationError("Grid search not implemented yet".to_string()));
            }
        };

        // Create final model with best parameters
        let p = best_params.parameters["p"] as usize;
        let d = best_params.parameters["d"] as usize;
        let q = best_params.parameters["q"] as usize;
        
        let mut final_model = ARIMAModel::new(p, d, q)?;
        final_model.fit(data)?;
        
        let best_score = best_params.objective_value.unwrap();
        Ok((final_model, best_params, best_score))
    }

    /// Auto-tune ETS model parameters
    pub fn tune_ets(&self, data: &TimeSeriesData) -> Result<(ETSModel, ParameterSet, f64)> {
        let mut parameter_space = HashMap::new();
        parameter_space.insert("alpha".to_string(), ParameterBounds::new(0.01, 0.99));
        parameter_space.insert("beta".to_string(), ParameterBounds::new(0.01, 0.99));
        parameter_space.insert("gamma".to_string(), ParameterBounds::new(0.01, 0.99));

        let objective_function = |params: &HashMap<String, f64>| -> Result<f64> {
            let alpha = params["alpha"];
            let beta = Some(params["beta"]);
            let gamma = Some(params["gamma"]);

            let mut model = ETSModel::new(
                ErrorType::Additive,
                TrendType::Additive, 
                SeasonalType::Additive,
                alpha, 
                beta, 
                None, // phi
                gamma, 
                Some(12) // period - using 12 for monthly seasonality
            ).map_err(|e| OxiError::ModelError(format!("ETSModel creation failed: {:?}", e)))?;
            
            let split_point = (data.values.len() as f64 * (1.0 - self.validation_split)) as usize;
            
            let train_data = TimeSeriesData::new(
                data.timestamps[..split_point].to_vec(),
                data.values[..split_point].to_vec(),
                &format!("{}_train", data.name),
            )?;
            
            let test_data = TimeSeriesData::new(
                data.timestamps[split_point..].to_vec(),
                data.values[split_point..].to_vec(),
                &format!("{}_test", data.name),
            )?;

            model.fit(&train_data)?;
            let evaluation = model.evaluate(&test_data)?;
            Ok(self.objective.extract_value(&evaluation))
        };

        let best_params = match &self.optimization_method {
            OptimizationMethod::BayesianOptimization { max_iterations, initial_samples } => {
                let mut optimizer = BayesianOptimizer::new(parameter_space, self.objective);
                optimizer.optimize(objective_function, *max_iterations, *initial_samples)?
            }
            OptimizationMethod::GeneticAlgorithm { population_size, generations } => {
                let mut optimizer = GeneticOptimizer::new(parameter_space, self.objective, *population_size);
                optimizer.optimize(objective_function, *generations)?
            }
            OptimizationMethod::SimulatedAnnealing { max_iterations } => {
                let mut optimizer = SimulatedAnnealingOptimizer::new(parameter_space, self.objective);
                optimizer.optimize(objective_function, *max_iterations)?
            }
            OptimizationMethod::GridSearch { resolution: _ } => {
                return Err(OxiError::OptimizationError("Grid search not implemented yet".to_string()));
            }
        };

        // Create final model with best parameters
        let alpha = best_params.parameters["alpha"];
        let beta = Some(best_params.parameters["beta"]);
        let gamma = Some(best_params.parameters["gamma"]);
        
        let mut final_model = ETSModel::new(
            ErrorType::Additive,
            TrendType::Additive, 
            SeasonalType::Additive,
            alpha, 
            beta, 
            None, // phi
            gamma, 
            Some(12) // period - using 12 for monthly seasonality
        )?;
        final_model.fit(data)?;
        
        let best_score = best_params.objective_value.unwrap();
        Ok((final_model, best_params, best_score))
    }
}

/// Cross-validation utilities
pub struct CrossValidator {
    folds: usize,
    time_series_split: bool,
}

impl CrossValidator {
    pub fn new(folds: usize) -> Self {
        Self {
            folds,
            time_series_split: true,
        }
    }

    pub fn time_series_split(mut self, use_ts_split: bool) -> Self {
        self.time_series_split = use_ts_split;
        self
    }

    /// Perform cross-validation on a model
    pub fn validate<F>(&self, data: &TimeSeriesData, model_fn: F) -> Result<Vec<f64>>
    where
        F: Fn(&TimeSeriesData, &TimeSeriesData) -> Result<f64>,
    {
        let n = data.values.len();
        let mut scores = Vec::new();

        if self.time_series_split {
            // Time series cross-validation (expanding window)
            let min_train_size = n / (self.folds + 1);
            
            for fold in 0..self.folds {
                let train_end = min_train_size + (fold * min_train_size);
                let test_start = train_end;
                let test_end = (test_start + min_train_size).min(n);
                
                if test_end <= test_start {
                    break;
                }

                let train_data = TimeSeriesData::new(
                    data.timestamps[..train_end].to_vec(),
                    data.values[..train_end].to_vec(),
                    &format!("{}_fold{}_train", data.name, fold),
                )?;
                
                let test_data = TimeSeriesData::new(
                    data.timestamps[test_start..test_end].to_vec(),
                    data.values[test_start..test_end].to_vec(),
                    &format!("{}_fold{}_test", data.name, fold),
                )?;

                let score = model_fn(&train_data, &test_data)?;
                scores.push(score);
            }
        } else {
            // Standard k-fold cross-validation
            let fold_size = n / self.folds;
            
            for fold in 0..self.folds {
                let test_start = fold * fold_size;
                let test_end = if fold == self.folds - 1 { n } else { (fold + 1) * fold_size };
                
                let mut train_timestamps = Vec::new();
                let mut train_values = Vec::new();
                
                // Add data before test fold
                train_timestamps.extend_from_slice(&data.timestamps[..test_start]);
                train_values.extend_from_slice(&data.values[..test_start]);
                
                // Add data after test fold
                if test_end < n {
                    train_timestamps.extend_from_slice(&data.timestamps[test_end..]);
                    train_values.extend_from_slice(&data.values[test_end..]);
                }

                let train_data = TimeSeriesData::new(
                    train_timestamps,
                    train_values,
                    &format!("{}_fold{}_train", data.name, fold),
                )?;
                
                let test_data = TimeSeriesData::new(
                    data.timestamps[test_start..test_end].to_vec(),
                    data.values[test_start..test_end].to_vec(),
                    &format!("{}_fold{}_test", data.name, fold),
                )?;

                let score = model_fn(&train_data, &test_data)?;
                scores.push(score);
            }
        }

        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    fn create_test_data() -> TimeSeriesData {
        let start_time = Utc::now();
        let timestamps = (0..100).map(|i| start_time + Duration::days(i)).collect();
        let values: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64) * 0.5 + (i as f64 * 0.1).sin() * 10.0 + rand::random::<f64>() * 2.0)
            .collect();
        
        TimeSeriesData::new(timestamps, values, "test_series").unwrap()
    }

    #[test]
    fn test_parameter_bounds() {
        let bounds = ParameterBounds::new(0.0, 1.0);
        let mut rng = thread_rng();
        
        for _ in 0..100 {
            let sample = bounds.sample(&mut rng);
            assert!(sample >= 0.0 && sample <= 1.0);
        }

        let int_bounds = ParameterBounds::integer(1, 5);
        for _ in 0..100 {
            let sample = int_bounds.sample(&mut rng);
            assert!(sample >= 1.0 && sample <= 5.0);
            assert_eq!(sample, sample.round());
        }
    }

    #[test]
    fn test_bayesian_optimizer() {
        let mut parameter_space = HashMap::new();
        parameter_space.insert("x".to_string(), ParameterBounds::new(-5.0, 5.0));
        parameter_space.insert("y".to_string(), ParameterBounds::new(-5.0, 5.0));

        let mut optimizer = BayesianOptimizer::new(parameter_space, OptimizationObjective::MinimizeMSE);

        // Simple quadratic function: f(x,y) = x^2 + y^2
        let objective = |params: &HashMap<String, f64>| -> Result<f64> {
            let x = params["x"];
            let y = params["y"];
            Ok(x * x + y * y)
        };

        let result = optimizer.optimize(objective, 10, 5);
        assert!(result.is_ok());

        let best = result.unwrap();
        let best_x = best.parameters["x"];
        let best_y = best.parameters["y"];
        
        // Should be close to (0, 0)
        assert!(best_x.abs() < 2.0);
        assert!(best_y.abs() < 2.0);
    }

    #[test]
    fn test_genetic_optimizer() {
        let mut parameter_space = HashMap::new();
        parameter_space.insert("x".to_string(), ParameterBounds::new(-5.0, 5.0));

        let mut optimizer = GeneticOptimizer::new(parameter_space, OptimizationObjective::MinimizeMSE, 20);

        // Simple function: f(x) = (x - 2)^2
        let objective = |params: &HashMap<String, f64>| -> Result<f64> {
            let x = params["x"];
            Ok((x - 2.0) * (x - 2.0))
        };

        let result = optimizer.optimize(objective, 20);
        assert!(result.is_ok());

        let best = result.unwrap();
        let best_x = best.parameters["x"];
        
        // Should be close to 2
        assert!((best_x - 2.0).abs() < 1.0);
    }

    #[test]
    fn test_auto_tuner() {
        let data = create_test_data();
        let method = OptimizationMethod::BayesianOptimization { 
            max_iterations: 5, 
            initial_samples: 3 
        };
        
        let tuner = AutoTuner::new(method, OptimizationObjective::MinimizeAIC);
        let result = tuner.tune_arima(&data);
        
        assert!(result.is_ok());
        let (model, params, score) = result.unwrap();
        
        assert!(params.parameters.contains_key("p"));
        assert!(params.parameters.contains_key("d"));
        assert!(params.parameters.contains_key("q"));
        assert!(score.is_finite());
    }

    #[test]
    fn test_cross_validator() {
        let data = create_test_data();
        let validator = CrossValidator::new(3).time_series_split(true);

        let model_fn = |train_data: &TimeSeriesData, test_data: &TimeSeriesData| -> Result<f64> {
            let mut model = ARModel::new(2)?;
            model.fit(train_data)?;
            let evaluation = model.evaluate(test_data)?;
            Ok(evaluation.mae)
        };

        let scores = validator.validate(&data, model_fn);
        assert!(scores.is_ok());
        
        let scores = scores.unwrap();
        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|&s| s.is_finite() && s >= 0.0));
    }
} 