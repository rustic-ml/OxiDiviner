pub mod daily;
pub mod minute;

// Re-export ETS components
pub use daily::ETSModel as DailyETSModel;
pub use minute::ETSModel as MinuteETSModel;

// ETS components enum
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ETSComponent {
    None,
    Additive,
    Multiplicative,
    Damped,
}

// Helper function to convert component type to char for model naming
pub fn component_to_char(component: ETSComponent) -> char {
    match component {
        ETSComponent::None => 'N',
        ETSComponent::Additive => 'A',
        ETSComponent::Multiplicative => 'M',
        ETSComponent::Damped => 'D',
    }
}

// Model evaluation metrics structure
pub struct ModelEvaluation {
    pub model_name: String,
    pub mae: f64,
    pub mse: f64,
    pub rmse: f64,
    pub mape: f64,
}

// Error metrics functions
pub fn mean_absolute_error(actual: &[f64], forecast: &[f64]) -> f64 {
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

pub fn mean_squared_error(actual: &[f64], forecast: &[f64]) -> f64 {
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

pub fn mean_absolute_percentage_error(actual: &[f64], forecast: &[f64]) -> f64 {
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