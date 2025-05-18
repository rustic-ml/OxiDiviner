use std::error::Error;

// Main function for the demo
fn main() -> Result<(), Box<dyn Error>> {
    println!("OxiDiviner - Time Series Forecasting");
    println!("====================================\n");

    println!("This demo showcases the standardized model interface in OxiDiviner.\n");
    println!("Each model in OxiDiviner now has:");
    println!("1. One standard public entry point via the `predict()` method");
    println!("2. Standardized output via the `ModelOutput` struct\n");

    println!("The standardized interface helps traders by:");
    println!("- Providing consistent access to forecasts from any model");
    println!("- Including confidence intervals when available");
    println!("- Including evaluation metrics to assess forecast quality");
    println!("- Offering additional metadata useful for trading decisions\n");

    println!("Example ModelOutput contains:");
    println!("- Model name:            Simple string identifier");
    println!("- Forecasts:             Vector of predicted values");
    println!("- Confidence intervals:  Optional lower/upper bounds");
    println!("- Evaluation metrics:    MAE, MSE, RMSE, MAPE, etc.");
    println!("- Metadata:              Custom key-value pairs for trader info\n");

    println!("To use any model with the standardized interface:");
    println!("```rust");
    println!("// 1. Create and fit the model");
    println!("let mut model = SomeModel::new(params);");
    println!("model.fit(&training_data)?;");
    println!("");
    println!("// 2. Generate standardized predictions");
    println!("let output = model.predict(horizon, Some(&test_data))?;");
    println!("");
    println!("// 3. Use the standardized output");
    println!("println!(\"Model: {{}}\", output.model_name);");
    println!("println!(\"Forecast: {{:?}}\", output.forecasts);");
    println!("if let Some(eval) = &output.evaluation {{");
    println!("    println!(\"RMSE: {{}}\", eval.rmse);");
    println!("}}");
    println!("```\n");

    println!("This standardization ensures all models provide consistent output");
    println!("that can be easily integrated into trading systems and strategies.");

    Ok(())
}
