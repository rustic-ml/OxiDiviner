# OxiDiviner Development Guidelines

## Important Dependency Rules
1. **Rustalib**: This project MUST use the rustalib library (version 1.0.8 or newer) for:
   - Reading CSV files
   - Reading Parquet files
   - Converting file data to dataframes
   - All technical indicators calculations

2. **Exclusive Use**: Do NOT implement custom alternatives or use other libraries for these functionalities. Always use rustalib for these purposes.

## Code Style
- Follow Rust idiomatic patterns
- Use snake_case for variables and functions
- Use PascalCase for types and traits
- Maximum line length: 100 characters
- No tabs, use 4 spaces for indentation

## Documentation
All public APIs must include documentation with:
- A summary of what the function or type does
- Description of parameters and return values
- Examples of usage
- Mathematical foundation (where applicable)

## Testing
- All code must have unit tests
- Tests should be placed in the `/tests` directory, not alongside implementation
- Each logical component should have its own test file

## Performance Considerations
- Optimize for numerical stability and accuracy first
- Then optimize for performance
- Consider implementing parallel versions of computationally intensive operations

## Versioning and Dependencies
- Use semantic versioning
- Pin dependencies to compatible versions
- Document all breaking changes

## Time Series Specific Guidelines
- Use appropriate data structures for time series data
- Handle missing values gracefully
- Implement proper validation for time series models
- Document assumptions and mathematical foundations of all models

## Technical Indicators
- ALL technical indicators MUST use rustalib implementations
- Do NOT create custom implementations of common indicators
- Standard indicators (RSI, MACD, Bollinger Bands, etc.) MUST use rustalib API

## File Processing
- CSV and Parquet file reading MUST use rustalib
- Use proper error handling for file operations
- Validate input data before processing

Remember that consistency across the codebase is more important than personal style preferences. Follow these guidelines to ensure the project maintains high quality and consistency.

## Git Workflow

### Do:
- Write clear, descriptive commit messages
- Create feature branches for new work
- Reference issue numbers in commit messages when applicable
- Keep pull requests focused on a single feature or fix
- Update tests when modifying existing functionality

### Don't:
- Commit directly to the main branch
- Create large pull requests with multiple unrelated changes
- Commit code that doesn't compile or fails tests

## Community Interaction

### Do:
- Be respectful and constructive in code reviews
- Welcome newcomers and help them navigate the codebase
- Acknowledge contributions and provide feedback

### Don't:
- Merge code without review
- Dismiss feedback without consideration
- Make significant architectural changes without discussion

---

These guidelines aim to maintain code quality and consistency throughout the OxiDiviner project. They are meant to evolve over time as the project grows and matures. 