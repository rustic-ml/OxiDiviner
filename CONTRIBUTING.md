# Contributing to OxiDiviner

Thank you for your interest in contributing to OxiDiviner! This document provides guidelines and instructions for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork to your local machine
3. Add the original repository as an upstream remote:
   ```bash
   git remote add upstream https://github.com/[original-owner]/OxiDiviner.git
   ```
4. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. Make your changes, following the [code guidelines](GUIDELINES.md)
2. Add tests for your changes
3. Ensure all tests pass with `cargo test`
4. Format your code with `cargo fmt`
5. Run `cargo clippy` and address any lints
6. Commit your changes with a descriptive message
7. Push to your fork
8. Submit a pull request

## Pull Request Process

1. Ensure your PR description clearly describes the problem and solution
2. Include any relevant issue numbers in the PR description
3. Update documentation as needed
4. Make sure CI checks pass
5. Wait for review from maintainers

## Code Structure

The OxiDiviner codebase is organized as follows:

- `/src`: Core library code
  - `/data`: Time series data structures and manipulation
  - `/models`: Implementation of forecasting models
  - `/utils`: Utility functions and helpers
  - `/error`: Error types and handling
- `/examples`: Example usage of the library
- `/tests`: Integration tests

## Adding New Models

When adding a new time series model:

1. Create a new module in the appropriate directory under `/src/models/`
2. Implement the model following the existing patterns
3. Ensure your model implements the appropriate traits
4. Add comprehensive unit tests
5. Add an example in the `/examples` directory
6. Update documentation to include the new model

## Testing

All code should be thoroughly tested:

1. Unit tests alongside the code they test
2. Integration tests in the `/tests` directory
3. Consider property-based tests where appropriate
4. Benchmark tests for performance-critical code

## Documentation

Good documentation is essential:

1. All public APIs should have doc comments
2. Include examples in documentation
3. Explain the mathematical foundation of any statistical methods
4. Document any assumptions or limitations

## Reporting Issues

When reporting issues, please include:

1. A clear description of the problem
2. Steps to reproduce
3. Expected vs. actual behavior
4. Version information (Rust version, crate version)
5. Any relevant error messages or logs

## Code of Conduct

Please note that this project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Questions?

If you have questions about contributing, please open an issue labeled "question" or contact the maintainers directly.

Thank you for contributing to OxiDiviner! 