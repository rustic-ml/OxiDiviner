# OxiDiviner Project Guidelines

This document outlines the baseline rules, best practices, and conventions to follow when contributing to the OxiDiviner project.

## Code Style and Organization

### Do:
- Use the Rust formatting standard (`cargo fmt`)
- Follow Rust naming conventions (snake_case for variables/functions, CamelCase for types/structs)
- Write comprehensive unit tests for all new functionality
- Document all public APIs with proper Rust doc comments
- Organize code logically by functionality (models, data, utils, etc.)
- Use meaningful variable and function names that indicate purpose
- Handle errors properly using the `Result` type from the error module

### Don't:
- Use unsafe code without clear justification and thorough documentation
- Leave `TODO` comments in production code; use GitHub issues instead
- Create large, monolithic functions (aim for <50 lines where possible)
- Add dependencies without discussion (minimize dependency bloat)
- Reinvent functionality already present in the standard library

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

## Performance Considerations

### Do:
- Consider time and memory complexity for operations on large time series
- Use iterators and functional programming patterns where appropriate
- Profile code for hot spots before optimizing
- Use appropriate data structures for the task

### Don't:
- Prematurely optimize code before it's proven to be a bottleneck
- Sacrifice code clarity for minor performance gains
- Use heap allocations unnecessarily

## Documentation

### Do:
- Document the mathematical foundation of implemented models
- Include examples in documentation for complex functionality
- Keep the README and other documentation up-to-date
- Document assumptions and limitations of models

### Don't:
- Use overly technical language without explanation
- Leave parameters or return values undocumented

## Time Series Model Implementation

### Do:
- Implement proper validation for model parameters
- Ensure models correctly handle edge cases (e.g., missing data)
- Follow established statistical best practices for each model type
- Provide methods to assess model fit and forecast accuracy
- Ensure reproducibility of results (use seeded RNGs if needed)

### Don't:
- Implement models without understanding the underlying math
- Ignore numerical stability considerations
- Hard-code assumptions that should be configurable parameters

## API Design

### Do:
- Design for extensibility and composition
- Maintain backward compatibility when possible
- Use appropriate Rust traits to enable generic programming
- Provide builder patterns for complex object construction

### Don't:
- Expose implementation details in public APIs
- Change public interfaces without a deprecation path
- Create overly complex type hierarchies

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