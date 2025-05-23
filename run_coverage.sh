#!/bin/bash

# OxiDiviner Code Coverage Runner (Updated for Single Crate)
# This script runs comprehensive code coverage analysis on the unified oxidiviner crate

echo "🚀 Running OxiDiviner Code Coverage Analysis..."
echo "================================================"
echo "📦 Target: Single Crate Architecture (oxidiviner)"
echo ""

# Check if tarpaulin is installed
if ! command -v cargo-tarpaulin &> /dev/null; then
    echo "📦 Installing cargo-tarpaulin..."
    cargo install cargo-tarpaulin
fi

# Clean previous coverage data
echo "🧹 Cleaning previous coverage data..."
rm -rf target/coverage/
rm -rf oxidiviner/target/coverage/
rm -rf oxidiviner/target/tarpaulin/
rm -f *.profraw

# Change to oxidiviner directory
cd oxidiviner

echo "📊 Running comprehensive coverage analysis..."
echo "   - Library tests"
echo "   - Integration tests" 
echo "   - Doctests"
echo "   - All model types (ARIMA, MA, ES, GARCH, AR)"
echo ""

# Run coverage analysis with comprehensive options
cargo tarpaulin \
    --config ../tarpaulin.toml \
    --all-features \
    --timeout 600 \
    --out Html \
    --out Lcov \
    --out Json \
    --out Stdout \
    --output-dir target/coverage \
    --skip-clean \
    --verbose

# Check coverage results
if [ -f "target/coverage/tarpaulin-report.html" ]; then
    echo ""
    echo "✅ Coverage report generated successfully!"
    echo "📁 HTML Report: oxidiviner/target/coverage/tarpaulin-report.html"
    echo "📁 LCOV Report: oxidiviner/target/coverage/lcov.info" 
    echo "📁 JSON Report: oxidiviner/target/coverage/cobertura.json"
    echo "📊 View in browser: file://$(pwd)/target/coverage/tarpaulin-report.html"
else
    echo "⚠️  HTML report not found, but stdout coverage was displayed above"
fi

# Generate quick summary
echo ""
echo "🎯 Quick Coverage Check:"
echo "========================"

# Count total tests
TOTAL_TESTS=$(cargo test --all-features 2>&1 | grep -E "test result:|running" | tail -1 | grep -o "[0-9]\+ passed" | head -1 | grep -o "[0-9]\+")

if [ -z "$TOTAL_TESTS" ]; then
    TOTAL_TESTS="N/A"
fi

echo "🧪 Total Tests: $TOTAL_TESTS"

# Extract coverage from stdout (if available)
if [ -f "target/coverage/cobertura.json" ]; then
    echo "📊 Detailed coverage available in JSON/HTML reports"
else
    echo "📊 Coverage details in stdout above"
fi

echo ""
echo "📋 Coverage Focus Areas:"
echo "   ✅ Core data structures and validation"
echo "   ✅ Mathematical functions and metrics"
echo "   ✅ All forecasting models (ARIMA, AR, MA, ES, GARCH)"
echo "   ✅ API layers (Quick API, Enhanced API)"
echo "   ✅ Error handling and edge cases"
echo "   ✅ Integration tests for complete workflows"
echo ""
echo "📖 See updated CODE_COVERAGE_REPORT.md for detailed analysis"
echo "🔄 To run again: ./run_coverage.sh" 