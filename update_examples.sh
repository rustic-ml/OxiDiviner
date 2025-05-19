#!/bin/bash

# This script updates all the examples and tests to use the new crate structure
echo "Updating examples and tests for new workspace structure..."

# Add required dependencies to the workspace 
echo "Adding dependencies to workspace Cargo.toml..."
cat <<EOT >> Cargo.toml

[dev-dependencies]
oxidiviner-core = { path = "./oxidiviner-core" }
oxidiviner-math = { path = "./oxidiviner-math" }
oxidiviner-moving-average = { path = "./oxidiviner-moving-average" }
oxidiviner-exponential-smoothing = { path = "./oxidiviner-exponential-smoothing" }
rand = { workspace = true }
chrono = { workspace = true }
EOT

# Replace import references in examples
echo "Updating examples..."
find examples -name "*.rs" -type f -exec sed -i \
    -e 's/use oxidiviner::/use oxidiviner_core::/g' \
    -e 's/use oxidiviner::ModelsOHLCVData;/use oxidiviner_core::OHLCVData;/g' \
    -e 's/use oxidiviner::TimeSeriesData;/use oxidiviner_core::TimeSeriesData;/g' \
    -e 's/use oxidiviner::models::exponential_smoothing::simple::/use oxidiviner_exponential_smoothing::simple::/g' \
    -e 's/use oxidiviner::models::exponential_smoothing::holt::/use oxidiviner_exponential_smoothing::holt::/g' \
    -e 's/use oxidiviner::models::exponential_smoothing::holt_winters::/use oxidiviner_exponential_smoothing::holt_winters::/g' \
    -e 's/use oxidiviner::models::exponential_smoothing::ets::/use oxidiviner_exponential_smoothing::ets::/g' \
    -e 's/use oxidiviner::models::moving_average::/use oxidiviner_moving_average::/g' \
    -e 's/ModelsOHLCVData {/OHLCVData::new(/g' \
    -e 's/SESModel::new(\([^,)]*\)), None/SimpleESModel::new(\1/g' \
    -e 's/MAModel::new(\([^,)]*\)), None/MAModel::new(\1/g' \
    {} \;

# Update name field in OHLCV objects
find examples -name "*.rs" -type f -exec sed -i \
    -e 's/name: "\([^"]*\)".to_string(),/Some("\1".to_string())/g' \
    {} \;

# Update tests
echo "Updating tests..."
find tests -name "*.rs" -type f -exec sed -i \
    -e 's/use oxidiviner::/use oxidiviner_core::/g' \
    -e 's/use oxidiviner::ModelsOHLCVData;/use oxidiviner_core::OHLCVData;/g' \
    -e 's/use oxidiviner::TimeSeriesData;/use oxidiviner_core::TimeSeriesData;/g' \
    -e 's/use oxidiviner::models::exponential_smoothing::simple::/use oxidiviner_exponential_smoothing::simple::/g' \
    -e 's/use oxidiviner::models::exponential_smoothing::holt::/use oxidiviner_exponential_smoothing::holt::/g' \
    -e 's/use oxidiviner::models::exponential_smoothing::holt_winters::/use oxidiviner_exponential_smoothing::holt_winters::/g' \
    -e 's/use oxidiviner::models::exponential_smoothing::ets::/use oxidiviner_exponential_smoothing::ets::/g' \
    -e 's/use oxidiviner::models::moving_average::/use oxidiviner_moving_average::/g' \
    -e 's/ModelsOHLCVData {/OHLCVData::new(/g' \
    -e 's/SESModel::new(\([^,)]*\)), None/SimpleESModel::new(\1/g' \
    -e 's/MAModel::new(\([^,)]*\)), None/MAModel::new(\1/g' \
    {} \;

# Update name field in OHLCV objects for tests
find tests -name "*.rs" -type f -exec sed -i \
    -e 's/name: "\([^"]*\)".to_string(),/Some("\1".to_string())/g' \
    -e 's/"\([A-Z0-9]*\)"/Some("\1".to_string())/g' \
    {} \;

echo "Done! You may need to manually check the files for any remaining issues."

# Add note about manually fixing up any remaining issues
echo "Note: You may need to manually fix up any files that couldn't be properly updated by the script."
echo "Pay special attention to model constructors and data structure creation." 