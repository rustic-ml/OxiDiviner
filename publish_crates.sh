#!/bin/bash
set -e  # Exit on error

# Parse command line arguments
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run]"
            exit 1
            ;;
    esac
done

if $DRY_RUN; then
    echo "Running in dry-run mode. No changes will be published."
fi

# Define the order of crates to publish
CRATES=(
    "oxidiviner-math"
    "oxidiviner-core"
    "oxidiviner-autoregressive"
    "oxidiviner-exponential-smoothing"
    "oxidiviner-moving-average"
    "oxidiviner-garch"
    "oxidiviner"
)

# The new version to set for all crates
NEW_VERSION="0.3.7"

# Function to update a crate's Cargo.toml to enable publishing
update_cargo_toml() {
    local crate_path=$1
    echo "Updating $crate_path/Cargo.toml to enable publishing..."
    
    if ! $DRY_RUN; then
        # Remove the publish = false line
        sed -i '/publish = false/d' "$crate_path/Cargo.toml"
    else
        echo "[DRY RUN] Would remove 'publish = false' from $crate_path/Cargo.toml"
    fi
}

# Function to update dependency versions in a crate
update_dependencies() {
    local crate_path=$1
    echo "Updating dependency versions in $crate_path/Cargo.toml..."
    
    if ! $DRY_RUN; then
        # Replace version numbers for internal dependencies
        for dep in "${CRATES[@]}"; do
            if [ "$dep" != "oxidiviner" ]; then  # Don't update references to the main crate
                # Update dependencies with explicit version numbers
                sed -i "s/$dep = { path = \"../$dep\", version = \"[0-9.]*\" }/$dep = { path = \"../$dep\", version = \"$NEW_VERSION\" }/g" "$crate_path/Cargo.toml"
            fi
        done
    else
        echo "[DRY RUN] Would update internal dependency versions to $NEW_VERSION in $crate_path/Cargo.toml"
    fi
}

# Function to publish a crate
publish_crate() {
    local crate_path=$1
    local crate_name=$(basename "$crate_path")
    echo "============================================="
    echo "Publishing $crate_name..."
    echo "============================================="
    
    # Update Cargo.toml to enable publishing
    update_cargo_toml "$crate_path"
    
    # Update dependency versions
    update_dependencies "$crate_path"
    
    # Display the updated Cargo.toml if not in dry-run mode
    if ! $DRY_RUN; then
        echo "Updated Cargo.toml content:"
        cat "$crate_path/Cargo.toml"
    fi
    
    # Ask for confirmation before publishing
    if ! $DRY_RUN; then
        read -p "Proceed with publishing $crate_name? (y/n): " confirm
        if [[ $confirm != [yY] ]]; then
            echo "Skipping $crate_name..."
            return
        fi
        
        # Navigate to the crate directory and publish
        cd "$crate_path"
        cargo publish --allow-dirty
        cd ..
        
        # Wait a bit for crates.io to process the upload
        echo "Waiting for crates.io to process the upload..."
        sleep 30
    else
        echo "[DRY RUN] Would publish $crate_name with 'cargo publish --allow-dirty'"
    fi
}

# Main publishing loop
for crate in "${CRATES[@]}"; do
    publish_crate "$crate"
done

if $DRY_RUN; then
    echo "Dry run completed successfully. No changes were published."
else
    echo "All crates published successfully!" 