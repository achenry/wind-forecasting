#!/bin/bash

# Instructions for installing environments on the CU Boulder Alpine cluster (for jubo7621)
# Assumes user has already logged in via SSH and acompile as follows:
# ssh jubo7621@login.rc.colorado.edu
# acompile

set -e # Exit immediately if a command exits with !0 status

module purge
ml mambaforge

ENV_DIR="install_rc"

# Function to create and activate environment
create_and_activate_env() {
    env_file="$1"
    env_name="${env_file%.yaml}"
    echo "Creating environment: $env_name"
    if [ ! -f "$ENV_DIR/$env_file" ]; then
        echo "Error: $ENV_DIR/$env_file not found"
        return 1
    fi
    if ! mamba env create -f "$ENV_DIR/$env_file"; then
        echo "Error: Failed to create environment $env_name"
        return 1
    fi
    echo "Activating environment: $env_name"
    mamba activate "$env_name"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to activate environment $env_name"
        return 1
    fi
    return 0
}

# Create and activate environments
for env_file in wind_forecasting_cuda.yaml wind_forecasting_rocm.yaml; do
    if create_and_activate_env "$env_file"; then
        # Install additional packages if needed
        # pip install matplotlib==3.7.1 pyside6==6.5.0
        mamba deactivate
    else
        echo "Skipping environment: ${env_file%.yml}"
    fi
done

echo "Environment creation complete"
