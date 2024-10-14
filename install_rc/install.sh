#!/bin/bash

# Instructions for installing environments on the CU Boulder Alpine cluster (for jubo7621)
# Assumes user has already logged in via SSH and acompile as follows:
# ssh jubo7621@login.rc.colorado.edu
# acompile

set -e # Exit immediately if a command exits with a non-zero status

module purge
ml mambaforge

ENV_DIR="install_rc"

# Create and activate environments
for env_file in wind_forecasting_cuda.yml wind_forecasting_rocm.yml; do
    env_name="${env_file%.yml}"
    echo "Creating environment: $env_name"
    if [ ! -f "$ENV_DIR/$env_file" ]; then
        echo "Error: $ENV_DIR/$env_file not found"
        continue
    fi
    if ! mamba env create -n "$env_name" -f "$ENV_DIR/$env_file"; then
        echo "Error: Failed to create environment $env_name"
        continue
    fi
    echo "Activating environment: $env_name"
    if ! mamba activate "$env_name"; then
        echo "Error: Failed to activate environment $env_name"
        continue
    fi
    mamba deactivate
done

cd /projects/$USER/wind-forecasting/wind-forecasting/models

# ***Uncomment these lines if you need to install requirements for specific models***
# python -m pip install -r ./spacetimeformer/requirements.txt
# python ./spacetimeformer/setup.py develop
# python -m pip install -r ./Informer2020/requirements.txt
# python -m pip install -r ./Autoformer/requirements.txt

# ***Uncomment this line if you need to update submodules***
# git pull --recurse-submodules

# ***Uncomment these lines if you need to clone specific repositories***
# git clone https://github.com/achenry/spacetimeformer.git
# git clone https://github.com/achenry/Autoformer.git
# git clone https://github.com/achenry/Informer2020.git

echo "Environment creation complete"