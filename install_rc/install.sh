#!/bin/bash

# Instructions for installing environments on the CU Boulder Alpine cluster (for jubo7621)
# Assumes user has already logged in via SSH and acompile as follows:#
# ssh jubo7621@login.rc.colorado.edu
# acompile

module purge
ml mambaforge

# Create and activate environments
for env_file in wind_forecasting_cuda_test.yml wind_forecasting_env_test.yml wind_forecasting_rocm_test.yml wind_preprocessing_test.yml
do
    env_name="${env_file%.yml}"
    echo "Creating environment: $env_name"
    if [ -f "$env_file" ]; then
        mamba env create -f "$env_file" -n "$env_name"
    else
        echo "Environment file $env_file not found"
    fi
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