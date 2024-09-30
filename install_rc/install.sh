#!/bin/bash

# Instructions for installing environments on the CU Boulder Alpine cluster (for jubo7621)

ssh jubo7621@login.rc.colorado.edu
# https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux

# Create and activate environments
for env_file in wind_forecasting_cuda.yml wind_forecasting_env.yml wind_forecasting_rocm.yml wind_preprocessing_env.yml
do
    env_name="${env_file%.yml}"
    mamba env create -f "@install_rc/$env_file" -n "$env_name"
    mamba activate "$env_name"
    mamba deactivate
done

# Clone the repository
cd /projects/$USER/
git clone --recurse-submodules https://github.com/achenry/wind-forecasting.git

cd wind-forecasting/wind-forecasting/models

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
