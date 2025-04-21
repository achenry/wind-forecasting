#!/bin/bash

module purge
ml mambaforge

# List of environment names
environments=("wind_forecasting_cuda" "wind_forecasting_rocm")

export_dir="/projects/$USER/wind-forecasting/install_rc"
mkdir -p "$export_dir"

# Export each environment to a YAML file
for env_name in "${environments[@]}"
do
    echo "Exporting environment: $env_name"
    mamba activate "$env_name"
    mamba env export > "$export_dir/${env_name}.yml"
    mamba deactivate
    echo "Exported $env_name to $export_dir/${env_name}.yml"
done

echo "Environment export complete"