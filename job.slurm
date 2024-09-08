#!/bin/bash

#SBATCH --partition=atesting_mi100 # partition name, change for ami100, aa100, atesting_a100 or atesting_mi100
#SBATCH --job-name=STTRE_test_Juan
#SBATCH --output=output.%j.out
#SBATCH --error=error.%j.err
##SBATCH --time=0:05:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
##SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --mem=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jubo7621@colorado.edu

# Load necessary modules
module purge
ml mambaforge

# Activate your Mamba environment
mamba activate wind_forecasting_env

# Run your Python script
echo "========== Running Python script =========="
python STTRE_test.py
echo "========== Python script completed =========="