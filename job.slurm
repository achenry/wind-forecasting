#!/bin/bash
#SBATCH --partition=ami100
#SBATCH --job-name=STTRE_test_Juan
#SBATCH --output=output.%j.out
#SBATCH --error=error.%j.err
#SBATCH --time=0:10:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
##SBATCH --ntasks-per-node=1
##SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jubo7621@colorado.edu

# Load necessary modules
module purge
ml mambaforge

# Activate your Mamba environment
mamba activate wind_forecasting_env

# Run your Python script
python STTRE_test.py