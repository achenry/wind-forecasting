#!/bin/bash

## sbatch job.slurm
## sinteractive --partition=ami100 --time=00:20:00 --ntasks=5 --gres=gpu:1
## sinteractive --partition=aa100 --time=00:20:00 --ntasks=20 --gres=gpu:2

#SBATCH --partition=atesting_aa100 # partition name, change for ami100, aa100, atesting_a100 or atesting_mi100
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
#SBATCH --mem=2G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jubo7621@colorado.edu

# Load necessary modules
module purge
ml mambaforge
mamba activate wind_forecasting_cuda
## ml rocm
ml cuda

cd /projects/jubo7621/wind-forecasting

# Activate your Mamba environment
mamba activate wind_forecasting_env

# Run your Python script with timing
echo "========== Running Python script =========="
start_time=$(date +%s)
python STTRE_test.py # Run my Python script
end_time=$(date +%s)
echo "========== Python script completed =========="

# Calculate and display execution time
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"

## squeue -u jubo7621
## scancel jobid
## For detailed resource usage, run: sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,AveRSS,AveVMSize"