#!/bin/bash

## sbatch job_amd.slurm
## sinteractive --partition=ami100 --time=00:20:00 --ntasks=5 --gres=gpu:1

#SBATCH --partition=ami100 # ami100 or atesting_mi100
#SBATCH --job-name=STTRE_test_Juan_AMD
#SBATCH --output=output_amd.%j.out
#SBATCH --error=error_amd.%j.err
#SBATCH --time=0:30:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
##SBATCH --ntasks-per-node=1
##SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jubo7621@colorado.edu

# Load necessary modules
module purge
ml mambaforge
ml rocm

mamba activate wind_forecasting_rocm

cd /projects/jubo7621/wind-forecasting

export HIP_VISIBLE_DEVICES=0
# export HSA_OVERRIDE_GFX_VERSION=9.0.0 # 10.3.0

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