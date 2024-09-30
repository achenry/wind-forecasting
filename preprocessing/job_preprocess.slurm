#!/bin/bash

## sbatch job.slurm
## sinteractive --partition=amilan --time=00:20:00 --ntasks=5
## sinteractive --partition=amilan --time=00:20:00 --ntasks=20

#SBATCH --partition=amilan
#SBATCH --job-name=preprocess_floris_data
#SBATCH --output=output_preprocess_floris.%j.out
#SBATCH --error=error_preprocess_floris.%j.err
#SBATCH --time=0:30:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
##SBATCH --ntasks-per-node=1
##SBATCH --nodes=1
#SBATCH --mem=2G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jubo7621@colorado.edu

# Load necessary modules
module purge
ml mambaforge
mamba activate wind_preprocessing

cd /projects/jubo7621/wind-forecasting/

# Run your Python script with timing
echo "========== Running Python script =========="
start_time=$(date +%s)
python ./preprocessing/preprocessing_floris.py # Run my Python script
end_time=$(date +%s)
echo "========== Python script completed =========="

# Calculate and display execution time
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"

## squeue -u jubo7621
## scancel jobid
## For detailed resource usage, run: sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,AveRSS,AveVMSize"

## sacct --user=jubo7621
## scontrol show job job_id