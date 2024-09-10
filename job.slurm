#!/bin/bash

## sbatch job.slurm
## sinteractive --partition=ami100 --time=00:20:00 --ntasks=5 --gres=gpu:1
## sinteractive --partition=aa100 --time=00:20:00 --ntasks=20 --gres=gpu:2

#SBATCH --partition=aa100 # partition name, change for ami100, aa100, atesting_a100 or atesting_mi100
#SBATCH --job-name=STTRE_test_Juan_Nvidia
#SBATCH --output=output.%j.out
#SBATCH --error=error.%j.err
#SBATCH --time=0:30:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
##SBATCH --ntasks-per-node=1
##SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=2G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jubo7621@colorado.edu

# Load necessary modules
module purge
ml mambaforge
ml cuda

mamba activate wind_forecasting_cuda

cd /projects/jubo7621/wind-forecasting

export CUDA_VISIBLE_DEVICES=0,1  # Adjust based on the number of GPUs you're using (0,1,2 for 3 GPUs)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

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

## sacct --user=jubo7621
## scontrol show job job_id