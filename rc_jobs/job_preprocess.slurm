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
#SBATCH --ntasks-per-node=64
#SBATCH --mem=0G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jubo7621@colorado.edu

# Load necessary modules
module purge
ml mambaforge
mamba activate wind_preprocessing

cd /projects/jubo7621/wind-forecasting/

# Set the number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

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

## To view position in the queue:
## squeue -p amilan -t PENDING --sort=p,t -o "%.18i %.9P %.8j %.8u %.2t %.19S %.19V %.10M %.6D %R" | nl | grep -C 5 jubo7621

## To view current jobs running:
## squeue -p amilan -t RUNNING

## To estimate the start time of your job:
## ./rc_jobs/estimate_job_start_time.sh