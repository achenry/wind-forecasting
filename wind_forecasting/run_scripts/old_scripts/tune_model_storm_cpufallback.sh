#!/bin/bash

#SBATCH --partition=all_cpu.p       # Partition for CPU nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Typically 1 task for CPU parallelism
#SBATCH --cpus-per-task=128         # Request all available CPUs per node (adjust as needed)
#SBATCH --mem-per-cpu=8016          # Memory per CPU (Total Mem = ntasks * cpus-per-task * mem-per-cpu)
#SBATCH --gres=gpu:0                # Request 0 GPUs for CPU job
#SBATCH --time=1-00:00              # Time limit (1 day)
#SBATCH --job-name=tactis_tune_flasc_cpu
#SBATCH --output=/user/taed7566/wind-forecasting/logging/slurm_logs/tactis_tune_flasc_cpu_%j.out
#SBATCH --error=/user/taed7566/wind-forecasting/logging/slurm_logs/tactis_tune_flasc_cpu_%j.err
#SBATCH --hint=multithread          # Enable hyperthreading for CPU workloads

# For UOL HPC, when using all_gpu.p partition, has a time limit of 1 day (1-00:00)
# While cfdg.p [1-2] partition has a time limit of 21 days (21-00:00). Use cfdg002 for x4 H100 GPUs
# mpcg.p [1-6] is allowed for 7 days (7-00:00) and up to 21 days with '--qos=long_mpcg.q', but might be restricted for my group

# --- Configuration ---
# Set this to "--restart_tuning" to clear and restart the Optuna study, otherwise set to "" to continue previous one
RESTART_TUNING_FLAG="--restart_tuning" # "" Or "--restart_tuning"

# --- Base Directories ---
BASE_DIR="/user/taed7566/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${BASE_DIR}/logging"
export CONFIG_FILE="${BASE_DIR}/examples/inputs/training_inputs_juan_flasc.yaml"
export MODEL_NAME="tactis"
export RESTART_TUNING_FLAG="--restart_tuning" # "" Or "--restart_tuning"

# --- Create Logging Directories ---
# Ensure the main slurm_logs directory exists
mkdir -p ${LOG_DIR}/slurm_logs
# Create the job-specific directory for worker logs and final main logs
mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/optuna
mkdir -p ${LOG_DIR}/checkpoints

# --- Change to Working Directory ---
cd ${WORK_DIR} || exit 1 # Exit if cd fails

# --- Set Shared Environment Variables ---
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR} # WandB will create a 'wandb' subdirectory here automatically

# --- Print Job Info ---
echo "--- SLURM JOB INFO ---"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "JOB NAME: ${SLURM_JOB_NAME}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE LIST: ${SLURM_JOB_NODELIST}"
echo "NUM NODES: ${SLURM_JOB_NUM_NODES}"
echo "NUM GPUS (Requested via ntasks): ${SLURM_NTASKS_PER_NODE}"
echo "NUM TASKS PER NODE: ${SLURM_NTASKS_PER_NODE}"
echo "CPUS PER TASK: ${SLURM_CPUS_PER_TASK}"
echo "------------------------"
echo "BASE_DIR: ${BASE_DIR}"
echo "WORK_DIR: ${WORK_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "RESTART_TUNING_FLAG: '${RESTART_TUNING_FLAG}'"
echo "------------------------"

# --- Setup Main Environment ---
echo "Setting up main environment..."
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load PostgreSQL/16.1-GCCcore-13.1.0
module load Mamba/24.3.0-0
# module load CUDA/12.4.0 # Remove CUDA module for CPU
module load git
echo "Modules loaded."

# Capture LD_LIBRARY_PATH after modules are loaded
export CAPTURED_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo "Captured LD_LIBRARY_PATH: ${CAPTURED_LD_LIBRARY_PATH}"

# Find PostgreSQL binary directory after loading the module
PG_INITDB_PATH=$(which initdb)
if [[ -z "$PG_INITDB_PATH" ]]; then
  echo "FATAL: Could not find 'initdb' after loading PostgreSQL module. Check module system." >&2
  exit 1
fi
# Extract the directory path (e.g., /path/to/postgres/bin)
export POSTGRES_BIN_DIR=$(dirname "$PG_INITDB_PATH")
echo "Found PostgreSQL bin directory: ${POSTGRES_BIN_DIR}"

eval "$(conda shell.bash hook)"
conda activate wf_env_2_cpu
echo "Conda environment 'wf_env_2_cpu' activated."
# --- End Main Environment Setup ---

echo "=== STARTING PARALLEL OPTUNA TUNING WORKERS ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Parallel Worker Launch using srun ---

echo "Launching distributed workers using srun..."
export WORLD_SIZE=$SLURM_NTASKS # Total number of tasks across all nodes (nodes * ntasks-per-node)
echo "Total tasks (WORLD_SIZE): ${WORLD_SIZE}"

eval "$(conda shell.bash hook)"
conda activate wf_env_2_cpu
echo "Conda environment 'wf_env_2_cpu' activated for srun."

srun --cpu-bind=cores --distribution=block:block python ${WORK_DIR}/run_scripts/run_model.py \
  --config ${CONFIG_FILE} \
  --model ${MODEL_NAME} \
  --mode tune \
  --seed 12 \
  ${RESTART_TUNING_FLAG}

SRUN_EXIT_CODE=$?
echo "srun finished with exit code: ${SRUN_EXIT_CODE}"
# --- End srun launch ---

# Set final exit code based on srun result
FINAL_EXIT_CODE=${SRUN_EXIT_CODE}

echo "=== TUNING SCRIPT COMPLETED ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Move main SLURM logs to the job ID directory with rest of worker logs ---
echo "Moving main SLURM logs to ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/"
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.out; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/" || echo "Warning: Could not move .out file."
done
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.err; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/" || echo "Warning: Could not move .err file."
done
echo "--------------------------------------------------"

exit $FINAL_EXIT_CODE

# sbatch wind_forecasting/run_scripts/tune_model_storm.sh
# squeue -p cfdg.p,mpcg.p,all_gpu.p -o "%.10a %.10P %.25j %.8u %.2t %.10M %.6D %R"