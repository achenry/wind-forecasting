#!/bin/bash

#SBATCH --partition=cfdg.p          # Partition for H100/A100 GPUs cfdg.p / all_gpu.p / mpcg.p(not allowed)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Match number of GPUs requested below (for DDP training)
#SBATCH --cpus-per-task=32           # CPUs per task (adjust if needed for data loading)
#SBATCH --mem-per-cpu=4096          # Memory per CPU
#SBATCH --gres=gpu:H100:1           # Request 4 H100 GPUs
#SBATCH --time=2-00:00              # Time limit (adjust as needed for training)
#SBATCH --job-name=flasc_train      # Updated job name
#SBATCH --output=/user/taed7566/Forecasting/wind-forecasting/logs/slurm_logs/flasc_train_%j.out # Updated output log path
#SBATCH --error=/user/taed7566/Forecasting/wind-forecasting/logs/slurm_logs/flasc_train_%j.err  # Updated error log path
#SBATCH --hint=nomultithread        # Disable hyperthreading
#SBATCH --distribution=block:block  # Improve GPU-CPU affinity
#SBATCH --gres-flags=enforce-binding # Enforce binding of GPUs to tasks

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting/wind-forecasting" # Absolute path to the base directory
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${BASE_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_juan_flasc_test_storm.yaml"
export MODEL_NAME="tactis" # Or pass as argument if needed: "$1"
export NUMEXPR_MAX_THREADS=128

# --- Create Logging Directories ---
# Create the main SLURM log directory if it doesn't exist
mkdir -p ${LOG_DIR}/slurm_logs
mkdir -p ${LOG_DIR}/checkpoints

# --- Change to Working Directory ---
cd ${WORK_DIR} || exit 1 # Exit if cd fails

# --- Set Shared Environment Variables ---
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR} # WandB will create a 'wandb' subdirectory here automatically

# --- Print Job Info ---
echo "--- SLURM JOB INFO (Training) ---"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "JOB NAME: ${SLURM_JOB_NAME}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE LIST: ${SLURM_JOB_NODELIST}"
echo "NUM NODES: ${SLURM_JOB_NUM_NODES}"
echo "NUM TASKS PER NODE (GPUs for DDP): ${SLURM_NTASKS_PER_NODE}"
echo "CPUS PER TASK: ${SLURM_CPUS_PER_TASK}"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
echo "------------------------"
echo "BASE_DIR: ${BASE_DIR}"
echo "WORK_DIR: ${WORK_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "------------------------"

# --- Setup Main Environment ---
echo "Setting up main environment..."
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load Mamba/24.3.0-0
module load CUDA/12.4.0
module load git
echo "Modules loaded."

eval "$(conda shell.bash hook)"
conda activate wf_env_storm
echo "Conda environment 'wf_env_storm' activated."
export CAPTURED_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

# --- End Main Environment Setup ---
# --- Find and Export PostgreSQL Bin Directory ---
# echo "Attempting to find PostgreSQL binaries within the conda environment..."
# PG_INITDB_PATH=$(which initdb)
# if [[ -z "$PG_INITDB_PATH" ]]; then
#   echo "FATAL: Could not find 'initdb' after activating conda environment 'wf_env_storm'." >&2
#   echo "Ensure PostgreSQL client tools are installed in this environment (e.g., 'conda install postgresql')." >&2
#   exit 1
# fi
# # Extract the directory path (e.g., /path/to/conda/env/bin)
# export POSTGRES_BIN_DIR=$(dirname "$PG_INITDB_PATH")
# echo "Found and exported POSTGRES_BIN_DIR: ${POSTGRES_BIN_DIR}"
# --- End PostgreSQL Setup ---

echo "=== STARTING MODEL TRAINING ==="
date +"%Y-%m-%d %H:%M:%S"

# Use srun to launch the training script. PyTorch Lightning's SLURMEnvironment
# should detect the environment variables set by srun for distributed training (DDP).
srun python ${WORK_DIR}/run_scripts/run_model.py \
  --config ${CONFIG_FILE} \
  --model ${MODEL_NAME} \
  --mode train \
  --use_tuned_parameters \
  --override gradient_clip_val_stage1 gradient_clip_val_stage2 # Override gradient clipping values from YAML

TRAIN_EXIT_CODE=$?

echo "=== TRAINING SCRIPT FINISHED WITH EXIT CODE: ${TRAIN_EXIT_CODE} ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Move main SLURM logs if needed (optional, SLURM might handle this) ---
# Consider if you want to move the main .out/.err files after completion
# mkdir -p "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}"
# mv "${LOG_DIR}/slurm_logs/flasc_train_${SLURM_JOB_ID}.out" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/" 2>/dev/null
# mv "${LOG_DIR}/slurm_logs/flasc_train_${SLURM_JOB_ID}.err" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/" 2>/dev/null
# echo "--------------------------------------------------"

exit $TRAIN_EXIT_CODE

# Example usage:
# sbatch wind-forecasting/wind_forecasting/run_scripts/train_scripts/train_model_storm.sh
# (Optionally pass model name: sbatch wind-forecasting/wind_forecasting/run_scripts/train_scripts/train_model_storm.sh tactis)
