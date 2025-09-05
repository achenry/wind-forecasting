#!/bin/bash

#SBATCH --partition=all_gpu.p          # Partition for H100/A100 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # REDUCED: Only 1 GPU for quick test
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --gres=gpu:1                   # Request only 1 GPU
#SBATCH --time=0-02:00                 # REDUCED: Only 2 hours for quick test
#SBATCH --job-name=quicktest_copula_fix
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/quicktest_copula_fix_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/quicktest_copula_fix_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/storm_configs/training_inputs_juan_flasc_tune_storm_local_db_360_quicktest.yaml"  # QUICKTEST CONFIG
export MODEL_NAME="tactis"
export RESTART_TUNING_FLAG="--restart_tuning"  # FORCE NEW STUDY for clean test
export AUTO_EXIT_WHEN_DONE="true"
export NUMEXPR_MAX_THREADS=128

# --- Database Credentials File ---
DB_LOGIN_FILE="/user/taed7566/Forecasting/Docs/db_login"

# --- Parse Database Credentials ---
echo "Reading database credentials from ${DB_LOGIN_FILE}..."
if [ ! -f "${DB_LOGIN_FILE}" ]; then
    echo "ERROR: Database login file not found: ${DB_LOGIN_FILE}"
    exit 1
fi

# Parse the login file
DB_NAME=$(grep "^DB=" "${DB_LOGIN_FILE}" | cut -d'=' -f2)
DB_HOST=$(grep "^FQDN=" "${DB_LOGIN_FILE}" | cut -d'=' -f2)
DB_PORT=$(grep "^DBPORT=" "${DB_LOGIN_FILE}" | cut -d'=' -f2)

# Parse user credentials (format: username:password)
USER_LINE=$(grep "^USER=" "${DB_LOGIN_FILE}" | cut -d'=' -f2)
DB_USER=$(echo "${USER_LINE}" | cut -d':' -f1)
DB_PASSWORD=$(echo "${USER_LINE}" | cut -d':' -f2)

# Export the password as environment variable for the Python scripts
export LOCAL_PG_PASSWORD="${DB_PASSWORD}"

echo "Database connection info:"
echo "  Host: ${DB_HOST}"
echo "  Port: ${DB_PORT}"
echo "  Database: ${DB_NAME}"
echo "  User: ${DB_USER}"
echo "  Password: [HIDDEN]"

# --- Create Logging Directories ---
mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

# --- Change to Working Directory ---
cd ${WORK_DIR} || exit 1

# --- Set Shared Environment Variables ---
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

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
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
echo "------------------------"
echo "BASE_DIR: ${BASE_DIR}"
echo "WORK_DIR: ${WORK_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "RESTART_TUNING_FLAG: '${RESTART_TUNING_FLAG}'"
echo "AUTO_EXIT_WHEN_DONE: '${AUTO_EXIT_WHEN_DONE}'"
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

# Find PostgreSQL binary directory
PG_INITDB_PATH=$(which initdb)
if [[ -z "$PG_INITDB_PATH" ]]; then
  echo "FATAL: Could not find 'initdb' after loading PostgreSQL module. Check module system." >&2
  exit 1
fi
export POSTGRES_BIN_DIR=$(dirname "$PG_INITDB_PATH")
echo "Found PostgreSQL bin directory: ${POSTGRES_BIN_DIR}"
echo "Setting PostgreSQL environment variables..."
export PGPASSWORD="${LOCAL_PG_PASSWORD}"

echo "=== STARTING QUICK TEST TUNING (SINGLE WORKER) ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Launch Single Worker ---
NUM_GPUS=1  # Only 1 GPU for quick test
export WORLD_SIZE=${NUM_GPUS}
CURRENT_WORKER_SEED=12

echo "Starting single worker test on GPU 0 with seed ${CURRENT_WORKER_SEED}"

# Set Worker-Specific Environment
export CUDA_VISIBLE_DEVICES=0
export WORKER_RANK=0

echo "Running python script with WORKER_RANK=${WORKER_RANK}..."

# --- Run the tuning script directly (no background) ---
python ${WORK_DIR}/run_scripts/run_model.py \
  --config ${CONFIG_FILE} \
  --model ${MODEL_NAME} \
  --mode tune \
  --seed ${CURRENT_WORKER_SEED} \
  ${RESTART_TUNING_FLAG} \
  --single_gpu

# Check exit status
status=$?
if [ $status -ne 0 ]; then
    echo "Quick test FAILED with status $status"
    FINAL_EXIT_CODE=1
else
    echo "Quick test COMPLETED successfully"
    FINAL_EXIT_CODE=0
fi

echo "=== QUICK TEST COMPLETED ==="
date +"%Y-%m-%d %H:%M:%S"

exit $FINAL_EXIT_CODE