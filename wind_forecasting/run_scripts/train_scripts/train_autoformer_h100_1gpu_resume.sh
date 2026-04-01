#!/bin/bash

#SBATCH --partition=all_gpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8192
#SBATCH --gres=gpu:H100:1
#SBATCH --time=1-00:00
#SBATCH --job-name=autoformer_resume
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/autoformer_resume_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/autoformer_resume_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TRAINING RESUME - AUTOFORMER ON H100 (1 GPU)
# Resumes from latest checkpoint of the previous training run
# =============================================================================

export MODEL_NAME="autoformer"

# Find the latest checkpoint automatically
CKPT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/train_awaken_storm_smoothed_pred60_external_autoformer/20260313_094046_0_0"
CHECKPOINT_PATH="${CKPT_DIR}/last.ckpt"

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_storm_awaken_smoothed_pred60_external.yaml"

export NUMEXPR_MAX_THREADS=128

mkdir -p ${LOG_DIR}/slurm_logs

cd ${WORK_DIR} || exit 1

export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

echo "=============================================="
echo "AUTOFORMER TRAINING RESUME - H100 (1 GPU)"
echo "=============================================="
echo "JOB ID: ${SLURM_JOB_ID}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE LIST: ${SLURM_JOB_NODELIST}"
echo "----------------------------------------------"
echo "MODEL: ${MODEL_NAME}"
echo "CHECKPOINT: ${CHECKPOINT_PATH}"
echo "----------------------------------------------"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
echo "=============================================="

# --- Setup Environment ---
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load Mamba/24.3.0-0
module load CUDA/12.4.0
module load git

eval "$(conda shell.bash hook)"
conda activate wf_env_storm

export CAPTURED_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# PostgreSQL password for reading tuned parameters from Optuna DB
export LOCAL_PG_PASSWORD="fXTNFv9L"
export PGPASSWORD="${LOCAL_PG_PASSWORD}"

echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

# Verify checkpoint exists
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT_PATH}"
    echo "Available checkpoints:"
    ls -lt ${CKPT_DIR}/*.ckpt 2>/dev/null
    exit 1
fi

echo "=== RESUMING AUTOFORMER TRAINING ==="
date +"%Y-%m-%d %H:%M:%S"

python ${WORK_DIR}/run_scripts/run_model.py \
  --config ${CONFIG_FILE} \
  --model ${MODEL_NAME} \
  --mode train \
  --use_tuned_parameters \
  --checkpoint ${CHECKPOINT_PATH} \
  --seed 42 \
  --override trainer.max_epochs=200 \
      trainer.limit_train_batches=null \
      trainer.val_check_interval=1.0 \
      trainer.strategy=auto \
      trainer.devices=1

TRAIN_EXIT_CODE=$?

echo "=== TRAINING FINISHED WITH EXIT CODE: ${TRAIN_EXIT_CODE} ==="
date +"%Y-%m-%d %H:%M:%S"

exit $TRAIN_EXIT_CODE
