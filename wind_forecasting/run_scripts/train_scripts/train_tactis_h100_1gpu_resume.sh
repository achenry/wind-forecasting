#!/bin/bash

#SBATCH --partition=cfdg.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8192
#SBATCH --gres=gpu:H100:1
#SBATCH --time=7-00:00
#SBATCH --job-name=tactis_train_resume
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_train_resume_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_train_resume_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding
#SBATCH --signal=B:USR1@300

# =============================================================================
# TRAINING SCRIPT - TACTIS-2 ON H100 (RESUME FROM CHECKPOINT)
# Resumes TACTiS-2 training from a checkpoint with best tuned hyperparameters
# Uses 1x H100 GPU via cfdg.p partition
# --signal=B:USR1@300 triggers Lightning's auto-save 5 min before walltime
# =============================================================================

export MODEL_NAME="tactis"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/rerun_from_epoch=94_step=616518.ckpt}"

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_storm_awaken_smoothed_pred60_tactis.yaml"

export NUMEXPR_MAX_THREADS=128

# --- Create Logging Directories ---
mkdir -p ${LOG_DIR}/slurm_logs
mkdir -p ${LOG_DIR}/checkpoints

# --- Change to Working Directory ---
cd ${WORK_DIR} || exit 1

# --- Set Environment Variables ---
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

# --- Print Job Info ---
echo "=============================================="
echo "TACTIS-2 TRAINING (RESUME) - H100 SINGLE GPU"
echo "=============================================="
echo "JOB ID: ${SLURM_JOB_ID}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE LIST: ${SLURM_JOB_NODELIST}"
echo "NUM GPUs: ${SLURM_NTASKS_PER_NODE}"
echo "----------------------------------------------"
echo "MODEL: ${MODEL_NAME}"
echo "CHECKPOINT: ${CHECKPOINT_PATH}"
echo "CONFIG: ${CONFIG_FILE}"
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

echo "=== STARTING TACTIS-2 TRAINING (RESUME) ==="
date +"%Y-%m-%d %H:%M:%S"

python ${WORK_DIR}/run_scripts/run_model.py \
  --config ${CONFIG_FILE} \
  --model ${MODEL_NAME} \
  --mode train \
  --use_tuned_parameters \
  --seed 42 \
  --checkpoint ${CHECKPOINT_PATH} \
  --override trainer.max_epochs=26 \
      trainer.limit_train_batches=20000 \
      trainer.val_check_interval=1.0 \
      trainer.strategy=auto \
      trainer.devices=1 \
      model.tactis.skip_copula=false \
      model.tactis.stage2_start_epoch=30 \
      model.tactis.initial_stage=2 \
      callbacks.model_checkpoint.init_args.monitor=val_total_nll

TRAIN_EXIT_CODE=$?

echo "=== TRAINING FINISHED WITH EXIT CODE: ${TRAIN_EXIT_CODE} ==="
date +"%Y-%m-%d %H:%M:%S"

exit $TRAIN_EXIT_CODE
