#!/bin/bash

#SBATCH --partition=all_gpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=tactis_feasibility
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_feasibility_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_feasibility_%j.err
#SBATCH --hint=nomultithread

# =============================================================================
# TACTIS-2 FEASIBILITY SMOKE TEST
# Quick 5-epoch test: Stage 1 (epochs 0-2), transition at epoch 3, Stage 2 (3-4)
# Verifies: data loading, forward pass, stage transition, copula training
# =============================================================================

BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
WORK_DIR="${BASE_DIR}/wind_forecasting"
CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_storm_awaken_smoothed_pred60_tactis.yaml"

cd ${WORK_DIR} || exit 1
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}

echo "=============================================="
echo "TACTIS-2 FEASIBILITY SMOKE TEST"
echo "=============================================="
echo "JOB ID: ${SLURM_JOB_ID}"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
nvidia-smi
echo "=============================================="

# Setup environment
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load Mamba/24.3.0-0
module load CUDA/12.4.0
module load git

eval "$(conda shell.bash hook)"
conda activate wf_env_storm

export LOCAL_PG_PASSWORD="fXTNFv9L"
export PGPASSWORD="${LOCAL_PG_PASSWORD}"
export CUDA_VISIBLE_DEVICES=0

echo "=== STARTING FEASIBILITY TEST ==="
date +"%Y-%m-%d %H:%M:%S"

python ${WORK_DIR}/run_scripts/run_model.py \
  --config ${CONFIG_FILE} \
  --model tactis \
  --mode train \
  --seed 42 \
  --single_gpu \
  --override trainer.max_epochs=5 \
      trainer.limit_train_batches=50 \
      trainer.val_check_interval=25 \
      trainer.devices=1 \
      trainer.strategy=auto \
      model.tactis.stage2_start_epoch=3

EXIT_CODE=$?

echo "=== FEASIBILITY TEST FINISHED ==="
echo "Exit code: ${EXIT_CODE}"
date +"%Y-%m-%d %H:%M:%S"

if [ $EXIT_CODE -eq 0 ]; then
    echo "FEASIBILITY TEST PASSED"
else
    echo "FEASIBILITY TEST FAILED"
fi

exit $EXIT_CODE
