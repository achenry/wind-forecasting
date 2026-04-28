#!/bin/bash

#SBATCH --partition=cfdg.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8192
#SBATCH --gres=gpu:H100:1
#SBATCH --time=8:00:00
#SBATCH --job-name=tactis_finetune_lr
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_finetune_lr_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_finetune_lr_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding
#SBATCH --signal=B:USR1@300

# =============================================================================
# FINE-TUNE SCRIPT - TACTIS-2 ON H100 (LOWER LR FROM EPOCH 119)
# Resumes from last_finetune_ready (state_dict only, fresh optimizer+scheduler)
# Targets epoch 129 with lr_stage2=1e-7 + cosine decay to 5e-8
# Pairs with lightning_module.py configure_optimizers stage==2 scheduler-binding fix
# =============================================================================

export MODEL_NAME="tactis"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/last_finetune_ready_epoch=119_step=1116518.ckpt}"

BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_storm_awaken_smoothed_pred60_tactis.yaml"

export NUMEXPR_MAX_THREADS=128

mkdir -p ${LOG_DIR}/slurm_logs
mkdir -p ${LOG_DIR}/checkpoints

cd ${WORK_DIR} || exit 1

export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

echo "=============================================="
echo "TACTIS-2 FINE-TUNE (LOWER LR) - H100 SINGLE GPU"
echo "=============================================="
echo "JOB ID: ${SLURM_JOB_ID}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE LIST: ${SLURM_JOB_NODELIST}"
echo "----------------------------------------------"
echo "MODEL: ${MODEL_NAME}"
echo "CHECKPOINT: ${CHECKPOINT_PATH}"
echo "CONFIG: ${CONFIG_FILE}"
echo "TARGET EPOCH: 129 (10 more from 119)"
echo "LR_STAGE2: 1e-7 (peak), 5e-8 (cosine min)"
echo "----------------------------------------------"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
echo "=============================================="

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

export LOCAL_PG_PASSWORD="fXTNFv9L"
export PGPASSWORD="${LOCAL_PG_PASSWORD}"

echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

echo "=== STARTING TACTIS-2 FINE-TUNE (LOWER LR) ==="
date +"%Y-%m-%d %H:%M:%S"

python ${WORK_DIR}/run_scripts/run_model.py \
  --config ${CONFIG_FILE} \
  --model ${MODEL_NAME} \
  --mode train \
  --use_tuned_parameters \
  --seed 42 \
  --checkpoint ${CHECKPOINT_PATH} \
  --override trainer.max_epochs=10 \
      trainer.limit_train_batches=20000 \
      trainer.val_check_interval=1.0 \
      trainer.strategy=auto \
      trainer.devices=1 \
      model.tactis.skip_copula=false \
      model.tactis.stage2_start_epoch=30 \
      model.tactis.initial_stage=2 \
      model.tactis.lr_stage2=1e-7 \
      model.tactis.warmup_steps_s2=0 \
      model.tactis.steps_to_decay_s2=200000 \
      model.tactis.eta_min_fraction_s2=0.5 \
      callbacks.model_checkpoint.init_args.monitor=val_total_nll

TRAIN_EXIT_CODE=$?

echo "=== FINE-TUNE FINISHED WITH EXIT CODE: ${TRAIN_EXIT_CODE} ==="
date +"%Y-%m-%d %H:%M:%S"

exit $TRAIN_EXIT_CODE
