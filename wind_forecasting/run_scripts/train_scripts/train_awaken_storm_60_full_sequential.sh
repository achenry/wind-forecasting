#!/bin/bash

#SBATCH --partition=cfdg.p          # Partition for H100/A100 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3            # 4 tasks (SLURM launches 4 processes for DDP)
#SBATCH --cpus-per-task=8              # CPUs per GPU task
#SBATCH --mem-per-cpu=8192             # Memory per CPU
#SBATCH --gres=gpu:H100:3              # Request 4 H100 GPUs total
#SBATCH --time=5-00:00:00              # 7 days for 100 epochs full data
#SBATCH --job-name=awaken_train_tactis_60_seq_full
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/awaken_train_tactis_60_seq_full_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/awaken_train_tactis_60_seq_full_%j.err
#SBATCH --hint=nomultithread           # Disable hyperthreading
#SBATCH --distribution=block:block     # Improve GPU-CPU affinity
#SBATCH --gres-flags=enforce-binding   # Enforce binding of GPUs to tasks

# =============================================================================
# SEQUENTIAL TRAINING SCRIPT: Stage 1→2 Transition in Single Job
# =============================================================================
# This script trains a TACTiS model with automatic Stage 1→2 transition:
#   - Epochs 0-49:  Stage 1 (marginals/flow/decoder) with skip_copula=false
#   - Epoch 50:     Automatic freeze of marginal/flow, switch to Stage 2
#   - Epochs 50-99: Stage 2 (copula only) with marginals frozen
#
# Hyperparameters from:
#   - Stage 1 (trial 157): context_length_factor=25, encoder_type=temporal, etc.
#   - Stage 2 (trial 146): copula architecture and optimizer params
#
# Key differences from tuning scripts:
#   - skip_copula=false (allows transition)
#   - lock_skip_copula=false (allows automatic transition)
#   - initial_stage=1 (start in Stage 1)
#   - stage2_start_epoch=50 (transition point)
#   - limit_train_batches=null (full dataset)
#   - sampler=sequential (deterministic order)
# =============================================================================

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_juan_awaken_train_storm_pred60_full.yaml"
export MODEL_NAME="tactis"
export AUTO_EXIT_WHEN_DONE="true"
export NUMEXPR_MAX_THREADS=128

# --- Create Logging Directories ---
mkdir -p ${LOG_DIR}/slurm_logs
mkdir -p ${LOG_DIR}/checkpoints

# --- Change to Working Directory ---
cd ${WORK_DIR} || exit 1

# --- Set Shared Environment Variables ---
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

# --- Print Job Info ---
echo "======================================================================="
echo "SEQUENTIAL TRAINING: TACTiS Stage 1→2 (Full Dataset)"
echo "======================================================================="
echo "--- SLURM JOB INFO ---"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "JOB NAME: ${SLURM_JOB_NAME}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE LIST: ${SLURM_JOB_NODELIST}"
echo "NUM NODES: ${SLURM_JOB_NUM_NODES}"
echo "NUM TASKS PER NODE: ${SLURM_NTASKS_PER_NODE} (SLURM-native DDP with srun)"
echo "CPUS PER TASK: ${SLURM_CPUS_PER_TASK}"
echo "GPUS PER TASK: 1 (3 GPUs total distributed across 3 tasks)"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
echo "-----------------------------------------------------------------------"
echo "BASE_DIR: ${BASE_DIR}"
echo "WORK_DIR: ${WORK_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "-----------------------------------------------------------------------"
echo ""
echo "=== TRAINING CONFIGURATION ==="
echo "Total Epochs: 100"
echo "  - Stage 1 (Epochs 0-49): Marginals/Flow/Decoder"
echo "  - Stage 2 (Epochs 50-99): Copula (Marginals Frozen)"
echo ""
echo "Dataset: FULL (no limit_train_batches)"
echo "Sampler: Sequential (deterministic)"
echo "Batch Size: 64 (from tuning trial 157)"
echo "Context Length: 600s (25 steps × 15s)"
echo "Prediction Length: 60s (4 steps × 15s)"
echo ""
echo "=== HYPERPARAMETERS FROM TUNING ==="
echo "Stage 1 (Trial 157, val_loss=-193.864):"
echo "  - context_length_factor: 25"
echo "  - encoder_type: temporal"
echo "  - batch_size: 64"
echo "  - dropout_rate: 0.008"
echo "  - lr_stage1: 0.000381"
echo "  - marginal_num_heads: 6"
echo "  - decoder_num_bins: 300"
echo ""
echo "Stage 2 (Trial 146, val_loss=-193.546):"
echo "  - copula_num_layers: 1"
echo "  - copula_num_heads: 3"
echo "  - ac_mlp_num_layers: 4"
echo "  - ac_mlp_dim: 256"
echo "  - lr_stage2: 0.000386"
echo ""
echo "=== AUTOMATIC STAGE TRANSITION ==="
echo "The model will automatically:"
echo "  1. Train Stage 1 for epochs 0-49"
echo "  2. At epoch 50, freeze marginal/flow parameters"
echo "  3. Switch optimizer to Stage 2 settings"
echo "  4. Train copula for epochs 50-99"
echo "======================================================================="
echo ""

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

# SLURM-native DDP: Using --gpu-bind ensures each task sees exactly 1 GPU
# SLURM will automatically set CUDA_VISIBLE_DEVICES correctly for each task
# srun will launch 3 processes (one per task), each coordinated by Lightning DDP
echo "SLURM-native DDP: Each of 3 tasks will see 1 GPU (--gpu-bind=map_gpu:0,1,2)"
echo "Using srun to launch coordinated processes for PyTorch Lightning DDP"

# --- GPU Monitoring Instructions ---
echo ""
echo "--- MANUAL MONITORING INSTRUCTIONS ---"
echo "To monitor GPU usage, open a NEW terminal and run:"
echo "ssh -L 8088:localhost:8088 ${USER}@${SLURM_JOB_NODELIST}"
echo "Then:"
echo "  mamba activate wf_env_storm"
echo "  gpustat -P --no-processes --watch 0.5"
echo "---------------------------------------"
echo ""

echo "=== STARTING SEQUENTIAL TRAINING ==="
date +"%Y-%m-%d %H:%M:%S"
echo ""

# Launch training with srun - SLURM will spawn 3 coordinated processes for DDP
# Use --gpu-bind=map_gpu to bind each task to a specific GPU (0,1,2)
# This ensures each of the 3 tasks sees exactly 1 GPU

# Set PyTorch distributed environment variables for DDP
# These are required for PyTorch/Lightning to detect the distributed setup
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=29500

# Use srun with --export=ALL to pass environment variables to all tasks
# srun will automatically set SLURM_PROCID (0,1,2) which we'll map to RANK
srun --export=ALL --gpu-bind=map_gpu:0,1,2 bash -c '
  export RANK=${SLURM_PROCID}
  export LOCAL_RANK=${SLURM_LOCALID}
  export WORLD_SIZE=${SLURM_NTASKS}
  echo "Task ${SLURM_PROCID}: RANK=${RANK}, LOCAL_RANK=${LOCAL_RANK}, WORLD_SIZE=${WORLD_SIZE}"
  python ${WORK_DIR}/run_scripts/run_model.py \
    --config ${CONFIG_FILE} \
    --model ${MODEL_NAME} \
    --mode train \
    --seed 666
'

TRAIN_EXIT_CODE=$?

echo ""
echo "=== TRAINING SCRIPT FINISHED WITH EXIT CODE: ${TRAIN_EXIT_CODE} ==="
date +"%Y-%m-%d %H:%M:%S"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "                    TRAINING COMPLETED SUCCESSFULLY"
    echo "======================================================================="
    echo ""
    echo "Next steps:"
    echo "  1. Check WandB for training curves and Stage 1→2 transition"
    echo "  2. Inspect checkpoints in:"
    echo "     ${OUTPUT_DIR}/checkpoints/*_60_sequential_full/"
    echo "  3. Look for val_loss improvements across both stages"
    echo "  4. Verify Stage 2 learning rate and parameter freezing in logs"
    echo ""
    echo "Expected checkpoints:"
    echo "  - seq_train_epoch=49-*  (End of Stage 1)"
    echo "  - seq_train_epoch=50-*  (Start of Stage 2)"
    echo "  - seq_train_epoch=99-*  (Final model)"
    echo "  - last.ckpt             (Latest checkpoint)"
    echo "======================================================================="
else
    echo ""
    echo "======================================================================="
    echo "                      TRAINING FAILED"
    echo "======================================================================="
    echo "Exit code: ${TRAIN_EXIT_CODE}"
    echo "Check logs for errors:"
    echo "  ${LOG_DIR}/slurm_logs/awaken_train_tactis_60_seq_full_${SLURM_JOB_ID}.err"
    echo "======================================================================="
fi

exit $TRAIN_EXIT_CODE
