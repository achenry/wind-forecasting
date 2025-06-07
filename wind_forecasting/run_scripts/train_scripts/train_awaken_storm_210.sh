#!/bin/bash

#SBATCH --partition=all_gpu.p          # Partition for H100/A100 GPUs cfdg.p / all_gpu.p / mpcg.p(not allowed)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # Match number of GPUs requested below (for DDP training)
#SBATCH --cpus-per-task=8           # CPUs per task (adjust if needed for data loading)
#SBATCH --mem-per-cpu=8192          # Memory per CPU
#SBATCH --gres=gpu:4           # Request 4 H100 GPUs
#SBATCH --time=7-00:00              # Time limit (adjust as needed for training)
#SBATCH --job-name=awaken_train_tactis_210      # Updated job name
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/awaken_train_tactis_210_%j.out # Updated output log path
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/awaken_train_tactis_210_%j.err  # Updated error log path
#SBATCH --hint=nomultithread        # Disable hyperthreading
#SBATCH --distribution=block:block  # Improve GPU-CPU affinity
#SBATCH --gres-flags=enforce-binding # Enforce binding of GPUs to tasks

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting/wind-forecasting" # Absolute path to the base directory
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_juan_awaken_tune_storm_pred210.yaml"
export MODEL_NAME="tactis"
export RESTART_TUNING_FLAG="" # "" Or "--restart_tuning"
export AUTO_EXIT_WHEN_DONE="true"  # Set to "true" to exit script when all workers finish, "false" to keep running until timeout
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

# model.tactis.lr_stage1=4.270656650991065e-06 \
# model.tactis.lr_stage2=4.899249681991742e-06 \

srun python ${WORK_DIR}/run_scripts/run_model.py \
  --config ${CONFIG_FILE} \
  --model ${MODEL_NAME} \
  --mode train \
  --seed 666 \
  --override dataset.sampler=sequential \
      trainer.max_epochs=40 \
      trainer.limit_train_batches=null \
      trainer.val_check_interval=1.0 \
      dataset.batch_size=64 \
      dataset.context_length_factor=5 \
      model.tactis.lr_stage1=4.270656650991065e-06 \
      model.tactis.lr_stage2=4.899249681991742e-06 \
      model.tactis.weight_decay_stage1=0.0 \
      model.tactis.weight_decay_stage2=5e-06 \
      model.tactis.stage2_start_epoch=20 \
      model.tactis.warmup_steps_s1=0.10 \
      model.tactis.warmup_steps_s2=0.10 \
      model.tactis.steps_to_decay_s1=0.90 \
      model.tactis.steps_to_decay_s2=0.90 \
      model.tactis.eta_min_fraction_s1=0.0016799548032196548 \
      model.tactis.eta_min_fraction_s2=0.00013329608232447702 \
      model.tactis.flow_series_embedding_dim=64 \
      model.tactis.copula_series_embedding_dim=64 \
      model.tactis.flow_input_encoder_layers=4 \
      model.tactis.copula_input_encoder_layers=2 \
      model.tactis.marginal_embedding_dim_per_head=512 \
      model.tactis.marginal_num_heads=3 \
      model.tactis.marginal_num_layers=4 \
      model.tactis.copula_embedding_dim_per_head=128 \
      model.tactis.copula_num_heads=6 \
      model.tactis.copula_num_layers=3 \
      model.tactis.decoder_dsf_num_layers=4 \
      model.tactis.decoder_dsf_hidden_dim=128 \
      model.tactis.decoder_mlp_num_layers=3 \
      model.tactis.decoder_mlp_hidden_dim=256 \
      model.tactis.decoder_transformer_num_layers=4 \
      model.tactis.decoder_transformer_embedding_dim_per_head=64 \
      model.tactis.decoder_transformer_num_heads=4 \
      model.tactis.decoder_num_bins=50 \
      model.tactis.encoder_type=standard \
      model.tactis.dropout_rate=0.007 \
      model.tactis.ac_mlp_num_layers=3 \
      model.tactis.ac_mlp_dim=128 \
      model.tactis.stage1_activation_function=relu \
      model.tactis.stage2_activation_function=relu \
      model.tactis.gradient_clip_val_stage1=1.0 \
      model.tactis.gradient_clip_val_stage2=1.0

TRAIN_EXIT_CODE=$?

echo "=== TRAINING SCRIPT FINISHED WITH EXIT CODE: ${TRAIN_EXIT_CODE} ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Load HPARAMS manually logic ---
# python -c "
#   import torch
#   checkpoint = torch.load('/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/checkpoints/tactis_210/trial_190/trial_190_epoch=19-step=200000-val_loss=-29.71.ckpt', map_location='cpu')
#   hparams = checkpoint.get('hyper_parameters', checkpoint.get('hparams'))
#   print('Hyperparameters from checkpoint:')
#   for k, v in hparams['model_config'].items():
#       print(f'model.tactis.{k}={v}')
#   "

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
