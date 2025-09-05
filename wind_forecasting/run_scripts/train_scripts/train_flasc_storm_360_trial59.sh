#!/bin/bash

#SBATCH --partition=all_gpu.p          # Partition for H100/A100 GPUs cfdg.p / all_gpu.p / mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Match number of GPUs requested below (for DDP training)
#SBATCH --cpus-per-task=8           # CPUs per task (adjust if needed for data loading)
#SBATCH --mem-per-cpu=8192          # Memory per CPU
#SBATCH --gres=gpu:H100:1
#SBATCH --time=1-00:00              # Time limit (adjust as needed for training)
#SBATCH --job-name=flasc_train_tactis_360_trial59      # Updated job name for trial 59
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/flasc_train_tactis_360_trial59_%j.out # Updated output log path
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/flasc_train_tactis_360_trial59_%j.err  # Updated error log path
#SBATCH --hint=nomultithread        # Disable hyperthreading
#SBATCH --distribution=block:block  # Improve GPU-CPU affinity
#SBATCH --gres-flags=enforce-binding # Enforce binding of GPUs to tasks

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting/wind-forecasting" # Absolute path to the base directory
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/storm_configs/training_inputs_juan_flasc_tune_storm_local_db_360.yaml"
export MODEL_NAME="tactis"
export RESTART_TUNING_FLAG="" # "" Or "--restart_tuning"
export AUTO_EXIT_WHEN_DONE="true"  # Set to "true" to exit script when all workers finish, "false" to keep running until timeout
export NUMEXPR_MAX_THREADS=128
# export NCCL_DEBUG=INFO # Enable verbose logging for the NCCL backend for debugging

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
# Create the main SLURM log directory if it doesn't exist
mkdir -p ${LOG_DIR}/slurm_logs
mkdir -p ${LOG_DIR}/checkpoints

# --- Change to Working Directory ---
cd ${WORK_DIR} || exit 1 # Exit if cd fails

# --- Set Shared Environment Variables ---
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR} # WandB will create a 'wandb' subdirectory here automatically

# --- Print Job Info ---
echo "--- SLURM JOB INFO (Training with Trial 59 Parameters) ---"
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
echo "USING TRIAL 59 PARAMETERS"
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

# Set PostgreSQL binary directory for Optuna database setup
export POSTGRES_BIN_DIR="/user/taed7566/.conda/envs/wf_env_storm/bin"
echo "POSTGRES_BIN_DIR set to: $POSTGRES_BIN_DIR"

# --- End Main Environment Setup ---

echo "=== STARTING MODEL TRAINING WITH TRIAL 59 PARAMETERS ==="
date +"%Y-%m-%d %H:%M:%S"

# Display the parameters being used
echo "Using Trial 59 hyperparameters:"
echo "  lr_stage1: 3.22e-06"
echo "  lr_stage2: 1.60e-04"
echo "  weight_decay_stage1: 1.00e-06"
echo "  weight_decay_stage2: 5.00e-06"
echo "  copula_num_layers: 3"
echo "  copula_num_heads: 2"
echo "  ac_mlp_num_layers: 5"
echo "  decoder_num_bins: 50"
echo "  batch_size: 64"
echo "  context_length_factor: 5"
echo "  Trial 59 validation loss: -27.33"

# Use srun to launch the training script with Trial 59's specific parameters
# Note: We DO NOT use --use_tuned_parameters since we want specific trial parameters
srun python ${WORK_DIR}/run_scripts/run_model.py \
  --config ${CONFIG_FILE} \
  --model ${MODEL_NAME} \
  --mode train \
  --seed 666 \
  --override dataset.sampler=random \
      trainer.max_epochs=100 \
      trainer.limit_train_batches=null \
      trainer.val_check_interval=1.0 \
      tactis.stage2_start_epoch=40 \
      tactis.lr_stage1=3.22e-06 \
      tactis.lr_stage2=1.60e-04 \
      tactis.weight_decay_stage1=1.00e-06 \
      tactis.weight_decay_stage2=5.00e-06 \
      tactis.copula_num_layers=3 \
      tactis.copula_num_heads=2 \
      tactis.ac_mlp_num_layers=5 \
      tactis.ac_mlp_dim=32 \
      tactis.decoder_num_bins=50 \
      tactis.copula_embedding_dim_per_head=16 \
      tactis.copula_input_encoder_layers=4 \
      tactis.copula_series_embedding_dim=64 \
      tactis.decoder_dsf_hidden_dim=128 \
      tactis.decoder_dsf_num_layers=3 \
      tactis.decoder_mlp_hidden_dim=8 \
      tactis.decoder_mlp_num_layers=3 \
      tactis.decoder_transformer_embedding_dim_per_head=128 \
      tactis.decoder_transformer_num_heads=4 \
      tactis.decoder_transformer_num_layers=5 \
      tactis.dropout_rate=0.008 \
      tactis.encoder_type=temporal \
      tactis.eta_min_fraction_s1=0.012357526343077298 \
      tactis.eta_min_fraction_s2=1.43e-05 \
      tactis.flow_input_encoder_layers=4 \
      tactis.flow_series_embedding_dim=8 \
      tactis.gradient_clip_val_stage1=1.0 \
      tactis.gradient_clip_val_stage2=0 \
      tactis.marginal_embedding_dim_per_head=128 \
      tactis.marginal_num_heads=6 \
      tactis.marginal_num_layers=3 \
      tactis.stage1_activation_function=relu \
      tactis.stage2_activation_function=relu \
      dataset.batch_size=64 \
      dataset.context_length_factor=5 \
      trainer.gradient_clip_val=1.0 \
      experiment.run_name=train_flasc_tactis_360_storm_trial59 \
      experiment.project_name=train_tactis_flasc_storm_trial59 \
      logging.chkp_dir_suffix=_360_train_trial59 \
      experiment.extra_tags="[\"flasc\", \"train\", \"tactis\", \"360s\", \"storm\", \"trial59\", \"improved_copula\"]"

TRAIN_EXIT_CODE=$?

echo "=== TRAINING SCRIPT FINISHED WITH EXIT CODE: ${TRAIN_EXIT_CODE} ==="
date +"%Y-%m-%d %H:%M:%S"

exit $TRAIN_EXIT_CODE

# Example usage:
# sbatch wind-forecasting/wind_forecasting/run_scripts/train_scripts/train_flasc_storm_360_trial59.sh