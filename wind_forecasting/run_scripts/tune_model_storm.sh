#!/bin/bash
#SBATCH --partition=all_gpu.p         # Partition for H100/A100 GPUs cfdg.p / all_gpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2         # Match number of GPUs
#SBATCH --cpus-per-task=32          # 32 CPUs per task (4 tasks × 32 = 128 CPUs)
#SBATCH --mem-per-cpu=8016          # Total memory = 128 × 8016 ≈ 1 TB
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00
#SBATCH --job-name=informer_tune_flasc_test
#SBATCH --output=/user/taed7566/wind-forecasting/logging/slurm_logs/informer_tune_flasc_%j.out
#SBATCH --error=/user/taed7566/wind-forecasting/logging/slurm_logs/informer_tune_flasc_%j.err
#SBATCH --hint=nomultithread        # Disable hyperthreading
#SBATCH --distribution=block:block  # Improve GPU-CPU affinity
#SBATCH --gres-flags=enforce-binding # Enforce binding of GPUs to tasks

BASE_DIR="/user/taed7566/wind-forecasting"
WORK_DIR="${BASE_DIR}/wind_forecasting"
LOG_DIR="${BASE_DIR}/logging"

# Create logging directories
mkdir -p ${LOG_DIR}/slurm_logs
mkdir -p ${LOG_DIR}/wandb
mkdir -p ${LOG_DIR}/optuna
mkdir -p ${LOG_DIR}/checkpoints

# Change to working directory
cd ${WORK_DIR}

# --- Module loading ---
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load Mamba/24.3.0-0
module load CUDA/12.4.0
# module load OpenMPI/4.1.4-GCC-13.1.0
# ----------------------

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate wf_env_2

# Set paths
export PYTHONPATH=/user/taed7566/pytorch-transformer-ts:${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}/wandb

# Print environment info
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_JOB_GRES=${SLURM_JOB_GRES}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"

echo "=== ENVIRONMENT ==="
module list
echo "=== STARTING TUNING ==="
date +"%Y-%m-%d %H:%M:%S"

# Configure how many workers to run per GPU
NUM_WORKERS_PER_GPU=1

# Used to track process IDs for all workers
declare -a WORKER_PIDS=()

# Total number of GPUs available
NUM_GPUS=${SLURM_NTASKS_PER_NODE}

# The restart flag should only be set for the first run
RESTART_FLAG="--restart_tuning"

# Prepare the database once before starting parallel execution
echo "Initializing Optuna database..."
python ${WORK_DIR}/run_scripts/run_model.py \
  --config ${BASE_DIR}/examples/inputs/training_inputs_juan_flasc.yaml \
  --model informer \
  --mode tune \
  --seed 42 \
  ${RESTART_FLAG} \
  --init_only

# Clear the restart flag now that database is initialized
RESTART_FLAG=""

# Use srun with heterogeneous job steps
# This is more SLURM-friendly than multiple background srun commands
echo "Launching parallel tuning workers across ${NUM_GPUS} GPUs..."

# Create a launcher script for all workers
LAUNCHER_SCRIPT="${BASE_DIR}/launcher_script_$$.sh"
cat > ${LAUNCHER_SCRIPT} << EOF
#!/bin/bash
# Worker ID is passed as SLURM_PROCID
WORKER_INDEX=\${SLURM_PROCID}
# Assign GPU based on worker ID
GPU_ID=\${WORKER_INDEX}
# Create a unique seed for this worker
WORKER_SEED=\$((42 + WORKER_INDEX*10))

echo "Worker \${WORKER_INDEX} starting on GPU \${GPU_ID} with seed \${WORKER_SEED}"

# Set environment variables
export CUDA_VISIBLE_DEVICES=\${GPU_ID}
export WANDB_DIR=${WANDB_DIR}

# Run the tuning script
python ${WORK_DIR}/run_scripts/run_model.py \
  --config ${BASE_DIR}/examples/inputs/training_inputs_juan_flasc.yaml \
  --model informer \
  --mode tune \
  --seed \${WORKER_SEED} \
  --single_gpu
EOF

chmod +x ${LAUNCHER_SCRIPT}

# Launch a single srun job with multiple tasks
# This avoids the "step creation temporarily disabled" issue
srun --ntasks=${NUM_GPUS} \
     --ntasks-per-node=${NUM_GPUS} \
     --cpus-per-task=$((${SLURM_CPUS_PER_TASK}/${NUM_GPUS})) \
     --mem-per-cpu=${SLURM_MEM_PER_CPU} \
     --gpus-per-task=1 \
     ${LAUNCHER_SCRIPT}

# Clean up the temporary launcher script
rm ${LAUNCHER_SCRIPT}

echo "All tuning processes completed"

date +"%Y-%m-%d %H:%M:%S"
echo "=== TUNING COMPLETED ==="

# --- Commands to check the job ---
# sbatch wind_forecasting/run_scripts/tune_model_storm.sh
# sinfo -p cfdg.p
# squeue -u taed7566
# tail -f /user/taed7566/wind-forecasting/logging/slurm_logs/informer_tune_flasc_%j.out# srun -p all_gpu.p -N 1 -n 1 --gpus-per-node 1 -c 128 --mem=32G --time=5:00:00 --pty bash
