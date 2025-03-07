#!/bin/bash
#SBATCH --partition=all_gpu.p         # Partition for H100/A100 GPUs cfdg.p / all_gpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # Match number of GPUs
#SBATCH --cpus-per-task=32          # 32 CPUs per task (4 tasks × 32 = 128 CPUs)
#SBATCH --mem-per-cpu=8016          # Total memory = 128 × 8016 ≈ 1 TB
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00
#SBATCH --job-name=informer_tune_flasc
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

# Set WandB directory explicitly
export WANDB_DIR=${LOG_DIR}/wandb

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

# Example with 2 workers per GPU:
NUM_WORKERS_PER_GPU=2
for i in $(seq 0 $((${SLURM_NTASKS_PER_NODE}-1))); do
    for j in $(seq 0 $((${NUM_WORKERS_PER_GPU}-1))); do
        # The restart flag should only be set for the very first worker (i=0, j=0)
        if [ $i -eq 0 ] && [ $j -eq 0 ]; then
            RESTART_FLAG="--restart_tuning"
        else
            RESTART_FLAG=""
        fi
        
        srun --exclusive -n 1 --export=ALL,CUDA_VISIBLE_DEVICES=$i,WANDB_DIR=${WANDB_DIR} python ${WORK_DIR}/run_scripts/run_model.py \
          --config ${BASE_DIR}/examples/inputs/training_inputs_juan_flasc.yaml \
          --model informer \
          --mode tune \
          ${RESTART_FLAG} &
        
        sleep 2
    done
done

wait

date +"%Y-%m-%d %H:%M:%S"
echo "=== TUNING COMPLETED ==="

# --- Commands to check the job ---
# sbatch wind_forecasting/run_scripts/tune_model_storm.sh
# sinfo -p cfdg.p
# squeue -u taed7566
# tail -f informer_tune_flasc_%j.out
# srun -p all_gpu.p -N 1 -n 1 --gpus-per-node 1 -c 128 --mem=32G --time=5:00:00 --pty bash