#!/bin/bash
#SBATCH --partition=all_gpu.p         # Partition for H100/A100 GPUs cfdg.p / all_gpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # Match number of GPUs
#SBATCH --cpus-per-task=32          # 32 CPUs per task (4 tasks × 32 = 128 CPUs)
#SBATCH --mem-per-cpu=1024          # Total memory = 128 × 3900 ≈ 1 TB
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00
#SBATCH --job-name=informer_train_flasc
#SBATCH --output=informer_train_flasc_%j.out
#SBATCH --error=informer_train_flasc_%j.err
#SBATCH --mail-user=juan.manuel.boullosa.novo@uol.de
#SBATCH --mail-type=END
#SBATCH --hint=nomultithread        # Disable hyperthreading
#SBATCH --distribution=block:block  # Improve GPU-CPU affinity

BASE_DIR="/user/taed7566/wind-forecasting"
WORK_DIR="${BASE_DIR}/wind_forecasting"
cd ${WORK_DIR}

# --- Module loading ---
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load Mamba/24.3.0-0
module load CUDA/12.4.0
module load OpenMPI/4.1.4-GCC-13.1.0
# ----------------------

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate wf_env_2

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_JOB_GRES=${SLURM_JOB_GRES}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"

echo "=== ENVIRONMENT ==="
module list
echo "=== STARTING TRAINING ==="
date +"%Y-%m-%d %H:%M:%S"

# Run with absolute paths using srun with MPI support
srun python ${WORK_DIR}/run_scripts/run_model.py \
  --config ${BASE_DIR}/examples/inputs/training_inputs_juan_flasc.yaml \
  --model informer \
  --mode train

# --- Commands to check the job ---
# sbatch wind_forecasting/run_scripts/train_model_storm.sh
# sinfo -p cfdg.p
# squeue -u taed7566
# tail -f tactis_train_flasc_%j.out
# srun -p all_gpu.p -N 1 -n 1 --gpus-per-node 1 -c 128 --mem=32G --time=5:00:00 --pty bash