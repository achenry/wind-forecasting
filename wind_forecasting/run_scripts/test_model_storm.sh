#!/bin/bash
#SBATCH --partition=all_gpu.p
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0-5:00
#SBATCH --job-name=tactis_test_flasc
#SBATCH --output=tactis_test_flasc_%j.out
#SBATCH --error=tactis_test_flasc_%j.err

BASE_DIR="/user/taed7566/wind-forecasting"
WORK_DIR="${BASE_DIR}/wind_forecasting"
cd ${WORK_DIR}

# --- Module loading ---
module load Mamba/24.3.0-0
module load foss/2023a
module load OpenMPI/4.1.4-GCC-13.1.0
module load CUDA/12.4.0
# ----------------------

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate wf_env_2

# Run with absolute paths using srun with MPI support
srun --mpi=pmi2 python ${WORK_DIR}/run_scripts/run_model.py \
  --config ${BASE_DIR}/examples/inputs/training_inputs_juan_flasc.yaml \
  --model tactis \
  --mode test \
  --checkpoint ${BASE_DIR}/logging/wind_forecasting/mgam2x4g/checkpoints/epoch=4-step=500.ckpt

# sbatch train_model_storm.sh
# sinfo -p cfdg.p
# squeue -u taed7566
# tail -f tactis_train_flasc_%j.out
# srun -p all_gpu.p -N 1 -n 1 --gpus-per-node 1 -c 128 --mem=32G --time=5:00:00 --pty bash