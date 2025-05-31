#!/bin/bash
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1 # this needs to match Trainer(num_nodes...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4 # necessary for gpus on rc
#SBATCH --ntasks-per-node=4 # this needs to match Trainer(devices=...)
#SBATCH --mem-per-cpu=85G
#SBATCH --time=36:00:00
#SBATCH --output=%j-%x.log

# sinteractive --partition=atesting_a100 --gres=gpu:1 --ntasks-per-node=1 --ntasks=1 --time=50:00

module purge
# module load miniforge
#module load intel impi
eval "$(conda shell.bash hook)"
conda activate wind_forecasting_env

export NUMEXPR_MAX_THREADS=128
export MODEL=$1
export MODEL_CONFIG_FILE=$2

API_FILE="../.wandb_api_key"
if [ -f "${API_FILE}" ]; then
  source "${API_FILE}"
else
  echo "ERROR: WANDB API‑key file not found at ${API_FILE}" >&2
  exit 1
fi

echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_JOB_GRES=${SLURM_JOB_GRES}"

srun python ../run_model.py --config $MODEL_CONFIG_FILE --mode train --model $MODEL --checkpoint best --override model.x.lr=1.0e-4 model.x.weight_decay=1.0e-8 #--use_tuned_parameters