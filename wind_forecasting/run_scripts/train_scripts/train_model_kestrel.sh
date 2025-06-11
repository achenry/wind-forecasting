#!/bin/bash 
#SBATCH --account=ssc
#SBATCH --output=%j-%x.out
#SBATCH --cpus-per-task=4           # CPUs per task (adjust if needed for data loading)
#SBATCH --time=01:00:00
#SBATCH --nodes=1 # this needs to match Trainer(num_nodes...)
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2 # this needs to match Trainer(devices=...), and number of GPUs
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=debug
##SBATCH --time=36:00:00
##SBATCH --nodes=2 # this needs to match Trainer(num_nodes...)
##SBATCH --gres=gpu:4
##SBATCH --ntasks-per-node=4 # this needs to match Trainer(devices=...), and number of GPUs
##SBATCH --mem-per-cpu=85G
##SBATCH --mem=0 # refers to CPU (not GPU) memory, automatically given all GPU memory in a SLURM job, 85G
##SBATCH --ntasks=1

# salloc --account=ssc --time=01:00:00 --gpus=2 --ntasks-per-node=2 --partition=debug

#eval "$(conda shell.bash hook)"

module purge
ml PrgEnv-intel mamba
mamba activate wind_forecasting_env
export PYTHONPATH=$(which python)

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

srun $PYTHONPATH ../run_model.py --config $MODEL_CONFIG_FILE --mode train --model $MODEL --use_tuned_parameters 
