#!/bin/bash 
#SBATCH --account=ssc
#SBATCH --time=12:00:00
#SBATCH --output=%j-%x.out
##SBATCH --partition=debug
#SBATCH --nodes=1 # this needs to match Trainer(num_nodes...)
#SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=4 # this needs to match Trainer(devices=...), and number of GPUs
#SBATCH --mem-per-cpu=85G
##SBATCH --mem=0 # refers to CPU (not GPU) memory, automatically given all GPU memory in a SLURM job, 85G
##SBATCH --ntasks=1

# salloc --account=ssc --time=01:00:00 --gpus=2 --ntasks-per-node=2 --partition=debug

module purge
module load PrgEnv-intel
ml mamba
mamba activate wind_forecasting_env

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

srun python ../run_model.py --config $2 --mode train --model $1 --use_tuned_parameters
