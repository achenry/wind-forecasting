#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=10
#SBATCH --time=01:00:00
#SBATCH --output=%j-%x.log
#SBATCH --partition=atesting_a100

# sinteractive --partition=atesting_a100 --gres=gpu:1 --ntasks-per-node=10 --ntasks=10 --time=50:00


module purge
ml mambaforge
mamba activate wind_forecasting

echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_JOB_GRES=${SLURM_JOB_GRES}"

srun python run_model.py --config ../../examples/inputs/training_inputs_rc.yaml --train --model informer --test
# srun python informer.py
#python train_spacetimeformer.py spacetimeformer windfarm --debug --run_name spacetimeformer_windfarm_debug --context_points 600 --target_points 600

