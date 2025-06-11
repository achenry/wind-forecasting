#!/bin/bash 
#SBATCH --account=ssc
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1 # this needs to match Trainer(num_nodes...)
##SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2 # this needs to match Trainer(devices=...), and number of GPUs
#SBATCH --mem-per-cpu=20G
#SBATCH --output=%j-%x.out
##SBATCH --time=36:00:00
##SBATCH --nodes=1 # this needs to match Trainer(num_nodes...)
##SBATCH --gres=gpu:4
##SBATCH --ntasks-per-node=4 # this needs to match Trainer(devices=...), and number of GPUs
##SBATCH --mem-per-cpu=20G

# salloc --account=ssc --time=01:00:00 --gpus=2 --ntasks-per-node=2 --partition=debug

module purge
ml PrgEnv-intel mamba
#eval "$(conda shell.bash hook)"
mamba activate wind_forecasting_env

srun python train_test.py
