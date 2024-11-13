#!/bin/bash 
#SBATCH --account=ssc
#SBATCH --nodes=1
#SBATCH --gpus=h100:1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=80G
#SBATCH --output=%j-%x.log

module purge
ml mamba
mamba activate wind_forecasting_env
cd /home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/wind_forecasting/models
srun python run_model.py
#python train_spacetimeformer.py spacetimeformer windfarm --debug --run_name spacetimeformer_windfarm_debug --context_points 600 --target_points 600
# salloc --account=ssc --time=01:00:00 --mem=80G --gpus=1 --ntasks-per-node=32 --partition=debug
