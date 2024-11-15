#!/bin/bash 
#SBATCH --account=ssc
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=%j-%x.log
#SBATCH --partition=debug

module purge
ml mamba
mamba activate wind_forecasting
cd /home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/wind_forecasting/models

echo $SLURM_NTASKS
#srun --ntasks=$SLURM_NTASKS python run_model.py
python run_model.py
#python train_spacetimeformer.py spacetimeformer windfarm --debug --run_name spacetimeformer_windfarm_debug --context_points 600 --target_points 600
# salloc --account=ssc --time=01:00:00 --mem=80G --gpus=1 --ntasks-per-node=32 --partition=debug
