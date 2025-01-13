#!/bin/bash 
#SBATCH --account=ssc
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --output=%j-%x.log
#SBATCH --partition=debug

module purge
ml PrgEnv-intel
ml mamba
mamba activate wind_forecasting
cd /home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/wind_forecasting/models/pytorch-transformer-ts/informer

echo $SLURM_NTASKS
srun python run_model.py --config ../../examples/inputs/training_inputs_kestrel.yaml --train --model informer --test
# srun python informer.py
#python train_spacetimeformer.py spacetimeformer windfarm --debug --run_name spacetimeformer_windfarm_debug --context_points 600 --target_points 600
# salloc --account=ssc --time=01:00:00 --mem-per-cpu=64G --gpus=2 --ntasks-per-node=2 --partition=debug
