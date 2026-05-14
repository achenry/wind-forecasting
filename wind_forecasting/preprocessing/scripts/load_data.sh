#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
##SBATCH --cpus-per-task=104
#SBATCH --mem=0
#SBATCH --account=awaken
#SBATCH --time=04:00:00
#SBATCH --partition=bigmem
##SBATCH --partition=standard
#SBATCH --output=load_data-%j.out
##SBATCH --tmp=1T
#SBATCH --exclusive

module purge
# ml PrgEnv-intel mamba
ml mamba
mamba activate wind_forecasting_env
echo $SLURM_NTASKS
export NUMEXPR_MAX_THREADS=96
export MAX_WORKERS=96

#python ../preprocessing_main.py --config $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml --reload_data --multiprocessor cf
#srun 
python ../preprocessing_main.py --config $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml --reload_data --multiprocessor cf # mpi

