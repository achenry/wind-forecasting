#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=48
##SBATCH --ntasks=1
##SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --qos=mem
##SBATCH --time=01:00:00
#SBATCH --partition=amem
##SBATCH --partition=atesting
#SBATCH --output=preprocess_data.out
##SBATCH --tmp=1T

module purge
module load intel impi
module load miniforge
#conda init
mamba activate wind_forecasting
echo $SLURM_NTASKS
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

python preprocessing_main.py --config /projects/aohe7145/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_rc_awaken_new.yaml --multiprocessor cf --preprocess_data #--regenerate_filters
# python preprocessing_main.py --config /$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_flasc.yaml --multiprocessor cf --preprocess_data #--regenerate_filters
# srun preprocessing_main.py --config /$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_awaken_new.yaml --multiprocessor mpi --preprocess_data #--regenerate_filters

# salloc --nodes=1 --ntasks=1 --time=01:00:00 --partition=atesting 
