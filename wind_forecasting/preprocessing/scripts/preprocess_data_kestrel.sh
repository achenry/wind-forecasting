#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=96 # NOTE use 1 for st_dev calc
#SBATCH --mem=0
#SBATCH --account=awaken
#SBATCH --time=48:00:00
#SBATCH --partition=bigmem
##SBATCH --time=01:00:00
##SBATCH --partition=standard
#SBATCH --output=preprocess_data-%j.out
##SBATCH --tmp=1T

module purge
module load mamba
mamba activate wind_forecasting_env

# export POLARS_MAX_THREADS=96
# export OMP_NUM_THREADS=96, try 48 here (for numpy) and 48 above (for polars), or try omitting thread max lines
# export NUMEXPR_MAX_THREADS=96
export MAX_WORKERS=96

python ../preprocessing_main.py --config /$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml --multiprocessor cf --preprocess_data # --regenerate_filters

