#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=104
#SBATCH --mem=0
#SBATCH --account=ssc
#SBATCH --time=02:00:00
#SBATCH --partition=bigmem
#SBATCH --partition=standard
#SBATCH --output=load_data.out
#SBATCH --tmp=1T

module purge
module load mamba
mamba activate wind_forecasting_env

python load_data.py --config /$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_kestrel_awaken.yaml --reload
