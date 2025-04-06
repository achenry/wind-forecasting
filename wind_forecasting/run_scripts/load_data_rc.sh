#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=12:00:00
#SBATCH --qos=mem
#SBATCH --partition=amem
#SBATCH --output=preprocess_data.out

module purge
module load intel impi
#module load miniforge
#conda init
mamba activate wind_forecasting
echo $SLURM_NTASKS
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

python load_data.py --config /projects/aohe7145/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/training_inputs_rc_awaken.yaml --reload
