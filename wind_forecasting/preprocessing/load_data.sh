#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=104
##SBATCH --mem=0
#SBATCH --account=ssc
#SBATCH --time=02:00:00
#SBATCH --partition=bigmem
#SBATCH --output=data_loader.out

module purge
module load mamba
mamba activate wind_forecasting_env

export LD_LIBRARy_PATH=$CONDA_PREFIX/lib

mpirun -np $SLURM_NTASKS python data_loader.py
#python data_loader.py

#rm /pl/active/paolab/awaken_data/kp.turbine.z02.b0/*.nc
#mv /scratch/alpine/aohe7145/awaken_data/kp.turbine.zo2.b0.raw.parquet /pl/active/paolab/awaken_data/kp.turbine.zo2.b0.raw.parquet
