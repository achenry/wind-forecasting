#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --partition=amem
#SBATCH --output=data_loader.out

module purge
module load mambaforge
mamba activate wind_forecasting_env
#module load gcc
#module load openmpi
module load intel
module load impi

#export LD_LIBRARy_PATH=$CONDA_PREFIX/lib

mpirun -np $SLURM_NTASKS python data_loader.py

# rm -rf 