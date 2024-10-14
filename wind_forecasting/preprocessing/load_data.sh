#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
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

#rm /pl/active/paolab/awaken_data/kp.turbine.z02.b0/*.nc
#mv /scratch/alpine/aohe7145/awaken_data/kp.turbine.zo2.b0.raw.parquet /pl/active/paolab/awaken_data/kp.turbine.zo2.b0.raw.parquet
