#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=52
##SBATCH --mem=0
#SBATCH --account=ssc
#SBATCH --time=01:00:00
##SBATCH --partition=bigmem
#SBATCH --partition=standard
#SBATCH --output=data_loader.out

module purge
module load mamba
mamba activate wind_forecasting_env
echo $SLURM_NTASKS

export MPICH_SHARED_MEM_COLL_OPT=mpi_bcast,mpi_barrier 
export MPICH_COLL_OPT_OFF=mpi_allreduce 
export LD_LIBRARy_PATH=$CONDA_PREFIX/lib

mpirun -np $SLURM_NTASKS python data_loader.py
#python data_loader.py

#rm /pl/active/paolab/awaken_data/kp.turbine.z02.b0/*.nc
#mv /scratch/alpine/aohe7145/awaken_data/kp.turbine.zo2.b0.raw.parquet /pl/active/paolab/awaken_data/kp.turbine.zo2.b0.raw.parquet
