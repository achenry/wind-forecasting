#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=104
#SBATCH --mem=0
#SBATCH --account=ssc
#SBATCH --time=06:00:00
#SBATCH --partition=bigmem
#SBATCH --partition=standard
#SBATCH --output=load_data.out
#SBATCH --tmp=1T

module purge
module load mamba
mamba activate wind_forecasting
echo $SLURM_NTASKS
export RUST_BACKTRACE=full
# salloc --partition=debug --mem=0 --time=00:30:00 --ntasks=104 --account=ssc
#export MPICH_SHARED_MEM_COLL_OPT=mpi_bcast,mpi_barrier 
#export MPICH_COLL_OPT_OFF=mpi_allreduce 
#export LD_LIBRARy_PATH=$CONDA_PREFIX/lib

# cd $LARGE_STORAGE/ahenry/wind_forecasting_env/wind-forecasting/wind_forecasting/preprocessing
# conda activate wind_forecasting_preprocessing
# python preprocessing_main.py --config /srv/data/nfs/ahenry/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_server_awaken_new.yaml --reload_data --multiprocessor cf 

mpirun -np $SLURM_NTASKS python preprocessing_main.py --config /$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_awaken_new.yaml --reload_data --multiprocessor mpi

mv /tmp/scratch/$SLURM_JOB_ID/*.parquet /projects/ssc/ahenry/wind_forecasting/awaken_data/ 
