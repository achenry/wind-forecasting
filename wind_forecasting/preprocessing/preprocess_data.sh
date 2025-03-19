#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=104 # NOTE use 1 for st_dev calc
#SBATCH --mem=0
#SBATCH --account=ssc
#SBATCH --time=12:00:00
#SBATCH --partition=bigmem
#SBATCH --partition=standard
#SBATCH --output=preprocess_data.out
#SBATCH --tmp=1T

module purge
module load mamba
mamba activate wind_forecasting
#echo $SLURM_NTASKS
#export RUST_BACKTRACE=full

#module load openmpi/4.1.6-intel
#export MPICC=$(which mpicc)

#export MPICH_SHARED_MEM_COLL_OPT=mpi_bcast,mpi_barrier 
#export MPICH_COLL_OPT_OFF=mpi_allreduce 
#export LD_LIBRARy_PATH=$CONDA_PREFIX/lib

# cd $LARGE_STORAGE/ahenry/wind_forecasting_env/wind-forecasting/wind_forecasting/preprocessing
# conda activate wind_forecasting_preprocessing
# python preprocessing_main.py --config /srv/data/nfs/ahenry/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_server_awaken_new.yaml --reload_data --multiprocessor cf 

python preprocessing_main.py --config /$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_awaken_new.yaml --multiprocessor cf --preprocess_data #--regenerate_filters
# python preprocessing_main.py --config /$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_flasc.yaml --multiprocessor cf --preprocess_data #--regenerate_filters
# srun preprocessing_main.py --config /$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_awaken_new.yaml --multiprocessor mpi --preprocess_data #--regenerate_filters

