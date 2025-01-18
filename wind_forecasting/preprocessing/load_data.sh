#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=104
##SBATCH --mem=0
#SBATCH --account=ssc
#SBATCH --time=02:00:00
##SBATCH --partition=bigmem
#SBATCH --partition=standard
#SBATCH --output=data_loader_scratch.out
#SBATCH --tmp=1T

module purge
module load mamba
mamba activate wind_forecasting_env
echo $SLURM_NTASKS
export RUST_BACKTRACE=full

#export MPICH_SHARED_MEM_COLL_OPT=mpi_bcast,mpi_barrier 
#export MPICH_COLL_OPT_OFF=mpi_allreduce 
#export LD_LIBRARy_PATH=$CONDA_PREFIX/lib

# cd $LARGE_STORAGE/ahenry/wind_forecasting_env/wind-forecasting/wind_forecasting/preprocessing
# conda activate wind_forecasting_preprocessing
# python preprocessing_main.py --config /srv/data/nfs/ahenry/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_server_awaken_new.yaml --reload_data --multiprocessor cf 

mpirun -np $SLURM_NTASKS python preprocessing_main.py --config /$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_awaken_new.yaml --reload_data --multiprocessor mpi
#python data_loader.py

#rm /pl/active/paolab/awaken_data/kp.turbine.z02.b0/*.nc
#mv /scratch/alpine/aohe7145/awaken_data/kp.turbine.zo2.b0.raw.parquet /pl/active/paolab/awaken_data/kp.turbine.zo2.b0.raw.parquet
mv /tmp/scratch/$SLURM_JOB_ID/00_engie_scada_processed.parquet /projects/ssc/ahenry/wind_forecasting/awaken_data/ 
