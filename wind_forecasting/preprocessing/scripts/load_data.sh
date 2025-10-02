#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=104
#SBATCH --mem=0
#SBATCH --account=awaken
#SBATCH --time=06:00:00
#SBATCH --partition=bigmem
##SBATCH --partition=standard
#SBATCH --output=load_data-%j.out
#SBATCH --tmp=1T
#SBATCH --exclusive

module purge
ml PrgEnv-intel mamba
mamba activate wind_forecasting_env
#echo $SLURM_NNODES * $SLURM_NTASKS
echo $SLURM_NTASKS
#ntasks=$((SLURM_NTASKS*SLURM_NNODES))
#echo $ntasks
#module load openmpi/4.1.6-intel
#export MPICC=$(which mpicc)
export NUMEXPR_MAX_THREADS=104
# export RUST_BACKTRACE=full
# salloc --partition=debug --mem=0 --time=00:30:00 --ntasks=104 --account=awaken
#export MPICH_SHARED_MEM_COLL_OPT=mpi_bcast,mpi_barrier 
#export MPICH_COLL_OPT_OFF=mpi_allreduce 
#export LD_LIBRARy_PATH=$CONDA_PREFIX/lib

# cd $LARGE_STORAGE/ahenry/wind_forecasting_env/wind-forecasting/wind_forecasting/preprocessing
# conda activate wind_forecasting_preprocessing
# python preprocessing_main.py --config /srv/data/nfs/ahenry/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_server_awaken_new.yaml --reload_data --multiprocessor cf 
# /home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml 
# mpirun -np $SLURM_NTASKS
srun python ../preprocessing_main.py --config $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml --reload_data --multiprocessor mpi

#srun python preprocessing_main.py --config /$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_kestrel_awaken_new.yaml --reload_data --multiprocessor mpi
#srun python preprocessing_main.py --config /$HOME/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/preprocessing_inputs_rc_awaken.yaml --reload_data --multiprocessor mpi

#mv /tmp/scratch/$SLURM_JOB_ID/*.parquet /projects/awaken/ahenry/wind_forecasting/awaken_data/ 
#mv /tmp/scratch/$SLURM_JOB_ID/*.parquet /projects/awaken/ahenry/wind_forecasting/flasc_data/ 
