#!/bin/bash
#SBATCH --nodes=1 # this needs to match Trainer(num_nodes...)
#SBATCH --gres=gpu:3
#SBATCH --ntasks=3 # necessary for gpus on rc
#SBATCH --ntasks-per-node=3 # this needs to match Trainer(devices=...)
##SBATCH --mem-per-cpu=85G TODO what can I ask for here?
#SBATCH --time=01:00:00
#SBATCH --output=%j-%x.log
#SBATCH --partition=atesting_a100

# sinteractive --partition=atesting_a100 --gres=gpu:1 --ntasks-per-node=1 --ntasks=1 --time=50:00

module purge
module load miniforge
module load intel impi
conda activate wind_forecasting

export WANDB_API_KEY=a9aec8e98a88077de29031385225167c720030f7

echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
export WANDB_API_KEY=a9aec8e98a88077de29031385225167c720030f7
# echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
# echo "SLURM_JOB_GRES=${SLURM_JOB_GRES}"
export PYTHON_EXECUTABLE=$(which python)

# NOTE run the following first: python load_data.py --config ../../examples/inputs/training_inputs_rc_awaken.yaml --reload
# mpirun -np $SLURM_NTASKS 
srun $PYTHON_EXECUTABLE run_model.py --config ../../examples/inputs/training_inputs_rc_flasc.yaml --mode train --model $1 
# srun python informer.py
#python train_spacetimeformer.py spacetimeformer windfarm --debug --run_name spacetimeformer_windfarm_debug --context_points 600 --target_points 600

