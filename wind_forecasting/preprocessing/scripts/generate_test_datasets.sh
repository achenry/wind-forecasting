#!/bin/bash 
#SBATCH --account=awaken
#SBATCH --time=01:00:00
#SBATCH --output=%j-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=104
##SBATCH --mem-per-cpu=10G
#SBATCH --mem=0
#SBATCH --partition=debug

# salloc --account=awaken --time=01:00:00 --gpus=2 --ntasks-per-node=2 --partition=debug

# Print environment info
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_JOB_GRES=${SLURM_JOB_GRES}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"

echo "=== ENVIRONMENT ==="
module list

#export MODEL_CONFIG_PATH="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_predGreedy.yaml $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_predLUT.yaml"
export MODEL_CONFIG_PATH="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_predLUT.yaml"
export DATA_CONFIG_PATH="$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml"
#export NUMEXPR_MAX_THREADS=104

echo "MODELS=${MODELS}"
echo "MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH}"
echo "DATA_CONFIG_PATH=${DATA_CONFIG_PATH}"

# prepare training data first
date +"%Y-%m-%d %H:%M:%S"
module purge
ml mamba
#ml PrgEnv-intel mamba
#eval "$(conda shell.bash hook)"
mamba activate wind_forecasting_env

#mpirun -np $SLURM_NTASKS
WORKER_RANK=0
export PYFILE_PATH="$HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/whoc/wind_forecast/run_forecaster_validation.py"
python $PYFILE_PATH --resplit_data --ram_limit 65 --model_config ${MODEL_CONFIG_PATH} --data_config ${DATA_CONFIG_PATH} --simulation_timestep 1 \
						--save_dir /projects/awaken/ahenry/wind_forecasting/logging --multiprocessor cf --prediction_type distribution \
					        --use_trained_models --max_splits 30

