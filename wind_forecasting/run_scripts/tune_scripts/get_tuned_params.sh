#!/bin/bash 
#SBATCH --account=ssc
#SBATCH --time=01:00:00
#SBATCH --output=%j-%x.out
#SBATCH --nodes=1 # this needs to match Trainer(num_nodes...)
#SBATCH --ntasks-per-node=1 # this needs to match Trainer(devices=...)

# salloc --account=ssc --time=01:00:00 --ntasks-per-node=1 --partition=debug

# --- Command Line Args ---
export MODEL_NAME=$1
export CONFIG_FILE=$2

#export CONFIG_FILE=$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred60.yaml
#export MODEL_NAME=informer

# --- Base Directories ---
export BASE_DIR="/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"

# --- Set Shared Environment Variables ---
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
# --- Setup Main Environment ---
echo "Setting up main environment..."
module purge
echo "Modules loaded."

eval "$(conda shell.bash hook)"
conda activate wind_forecasting_env
echo "Conda environment 'wind_forecasting_env' activated."
# --- End Main Environment Setup ---


python ${WORK_DIR}/run_scripts/get_tuned_params.py \
        --config ${CONFIG_FILE} \
        --model ${MODEL_NAME}