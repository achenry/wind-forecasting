#!/bin/bash 
#SBATCH --account=awaken
#SBATCH --time=02:00:00
#SBATCH --output=%j-%x.out
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=104
#SBATCH --ntasks=104
#SBATCH --mem=0
#SBATCH --exclusive

# salloc --account=awaken --time=01:00:00 --gpus=2 --ntasks-per-node=2 --partition=debug

module purge
eval "$(conda shell.bash hook)"
conda activate wind_forecasting_env

#export NUMEXPR_MAX_THREADS=128

API_FILE="../.wandb_api_key"
if [ -f "${API_FILE}" ]; then
  source "${API_FILE}"
else
  echo "ERROR: WANDB API‑key file not found at ${API_FILE}" >&2
  exit 1
fi

echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"

WORKER_RANK=0
# MDL_CONF_G=$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_predGreedy.yaml
MDL_CONF_L=$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_predLUT.yaml

#python ../run_model.py --config $MDL_CONF_G $MDL_CONF_G $MDL_CONF_L $MDL_CONF_L --model informer informer tactis tactis --mode dataset --reload_data 
# python ../run_model.py --config $MDL_CONF_G $MDL_CONF_L --model tactis tactis --mode dataset --reload_data 
python ../run_model.py --config $MDL_CONF_L --model tactis --mode dataset --reload_data 
