export MODEL_CONFIG_FILE=$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_pred60.yaml
export MODEL=informer

module purge
ml mamba
mamba activate wind_forecasting_env

export NUMEXPR_MAX_THREADS=128

API_FILE="../.wandb_api_key"
if [ -f "${API_FILE}" ]; then
  source "${API_FILE}"
else
  echo "ERROR: WANDB API‑key file not found at ${API_FILE}" >&2
  exit 1
fi

srun python ../run_model.py --config $MODEL_CONFIG_FILE --mode train --model $MODEL --use_tuned_parameters
