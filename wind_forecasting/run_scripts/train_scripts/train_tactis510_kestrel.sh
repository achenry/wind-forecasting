#!/bin/bash 
#SBATCH --account=ssc
#SBATCH --output=%j-%x.out
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1 # this needs to match Trainer(num_nodes...)
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2 # this needs to match Trainer(devices=...), and number of GPUs
#SBATCH --mem-per-cpu=40G
#SBATCH --time=48:00:00
#SBATCH --nodes=2 # this needs to match Trainer(num_nodes...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4 # this needs to match Trainer(devices=...), and number of GPUs
#SBATCH --mem-per-cpu=85G

# salloc --account=ssc --time=01:00:00 --gpus=2 --ntasks-per-node=2 --partition=debug

module purge
#module load PrgEnv-intel
#ml mamba
eval "$(conda shell.bash hook)"
conda activate wind_forecasting_env

export NUMEXPR_MAX_THREADS=128
export MODEL=$1
export MODEL_CONFIG_FILE=$2

API_FILE="../.wandb_api_key"
if [ -f "${API_FILE}" ]; then
  source "${API_FILE}"
else
  echo "ERROR: WANDB API‑key file not found at ${API_FILE}" >&2
  exit 1
fi

echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_JOB_GRES=${SLURM_JOB_GRES}"

srun python ../run_model.py --config $MODEL_CONFIG_FILE --mode train --model $MODEL \
  --seed 666 \
  --override dataset.sampler=sequential \
      trainer.max_epochs=40 \
      trainer.limit_train_batches=null \
      trainer.val_check_interval=1.0 \
      #model.tactis.lr_stage1=4.559298934473364e-06 \
      model.tactis.lr_stage1=6.383018508262709e-06 \
      #model.tactis.lr_stage2=4.805723253254209e-06 \
      model.tactis.lr_stage2=6.728012554555892e-06 \
      model.tactis.weight_decay_stage1=0.0 \
      model.tactis.weight_decay_stage2=5e-06 \
      model.tactis.stage=1 \
      model.tactis.stage2_start_epoch=20 \
      model.tactis.warmup_steps_s1=785380 \
      model.tactis.warmup_steps_s2=785380 \
      model.tactis.steps_to_decay_s1=2356140 \
      model.tactis.steps_to_decay_s2=2356140 \
      model.tactis.stage1_activation_function=relu \
      model.tactis.stage2_activation_function=relu \
      model.tactis.eta_min_fraction_s1=0.0035969620681086476 \
      model.tactis.eta_min_fraction_s2=0.00015866914804312245 \
      dataset.batch_size=64 \
      dataset.context_length_factor=5.0 \
      model.tactis.context_length=85 \
      model.tactis.prediction_length=17 \
      model.tactis.flow_series_embedding_dim=5 \
      model.tactis.copula_series_embedding_dim=256 \
      model.tactis.flow_input_encoder_layers=4 \
      model.tactis.copula_input_encoder_layers=2 \
      model.tactis.marginal_embedding_dim_per_head=256 \
      model.tactis.marginal_num_heads=6 \
      model.tactis.marginal_num_layers=4 \
      model.tactis.copula_embedding_dim_per_head=256 \
      model.tactis.copula_num_heads=5 \
      model.tactis.copula_num_layers=1 \
      model.tactis.decoder_dsf_num_layers=3 \
      model.tactis.decoder_dsf_hidden_dim=256 \
      model.tactis.decoder_mlp_num_layers=2 \
      model.tactis.decoder_mlp_hidden_dim=32 \
      model.tactis.decoder_transformer_num_layers=3 \
      model.tactis.decoder_transformer_embedding_dim_per_head=32 \
      model.tactis.decoder_transformer_num_heads=4 \
      model.tactis.decoder_num_bins=200 \
      model.tactis.bagging_size=null \
      model.tactis.input_encoding_normalization=True \
      model.tactis.loss_normalization=both \
      model.tactis.encoder_type=standard \
      model.tactis.dropout_rate=0.005 \
      model.tactis.ac_mlp_num_layers=3 \
      model.tactis.ac_mlp_dim=64 \
      model.tactis.gradient_clip_val_stage1=1.0 \
      model.tactis.gradient_clip_val_stage2=1.0
