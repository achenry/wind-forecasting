#!/bin/bash 
#SBATCH --account=awaken
#SBATCH --output=%j-%x-4gpu8cpu_test.out
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8 # 2-5 is usually good
#SBATCH --nodes=1 # this needs to match Trainer(num_nodes...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4 # this needs to match Trainer(devices=...), and number of GPUs
#SBATCH --mem=80G #8192 # refers to CPU (not GPU) memory, automatically given all GPU memory in a SLURM job, 85G

##SBATCH --partition=debug
##SBATCH --time=01:00:00
##SBATCH --mem-per-cpu=5G
##SBATCH --ntasks-per-node=2 # this needs to match Trainer(devices=...), and number of GPUs
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=8

# salloc --account=awaken --time=01:00:00 --gpus=2 --cpus-per-task=4 --ntasks-per-node=2 --partition=debug --mem-per-cpu=20G
# export MODEL=informer
# export MODEL_CONFIG_FILE=$HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_predLUT.yaml

#eval "$(conda shell.bash hook)"

module purge
ml PrgEnv-intel mamba
mamba activate wind_forecasting_env
export PYTHONPATH=$(which python)

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

srun $PYTHONPATH ../run_model.py --config $MODEL_CONFIG_FILE --mode train --model $MODEL --use_tuned_parameters 
ulimit -n 65535
echo "Open file limit is now: $(ulimit -n)"

# trainer.limit_val_batches=100 \
#srun bash -c "LINE_PROFILE=0 python ../run_model.py --config $MODEL_CONFIG_FILE --mode train --model $MODEL --use_tuned_parameters --checkpoint latest \
#                            --override dataset.context_length_factor=10 \
#                                       dataset.sampler=sequential \
#                                       trainer.max_epochs=100 \
#                                       trainer.limit_train_batches=null \
#                                       trainer.val_check_interval=1.0 \
#                                       dataset.batch_size=512"
