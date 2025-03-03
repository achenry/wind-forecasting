#!/bin/bash
#SBATCH --partition=cfdg.p         # Partition for H100 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4        # GPUs per node
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=960G
#SBATCH --time=24:00:00
#SBATCH --job-name=tactis_train
#SBATCH --output=tactis_train_%j.out
#SBATCH --error=tactis_train_%j.err


cd /user/taed7566/wind-forecasting/wind_forecasting
mamba activate wf_env_2

srun python run_scripts/run_model.py \
  --config examples/inputs/training_inputs_juan_flasc.yaml \
  --model tactis \
  --mode train


# sbatch train_model_storm.sh
# sinfo -p cfdg.p
# squeue -u taed7566
# tail -f tactis_train_1234567890.out