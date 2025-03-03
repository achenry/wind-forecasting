#!/bin/bash
#SBATCH --partition=cfdg.p         # Partition for H100/A100 GPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:4         # Request 4 GPUs
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=0-12:00
#SBATCH --job-name=tactis_train_flasc
#SBATCH --output=tactis_train_flasc_%j.out
#SBATCH --error=tactis_train_flasc_%j.err
#SBATCH --mail-user=juan.manuel.boullosa.novo@uol.de
#SBATCH --mail-type=END,FAIL

BASE_DIR="/user/taed7566/wind-forecasting"
WORK_DIR="${BASE_DIR}/wind_forecasting"
cd ${WORK_DIR}

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate wf_env_2

# Setup output file
OUTPUT_FILE="tactis_train_flasc_${SLURM_JOB_ID}.out"
echo "--- GPU Verification ---" >> $OUTPUT_FILE
nvidia-smi -L >> $OUTPUT_FILE
echo "--- Starting Training ---" >> $OUTPUT_FILE

# Run with absolute paths
srun python ${WORK_DIR}/run_scripts/run_model.py \
  --config ${BASE_DIR}/examples/inputs/training_inputs_juan_flasc.yaml \
  --model tactis \
  --mode train

# sbatch train_model_storm.sh
# sinfo -p cfdg.p
# squeue -u taed7566
# tail -f tactis_train_flasc_%j.out