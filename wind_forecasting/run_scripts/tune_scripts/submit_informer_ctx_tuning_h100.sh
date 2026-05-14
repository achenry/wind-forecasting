#!/bin/bash
# =============================================================================
# BATCH SUBMISSION: Informer Context Length Tuning on H100 GPUs
# Submits N individual 1-GPU H100 jobs on all_gpu.p
# All jobs join the SAME Optuna study (no --restart_tuning)
# =============================================================================

NUM_JOBS=${1:-6}
BASE_SEED=42

echo "=============================================="
echo "INFORMER CONTEXT LENGTH TUNING - H100 BATCH"
echo "Submitting ${NUM_JOBS} individual 1-GPU H100 jobs"
echo "=============================================="

for i in $(seq 0 $((NUM_JOBS-1))); do
    SEED=$((BASE_SEED + i*100))

    TEMP_SCRIPT=$(mktemp /tmp/tune_informer_ctx_h100_XXXXXX.sh)
    cat > "$TEMP_SCRIPT" << JOBSCRIPT
#!/bin/bash
#SBATCH --partition=all_gpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:H100:1
#SBATCH --time=1-00:00
#SBATCH --job-name=informer_ctx_h100_${i}
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/informer_ctx_h100_${i}_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/informer_ctx_h100_${i}_%j.err
#SBATCH --hint=nomultithread

# --- Configuration ---
export MODEL_NAME="informer"
export NUMEXPR_MAX_THREADS=128

BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
WORK_DIR="\${BASE_DIR}/wind_forecasting"
LOG_DIR="\${OUTPUT_DIR}/logs"
CONFIG_FILE="\${BASE_DIR}/config/training/training_inputs_storm_awaken_smoothed_pred60_external.yaml"

cd \${WORK_DIR} || exit 1
export PYTHONPATH=\${WORK_DIR}:\${PYTHONPATH}
export WANDB_DIR=\${LOG_DIR}

echo "=============================================="
echo "INFORMER CONTEXT LENGTH TUNING - H100 Worker ${i}"
echo "JOB ID: \${SLURM_JOB_ID}"
echo "NODE: \${SLURM_JOB_NODELIST}"
echo "SEED: ${SEED}"
echo "=============================================="

# --- Setup Environment ---
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load Mamba/24.3.0-0
module load CUDA/12.4.0
module load git

eval "\$(conda shell.bash hook)"
conda activate wf_env_storm

export PGPASSWORD="\${LOCAL_PG_PASSWORD}"
export WORKER_RANK=0
export CUDA_VISIBLE_DEVICES=0

GPU_TYPE=\$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: \${GPU_TYPE}"
echo "=== STARTING TUNING ==="
date +"%Y-%m-%d %H:%M:%S"

python \${WORK_DIR}/run_scripts/run_model.py \\
  --config \${CONFIG_FILE} \\
  --model informer \\
  --mode tune \\
  --seed ${SEED} \\
  --single_gpu

EXIT_CODE=\$?
echo "=== TUNING FINISHED WITH EXIT CODE: \${EXIT_CODE} ==="
date +"%Y-%m-%d %H:%M:%S"
exit \$EXIT_CODE
JOBSCRIPT

    JOB_OUTPUT=$(sbatch --export=ALL "$TEMP_SCRIPT" 2>&1)
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+$')
    echo "  Worker ${i}: seed=${SEED} -> ${JOB_OUTPUT}"

    (sleep 5 && rm -f "$TEMP_SCRIPT") &
    sleep 1
done

echo "=============================================="
echo "All ${NUM_JOBS} jobs submitted."
echo "They will join the existing Informer Optuna study."
echo "Context length factors: [10, 20, 40, 60]"
echo "=============================================="
