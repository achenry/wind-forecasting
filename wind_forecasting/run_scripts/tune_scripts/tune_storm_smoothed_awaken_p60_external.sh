#!/bin/bash

#SBATCH --partition=cfdg.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00
#SBATCH --job-name=smoothed60_tune_ext
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/smoothed60_tune_ext_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/smoothed60_tune_ext_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# FULL TUNING SCRIPT - EXTERNAL NORMALIZATION
# Purpose: Complete hyperparameter tuning with 25 epochs and 200 trials
# Usage: sbatch tune_storm_smoothed_awaken_p60_external.sh <model> [--restart_tuning]
#        model: informer, autoformer, spacetimeformer
# Example: sbatch tune_storm_smoothed_awaken_p60_external.sh informer
#          sbatch tune_storm_smoothed_awaken_p60_external.sh autoformer --restart_tuning
# =============================================================================

# --- Arguments ---
export MODEL_NAME="${1:-informer}"

# Check for --restart_tuning flag
export RESTART_TUNING_FLAG=""
for arg in "$@"; do
    if [ "$arg" == "--restart_tuning" ]; then
        export RESTART_TUNING_FLAG="--restart_tuning"
        echo "Restart tuning flag enabled - will create new study"
    fi
done

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_storm_awaken_smoothed_pred60_external.yaml"

export AUTO_EXIT_WHEN_DONE="true"
export NUMEXPR_MAX_THREADS=128

# --- Create Logging Directories ---
mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

# --- Change to Working Directory ---
cd ${WORK_DIR} || exit 1

# --- Set Environment Variables ---
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

# --- Print Job Info ---
echo "=============================================="
echo "FULL TUNING - EXTERNAL NORMALIZATION"
echo "=============================================="
echo "JOB ID: ${SLURM_JOB_ID}"
echo "JOB NAME: ${SLURM_JOB_NAME}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE LIST: ${SLURM_JOB_NODELIST}"
echo "NUM GPUS: ${SLURM_NTASKS_PER_NODE}"
echo "----------------------------------------------"
echo "MODEL: ${MODEL_NAME}"
echo "NORMALIZATION: external"
echo "CONFIG: ${CONFIG_FILE}"
echo "RESTART FLAG: ${RESTART_TUNING_FLAG}"
echo "----------------------------------------------"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
echo "=============================================="

# --- Setup Environment ---
echo "Setting up environment..."
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load Mamba/24.3.0-0
module load CUDA/12.4.0
module load git
echo "Modules loaded."

eval "$(conda shell.bash hook)"
conda activate wf_env_storm
echo "Conda environment 'wf_env_storm' activated."

# PostgreSQL password (Oldenburg University)
export PGPASSWORD="${LOCAL_PG_PASSWORD}"

echo "=== STARTING PARALLEL OPTUNA TUNING WORKERS ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Parallel Worker Launch ---
NUM_GPUS=${SLURM_NTASKS_PER_NODE}
export WORLD_SIZE=${NUM_GPUS}
declare -a WORKER_PIDS=()

echo "Launching ${NUM_GPUS} tuning workers for FULL TUNING (25 epochs, 200 trials)..."

for i in $(seq 0 $((${NUM_GPUS}-1))); do
    CURRENT_WORKER_SEED=$((42 + i*100))

    echo "Starting worker ${i} on GPU ${i} with seed ${CURRENT_WORKER_SEED}"

    WORKER_RANK=${i} CUDA_VISIBLE_DEVICES=${i} nohup bash -c "
        echo \"Worker ${i} starting environment setup...\"
        module purge
        module load slurm/hpc-2023/23.02.7
        module load hpc-env/13.1
        module load mpi4py/3.1.4-gompi-2023a
        module load Mamba/24.3.0-0
        module load CUDA/12.4.0
        module load git
        echo \"Worker ${i}: Modules loaded.\"

        eval \"\$(conda shell.bash hook)\"
        conda activate wf_env_storm
        echo \"Worker ${i}: Conda environment activated.\"

        echo \"Worker ${i}: Running FULL TUNING (config defaults: 25 epochs, 200 trials)...\"

        python ${WORK_DIR}/run_scripts/run_model.py \\
          --config ${CONFIG_FILE} \\
          --model ${MODEL_NAME} \\
          --mode tune \\
          --seed ${CURRENT_WORKER_SEED} \\
          ${RESTART_TUNING_FLAG} \\
          --single_gpu

        status=\$?
        if [ \$status -ne 0 ]; then
            echo \"Worker ${i} FAILED with status \$status\"
        else
            echo \"Worker ${i} COMPLETED successfully\"
        fi
        exit \$status
    " > "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/worker_${i}_${SLURM_JOB_ID}.log" 2>&1 &

    WORKER_PIDS+=($!)
    sleep 2
done

echo "--- Worker Processes Launched ---"
echo "Number of workers: ${#WORKER_PIDS[@]}"
echo "Process IDs: ${WORKER_PIDS[@]}"

# --- Wait for workers ---
while true; do
    ALIVE_WORKERS=0
    for pid in ${WORKER_PIDS[@]}; do
        if kill -0 $pid 2>/dev/null; then
            ((ALIVE_WORKERS++))
        fi
    done

    echo "$(date +"%Y-%m-%d %H:%M:%S") - Workers still running: ${ALIVE_WORKERS}/${#WORKER_PIDS[@]}"

    if [ $ALIVE_WORKERS -eq 0 ]; then
        echo "All workers have finished."
        break
    fi
    sleep 60
done

# --- Final Status Check ---
echo "--- Worker Completion Status ---"
FAILED_WORKERS=0
SUCCESSFUL_WORKERS=0
for i in $(seq 0 $((${NUM_GPUS}-1))); do
    WORKER_LOG="${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/worker_${i}_${SLURM_JOB_ID}.log"
    if [ -f "$WORKER_LOG" ]; then
        if grep -q "COMPLETED successfully" "$WORKER_LOG"; then
            echo "Worker ${i}: SUCCESS"
            ((SUCCESSFUL_WORKERS++))
        elif grep -q "FAILED with status" "$WORKER_LOG"; then
            echo "Worker ${i}: FAILED"
            ((FAILED_WORKERS++))
        else
            echo "Worker ${i}: UNKNOWN status"
            ((FAILED_WORKERS++))
        fi
    else
        echo "Worker ${i}: FAILED (log file not found)"
        ((FAILED_WORKERS++))
    fi
done

echo "=============================================="
echo "FULL TUNING COMPLETED - EXTERNAL NORMALIZATION"
echo "Model: ${MODEL_NAME}"
echo "Successful workers: ${SUCCESSFUL_WORKERS}/${NUM_GPUS}"
echo "=============================================="
date +"%Y-%m-%d %H:%M:%S"

# Move logs
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.out; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/" 2>/dev/null
done
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.err; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/" 2>/dev/null
done

exit $FAILED_WORKERS
