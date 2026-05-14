#!/bin/bash

# =============================================================================
# FLEXIBLE TUNING SCRIPT - EXTERNAL NORMALIZATION
# Purpose: Hyperparameter tuning with configurable resources via arguments
#
# Usage: sbatch tune_storm_smoothed_awaken_p60_external_flex.sh [OPTIONS]
#
# OPTIONS:
#   -m, --model MODEL        Model name (default: informer)
#   -g, --gpus N             Number of GPUs to request (default: 4)
#   -p, --partition PART     Partition name (default: all_gpu.p)
#   -t, --time TIME          Time limit (default: 1-00:00 for all_gpu.p, 7-00:00 for cfdg.p)
#   -s, --seed SEED          Base seed for workers (default: 42)
#   -r, --restart            Restart tuning with new study
#   -n, --node NODE          Request specific node (optional)
#
# EXAMPLES:
#   # Use 1 GPU on mpcg003 for 24h:
#   sbatch tune_storm_smoothed_awaken_p60_external_flex.sh -g 1 -p all_gpu.p -n mpcg003
#
#   # Use 2 GPUs on cfdg.p for 7 days:
#   sbatch tune_storm_smoothed_awaken_p60_external_flex.sh -g 2 -p cfdg.p
#
#   # Use all 4 GPUs with default settings:
#   sbatch tune_storm_smoothed_awaken_p60_external_flex.sh -m autoformer
#
# =============================================================================

# --- Default values ---
MODEL_NAME="informer"
NUM_GPUS=4
PARTITION="all_gpu.p"
TIME_LIMIT=""
BASE_SEED=42
RESTART_TUNING=""
SPECIFIC_NODE=""

# --- Parse command line arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -g|--gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -p|--partition)
            PARTITION="$2"
            shift 2
            ;;
        -t|--time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        -s|--seed)
            BASE_SEED="$2"
            shift 2
            ;;
        -r|--restart)
            RESTART_TUNING="--restart_tuning"
            shift
            ;;
        -n|--node)
            SPECIFIC_NODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# --- Set time limit based on partition if not specified ---
if [[ -z "$TIME_LIMIT" ]]; then
    case "$PARTITION" in
        cfdg.p)
            TIME_LIMIT="7-00:00"
            ;;
        mpcg.p)
            TIME_LIMIT="7-00:00"
            ;;
        all_gpu.p|*)
            TIME_LIMIT="1-00:00"
            ;;
    esac
fi

# --- Calculate CPUs (8 per GPU) ---
NUM_CPUS=$((NUM_GPUS * 8))

# --- Build SBATCH options dynamically ---
SBATCH_OPTS=""
SBATCH_OPTS+="#SBATCH --partition=${PARTITION}\n"
SBATCH_OPTS+="#SBATCH --nodes=1\n"
SBATCH_OPTS+="#SBATCH --ntasks-per-node=${NUM_GPUS}\n"
SBATCH_OPTS+="#SBATCH --cpus-per-task=8\n"
SBATCH_OPTS+="#SBATCH --mem-per-cpu=8016\n"
SBATCH_OPTS+="#SBATCH --gres=gpu:${NUM_GPUS}\n"
SBATCH_OPTS+="#SBATCH --time=${TIME_LIMIT}\n"
SBATCH_OPTS+="#SBATCH --job-name=tune_${MODEL_NAME}_${NUM_GPUS}gpu\n"
SBATCH_OPTS+="#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tune_flex_%j.out\n"
SBATCH_OPTS+="#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tune_flex_%j.err\n"
SBATCH_OPTS+="#SBATCH --hint=nomultithread\n"
SBATCH_OPTS+="#SBATCH --distribution=block:block\n"
SBATCH_OPTS+="#SBATCH --gres-flags=enforce-binding\n"

if [[ -n "$SPECIFIC_NODE" ]]; then
    SBATCH_OPTS+="#SBATCH --nodelist=${SPECIFIC_NODE}\n"
fi

# --- Create temporary job script ---
TEMP_SCRIPT=$(mktemp /tmp/tune_flex_XXXXXX.sh)
cat > "$TEMP_SCRIPT" << 'SCRIPT_END'
#!/bin/bash
SCRIPT_END

# Add SBATCH options
echo -e "$SBATCH_OPTS" >> "$TEMP_SCRIPT"

# Add the rest of the script
cat >> "$TEMP_SCRIPT" << SCRIPT_BODY

# --- Configuration from launcher ---
export MODEL_NAME="${MODEL_NAME}"
export BASE_SEED=${BASE_SEED}
export RESTART_TUNING_FLAG="${RESTART_TUNING}"
export REQUESTED_GPUS=${NUM_GPUS}

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="\${BASE_DIR}/wind_forecasting"
export LOG_DIR="\${OUTPUT_DIR}/logs"
export CONFIG_FILE="\${BASE_DIR}/config/training/training_inputs_storm_awaken_smoothed_pred60_external.yaml"

export AUTO_EXIT_WHEN_DONE="true"
export NUMEXPR_MAX_THREADS=128

# --- Create Logging Directories ---
mkdir -p \${LOG_DIR}/slurm_logs/\${SLURM_JOB_ID}
mkdir -p \${LOG_DIR}/checkpoints

# --- Change to Working Directory ---
cd \${WORK_DIR} || exit 1

# --- Set Environment Variables ---
export PYTHONPATH=\${WORK_DIR}:\${PYTHONPATH}
export WANDB_DIR=\${LOG_DIR}

# --- Print Job Info ---
echo "=============================================="
echo "FLEXIBLE TUNING - EXTERNAL NORMALIZATION"
echo "=============================================="
echo "JOB ID: \${SLURM_JOB_ID}"
echo "JOB NAME: \${SLURM_JOB_NAME}"
echo "PARTITION: \${SLURM_JOB_PARTITION}"
echo "NODE LIST: \${SLURM_JOB_NODELIST}"
echo "REQUESTED GPUS: \${REQUESTED_GPUS}"
echo "ACTUAL TASKS: \${SLURM_NTASKS_PER_NODE}"
echo "----------------------------------------------"
echo "MODEL: \${MODEL_NAME}"
echo "NORMALIZATION: external"
echo "CONFIG: \${CONFIG_FILE}"
echo "BASE SEED: \${BASE_SEED}"
echo "RESTART FLAG: \${RESTART_TUNING_FLAG}"
echo "----------------------------------------------"
GPU_TYPE=\$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: \${GPU_TYPE}"
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

eval "\$(conda shell.bash hook)"
conda activate wf_env_storm
echo "Conda environment 'wf_env_storm' activated."

# PostgreSQL password (Oldenburg University)
export PGPASSWORD="\${LOCAL_PG_PASSWORD}"

echo "=== STARTING PARALLEL OPTUNA TUNING WORKERS ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Parallel Worker Launch ---
NUM_GPUS=\${SLURM_NTASKS_PER_NODE}
export WORLD_SIZE=\${NUM_GPUS}
declare -a WORKER_PIDS=()

echo "Launching \${NUM_GPUS} tuning workers..."

for i in \$(seq 0 \$((NUM_GPUS-1))); do
    CURRENT_WORKER_SEED=\$((BASE_SEED + i*100))

    echo "Starting worker \${i} on GPU \${i} with seed \${CURRENT_WORKER_SEED}"

    WORKER_RANK=\${i} CUDA_VISIBLE_DEVICES=\${i} nohup bash -c "
        echo \"Worker \${i} starting environment setup...\"
        module purge
        module load slurm/hpc-2023/23.02.7
        module load hpc-env/13.1
        module load mpi4py/3.1.4-gompi-2023a
        module load Mamba/24.3.0-0
        module load CUDA/12.4.0
        module load git
        echo \"Worker \${i}: Modules loaded.\"

        eval \"\\\$(conda shell.bash hook)\"
        conda activate wf_env_storm
        echo \"Worker \${i}: Conda environment activated.\"

        echo \"Worker \${i}: Running TUNING...\"

        python \${WORK_DIR}/run_scripts/run_model.py \\\\
          --config \${CONFIG_FILE} \\\\
          --model \${MODEL_NAME} \\\\
          --mode tune \\\\
          --seed \${CURRENT_WORKER_SEED} \\\\
          \${RESTART_TUNING_FLAG} \\\\
          --single_gpu

        status=\\\$?
        if [ \\\$status -ne 0 ]; then
            echo \"Worker \${i} FAILED with status \\\$status\"
        else
            echo \"Worker \${i} COMPLETED successfully\"
        fi
        exit \\\$status
    " > "\${LOG_DIR}/slurm_logs/\${SLURM_JOB_ID}/worker_\${i}_\${SLURM_JOB_ID}.log" 2>&1 &

    WORKER_PIDS+=(\$!)
    sleep 2
done

echo "--- Worker Processes Launched ---"
echo "Number of workers: \${#WORKER_PIDS[@]}"
echo "Process IDs: \${WORKER_PIDS[@]}"

# --- Wait for workers ---
while true; do
    ALIVE_WORKERS=0
    for pid in \${WORKER_PIDS[@]}; do
        if kill -0 \$pid 2>/dev/null; then
            ((ALIVE_WORKERS++))
        fi
    done

    echo "\$(date +\"%Y-%m-%d %H:%M:%S\") - Workers still running: \${ALIVE_WORKERS}/\${#WORKER_PIDS[@]}"

    if [ \$ALIVE_WORKERS -eq 0 ]; then
        echo "All workers have finished."
        break
    fi
    sleep 60
done

# --- Final Status Check ---
echo "--- Worker Completion Status ---"
FAILED_WORKERS=0
SUCCESSFUL_WORKERS=0
for i in \$(seq 0 \$((NUM_GPUS-1))); do
    WORKER_LOG="\${LOG_DIR}/slurm_logs/\${SLURM_JOB_ID}/worker_\${i}_\${SLURM_JOB_ID}.log"
    if [ -f "\$WORKER_LOG" ]; then
        if grep -q "COMPLETED successfully" "\$WORKER_LOG"; then
            echo "Worker \${i}: SUCCESS"
            ((SUCCESSFUL_WORKERS++))
        elif grep -q "FAILED with status" "\$WORKER_LOG"; then
            echo "Worker \${i}: FAILED"
            ((FAILED_WORKERS++))
        else
            echo "Worker \${i}: UNKNOWN status"
            ((FAILED_WORKERS++))
        fi
    else
        echo "Worker \${i}: FAILED (log file not found)"
        ((FAILED_WORKERS++))
    fi
done

echo "=============================================="
echo "TUNING COMPLETED - EXTERNAL NORMALIZATION"
echo "Model: \${MODEL_NAME}"
echo "Successful workers: \${SUCCESSFUL_WORKERS}/\${NUM_GPUS}"
echo "=============================================="
date +"%Y-%m-%d %H:%M:%S"

# Move logs
for f in \${LOG_DIR}/slurm_logs/*_\${SLURM_JOB_ID}.out; do
    [ -e "\$f" ] && mv "\$f" "\${LOG_DIR}/slurm_logs/\${SLURM_JOB_ID}/" 2>/dev/null
done
for f in \${LOG_DIR}/slurm_logs/*_\${SLURM_JOB_ID}.err; do
    [ -e "\$f" ] && mv "\$f" "\${LOG_DIR}/slurm_logs/\${SLURM_JOB_ID}/" 2>/dev/null
done

exit \$FAILED_WORKERS
SCRIPT_BODY

# --- Print info and submit ---
echo "=============================================="
echo "FLEXIBLE TUNING JOB SUBMISSION"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "GPUs: ${NUM_GPUS}"
echo "Partition: ${PARTITION}"
echo "Time Limit: ${TIME_LIMIT}"
echo "Base Seed: ${BASE_SEED}"
echo "Restart: ${RESTART_TUNING:-no}"
echo "Node: ${SPECIFIC_NODE:-any}"
echo "----------------------------------------------"

# Submit the job
sbatch --export=ALL "$TEMP_SCRIPT"

# Cleanup temp script after a delay
(sleep 5 && rm -f "$TEMP_SCRIPT") &
