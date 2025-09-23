#!/bin/bash

#SBATCH --partition=all_gpu.p          # Partition for H100/A100 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Match number of GPUs requested below
#SBATCH --cpus-per-task=8           # CPUs per task
#SBATCH --mem-per-cpu=8192          # Memory per CPU
#SBATCH --gres=gpu:1                # Request 3 GPUs
#SBATCH --time=1-00:00              # Time limit
#SBATCH --job-name=60awaken_tune_tactis_stage1
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/awaken_tune_tactis60_stage1_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/awaken_tune_tactis60_stage1_%j.err
#SBATCH --hint=nomultithread        # Disable hyperthreading
#SBATCH --distribution=block:block  # Improve GPU-CPU affinity
#SBATCH --gres-flags=enforce-binding # Enforce binding of GPUs to tasks

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_juan_awaken_tune_storm_pred60_stage1.yaml"
export MODEL_NAME="tactis"
export RESTART_TUNING_FLAG="" # "" Or "--restart_tuning"
export AUTO_EXIT_WHEN_DONE="true"
export NUMEXPR_MAX_THREADS=128

# --- Database Credentials File ---
DB_LOGIN_FILE="/user/taed7566/Forecasting/Docs/db_login"

# --- Parse Database Credentials ---
echo "Reading database credentials from ${DB_LOGIN_FILE}..."
if [ ! -f "${DB_LOGIN_FILE}" ]; then
    echo "ERROR: Database login file not found: ${DB_LOGIN_FILE}"
    exit 1
fi

# Parse the login file
DB_NAME=$(grep "^DB=" "${DB_LOGIN_FILE}" | cut -d'=' -f2)
DB_HOST=$(grep "^FQDN=" "${DB_LOGIN_FILE}" | cut -d'=' -f2)
DB_PORT=$(grep "^DBPORT=" "${DB_LOGIN_FILE}" | cut -d'=' -f2)

# Parse user credentials (format: username:password)
USER_LINE=$(grep "^USER=" "${DB_LOGIN_FILE}" | cut -d'=' -f2)
DB_USER=$(echo "${USER_LINE}" | cut -d':' -f1)
DB_PASSWORD=$(echo "${USER_LINE}" | cut -d':' -f2)

# Export the password as environment variable for the Python scripts
export LOCAL_PG_PASSWORD="${DB_PASSWORD}"

echo "Database connection info:"
echo "  Host: ${DB_HOST}"
echo "  Port: ${DB_PORT}"
echo "  Database: ${DB_NAME}"
echo "  User: ${DB_USER}"
echo "  Password: [HIDDEN]"

# --- Create Logging Directories ---
mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

# --- Change to Working Directory ---
cd ${WORK_DIR} || exit 1

# --- Set Shared Environment Variables ---
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

# --- Print Job Info ---
echo "--- STAGE 1 TUNING: MARGINALS ONLY ---"
echo "--- User and Group Info within Slurm Job ---"
id
groups
echo "--------------------------------------------"
echo "--- SLURM JOB INFO ---"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "JOB NAME: ${SLURM_JOB_NAME}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE LIST: ${SLURM_JOB_NODELIST}"
echo "NUM NODES: ${SLURM_JOB_NUM_NODES}"
echo "NUM GPUS (Requested via ntasks): ${SLURM_NTASKS_PER_NODE}"
echo "NUM TASKS PER NODE: ${SLURM_NTASKS_PER_NODE}"
echo "CPUS PER TASK: ${SLURM_CPUS_PER_TASK}"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
echo "------------------------"
echo "BASE_DIR: ${BASE_DIR}"
echo "WORK_DIR: ${WORK_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "RESTART_TUNING_FLAG: '${RESTART_TUNING_FLAG}'"
echo "AUTO_EXIT_WHEN_DONE: '${AUTO_EXIT_WHEN_DONE}'"
echo "------------------------"
echo "STAGE 1 CONFIGURATION:"
echo "  - Training marginals only (skip_copula=true)"
echo "  - Max epochs: 25"
echo "  - Prediction length: 60s"
echo "  - Context length: 600s"
echo "------------------------"

# --- GPU Monitoring Instructions ---
echo "--- MANUAL MONITORING INSTRUCTIONS ---"
echo "To monitor GPU usage, open a NEW terminal session on the login node and run:"
echo "ssh -L 8088:localhost:8088 ${USER}@${SLURM_JOB_NODELIST}"
echo "After connecting, activate the environment and run gpustat:"
echo "mamba activate wf_env_storm"
echo "gpustat -P --no-processes --watch 0.5"
echo "------------------------------------"

# --- Setup Main Environment ---
echo "Setting up main environment..."
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
export CAPTURED_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# Find PostgreSQL binary directory after loading the module
PG_INITDB_PATH=$(which initdb)
if [[ -z "$PG_INITDB_PATH" ]]; then
  echo "FATAL: Could not find 'initdb' after loading PostgreSQL module. Check module system." >&2
  exit 1
fi
export POSTGRES_BIN_DIR=$(dirname "$PG_INITDB_PATH")
echo "Found PostgreSQL bin directory: ${POSTGRES_BIN_DIR}"
echo "Setting PostgreSQL environment variables..."
export PGPASSWORD="${LOCAL_PG_PASSWORD}"

# --- End Main Environment Setup ---

echo "=== STARTING STAGE 1 PARALLEL OPTUNA TUNING WORKERS ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Parallel Worker Launch using nohup ---
NUM_GPUS=${SLURM_NTASKS_PER_NODE}
export WORLD_SIZE=${NUM_GPUS}  # Set total number of workers for tuning
declare -a WORKER_PIDS=()

echo "Launching ${NUM_GPUS} Stage 1 tuning workers..."

for i in $(seq 0 $((${NUM_GPUS}-1))); do
    # Create a unique seed for this worker
    CURRENT_WORKER_SEED=$((12 + i*100))

    echo "Starting Stage 1 worker ${i} on assigned GPU ${i} with seed ${CURRENT_WORKER_SEED}"
    # Launch worker in the background using nohup and a dedicated bash shell
    nohup bash -c "
        echo \"Stage 1 Worker ${i} starting environment setup...\"
        # --- Module loading ---
        module purge
        module load slurm/hpc-2023/23.02.7
        module load hpc-env/13.1
        module load mpi4py/3.1.4-gompi-2023a
        module load Mamba/24.3.0-0
        module load CUDA/12.4.0
        module load git
        echo \"Stage 1 Worker ${i}: Modules loaded.\"

        # --- Activate conda environment ---
        eval \"\$(conda shell.bash hook)\"
        conda activate wf_env_storm
        echo \"Stage 1 Worker ${i}: Conda environment 'wf_env_storm' activated.\"

        # --- Set Worker-Specific Environment ---
        export CUDA_VISIBLE_DEVICES=${i} # Assign specific GPU based on loop index
        export WORKER_RANK=${i}          # Export rank for Python script - MUST be inside the subshell
        
        # Note: PYTHONPATH and WANDB_DIR are inherited via export from parent script

        echo \"Stage 1 Worker ${i}: Running python script with WORKER_RANK=\${WORKER_RANK}...\"
        # --- Run the Stage 1 tuning script ---
        python ${WORK_DIR}/run_scripts/run_model.py \\
          --config ${CONFIG_FILE} \\
          --model ${MODEL_NAME} \\
          --mode tune \\
          --tuning_phase 1 \\
          --seed ${CURRENT_WORKER_SEED} \\
          ${RESTART_TUNING_FLAG} \\
          --single_gpu # Crucial for making Lightning use only the assigned GPU

        # Check exit status
        status=\$?
        if [ \$status -ne 0 ]; then
            echo \"Stage 1 Worker ${i} FAILED with status \$status\"
        else
            echo \"Stage 1 Worker ${i} COMPLETED successfully\"
        fi
        exit \$status
    " > "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/stage1_worker_${i}_${SLURM_JOB_ID}.log" 2>&1 &

    # Store the process ID
    WORKER_PIDS+=($!)

    # Small delay between starting workers
    sleep 2
done

echo "--- Stage 1 Worker Processes Launched ---"
echo "Number of workers: ${#WORKER_PIDS[@]}"
echo "Process IDs: ${WORKER_PIDS[@]}"
echo "Main script now waiting for workers to complete..."
echo "Check worker logs in ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/stage1_worker_*.log"
echo "-------------------------------"

# --- System Monitoring ---
echo "=== SYSTEM MONITORING SETUP ==="
date +"%Y-%m-%d %H:%M:%S"

# Capture detailed system information at the start
echo "--- DETAILED NODE INFORMATION ---"
echo "Hostname: $(hostname)"
echo "Kernel: $(uname -r)"
echo "CPU Model: $(grep "model name" /proc/cpuinfo | head -1 | cut -d ":" -f2 | sed 's/^[ \t]*//')"
echo "CPU Cores: $(nproc) physical, $(grep -c processor /proc/cpuinfo) logical"
echo "Memory Total: $(free -h | grep Mem | awk '{print $2}')"
echo "Slurm Job ID: ${SLURM_JOB_ID}"
echo "Slurm Partition: ${SLURM_JOB_PARTITION}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Worker PIDs: ${WORKER_PIDS[*]}"

# GPU Details
echo "--- GPU INFORMATION ---"
nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv

# Add trap to ensure monitoring process is terminated
trap "echo '--- Stopping System Monitoring ---'; kill \$MONITOR_PID 2>/dev/null" EXIT

# Start periodic monitoring in background
(
    eval "$(conda shell.bash hook)"
    conda activate wf_env_storm    
    echo "--- Starting Periodic Resource Monitoring (every 10 minutes) ---"    
    while true; do
        echo "====== SYSTEM STATUS: $(date +"%Y-%m-%d %H:%M:%S") ======"        
        echo "CPU Load: $(cat /proc/loadavg | awk '{print $1, $2, $3}')"        
        echo "Memory (GiB): $(free -g | grep Mem | awk '{print "Total:", $2, "Used:", $3, "Free:", $4, "Cache:", $6}')"        
        echo "Disk: $(df -h / | grep -v Filesystem | awk '{print "Used:", $3, "Free:", $4, "of", $2, "("$5")"}')"        
        echo "GPU Status:"
        gpustat --no-header        
        ALIVE_WORKERS=0
        for pid in ${WORKER_PIDS[@]}; do
            if kill -0 $pid 2>/dev/null; then
                ((ALIVE_WORKERS++))
            fi
        done
        echo "Stage 1 Workers: ${ALIVE_WORKERS}/${#WORKER_PIDS[@]} still running"
        echo "------------------------------------------"        
        sleep 600
    done
) &

MONITOR_PID=$!
echo "System monitoring started (PID: ${MONITOR_PID})"
echo "=== END MONITORING SETUP ==="
echo "---------------------------------------"

# --- Wait for all background workers to complete, with auto-exit option ---
if [ "${AUTO_EXIT_WHEN_DONE}" = "true" ]; then
    echo "Auto-exit when done is enabled. Script will terminate when all workers finish."
    
    # Check worker status every 60 seconds
    while true; do
        ALIVE_WORKERS=0
        for pid in ${WORKER_PIDS[@]}; do
            if kill -0 $pid 2>/dev/null; then
                ((ALIVE_WORKERS++))
            fi
        done
        
        echo "$(date +"%Y-%m-%d %H:%M:%S") - Stage 1 Workers still running: ${ALIVE_WORKERS}/${#WORKER_PIDS[@]}"
        
        # If no workers are alive, exit the loop
        if [ $ALIVE_WORKERS -eq 0 ]; then
            echo "All Stage 1 workers have finished. Proceeding to final status check."
            break
        fi
        
        # Sleep for 60 seconds before checking again
        sleep 60
    done
else
    echo "Auto-exit when done is disabled. Script will wait until all workers finish or timeout occurs."
    wait
fi

WAIT_EXIT_CODE=$?

# --- Final Status Check based on Worker Logs ---
echo "--- Stage 1 Worker Completion Status (from logs) ---"
FAILED_WORKERS=0
SUCCESSFUL_WORKERS=0
for i in $(seq 0 $((${NUM_GPUS}-1))); do
    WORKER_LOG="${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/stage1_worker_${i}_${SLURM_JOB_ID}.log"
    if [ -f "$WORKER_LOG" ]; then
        if grep -q "COMPLETED successfully" "$WORKER_LOG"; then
            echo "Stage 1 Worker ${i}: SUCCESS (based on log)"
            ((SUCCESSFUL_WORKERS++))
        elif grep -q "FAILED with status" "$WORKER_LOG"; then
            echo "Stage 1 Worker ${i}: FAILED (based on log)"
            ((FAILED_WORKERS++))
        else
            echo "Stage 1 Worker ${i}: UNKNOWN status (log exists but completion message not found)"
            ((FAILED_WORKERS++))
        fi
    else
        echo "Stage 1 Worker ${i}: FAILED (log file not found: $WORKER_LOG)"
        ((FAILED_WORKERS++))
    fi
done
echo "------------------------------------------"

TOTAL_WORKERS=${NUM_GPUS}
if [ $FAILED_WORKERS -gt 0 ]; then
    echo "SUMMARY: ${FAILED_WORKERS} out of ${TOTAL_WORKERS} Stage 1 worker(s) reported failure. Check individual worker logs and SLURM error file."
    FINAL_EXIT_CODE=1
else
    echo "SUMMARY: All ${TOTAL_WORKERS} Stage 1 workers reported success."
    echo ""
    echo "=== STAGE 1 TUNING COMPLETE ==="
    echo "Next steps:"
    echo "1. Note the study name from the logs"
    echo "2. Update Stage 2 script with the Stage 1 study name"
    echo "3. Run: sbatch tune_model_storm_awaken_p60_stage2.sh"
    FINAL_EXIT_CODE=0
fi
echo "=== STAGE 1 TUNING SCRIPT COMPLETED ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Move main SLURM logs to the job ID directory ---
echo "Moving main SLURM logs to ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/"
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.out; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/" || echo "Warning: Could not move .out file."
done
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.err; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/" || echo "Warning: Could not move .err file."
done
echo "--------------------------------------------------"

exit $FINAL_EXIT_CODE