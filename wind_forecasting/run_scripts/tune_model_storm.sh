#!/bin/bash

#SBATCH --partition=all_gpu.p       # Partition for H100/A100 GPUs cfdg.p / all_gpu.p / mpcg.p(not allowed)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # Match number of GPUs requested below
#SBATCH --cpus-per-task=32          # CPUs per task (4 tasks * 32 = 128 CPUs total)
#SBATCH --mem-per-cpu=8016          # Memory per CPU (Total Mem = ntasks * cpus-per-task * mem-per-cpu)
#SBATCH --gres=gpu:4           # Request 4 H100 GPUs
#SBATCH --time=1-00:00              # Time limit (1 day)
#SBATCH --job-name=autoformer_tune_flasc
#SBATCH --output=/user/taed7566/wind-forecasting/logging/slurm_logs/autoformer_tune_flasc_%j.out
#SBATCH --error=/user/taed7566/wind-forecasting/logging/slurm_logs/autoformer_tune_flasc_%j.err
#SBATCH --hint=nomultithread        # Disable hyperthreading
#SBATCH --distribution=block:block  # Improve GPU-CPU affinity
#SBATCH --gres-flags=enforce-binding # Enforce binding of GPUs to tasks

# --- Configuration ---
# Set this to "--restart_tuning" to clear and restart the Optuna study, otherwise set to "" to continue previous one
RESTART_TUNING_FLAG="--restart_tuning" # "" Or "--restart_tuning"

# --- Base Directories ---
BASE_DIR="/user/taed7566/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${BASE_DIR}/logging"
export CONFIG_FILE="${BASE_DIR}/examples/inputs/training_inputs_juan_flasc.yaml"
export MODEL_NAME="autoformer"

# --- Create Logging Directories ---
# Ensure the main slurm_logs directory exists
mkdir -p ${LOG_DIR}/slurm_logs
# Create the job-specific directory for worker logs and final main logs
mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/optuna
mkdir -p ${LOG_DIR}/checkpoints

# --- Change to Working Directory ---
cd ${WORK_DIR} || exit 1 # Exit if cd fails

# --- Set Shared Environment Variables ---
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR} # WandB will create a 'wandb' subdirectory here automatically

# --- Print Job Info ---
echo "--- SLURM JOB INFO ---"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "JOB NAME: ${SLURM_JOB_NAME}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE LIST: ${SLURM_JOB_NODELIST}"
echo "NUM NODES: ${SLURM_JOB_NUM_NODES}"
echo "NUM GPUS (Requested via ntasks): ${SLURM_NTASKS_PER_NODE}"
echo "NUM TASKS PER NODE: ${SLURM_NTASKS_PER_NODE}"
echo "CPUS PER TASK: ${SLURM_CPUS_PER_TASK}"
echo "------------------------"
echo "BASE_DIR: ${BASE_DIR}"
echo "WORK_DIR: ${WORK_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "RESTART_TUNING_FLAG: '${RESTART_TUNING_FLAG}'"
echo "------------------------"

# --- Setup Main Environment ---
echo "Setting up main environment..."
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load PostgreSQL/16.1-GCCcore-13.1.0
module load Mamba/24.3.0-0
module load CUDA/12.4.0
echo "Modules loaded."

# Capture LD_LIBRARY_PATH after modules are loaded
export CAPTURED_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo "Captured LD_LIBRARY_PATH: ${CAPTURED_LD_LIBRARY_PATH}"

# Find PostgreSQL binary directory after loading the module
PG_INITDB_PATH=$(which initdb)
if [[ -z "$PG_INITDB_PATH" ]]; then
  echo "FATAL: Could not find 'initdb' after loading PostgreSQL module. Check module system." >&2
  exit 1
fi
# Extract the directory path (e.g., /path/to/postgres/bin)
export POSTGRES_BIN_DIR=$(dirname "$PG_INITDB_PATH")
echo "Found PostgreSQL bin directory: ${POSTGRES_BIN_DIR}"

eval "$(conda shell.bash hook)"
conda activate wf_env_2
echo "Conda environment 'wf_env_2' activated."
# --- End Main Environment Setup ---

echo "=== STARTING PARALLEL OPTUNA TUNING WORKERS ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Parallel Worker Launch using nohup ---
NUM_GPUS=${SLURM_NTASKS_PER_NODE} # Should match --gres=gpu:N and --ntasks-per-node
declare -a WORKER_PIDS=()

echo "Launching ${NUM_GPUS} tuning workers..."

for i in $(seq 0 $((${NUM_GPUS}-1))); do
    # Create a unique seed for this worker
    CURRENT_WORKER_SEED=$((12 + i*100)) # Base seed + offset per worker (increased multiplier to avoid trials overlap on workers)

    echo "Starting worker ${i} on assigned GPU ${i} with seed ${CURRENT_WORKER_SEED}"

    # Launch worker in the background using nohup and a dedicated bash shell
    nohup bash -c "
        echo \"Worker ${i} starting environment setup...\"
        # --- Module loading ---
        module purge
        module load slurm/hpc-2023/23.02.7
        module load hpc-env/13.1
        module load Mamba/24.3.0-0
        module load CUDA/12.4.0
        echo \"Worker ${i}: Modules loaded.\"

        # --- Activate conda environment ---
        eval \"\$(conda shell.bash hook)\"
        conda activate wf_env_2
        echo \"Worker ${i}: Conda environment 'wf_env_2' activated.\"

        # --- Set Worker-Specific Environment ---
        export CUDA_VISIBLE_DEVICES=${i} # Assign specific GPU based on loop index
        export WORKER_RANK=${i}          # Export rank for Python script
        # Note: PYTHONPATH and WANDB_DIR are inherited via export from parent script

        echo \"Worker ${i}: Running python script with WORKER_RANK=${WORKER_RANK}...\"
        # --- Run the tuning script ---
        # Workers connect to the already initialized study using the PG URL
        # Pass --restart_tuning flag from the main script environment
        python ${WORK_DIR}/run_scripts/run_model.py \\
          --config ${CONFIG_FILE} \\
          --model ${MODEL_NAME} \\
          --mode tune \\
          --seed ${CURRENT_WORKER_SEED} \\
          ${RESTART_TUNING_FLAG} \\
          --single_gpu # Crucial for making Lightning use only the assigned GPU

        # Check exit status
        status=\$?
        if [ \$status -ne 0 ]; then
            echo \"Worker ${i} FAILED with status \$status\"
        else
            echo \"Worker ${i} COMPLETED successfully\"
        fi
        exit \$status
    " > "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/worker_${i}_${SLURM_JOB_ID}.log" 2>&1 &

    # Store the process ID
    WORKER_PIDS+=($!)

    # Small delay between starting workers
    sleep 2
done

echo "--- Worker Processes Launched ---"
echo "Number of workers: ${#WORKER_PIDS[@]}"
echo "Process IDs: ${WORKER_PIDS[@]}"
echo "Main script now waiting for workers to complete..."
echo "Check worker logs in ${LOG_DIR}/slurm_logs/worker_*.log"
echo "-------------------------------"

# --- Wait for all background workers to complete ---
wait
WAIT_EXIT_CODE=$? # Capture the exit code of the initial 'wait' command

# --- Final Status Check based on Worker Logs ---
echo "--- Worker Completion Status (from logs) ---"
FAILED_WORKERS=0
SUCCESSFUL_WORKERS=0
for i in $(seq 0 $((${NUM_GPUS}-1))); do
    # Correct path to include the job ID subdirectory
    WORKER_LOG="${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/worker_${i}_${SLURM_JOB_ID}.log"
    if [ -f "$WORKER_LOG" ]; then
        # Check for success message (adjust pattern if needed)
        if grep -q "COMPLETED successfully" "$WORKER_LOG"; then
            echo "Worker ${i}: SUCCESS (based on log)"
            ((SUCCESSFUL_WORKERS++))
        # Check for failure message (adjust pattern if needed)
        elif grep -q "FAILED with status" "$WORKER_LOG"; then
            echo "Worker ${i}: FAILED (based on log)"
            ((FAILED_WORKERS++))
        else
            echo "Worker ${i}: UNKNOWN status (log exists but completion message not found)"
            ((FAILED_WORKERS++)) # Treat unknown as failure
        fi
    else
        echo "Worker ${i}: FAILED (log file not found: $WORKER_LOG)"
        ((FAILED_WORKERS++))
    fi
done
echo "------------------------------------------"

TOTAL_WORKERS=${NUM_GPUS}
if [ $FAILED_WORKERS -gt 0 ]; then
    echo "SUMMARY: ${FAILED_WORKERS} out of ${TOTAL_WORKERS} worker(s) reported failure. Check individual worker logs and SLURM error file."
    FINAL_EXIT_CODE=1 # Force non-zero exit code for the SLURM job
else
    echo "SUMMARY: All ${TOTAL_WORKERS} workers reported success."
    FINAL_EXIT_CODE=0 # Use 0 if all logs indicate success
fi

echo "=== TUNING SCRIPT COMPLETED ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Move main SLURM logs to the job ID directory with rest of worker logs ---
echo "Moving main SLURM logs to ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/"
mv ${LOG_DIR}/slurm_logs/informer_tune_flasc_${SLURM_JOB_ID}.out ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/ 2>/dev/null || echo "Warning: Could not move .out file."
mv ${LOG_DIR}/slurm_logs/informer_tune_flasc_${SLURM_JOB_ID}.err ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/ 2>/dev/null || echo "Warning: Could not move .err file."
echo "--------------------------------------------------"

exit $FINAL_EXIT_CODE
