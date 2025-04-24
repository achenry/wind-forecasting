#!/bin/bash

#SBATCH --partition=all_gpu.p          # Partition for H100/A100 GPUs cfdg.p / all_gpu.p / mpcg.p(not allowed)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # Match number of GPUs requested below
#SBATCH --cpus-per-task=1           # CPUs per task (4 tasks * 32 = 128 CPUs total) [1 CPU/GPU more than enough]
#SBATCH --mem-per-cpu=4096          # Memory per CPU (Total Mem = ntasks * cpus-per-task * mem-per-cpu) [flasc uses only ~4-5 GiB max]
#SBATCH --gres=gpu:4:H100           # Request 4 H100 GPUs
#SBATCH --time=0-08:00              # Time limit (up to 7 days)
#SBATCH --job-name=tactis_tune_flasc_sql
#SBATCH --output=/user/taed7566/Forecasting/wind-forecasting/logs/slurm_logs/tactis_tune_flasc_sql_%j.out
#SBATCH --error=/user/taed7566/Forecasting/wind-forecasting/logs/slurm_logs/tactis_tune_flasc_sql_%j.err
#SBATCH --hint=nomultithread        # Disable hyperthreading
#SBATCH --distribution=block:block  # Improve GPU-CPU affinity
#SBATCH --gres-flags=enforce-binding # Enforce binding of GPUs to tasks

# For UOL HPC, when using all_gpu.p partition, has a time limit of 1 day (1-00:00)
# While cfdg.p [1-2] partition has a time limit of 21 days (21-00:00). Use cfdg002 for x4 H100 GPUs
# mpcg.p [1-6] is allowed for 7 days (7-00:00) and up to 21 days with '--qos=long_mpcg.q', but might be restricted for my group

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting/wind-forecasting" # Absolute path to the base directory
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${BASE_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_juan_flasc.yaml"
export MODEL_NAME="tactis"
export RESTART_TUNING_FLAG="--restart_tuning" # "" Or "--restart_tuning"

# --- Create Logging Directories ---
# Create the job-specific directory for worker logs and final main logs
mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
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
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
echo "------------------------"
echo "BASE_DIR: ${BASE_DIR}"
echo "WORK_DIR: ${WORK_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "RESTART_TUNING_FLAG: '${RESTART_TUNING_FLAG}'"
echo "------------------------"

# --- GPU  Monitoring Instructions ---
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
module load mpi4py/3.1.4-gompi-2023a
module load Mamba/24.3.0-0
module load CUDA/12.4.0
module load git
echo "Modules loaded."

# Capture LD_LIBRARY_PATH after modules are loaded
export CAPTURED_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
# echo "Captured LD_LIBRARY_PATH: ${CAPTURED_LD_LIBRARY_PATH}"

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
conda activate wf_env_storm
echo "Conda environment 'wf_env_storm' activated."
# --- End Main Environment Setup ---

echo "=== STARTING PARALLEL OPTUNA TUNING WORKERS ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Parallel Worker Launch using nohup ---
NUM_GPUS=${SLURM_NTASKS_PER_NODE}
export WORLD_SIZE=${NUM_GPUS}  # Set total number of workers for tuning
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
        module load mpi4py/3.1.4-gompi-2023a
        module load Mamba/24.3.0-0
        module load CUDA/12.4.0
        module load git
        echo \"Worker ${i}: Modules loaded.\"

        # --- Activate conda environment ---
        eval \"\$(conda shell.bash hook)\"
        conda activate wf_env_storm
        echo \"Worker ${i}: Conda environment 'wf_env_storm' activated.\"

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
    # Setup environment for monitoring commands
    eval "$(conda shell.bash hook)"
    conda activate wf_env_storm
    
    echo "--- Starting Periodic Resource Monitoring (every 10 minutes) ---"
    
    while true; do
        # Current timestamp
        echo "====== SYSTEM STATUS: $(date +"%Y-%m-%d %H:%M:%S") ======"
        
        # CPU load (1, 5, 15 min averages)
        echo "CPU Load: $(cat /proc/loadavg | awk '{print $1, $2, $3}')"
        
        # Memory usage summary
        echo "Memory (GiB): $(free -g | grep Mem | awk '{print "Total:", $2, "Used:", $3, "Free:", $4, "Cache:", $6}')"
        
        # Disk usage (root partition)
        echo "Disk: $(df -h / | grep -v Filesystem | awk '{print "Used:", $3, "Free:", $4, "of", $2, "("$5")"}')"
        
        # GPU usage - compact format
        echo "GPU Status:"
        gpustat --no-header
        
        # Worker process check (confirm they're still running)
        ALIVE_WORKERS=0
        for pid in ${WORKER_PIDS[@]}; do
            if kill -0 $pid 2>/dev/null; then
                ((ALIVE_WORKERS++))
            fi
        done
        echo "Workers: ${ALIVE_WORKERS}/${#WORKER_PIDS[@]} still running"
        echo "------------------------------------------"
        
        # Sleep for 10 minutes before next check
        sleep 600
    done
) &

MONITOR_PID=$!
echo "System monitoring started (PID: ${MONITOR_PID})"
echo "=== END MONITORING SETUP ==="
echo "---------------------------------------"

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
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.out; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/" || echo "Warning: Could not move .out file."
done
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.err; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/" || echo "Warning: Could not move .err file."
done
echo "--------------------------------------------------"

exit $FINAL_EXIT_CODE

# sbatch wind-forecasting/wind_forecasting/run_scripts/tune_scripts/tune_model_storm.sh
# sacct --node=cfdg002 --state=RUNNING --allusers --format=JobID,JobName,User,State,NodeList,AllocCPUS,AllocTRES%45,ReqCPUS,ReqMem%15,ReqTRES%45,TRESUsageInAve,TRESUsageInMax
# squeue -p cfdg.p,mpcg.p,all_gpu.p -o "%.10a %.10P %.25j %.8u %.2t %.10M %.6D %R"
# squeue --node=cfdg002
# scontrol show node cfdg002
# ssh -L 8088:localhost:8088 taed7566@cfdg002
# mamba activate wf_env_storm
# gpustat -P --no-processes --watch 0.5