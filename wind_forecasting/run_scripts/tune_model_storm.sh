#!/bin/bash
#SBATCH --partition=all_gpu.p         # Partition for H100/A100 GPUs cfdg.p / all_gpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2         # Match number of GPUs requested below
#SBATCH --cpus-per-task=32          # CPUs per task (2 tasks * 32 = 64 CPUs total)
#SBATCH --mem-per-cpu=8016          # Memory per CPU (Total Mem = ntasks * cpus-per-task * mem-per-cpu)
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH --time=1-00:00                # Time limit (1 day)
#SBATCH --job-name=informer_tune_flasc
#SBATCH --output=/user/taed7566/wind-forecasting/logging/slurm_logs/informer_tune_flasc_%j.out
#SBATCH --error=/user/taed7566/wind-forecasting/logging/slurm_logs/informer_tune_flasc_%j.err
#SBATCH --hint=nomultithread        # Disable hyperthreading
#SBATCH --distribution=block:block  # Improve GPU-CPU affinity
#SBATCH --gres-flags=enforce-binding # Enforce binding of GPUs to tasks

# --- Configuration ---
# Set this to "--restart_tuning" to clear and restart the Optuna study, otherwise set to ""
RESTART_TUNING_FLAG="--restart_tuning" # Or "--restart_tuning"

# --- Base Directories ---
BASE_DIR="/user/taed7566/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${BASE_DIR}/logging"
export CONFIG_FILE="${BASE_DIR}/examples/inputs/training_inputs_juan_flasc.yaml"
export MODEL_NAME="informer"

# --- Create Logging Directories ---
mkdir -p ${LOG_DIR}/slurm_logs
mkdir -p ${LOG_DIR}/wandb
mkdir -p ${LOG_DIR}/optuna
mkdir -p ${LOG_DIR}/checkpoints

# --- Change to Working Directory ---
cd ${WORK_DIR} || exit 1 # Exit if cd fails

# --- Set Shared Environment Variables ---
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}/wandb # Ensure WandB logs to the correct shared directory

# --- Print Job Info ---
echo "--- SLURM JOB INFO ---"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "JOB NAME: ${SLURM_JOB_NAME}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE LIST: ${SLURM_JOB_NODELIST}"
echo "NUM NODES: ${SLURM_JOB_NUM_NODES}"
echo "NUM GPUS (Requested): $(echo ${SLURM_JOB_GRES} | grep -o 'gpu:[0-9]*' | cut -d: -f2)"
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

# --- Initialize Optuna Study Database ---
echo "=== INITIALIZING/CHECKING OPTUNA STUDY DATABASE ==="
date +"%Y-%m-%d %H:%M:%S"
# Run initialization in a subshell to load environment without affecting the main script
(
    echo "Setting up environment for initialization..."
    # --- Module loading ---
    module purge
    module load slurm/hpc-2023/23.02.7
    module load hpc-env/13.1
    module load Mamba/24.3.0-0
    module load CUDA/12.4.0 # Load CUDA even if not using GPU for init
    echo "Modules loaded for initialization."

    # --- Activate conda environment ---
    eval "$(conda shell.bash hook)"
    conda activate wf_env_2
    echo "Conda environment 'wf_env_2' activated for initialization."

    # --- Run initialization script ---
    # Use --init_only flag in run_model.py
    # Pass the restart flag if set
    echo "Running database initialization..."
    python ${WORK_DIR}/run_scripts/run_model.py \
      --config ${CONFIG_FILE} \
      --model ${MODEL_NAME} \
      --mode tune \
      --seed 12 \
      ${RESTART_TUNING_FLAG} \
      --init_only

    INIT_STATUS=$?
    if [ $INIT_STATUS -ne 0 ]; then
        echo "DATABASE INITIALIZATION FAILED with status $INIT_STATUS"
        exit $INIT_STATUS # Exit the subshell with error
    else
        echo "Database initialization successful."
    fi
)

# Check if the initialization subshell failed
if [ $? -ne 0 ]; then
    echo "Exiting script due to database initialization failure."
    exit 1
fi
echo "================================================"


echo "=== STARTING PARALLEL OPTUNA TUNING WORKERS ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Parallel Worker Launch using nohup ---
NUM_GPUS=${SLURM_NTASKS_PER_NODE} # Should match --gres=gpu:N and --ntasks-per-node
declare -a WORKER_PIDS=()

echo "Launching ${NUM_GPUS} tuning workers..."

for i in $(seq 0 $((${NUM_GPUS}-1))); do
    # Create a unique seed for this worker
    CURRENT_WORKER_SEED=$((12 + i*10)) # Base seed + offset per worker

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
        # Note: PYTHONPATH and WANDB_DIR are inherited via export from parent script

        echo \"Worker ${i}: Running python script...\"
        # --- Run the tuning script ---
        # Workers connect to the already initialized study
        # Do NOT pass --restart_tuning here
        python ${WORK_DIR}/run_scripts/run_model.py \\
          --config ${CONFIG_FILE} \\
          --model ${MODEL_NAME} \\
          --mode tune \\
          --seed ${CURRENT_WORKER_SEED} \\
          --single_gpu # Crucial for making Lightning use only the assigned GPU

        # Check exit status
        status=\$?
        if [ \$status -ne 0 ]; then
            echo \"Worker ${i} FAILED with status \$status\"
        else
            echo \"Worker ${i} COMPLETED successfully\"
        fi
        exit \$status
    " > "${LOG_DIR}/slurm_logs/worker_${i}_${SLURM_JOB_ID}.log" 2>&1 &

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
WAIT_EXIT_CODE=$? # Capture the exit code of the 'wait' command itself

# --- Final Status Check ---
echo "--- Worker Completion Status ---"
FAILED_WORKERS=0
for pid in "${WORKER_PIDS[@]}"; do
    wait $pid
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "Worker PID $pid exited with status $STATUS (FAILED)"
        ((FAILED_WORKERS++))
    else
        echo "Worker PID $pid exited with status $STATUS (SUCCESS)"
    fi
done
echo "--------------------------------"

if [ $FAILED_WORKERS -gt 0 ]; then
    echo "${FAILED_WORKERS} workers failed. Check individual worker logs and SLURM error file."
    # Optionally force a non-zero exit code for the SLURM job
    # exit 1
else
    echo "All workers completed successfully."
fi

echo "=== TUNING SCRIPT COMPLETED ==="
date +"%Y-%m-%d %H:%M:%S"

# Exit with the code from the initial 'wait'. If any worker failed before 'wait'
# was reached, WAIT_EXIT_CODE might be 0, but FAILED_WORKERS will be > 0.
# Consider exiting with 1 if FAILED_WORKERS > 0.
if [ $FAILED_WORKERS -gt 0 ]; then
    exit 1
else
    exit $WAIT_EXIT_CODE
fi
