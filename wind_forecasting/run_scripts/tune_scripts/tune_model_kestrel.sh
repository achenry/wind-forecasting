#!/bin/bash 
#SBATCH --account=ssc
#SBATCH --time=01:00:00
#SBATCH --output=%j-%x.out
##SBATCH --partition=debug
#SBATCH --nodes=1 # this needs to match Trainer(num_nodes...)
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2 # this needs to match Trainer(devices=...)
#SBATCH --mem-per-cpu=85G

##SBATCH --mem=0 # refers to CPU (not GPU) memory, automatically given all GPU memory in a SLURM job, 85G
#SBATCH --ntasks=1

# salloc --account=ssc --time=01:00:00 --gpus=2 --ntasks-per-node=2 --partition=debug

# --- Command Line Args ---
export MODEL_NAME=$1
export CONFIG_FILE=$2

# --- Base Directories ---
export TUNING_PHASE=1
export BASE_DIR="/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting"
export LOG_DIR="${BASE_DIR}/logs"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export RESTART_TUNING_FLAG="--restart_tuning" # "" Or "--restart_tuning"
export AUTO_EXIT_WHEN_DONE="true"  # Set to "true" to exit script when all workers finish, "false" to keep running until timeout
export NUMEXPR_MAX_THREADS=128

# --- Create Logging Directories ---
# Create the job-specific directory for worker logs and final main logs
mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

# --- Change to Working Directory ---
# cd ${WORK_DIR} || exit 1 # Exit if cd fails

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
echo "AUTO_EXIT_WHEN_DONE: '${AUTO_EXIT_WHEN_DONE}'"
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
ml mamba
ml cuda
echo "Modules loaded."

eval "$(mamba shell.bash hook)"
mamba activate wind_forecasting_env
echo "Conda environment 'wind_forecasting_env' activated."
# --- End Main Environment Setup ---

export API_FILE="../.wandb_api_key"
if [[ -f "${API_FILE}" ]]; then   
  echo "WANDB API file exists";
  source "${API_FILE}"
else
  echo "ERROR: WANDB API‑key file not found at ${API_FILE}" >&2
  exit 1
fi

echo "=== STARTING PARALLEL OPTUNA TUNING WORKERS ==="
date +"%Y-%m-%d %H:%M:%S"


# --- Parallel Worker Launch using nohup ---
NUM_GPUS=${SLURM_NTASKS_PER_NODE}
export WORLD_SIZE=${NUM_GPUS}  # Set total number of workers for tuning
declare -a WORKER_PIDS=()

echo "Launching ${NUM_GPUS} tuning workers..."

# Launch multiple workers per GPU
for i in $(seq 0 $((${NUM_GPUS}-1))); do
      # Create a unique seed for this worker
      export CURRENT_WORKER_SEED=$((12 + i*100)) # Base seed + offset per worker (increased multiplier to avoid trials overlap on workers)
      
      echo "Starting worker ${i} on assigned GPU ${i} with seed ${CURRENT_WORKER_SEED}"
      export WORKER_RANK=${i}          # Export rank for Python script
      # Launch worker in the background using nohup and a dedicated bash shell
      
      # Launch worker with environment settings
      # CUDA_VISIBLE_DEVICES ensures each worker sees only one GPU
      # The worker ID (SLURM_PROCID) helps Optuna identify workers
      #srun --exclusive -n 1 --export=ALL,CUDA_VISIBLE_DEVICES=$i,SLURM_PROCID=${WORKER_INDEX},WANDB_DIR=${WANDB_DIR} \
      nohup bash -c "
      echo \"Worker ${i} starting environment setup...\"
      # --- Module loading ---
      module purge
      # ml PrgEnv-intel
      ml mamba
      ml cuda
      echo \"Worker ${i}: Modules loaded.\"

      # --- Activate conda environment ---
      # eval \"\$(conda shell.bash hook)\"
      mamba activate wind_forecasting_env
      echo \"Worker ${i}: Conda environment 'wind_forecasting_env' activated.\"

      # --- Set Worker-Specific Environment ---
      export CUDA_VISIBLE_DEVICES=${i} # Assign specific GPU based on loop index
      
      # Note: PYTHONPATH and WANDB_DIR are inherited via export from parent script

      echo \"Worker ${i}: Running python script with WORKER_RANK=${WORKER_RANK}...\"
      # --- Run the tuning script ---
      # Workers connect to the already initialized study using the PG URL
      # Pass --restart_tuning flag from the main script environment
      python ${WORK_DIR}/run_scripts/run_model.py \
        --config ${CONFIG_FILE} \
        --model ${MODEL_NAME} \
        --mode tune \
        --seed ${CURRENT_WORKER_SEED} \
        ${RESTART_TUNING_FLAG} \
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
      
      # Add a small delay between starting workers on the same GPU
      # to avoid initialization conflicts
      sleep 2
done

echo "Started ${#WORKER_PIDS[@]} worker processes"
echo "Process IDs: ${WORKER_PIDS[@]}"

# Wait for all workers to complete
wait

date +"%Y-%m-%d %H:%M:%S"
echo "=== TUNING COMPLETED ==="

# srun python informer.py
#python train_spacetimeformer.py spacetimeformer windfarm --debug --run_name spacetimeformer_windfarm_debug --context_points 600 --target_points 600

