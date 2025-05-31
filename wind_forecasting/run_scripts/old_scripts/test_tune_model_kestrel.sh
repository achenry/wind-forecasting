#!/bin/bash 
#SBATCH --account=ssc
#SBATCH --time=01:00:00
#SBATCH --output=%j-%x.out
#SBATCH --partition=debug
#SBATCH --nodes=1 # this needs to match Trainer(num_nodes...)
#SBATCH --ntasks-per-node=4 # this needs to match Trainer(devices=...)
#SBATCH --ntasks=4
# salloc --account=ssc --time=01:00:00 --gpus=2 --ntasks-per-node=2 --partition=debug

# --- Base Directories ---
export NUMEXPR_MAX_THREADS=128

# --- Setup Main Environment ---
echo "Setting up main environment..."
module purge
ml cuda
echo "Modules loaded."

eval "$(conda shell.bash hook)"
conda activate wind_forecasting_env
echo "Conda environment 'wind_forecasting_env' activated."
# --- End Main Environment Setup ---

echo "=== STARTING PARALLEL OPTUNA TEST TUNING WORKERS ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Parallel Worker Launch using nohup ---
NUM_CPUS=${SLURM_NTASKS_PER_NODE}
export WORLD_SIZE=${NUM_CPUS}  # Set total number of workers for tuning
declare -a WORKER_PIDS=()

echo "Launching ${NUM_CPUS} tuning workers..."

# Launch multiple workers per CPU
for i in $(seq 0 $((${NUM_CPUS}-1))); do
      # Create a unique seed for this worker
      export CURRENT_WORKER_SEED=$((12 + i*100)) # Base seed + offset per worker (increased multiplier to avoid trials overlap on workers)
      
      echo "Saving output for worker ${i} to './worker_${i}_${SLURM_JOB_ID}.log'"
      echo "Starting worker ${i} on assigned CPU ${i} with seed ${CURRENT_WORKER_SEED}"
      export WORKER_RANK=${i}          # Export rank for Python script
      # Launch worker in the background using nohup and a dedicated bash shell
      
      # Launch worker with environment settings
      # CUDA_VISIBLE_DEVICES ensures each worker sees only one CPU
      # The worker ID (SLURM_PROCID) helps Optuna identify workers
      nohup bash -c "
      echo \"Worker ${i} starting environment setup...\"
      # --- Module loading ---
      module purge
      ml cuda
      echo \"Worker ${i}: Modules loaded.\"

      # --- Activate conda environment ---
      eval \"\$(conda shell.bash hook)\"
      conda activate wind_forecasting_env
      echo \"Worker ${i}: Conda environment 'wind_forecasting_env' activated.\"

      # --- Set Worker-Specific Environment ---
      # export CUDA_VISIBLE_DEVICES=${i} # Assign specific CPU based on loop index
    
      # Note: PYTHONPATH and WANDB_DIR are inherited via export from parent script

      echo \"Worker ${i}: Running python script with WORKER_RANK=${WORKER_RANK}...\"
      
      # --- Run the tuning script ---
      # Workers connect to the already initialized study using the PG URL
      # Pass --restart_tuning flag from the main script environment
      python test_tuning.py

    # Check exit status
    status=\$?
    if [ \$status -ne 0 ]; then
        echo \"Worker ${i} FAILED with status \$status\"
    else
        echo \"Worker ${i} COMPLETED successfully\"
    fi
    exit \$status
  " > "./worker_${i}_${SLURM_JOB_ID}.log" 2>&1 &
    
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
echo "=== TEST TUNING COMPLETED ==="
