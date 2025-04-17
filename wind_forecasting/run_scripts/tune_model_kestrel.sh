#!/bin/bash 
#SBATCH --account=ssc
#SBATCH --time=01:00:00
#SBATCH --output=%j-%x.out
#SBATCH --partition=debug
#SBATCH --nodes=1 # this needs to match Trainer(num_nodes...)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1 # this needs to match Trainer(devices=...)
#SBATCH --mem-per-cpu=85G

##SBATCH --mem=0 # refers to CPU (not GPU) memory, automatically given all GPU memory in a SLURM job, 85G
##SBATCH --ntasks=1

# salloc --account=ssc --time=01:00:00 --gpus=2 --ntasks-per-node=2 --partition=debug

module purge
ml PrgEnv-intel
ml mamba
ml cuda

mamba activate wind_forecasting

export BASE_DIR="/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="/projects/ssc/ahenry/wind_forecasting/logging"

# Set paths
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}/wandb
export WANDB_API_KEY=a9aec8e98a88077de29031385225167c720030f7

# Print environment info
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_JOB_GRES=${SLURM_JOB_GRES}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"

echo "=== ENVIRONMENT ==="
module list
echo "=== STARTING TUNING ==="
date +"%Y-%m-%d %H:%M:%S"

# Configure how many workers to run per GPU
NUM_WORKERS_PER_GPU=1

# Used to track process IDs for all workers
declare -a WORKER_PIDS=()

# Total number of GPUs available
NUM_GPUS=${SLURM_NTASKS_PER_NODE}

# Launch multiple workers per GPU
for i in $(seq 0 $((${NUM_GPUS}-1))); do
    for j in $(seq 0 $((${NUM_WORKERS_PER_GPU}-1))); do
        # The restart flag should only be set for the very first worker (i=0, j=0)
        if [ $i -eq 0 ] && [ $j -eq 0 ]; then
            export RESTART_FLAG="--restart_tuning"
        else
            export RESTART_FLAG=""
        fi
        
        # Create a unique seed for each worker to ensure they explore different areas
        export WORKER_SEED=$((42 + i*10 + j))
        
        # Calculate worker index for logging
        export WORKER_INDEX=$((i*NUM_WORKERS_PER_GPU + j))
        
        echo "Starting worker ${WORKER_INDEX} on GPU ${i} with seed ${WORKER_SEED}"
        
        # Launch worker with environment settings
        # CUDA_VISIBLE_DEVICES ensures each worker sees only one GPU
        # The worker ID (SLURM_PROCID) helps Optuna identify workers
        #srun --exclusive -n 1 --export=ALL,CUDA_VISIBLE_DEVICES=$i,SLURM_PROCID=${WORKER_INDEX},WANDB_DIR=${WANDB_DIR} \
        nohup bash -c "
        module purge
        ml PrgEnv-intel
        ml mamba
        ml cuda
        mamba activate wind_forecasting
        export SLURM_NTASKS_PER_NODE=1
        export SLURM_NNODES=1
        export CUDA_VISIBLE_DEVICES=$i
        python ${WORK_DIR}/run_scripts/run_model.py --config ${BASE_DIR}/examples/inputs/training_inputs_kestrel_awaken.yaml --model $1 --mode tune --seed ${WORKER_SEED} ${RESTART_FLAG}" &
        
        # Store the process ID
        WORKER_PIDS+=($!)
        
        # Add a small delay between starting workers on the same GPU
        # to avoid initialization conflicts
        sleep 2
    done
done

echo "Started ${#WORKER_PIDS[@]} worker processes"
echo "Process IDs: ${WORKER_PIDS[@]}"

# Wait for all workers to complete
wait

date +"%Y-%m-%d %H:%M:%S"
echo "=== TUNING COMPLETED ==="

# srun python informer.py
#python train_spacetimeformer.py spacetimeformer windfarm --debug --run_name spacetimeformer_windfarm_debug --context_points 600 --target_points 600

