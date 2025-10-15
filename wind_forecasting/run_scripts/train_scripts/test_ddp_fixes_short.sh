#!/bin/bash

#SBATCH --partition=all_gpu.p          # Partition for H100/A100 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3            # 3 tasks for DDP on 3 GPUs
#SBATCH --cpus-per-task=8              # CPUs per GPU task
#SBATCH --mem-per-cpu=8192             # Memory per CPU
#SBATCH --gres=gpu:H100:3              # Request 3 H100 GPUs total
#SBATCH --time=00:30:00                # 30 minutes for quick test
#SBATCH --job-name=test_ddp_fixes
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/test_ddp_fixes_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/test_ddp_fixes_%j.err
#SBATCH --hint=nomultithread           # Disable hyperthreading
#SBATCH --distribution=block:block     # Improve GPU-CPU affinity
#SBATCH --gres-flags=enforce-binding   # Enforce binding of GPUs to tasks

# =============================================================================
# QUICK TEST SCRIPT: Validate DDP Optimization Fixes
# =============================================================================
# This script runs a SHORT test (30 min) to validate:
#   1. Data partitioning: Each worker gets unique subset of time series
#   2. Multi-GPU monitoring: WandB logs all 3 GPUs
#   3. Performance: Training speed >25 it/s (ideally >30 it/s)
#   4. Progress bar: Shows X/Y instead of X/?
# =============================================================================

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_juan_awaken_train_storm_pred60_full.yaml"
export MODEL_NAME="tactis"
export AUTO_EXIT_WHEN_DONE="true"
export NUMEXPR_MAX_THREADS=128

# --- Create Logging Directories ---
mkdir -p ${LOG_DIR}/slurm_logs
mkdir -p ${LOG_DIR}/checkpoints

# --- Change to Working Directory ---
cd ${WORK_DIR} || exit 1

# --- Set Shared Environment Variables ---
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

# --- Print Job Info ---
echo "======================================================================="
echo "QUICK TEST: DDP Optimization Fixes Validation"
echo "======================================================================="
echo "--- SLURM JOB INFO ---"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "JOB NAME: ${SLURM_JOB_NAME}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE LIST: ${SLURM_JOB_NODELIST}"
echo "NUM NODES: ${SLURM_JOB_NUM_NODES}"
echo "NUM TASKS PER NODE: ${SLURM_NTASKS_PER_NODE}"
echo "CPUS PER TASK: ${SLURM_CPUS_PER_TASK}"
echo "GPUS PER TASK: 1 (3 GPUs total)"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
echo "-----------------------------------------------------------------------"
echo "BASE_DIR: ${BASE_DIR}"
echo "WORK_DIR: ${WORK_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "-----------------------------------------------------------------------"
echo ""
echo "=== TEST CONFIGURATION ==="
echo "Duration: 30 minutes (quick validation)"
echo "Batch Size: 64"
echo "Context Length: 600s (25 steps × 15s)"
echo "Prediction Length: 60s (4 steps × 15s)"
echo "Workers per GPU: 7 (21 total)"
echo ""
echo "=== FIXES BEING TESTED ==="
echo "1. Data Partitioning: Each of 21 workers gets ~1,090 time series (not 22,880)"
echo "2. Multi-GPU Monitoring: WandB logs system/gpu.{0,1,2}.*"
echo "3. Performance: Expect >25 it/s (ideally >30 it/s)"
echo "4. Progress Bar: Should show X/618396 instead of X/?"
echo ""
echo "=== MONITORING CHECKLIST ==="
echo "After job starts, check logs for:"
echo "  - 'Worker X/21 assigned Y/22880 time series'"
echo "  - 'Added MultiGPUMonitor callback'"
echo "  - Training speed (it/s) in progress bar"
echo "  - WandB: https://wandb.ai/jmb0507-cu-boulder/"
echo "======================================================================="
echo ""

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

echo "SLURM-native DDP: Each of 3 tasks will see 1 GPU (--gpu-bind=map_gpu:0,1,2)"
echo "Using srun to launch coordinated processes for PyTorch Lightning DDP"

echo ""
echo "=== STARTING QUICK TEST ==="
date +"%Y-%m-%d %H:%M:%S"
echo ""

# Set PyTorch distributed environment variables for DDP
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=29500

# Launch training with srun
srun --export=ALL --gpu-bind=map_gpu:0,1,2 bash -c '
  export RANK=${SLURM_PROCID}
  export LOCAL_RANK=${SLURM_LOCALID}
  export WORLD_SIZE=${SLURM_NTASKS}
  echo "Task ${SLURM_PROCID}: RANK=${RANK}, LOCAL_RANK=${LOCAL_RANK}, WORLD_SIZE=${WORLD_SIZE}"
  python ${WORK_DIR}/run_scripts/run_model.py \
    --config ${CONFIG_FILE} \
    --model ${MODEL_NAME} \
    --mode train \
    --seed 666
'

TEST_EXIT_CODE=$?

echo ""
echo "=== TEST FINISHED WITH EXIT CODE: ${TEST_EXIT_CODE} ==="
date +"%Y-%m-%d %H:%M:%S"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "                    TEST COMPLETED SUCCESSFULLY"
    echo "======================================================================="
    echo ""
    echo "Next steps:"
    echo "  1. Check logs for data partitioning messages:"
    echo "     grep 'Worker.*assigned.*time series' ${LOG_DIR}/slurm_logs/test_ddp_fixes_${SLURM_JOB_ID}.err"
    echo ""
    echo "  2. Verify all workers got unique series (sum should be 22,880):"
    echo "     grep -oP 'assigned \K\d+(?=/22880)' ${LOG_DIR}/slurm_logs/test_ddp_fixes_${SLURM_JOB_ID}.err | awk '{s+=\$1} END {print s}'"
    echo ""
    echo "  3. Check training speed (should be >25 it/s):"
    echo "     grep -oP '\K\d+\.\d+(?=it/s)' ${LOG_DIR}/slurm_logs/test_ddp_fixes_${SLURM_JOB_ID}.out | tail -5"
    echo ""
    echo "  4. Verify multi-GPU monitoring in WandB:"
    echo "     Look for system/gpu.0.*, system/gpu.1.*, system/gpu.2.* metrics"
    echo ""
    echo "  5. If all checks pass, submit full 7-day training job:"
    echo "     sbatch train_awaken_storm_60_full_sequential.sh"
    echo "======================================================================="
else
    echo ""
    echo "======================================================================="
    echo "                      TEST FAILED"
    echo "======================================================================="
    echo "Exit code: ${TEST_EXIT_CODE}"
    echo "Check logs for errors:"
    echo "  ${LOG_DIR}/slurm_logs/test_ddp_fixes_${SLURM_JOB_ID}.err"
    echo "  ${LOG_DIR}/slurm_logs/test_ddp_fixes_${SLURM_JOB_ID}.out"
    echo "======================================================================="
fi

exit $TEST_EXIT_CODE
