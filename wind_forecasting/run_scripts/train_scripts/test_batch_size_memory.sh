#!/bin/bash

#SBATCH --partition=all_gpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --gres=gpu:H100:1
#SBATCH --time=0:15:00
#SBATCH --job-name=test_batch_memory
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/test_batch_memory_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/test_batch_memory_%j.err

# --- Setup Environment ---
cd /user/taed7566/Forecasting/wind-forecasting/wind_forecasting || exit 1
export PYTHONPATH=/user/taed7566/Forecasting/wind-forecasting/wind_forecasting:${PYTHONPATH}

module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load Mamba/24.3.0-0
module load CUDA/12.4.0

eval "$(conda shell.bash hook)"
conda activate wf_env_storm

echo "=== BATCH SIZE MEMORY TEST ==="
echo "Testing maximum batch size for TACTiS on H100"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Test different batch sizes
CONFIG_FILE="/user/taed7566/Forecasting/wind-forecasting/config/training/training_inputs_juan_awaken_tune_storm_pred510.yaml"

for batch_size in 64 128 256 512 1024; do
    echo "=== Testing batch_size=$batch_size ==="
    
    # Run a short test with this batch size
    timeout 120 python run_scripts/run_model.py \
        --config $CONFIG_FILE \
        --model tactis \
        --mode train \
        --seed 666 \
        --override \
            trainer.max_epochs=1 \
            trainer.limit_train_batches=5 \
            trainer.val_check_interval=1.0 \
            dataset.batch_size=$batch_size \
            dataset.context_length_factor=5 \
            model.tactis.gradient_clip_val_stage1=1.0 \
            model.tactis.gradient_clip_val_stage2=1.0 \
        > /tmp/batch_test_${batch_size}.log 2>&1
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # Get memory usage
        memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
        memory_percent=$((memory_used * 100 / memory_total))
        echo "  ✓ SUCCESS: batch_size=$batch_size"
        echo "    Memory used: ${memory_used}MB / ${memory_total}MB (${memory_percent}%)"
    elif [ $exit_code -eq 124 ]; then
        echo "  ⏰ TIMEOUT: batch_size=$batch_size (took too long)"
    else
        echo "  ✗ FAILED: batch_size=$batch_size (OOM or other error)"
        # Check if it's OOM
        if grep -q "out of memory\|OutOfMemoryError\|CUDA out of memory" /tmp/batch_test_${batch_size}.log; then
            echo "    Reason: Out of Memory"
            break
        else
            echo "    Reason: Other error (check /tmp/batch_test_${batch_size}.log)"
        fi
    fi
    
    # Clear GPU memory
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 2
done

echo "=== MEMORY TEST COMPLETED ==="
echo "Recommendation based on results above:"
echo "- Use 70-80% of maximum working batch size for safety margin"
echo "- Consider that distributed training may have additional memory overhead"