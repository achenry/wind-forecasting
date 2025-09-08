#!/bin/bash

# Stage 1 Monitoring Script
JOB_ID=11616516
LOG_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs"

echo "============================================"
echo "Stage 1 TACTiS Tuning Monitor"
echo "Job ID: $JOB_ID"
echo "============================================"
echo ""

# Check job status
echo "Current Job Status:"
squeue -j $JOB_ID -o "%.10i %.10P %.25j %.8u %.2t %.10M %.10l %.6D %R"
echo ""

# Check if job has started
if squeue -j $JOB_ID | grep -q " R "; then
    echo "Job is RUNNING!"
    echo ""
    
    # Get node name
    NODE=$(squeue -j $JOB_ID -h -o "%N")
    echo "Running on node: $NODE"
    echo ""
    
    # Check if main log files exist
    echo "Main SLURM logs:"
    if [ -f "${LOG_DIR}/awaken_tune_tactis60_stage1_${JOB_ID}.out" ]; then
        echo "  Output log exists"
        echo "  Last 5 lines:"
        tail -5 "${LOG_DIR}/awaken_tune_tactis60_stage1_${JOB_ID}.out" | sed 's/^/    /'
    fi
    echo ""
    
    # Check worker logs
    echo "Worker logs:"
    for i in 0 1 2; do
        WORKER_LOG="${LOG_DIR}/${JOB_ID}/stage1_worker_${i}_${JOB_ID}.log"
        if [ -f "$WORKER_LOG" ]; then
            echo "  Worker $i: $(tail -1 "$WORKER_LOG")"
        else
            echo "  Worker $i: Log not yet created"
        fi
    done
    echo ""
    
    echo "To monitor GPU usage on $NODE:"
    echo "  ssh -L 8088:localhost:8088 ${USER}@${NODE}"
    echo "  Then: mamba activate wf_env_storm && gpustat -P --no-processes --watch 0.5"
    echo ""
    echo "To tail worker logs:"
    echo "  tail -f ${LOG_DIR}/${JOB_ID}/stage1_worker_*_${JOB_ID}.log"
    echo ""
    echo "To find study name (after workers start):"
    echo "  grep 'Study name:' ${LOG_DIR}/${JOB_ID}/stage1_worker_0_${JOB_ID}.log"
    
elif squeue -j $JOB_ID | grep -q " PD "; then
    echo "Job is PENDING in queue"
    echo ""
    echo "Queue position and resource availability:"
    scontrol show job $JOB_ID | grep -E "JobState|Reason|StartTime" | sed 's/^/  /'
    echo ""
    echo "To check queue:"
    echo "  squeue -p all_gpu.p | head"
else
    echo "Job may have completed or failed. Checking logs..."
    echo ""
    
    # Check if logs exist
    if [ -d "${LOG_DIR}/${JOB_ID}" ]; then
        echo "Log directory exists. Checking completion status..."
        
        # Check for study name
        if [ -f "${LOG_DIR}/${JOB_ID}/stage1_worker_0_${JOB_ID}.log" ]; then
            STUDY_NAME=$(grep "Study name:" "${LOG_DIR}/${JOB_ID}/stage1_worker_0_${JOB_ID}.log" | tail -1 | cut -d':' -f2- | tr -d ' ')
            if [ ! -z "$STUDY_NAME" ]; then
                echo "✓ Stage 1 Study Name Found: $STUDY_NAME"
                echo ""
                echo "To launch Stage 2:"
                echo "  export STAGE1_STUDY_NAME=\"$STUDY_NAME\""
                echo "  sbatch wind_forecasting/run_scripts/tune_scripts/tune_model_storm_awaken_p60_stage2.sh"
            fi
        fi
        
        # Check worker completion
        echo ""
        echo "Worker completion status:"
        for i in 0 1 2; do
            WORKER_LOG="${LOG_DIR}/${JOB_ID}/stage1_worker_${i}_${JOB_ID}.log"
            if [ -f "$WORKER_LOG" ]; then
                if grep -q "COMPLETED successfully" "$WORKER_LOG"; then
                    echo "  Worker $i: ✓ COMPLETED"
                elif grep -q "FAILED" "$WORKER_LOG"; then
                    echo "  Worker $i: ✗ FAILED"
                else
                    echo "  Worker $i: ? UNKNOWN"
                fi
            fi
        done
    else
        echo "No log directory found at ${LOG_DIR}/${JOB_ID}"
        echo "Job may still be initializing..."
    fi
    
    # Check job accounting info
    echo ""
    echo "Job accounting info:"
    sacct -j $JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocCPUS,AllocNodes
fi

echo ""
echo "============================================"
echo "Run this script again to refresh: ./monitor_stage1.sh"
echo "============================================"