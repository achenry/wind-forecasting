#!/bin/bash

#SBATCH --partition=all_gpu.p
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1         # Just 1 task per node for testing
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2048
#SBATCH --gres=gpu:H100:1           # Just 1 GPU per node for testing
#SBATCH --time=0:10:00              # 10 minutes should be enough
#SBATCH --job-name=test_multinode_connectivity
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/test_multinode_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/test_multinode_%j.err

echo "=== MULTI-NODE CONNECTIVITY TEST ==="
echo "JOB ID: ${SLURM_JOB_ID}"
echo "NODE LIST: ${SLURM_JOB_NODELIST}"
echo "NUM NODES: ${SLURM_JOB_NUM_NODES}"

# Test 1: Basic node communication
echo "=== TEST 1: Node Communication ==="
srun --ntasks=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 bash -c '
    echo "Node $(hostname): Rank ${SLURM_PROCID}, Local Rank ${SLURM_LOCALID}"
    echo "Node $(hostname): Available GPUs: $(nvidia-smi --list-gpus | wc -l)"
'

# Test 2: Network interface detection
echo "=== TEST 2: Network Interface Detection ==="
srun --ntasks=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 bash -c '
    echo "=== Node $(hostname) Network Interfaces ==="
    ip addr show | grep "inet " | grep -v "127.0.0.1" | head -5
    echo "Available interfaces:"
    ls /sys/class/net/ | grep -E "^(ib|eth|en)" | head -3
'

# Test 3: Check if InfiniBand is available
echo "=== TEST 3: InfiniBand Detection ==="
srun --ntasks=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 bash -c '
    echo "Node $(hostname):"
    if ls /sys/class/net/ib* > /dev/null 2>&1; then
        echo "  ✓ InfiniBand interfaces found: $(ls /sys/class/net/ib*)"
    else
        echo "  ✗ No InfiniBand interfaces found"
    fi
    
    if command -v ibstat > /dev/null 2>&1; then
        echo "  ✓ InfiniBand tools available"
    else
        echo "  ✗ InfiniBand tools not found"
    fi
'

# Test 4: Basic connectivity between nodes
echo "=== TEST 4: Node-to-Node Connectivity ==="
if [ ${SLURM_JOB_NUM_NODES} -gt 1 ]; then
    srun --ntasks=1 --nodelist=$(echo ${SLURM_JOB_NODELIST} | cut -d',' -f1) bash -c '
        echo "Testing connectivity from $(hostname):"
        for node in $(scontrol show hostnames ${SLURM_JOB_NODELIST}); do
            if [ "$node" != "$(hostname)" ]; then
                if ping -c 1 -W 2 $node > /dev/null 2>&1; then
                    echo "  ✓ Can reach $node"
                else
                    echo "  ✗ Cannot reach $node"
                fi
            fi
        done
    '
else
    echo "Only one node allocated, skipping connectivity test"
fi

echo "=== CONNECTIVITY TEST COMPLETED ==="
echo "Check the output above to determine the correct network interface for NCCL_SOCKET_IFNAME"