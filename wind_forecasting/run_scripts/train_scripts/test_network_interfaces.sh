#!/bin/bash

#SBATCH --partition=all_gpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2048
#SBATCH --gres=gpu:H100:1
#SBATCH --time=0:05:00
#SBATCH --job-name=test_network_interfaces
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/test_network_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/test_network_%j.err

echo "=== NETWORK INTERFACE DETECTION TEST ==="
echo "NODE: $(hostname)"
echo "JOB ID: ${SLURM_JOB_ID}"

echo "=== Available Network Interfaces ==="
ip addr show | grep -E "^[0-9]+: " | grep -v "lo:"

echo "=== InfiniBand Interface Detection ==="
if ls /sys/class/net/ib* > /dev/null 2>&1; then
    echo "✓ InfiniBand interfaces found:"
    ls /sys/class/net/ib*
    for ib in /sys/class/net/ib*; do
        ib_name=$(basename $ib)
        echo "  Interface: $ib_name"
        if [ -f "$ib/operstate" ]; then
            state=$(cat $ib/operstate)
            echo "    State: $state"
        fi
        if [ -f "$ib/address" ]; then
            addr=$(cat $ib/address)
            echo "    Address: $addr"
        fi
    done
else
    echo "✗ No InfiniBand interfaces found"
fi

echo "=== Ethernet Interface Detection ==="
for eth in /sys/class/net/eth* /sys/class/net/en*; do
    if [ -d "$eth" ]; then
        eth_name=$(basename $eth)
        echo "  Interface: $eth_name"
        if [ -f "$eth/operstate" ]; then
            state=$(cat $eth/operstate)
            echo "    State: $state"
        fi
        if [ -f "$eth/address" ]; then
            addr=$(cat $eth/address)
            echo "    Address: $addr"
        fi
    fi
done

echo "=== IP Configuration ==="
ip addr show | grep "inet " | grep -v "127.0.0.1"

echo "=== Recommended NCCL Configuration ==="
if ls /sys/class/net/ib* > /dev/null 2>&1; then
    # InfiniBand found
    active_ib=""
    for ib in /sys/class/net/ib*; do
        ib_name=$(basename $ib)
        if [ -f "$ib/operstate" ] && [ "$(cat $ib/operstate)" = "up" ]; then
            active_ib=$ib_name
            break
        fi
    done
    
    if [ -n "$active_ib" ]; then
        echo "Recommended for InfiniBand:"
        echo "  export NCCL_SOCKET_IFNAME=$active_ib"
        echo "  export NCCL_IB_DISABLE=0"
    else
        echo "InfiniBand interfaces found but none are up"
        echo "  export NCCL_SOCKET_IFNAME=ib0  # Try this anyway"
        echo "  export NCCL_IB_DISABLE=0"
    fi
else
    # No InfiniBand, use Ethernet
    echo "Recommended for Ethernet:"
    echo "  export NCCL_SOCKET_IFNAME=eth0"
    echo "  export NCCL_IB_DISABLE=1"
fi

echo "=== TEST COMPLETED ==="