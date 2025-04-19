#!/bin/bash

# This script dynamically determines the best CPU partition and node count
# and prints the sbatch command to submit the job.
# You must manually execute the printed sbatch command.

CPU_PARTITIONS="cfds.p storm.p mouse.p"

DEFAULT_CPUS_PER_TASK=128
DEFAULT_MEM_PER_CPU=6000 # MB
DEFAULT_TIME_LIMIT="7-00:00" # D-HH:MM

echo "Querying CPU partitions for available nodes..."

best_partition=""
max_idle_nodes=0
partition_details=""

for partition in ${CPU_PARTITIONS}; do
    sinfo_output=$(sinfo -p ${partition} -h -o "%P %N(A/I/O/T)" 2>/dev/null)

    if [ -z "$sinfo_output" ]; then
        echo "Warning: Could not get info for partition ${partition}. Skipping."
        continue
    fi
    idle_nodes=$(echo "${sinfo_output}" | awk '{split($2, counts, "/"); print counts[2]}')
    
    mem_per_cpu=$(scontrol show part ${partition} | awk '/DefMemPerCPU=/ {print $1}' | cut -d'=' -f2)
    if [ -z "$mem_per_cpu" ]; then
        mem_per_cpu=${DEFAULT_MEM_PER_CPU}
    fi

    echo "Partition ${partition}: ${idle_nodes} idle nodes found (DefaultMemPerCPU: ${mem_per_cpu}M)"

    if [ "${idle_nodes}" -gt "${max_idle_nodes}" ]; then
        max_idle_nodes=${idle_nodes}
        best_partition=${partition}
        partition_details="--mem-per-cpu=${mem_per_cpu}"
    fi
done

echo "--------------------------------------------------"

if [ "${max_idle_nodes}" -eq 0 ]; then
    echo "No idle CPU nodes found in the specified partitions."
    echo "Please check partition status with 'sinfo -s' or 'sinfo -p <partition>'."
    exit 1
else
    echo "Best partition found: ${best_partition} with ${max_idle_nodes} idle nodes."
    echo "Recommended sbatch command:"
    echo "--------------------------------------------------"
    SBATCH_COMMAND="sbatch \\
    --partition=${best_partition} \\
    --nodes=${max_idle_nodes} \\
    --ntasks-per-node=1 \\
    --cpus-per-task=${DEFAULT_CPUS_PER_TASK} \\
    ${partition_details} \\
    --time=${DEFAULT_TIME_LIMIT} \\
    --job-name=tactis_tune_cpu_dynamic \\
    --output=/user/taed7566/wind-forecasting/logging/slurm_logs/tactis_tune_cpu_dynamic_%j.out \\
    --error=/user/taed7566/wind-forecasting/logging/slurm_logs/tactis_tune_cpu_dynamic_%j.err \\
    --hint=multithread \\
    # Add QoS if needed for longer time limits (e.g., --qos=long_cfds.q)
    /user/taed7566/wind-forecasting/wind_forecasting/run_scripts/tune_model_storm_cpufallback.sh"

    echo "${SBATCH_COMMAND}"
    echo "--------------------------------------------------"
    echo "Copy and paste the above command to submit your job."
    echo "Note: The actual job submission will happen when you run the printed command."
    exit 0
fi