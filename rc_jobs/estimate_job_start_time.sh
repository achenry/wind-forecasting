#!/bin/bash

your_job_id=$(squeue -u $USER -h -o %i | head -n1)
your_job_name=$(squeue -j $your_job_id -h -o %j)
your_job_nodes=$(squeue -j $your_job_id -h -o %D)
your_job_time=$(squeue -j $your_job_id -h -o %l)


total_nodes=$(sinfo -p amilan -h -o %D)
running_jobs=$(squeue -p amilan -t RUNNING -o "%D %e" --sort=e -h)

# Sum up total nodes used by running jobs
nodes_in_use=0
while read -r job_nodes _; do
    nodes_in_use=$((nodes_in_use + job_nodes))
done <<< "$running_jobs"

available_nodes=0 # Assume no nodes are available
current_time=$(date +%s)
estimated_start_time=$current_time

time_to_seconds() {
    echo $(echo $1 | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')
}

# Process running jobs
while read -r job_nodes job_end_time; do
    job_end_seconds=$(date -d "$job_end_time" +%s)
    
    if [ $available_nodes -ge $your_job_nodes ]; then
        break
    fi
    
    if [ $job_end_seconds -gt $estimated_start_time ]; then
        estimated_start_time=$job_end_seconds
        available_nodes=$((available_nodes + job_nodes))
    fi
done <<< "$running_jobs"

# Get pending jobs with higher or equal priority, sorted by priority and submission time
your_priority=$(squeue -j $your_job_id -h -o %Q)
pending_jobs=$(squeue -p amilan -t PENDING -o "%D %l %Q" --sort=-p,Q -h | awk -v prio="$your_priority" '$3 >= prio {print $1, $2}')

# Process pending jobs if necessary
if [ $available_nodes -lt $your_job_nodes ]; then
    while read -r job_nodes job_time; do
        job_seconds=$(time_to_seconds $job_time)
        estimated_start_time=$((estimated_start_time + job_seconds))
        available_nodes=$((available_nodes + job_nodes))
        
        if [ $available_nodes -ge $your_job_nodes ]; then
            break
        fi
    done <<< "$pending_jobs"
fi

# Calculate wait time
wait_time=$((estimated_start_time - current_time))
wait_time_human=$(date -u -d @${wait_time} +"%H:%M:%S")

# Calculate estimated end time
your_job_seconds=$(time_to_seconds $your_job_time)
estimated_end_time=$((estimated_start_time + your_job_seconds))

echo "Your job $your_job_id ($your_job_name) requires $your_job_nodes nodes."
echo "Current time: $(date)"
echo "Total nodes in partition: $total_nodes"
echo "Nodes currently in use: $nodes_in_use"
echo "Estimated start time: $(date -d @$estimated_start_time)"
echo "Estimated wait time: $wait_time_human"
echo "Estimated end time: $(date -d @$estimated_end_time)"