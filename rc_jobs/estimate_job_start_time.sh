#!/bin/bash

your_job_id=$(squeue -u $USER -h -o %i | head -n1)
your_job_name=$(squeue -j $your_job_id -h -o %j)
your_job_nodes=$(squeue -j $your_job_id -h -o %D)
your_job_time=$(squeue -j $your_job_id -h -o %l)


total_nodes=$(sinfo -p amilan -h -o %D)
running_jobs_info=$(squeue -p amilan -t RUNNING -o "%D %C %N" -h)
nodes_in_use=$(echo "$running_jobs_info" | awk '{sum += $1} END {print sum}')
cores_in_use=$(echo "$running_jobs_info" | awk '{sum += $2} END {print sum}')
unique_nodes_in_use=$(echo "$running_jobs_info" | awk '{print $3}' | tr ',' '\n' | sort -u | wc -l)
available_nodes=$((total_nodes - unique_nodes_in_use))

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
done <<< "$running_jobs_info"

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
echo "Nodes allocated to running jobs: $nodes_in_use"
echo "Unique nodes in use: $unique_nodes_in_use"
echo "Cores in use: $cores_in_use"
echo "Nodes currently available: $available_nodes"
echo "Estimated start time: $(date -d @$estimated_start_time)"
echo "Estimated wait time: $wait_time_human"
echo "Estimated end time: $(date -d @$estimated_end_time)"

# Debug information
echo -e "\nDetailed job information:"
echo "$running_jobs_info" | head -n 10