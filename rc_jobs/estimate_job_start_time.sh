#!/bin/bash

# Get your job ID and details
your_job_id=$(squeue -u $USER -h -o %i | head -n1)
your_job_name=$(squeue -j $your_job_id -h -o %j)
your_job_nodes=$(squeue -j $your_job_id -h -o %D)
your_job_time=$(squeue -j $your_job_id -h -o %l)

# Get total nodes in the partition
total_nodes=$(sinfo -p amilan -h -o %D)

# Get running jobs, sorted by end time
running_jobs=$(squeue -p amilan -t RUNNING -o "%D %e" --sort=e -h)

# Get pending jobs with higher priority, sorted by priority and submission time
pending_jobs=$(squeue -p amilan -t PENDING -o "%D %l %Q" --sort=-p,Q -h | awk -v your_prio=$(squeue -j $your_job_id -h -o %Q) '$3 >= your_prio {print $1, $2}')

# Initialize variables
available_nodes=$total_nodes
current_time=$(date +%s)
estimated_start_time=$current_time

# Function to convert HH:MM:SS to seconds
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
echo "Current time: $(date -d @$current_time)"
echo "Estimated start time: $(date -d @$estimated_start_time)"
echo "Estimated wait time: $wait_time_human"
echo "Estimated end time: $(date -d @$estimated_end_time)"