#!/bin/bash

# Get your job ID and name
your_job_id=$(squeue -u $USER -h -o %i | head -n1)
your_job_name=$(squeue -j $your_job_id -h -o %j)

# Get position in queue and jobs ahead
position=$(squeue -p amilan -t PENDING --sort=p,t -h -o %i | nl | grep $your_job_id | awk '{print $1}')
jobs_ahead=$((position - 1))

# Calculate total time of jobs ahead
total_minutes=$(squeue -p amilan -t PENDING --sort=p,t -h -o %l | head -n $jobs_ahead | awk -F':' '{sum += ($1 * 60) + $2} END {print sum}')

# Calculate estimated start time
start_time=$(date -d "+$total_minutes minutes" "+%Y-%m-%d %H:%M:%S")

echo "Your job $your_job_id with name $your_job_name is at position $position in the queue."
echo "Estimated start time: $start_time"