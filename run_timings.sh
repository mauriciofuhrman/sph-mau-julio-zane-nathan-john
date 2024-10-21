#!/bin/bash

# TIMINGS_FILE="timings.txt"
# echo "# n_particles execution_time(s)" > $TIMINGS_FILE

# Array of h values to iterate over
h_values=(0.5 0.4 0.3 0.2 0.1 0.05 0.02 0.005)

# Time delay in seconds between submissions
delay=120  # Adjust this value to control the delay

for h in "${h_values[@]}"; do
    sbatch submit_sph_job.sub $h
    echo "Submitted job for h = $h, sleeping for $delay seconds..."
    sleep $delay
done
