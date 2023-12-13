#!/bin/bash

# Check if both source and destination directories are provided as arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_dir> <destination_dir>"
    exit 1
fi

source_dir="$1"
destination_dir="$2"

# Create the destination directory if it doesn't exist
mkdir -p "$destination_dir"

# Loop through each jobs_* directory in the source directory
for job_dir in "$source_dir"/job_*; do
    # Check if the directory contains a "done" file
    if [ -f "$job_dir/done" ]; then
        # Move the entire directory to the destination directory
        mv "$job_dir" "$destination_dir/"
        echo "Moved $job_dir to $destination_dir/"
    else
        echo "Skipped $job_dir"
    fi
done

