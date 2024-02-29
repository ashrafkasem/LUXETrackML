#!/bin/bash

#SBATCH --partition=maxgpu  

#SBATCH --time=2:00:00

#SBATCH --nodes=1

unset LD_PRELOAD
trkml
source venv_maxwell/bin/activate 

# Define config file paths
config_files=(
    "configs/inference/${1}perc/config_maxwell_5prec.yaml"
    "configs/inference/${1}perc/config_maxwell_10prec.yaml"
    "configs/inference/${1}perc/config_maxwell_20prec.yaml"
    "configs/inference/${1}perc/config_maxwell_30prec.yaml"
    "configs/inference/${1}perc/config_maxwell_40prec.yaml"
    "configs/inference/${1}perc/config_maxwell_all.yaml"
)

# Iterate over each config file and execute the Python script
for config_file in "${config_files[@]}"; do
    echo "Using config file: $config_file"
    python inference.py "$config_file" 0
done