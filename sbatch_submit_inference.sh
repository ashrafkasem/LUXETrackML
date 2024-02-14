#!/bin/bash

#SBATCH --partition=maxgpu  

#SBATCH --time=1:00:00

#SBATCH --nodes=1

unset LD_PRELOAD
trkml
source venv_maxwell/bin/activate 
python inference.py $1 $2
