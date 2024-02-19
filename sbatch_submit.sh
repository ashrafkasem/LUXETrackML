#!/bin/bash

#SBATCH --partition=maxgpu  

#SBATCH --time=48:00:00

#SBATCH --nodes=1

#SBATCH --constraint='A100-SXM4-80GB'
unset LD_PRELOAD
trkml
source venv_maxwell/bin/activate 
python train.py $1 $2
