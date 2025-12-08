#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=01:00:00
#PBS -N repeat_control_mlp
#PBS -J 0-9

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/scripts

python ablation.py -s ${PBS_ARRAY_INDEX} -j 0 -v
