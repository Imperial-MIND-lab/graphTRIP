#!/bin/bash

#PBS -l select=1:ncpus=1:mem=1gb
#PBS -l walltime=00:30:00
#PBS -N temp_tlearner
#PBS -J 0-1855

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/scripts

NUM_SEEDS=10

python temp_tlearner.py -ci=${PBS_ARRAY_INDEX} -ns=${NUM_SEEDS} -v
