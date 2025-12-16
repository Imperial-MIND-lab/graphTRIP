#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=08:00:00
#PBS -N causal_graphtrip
#PBS -J 0-9

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/scripts

SEED=${PBS_ARRAY_INDEX}

python causal_graphtrip.py -s=${SEED}
