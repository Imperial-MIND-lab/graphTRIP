#!/bin/bash

#PBS -l select=1:ncpus=1:mem=1gb
#PBS -l walltime=00:15:00
#PBS -N eval_vgaes

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/scripts

NUM_SEEDS=10

for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
    python x_graphtrip.py -s=${SEED} -v
done
