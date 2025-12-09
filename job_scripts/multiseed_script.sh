#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=06:00:00
#PBS -N interpret
#PBS -J 0-69

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/scripts

# Calculate job_id (0-6) and seed (0-9) from array index (0-69)
# job_id = array_index // 10
# seed = array_index % 10
JOBID=$((PBS_ARRAY_INDEX / 10))
SEED=$((PBS_ARRAY_INDEX % 10))

python interpret.py -j ${JOBID} -s ${SEED} -v
