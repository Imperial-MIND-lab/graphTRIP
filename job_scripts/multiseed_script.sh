#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=08:00:00
#PBS -N causal_graphtrip
#PBS -J 0-79

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/scripts

# Calculate job_id (0-6) and seed (0-9) from array index (0-69)
# job_id = array_index // 10
# seed = array_index % 10
SEEDS=(0 1 2 3 4 5 6 7 8 9)
MAX_NUM_CONFIGS=8

CONFIG_ID=$((PBS_ARRAY_INDEX % MAX_NUM_CONFIGS))
SEED_INDEX=$((PBS_ARRAY_INDEX / MAX_NUM_CONFIGS))
SEED=${SEEDS[$SEED_INDEX]}
echo "CONFIG_ID: ${CONFIG_ID}, SEED: ${SEED}"

python causal_graphtrip.py -ci ${CONFIG_ID} -s ${SEED}
