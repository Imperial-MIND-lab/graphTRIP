#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=5:00:00
#PBS -N line2_option3b_part2
#PBS -J 0-71

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

# Settings
OBSERVER='FileStorageObserver'
EXNAME='transfer_and_finetune'
CONFIG_JSON='finetune_option_b.json'
SEEDS=(0 1)
MAX_NUM_CONFIGS=36

# Calculate config index (jobid) and seed index from PBS_ARRAY_INDEX
# For example, with 288 configs and 3 seeds: PBS_ARRAY_INDEX 0-863 maps to:
# - jobid: 0-287 (config index)
# - seed: determined by which "block" of 3 the array index falls into
JOBID=$((PBS_ARRAY_INDEX % MAX_NUM_CONFIGS))
SEED_INDEX=$((PBS_ARRAY_INDEX / MAX_NUM_CONFIGS))
SEED=${SEEDS[$SEED_INDEX]}

# Run the experiment with the specific config and seed
python run_experiment.py ${EXNAME} ${OBSERVER} --jobid=${JOBID} --config_json=${CONFIG_JSON} --seed=${SEED}