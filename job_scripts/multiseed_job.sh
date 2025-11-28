#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=6:00:00
#PBS -N reduce_complexity
#PBS -J 0-863  # Total jobs = MAX_NUM_CONFIGS × number_of_seeds (e.g., 10 configs × 10 seeds = 100 jobs)

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

# Settings
OBSERVER='NeptuneObserver'
EXNAME='train_jointly'
CONFIG_JSON='reduce_complexity.json'
SEEDS=(0 1 2)
MAX_NUM_CONFIGS=288

# Calculate config index (jobid) and seed index from PBS_ARRAY_INDEX
# For example, with 288 configs and 3 seeds: PBS_ARRAY_INDEX 0-863 maps to:
# - jobid: 0-287 (config index)
# - seed: determined by which "block" of 3 the array index falls into
JOBID=$((PBS_ARRAY_INDEX % MAX_NUM_CONFIGS))
SEED_INDEX=$((PBS_ARRAY_INDEX / MAX_NUM_CONFIGS))
SEED=${SEEDS[$SEED_INDEX]}

# Run the experiment with the specific config and seed
python run_experiment.py ${EXNAME} ${OBSERVER} --jobid=${JOBID} --config_json=${CONFIG_JSON} --seed=${SEED}