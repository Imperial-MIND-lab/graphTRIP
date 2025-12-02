#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=5:00:00
#PBS -N selected_jobs_multiseed
#PBS -J 0-249 

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

# Settings
OBSERVER='FileStorageObserver'
EXNAME='train_cfrnet'
CONFIG_JSON='cfrnet.json'
SEEDS=(0 1 2 3 4 5 6 7 8 9)
SELECTED_CONFIGS=(330 218 202 201 345 265 313 11 75 371 139 282 131 315 344 249 266 329 273 280 209 272 299 208 307)

# Calculate number of selected configs and seeds
NUM_SELECTED_CONFIGS=${#SELECTED_CONFIGS[@]}
NUM_SEEDS=${#SEEDS[@]}

# Calculate config index and seed index from PBS_ARRAY_INDEX
# PBS_ARRAY_INDEX ranges from 0 to (num_seeds * num_selected_configs - 1)
# For example, with 11 selected configs and 3 seeds: PBS_ARRAY_INDEX 0-32 maps to:
# - config_index: 0-10 (index into SELECTED_CONFIGS array)
# - seed_index: 0-2 (index into SEEDS array)
CONFIG_INDEX=$((PBS_ARRAY_INDEX % NUM_SELECTED_CONFIGS))
SEED_INDEX=$((PBS_ARRAY_INDEX / NUM_SELECTED_CONFIGS))
JOBID=${SELECTED_CONFIGS[$CONFIG_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

# Run the experiment with the specific config and seed
python run_experiment.py ${EXNAME} ${OBSERVER} --jobid=${JOBID} --config_json=${CONFIG_JSON} --seed=${SEED}