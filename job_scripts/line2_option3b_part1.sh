#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=4:00:00
#PBS -N line2_option3b_part1
#PBS -J 0-5

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

# Settings
OBSERVER='FileStorageObserver'
EXNAME='retrain_mlp'
CONFIG_JSON='pretrain_mlp.json'

# Run the experiment with the specific config and seed
python run_experiment.py ${EXNAME} ${OBSERVER} --jobid=${PBS_ARRAY_INDEX} --config_json=${CONFIG_JSON} --seed=${PBS_ARRAY_INDEX}
