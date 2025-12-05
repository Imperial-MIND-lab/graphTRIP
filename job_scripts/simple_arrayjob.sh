#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=4:00:00
#PBS -N pretrain_mlp
#PBS -J 0-2

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

CONFIG_JSON='pretrain_mlp.json'
OBSERVER='FileStorageObserver'
EXNAME='retrain_mlp'

python run_experiment.py ${EXNAME} ${OBSERVER} --jobid=${PBS_ARRAY_INDEX} --config_json=${CONFIG_JSON}
