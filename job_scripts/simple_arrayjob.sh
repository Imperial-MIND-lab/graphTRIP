#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=8:00:00
#PBS -N transfer_to_psilodep1_std01
#PBS -J 0-5

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

CONFIG_JSON='transfer_to_psilodep1_std01.json'
OBSERVER='NeptuneObserver'
EXNAME='transfer_and_finetune'

python run_experiment.py ${EXNAME} ${OBSERVER} --jobid=${PBS_ARRAY_INDEX} --config_json=${CONFIG_JSON}
