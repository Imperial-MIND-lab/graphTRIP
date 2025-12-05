#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=4:00:00
#PBS -N final_transfer_vgae
#PBS -J 0-15

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

CONFIG_JSON='transfer_vgae_and_pooling.json'
OBSERVER='NeptuneObserver'
EXNAME='transfer_vgae'

python run_experiment.py ${EXNAME} ${OBSERVER} --jobid=${PBS_ARRAY_INDEX} --config_json=${CONFIG_JSON}
