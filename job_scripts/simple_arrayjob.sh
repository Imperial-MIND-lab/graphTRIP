#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=8:00:00
#PBS -N transfer_vgae_option_2
#PBS -J 0-59

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

CONFIG_JSON='transfer_vgae_option_2.json'
OBSERVER='NeptuneObserver'
EXNAME='transfer_vgae'

python run_experiment.py ${EXNAME} ${OBSERVER} --jobid=${PBS_ARRAY_INDEX} --config_json=${CONFIG_JSON} --seed=0
