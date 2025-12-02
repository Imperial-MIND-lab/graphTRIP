#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=00:15:00
#PBS -N sklearn_jobs
#PBS -J 0-1951

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

CONFIG_JSON='sklearn_head_screen.json'
JOBID=${PBS_ARRAY_INDEX}
OBSERVER='NeptuneObserver'
EXNAME='train_tlearners'

python run_experiment.py ${EXNAME} ${OBSERVER} --jobid=${JOBID} --config_json=${CONFIG_JSON}
