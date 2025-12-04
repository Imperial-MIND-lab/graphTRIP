#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=8:00:00
#PBS -N transfer_vgae_screening
#PBS -J 0-159

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

NUM_CONFIGS_FIRST_GROUP=80

if [ ${PBS_ARRAY_INDEX} -lt $NUM_CONFIGS_FIRST_GROUP ]; then
    CONFIG_JSON='transfer_vgae_more_graphattrs.json'
    JOBID=${PBS_ARRAY_INDEX}
else
    CONFIG_JSON='transfer_vgae_and_pooling.json'
    JOBID=$((PBS_ARRAY_INDEX - $NUM_CONFIGS_FIRST_GROUP))
fi

OBSERVER='NeptuneObserver'
EXNAME='transfer_vgae'

python run_experiment.py ${EXNAME} ${OBSERVER} --jobid=${JOBID} --config_json=${CONFIG_JSON}
