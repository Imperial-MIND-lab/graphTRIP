#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=4:00:00
#PBS -N transfer_vgae_screening
#PBS -J 0-47

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

if [ ${PBS_ARRAY_INDEX} -lt 32 ]; then
    CONFIG_JSON='transfer_vgae.json'
    JOBID=${PBS_ARRAY_INDEX}
else
    CONFIG_JSON='transfer_vgae_and_pooling.json'
    JOBID=$((PBS_ARRAY_INDEX - 32))
fi

OBSERVER='NeptuneObserver'
EXNAME='transfer_vgae'

python run_experiment.py ${EXNAME} ${OBSERVER} --jobid=${JOBID} --config_json=${CONFIG_JSON}
