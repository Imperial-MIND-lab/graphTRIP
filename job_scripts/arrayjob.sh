#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=6:00:00
#PBS -N finetune_on_psilodep1
#PBS -J 0-251

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

NUM_CONFIGS_FIRST_GROUP=36

if [ ${PBS_ARRAY_INDEX} -lt $NUM_CONFIGS_FIRST_GROUP ]; then
    CONFIG_JSON='finetune_option_b.json'
    JOBID=${PBS_ARRAY_INDEX}
else
    CONFIG_JSON='finetune_option_c.json'
    JOBID=$((PBS_ARRAY_INDEX - $NUM_CONFIGS_FIRST_GROUP))
fi

OBSERVER='NeptuneObserver'
EXNAME='transfer_vgae'

python run_experiment.py ${EXNAME} ${OBSERVER} --jobid=${JOBID} --config_json=${CONFIG_JSON}
