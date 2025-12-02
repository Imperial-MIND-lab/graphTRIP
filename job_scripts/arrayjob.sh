#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=2:00:00
#PBS -N tlearners_screening
#PBS -J 0-575

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

if [ ${PBS_ARRAY_INDEX} -lt 288 ]; then
    CONFIG_JSON='tlearners.json'
    JOBID=${PBS_ARRAY_INDEX}
else
    CONFIG_JSON='tlearners_delta.json'
    JOBID=$((PBS_ARRAY_INDEX - 288))
fi

python run_experiment.py train_tlearners_torch NeptuneObserver --jobid=${JOBID} --config_json=${CONFIG_JSON}
