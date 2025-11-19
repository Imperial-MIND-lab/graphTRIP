#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=8:00:00
#PBS -N graphtrip_different_regs
#PBS -J 0-29

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

if [ ${PBS_ARRAY_INDEX} -lt 10 ]; then
    CONFIG_JSON='graphtrip.json'
else
    CONFIG_JSON='graphtrip_with_simple_mlps.json'
fi

python run_experiment.py train_jointly FileStorageObserver --jobid=${PBS_ARRAY_INDEX} --config_json=${CONFIG_JSON}
