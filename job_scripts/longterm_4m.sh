#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=4:00:00
#PBS -N longterm_4m
#PBS -J 0-9

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments
python run_experiment.py train_jointly FileStorageObserver --jobid=${PBS_ARRAY_INDEX} --config_json='graphtrip_4m.json'
