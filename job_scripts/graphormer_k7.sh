#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=2:00:00
#PBS -N graphormer_job49k7
#PBS -J 0-9

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments
python run_experiment.py train_jointly FileStorageObserver --jobid=${PBS_ARRAY_INDEX} --config_json='graphormer_job49k7.json'
