#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=8:00:00
#PBS -N graphtrip_simple_mlps
#PBS -J 0-19

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

python run_experiment.py train_jointly FileStorageObserver --jobid=${PBS_ARRAY_INDEX} --config_json='graphtrip_with_simple_mlps.json'
