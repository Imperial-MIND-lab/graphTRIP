#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=6:00:00
#PBS -N graphormer_seed0
#PBS -J 0-63

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments
python run_experiment.py train_jointly NeptuneObserver --jobid=${PBS_ARRAY_INDEX} --config_json='graphormer.json'
