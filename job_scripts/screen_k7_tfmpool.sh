#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=6:00:00
#PBS -N screen_k7_tfmpool_seed4
#PBS -J 0-1535

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments
python run_experiment.py train_jointly NeptuneObserver --jobid=${PBS_ARRAY_INDEX} --config_json='screen_tfmpool_k7_seed4.json'
