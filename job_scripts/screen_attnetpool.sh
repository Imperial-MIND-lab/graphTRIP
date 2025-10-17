#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=8:00:00
#PBS -N screen_attnetpool
#PBS -J 0-255

module load anaconda3/personal
source activate graphtrp

cd ~/projects/graphTRIP/experiments
python run_experiment.py train_jointly NeptuneObserver --jobid=${PBS_ARRAY_INDEX} --config_json='screen_attnetpool.json'
