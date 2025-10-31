#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=2:00:00
#PBS -N pretrain_for_psilodep1

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments
python run_experiment.py train_jointly FileStorageObserver --jobid=0 --config_json='pretrain_for_psilodep1.json'
