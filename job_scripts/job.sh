#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=8:00:00
#PBS -N sanity_check

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments
python run_experiment.py train_cfrnet FileStorageObserver --jobid=0 --config_json='causal_graphtrip.json' --output_dir='outputs/causal_graphtrip/sanity_check/'
