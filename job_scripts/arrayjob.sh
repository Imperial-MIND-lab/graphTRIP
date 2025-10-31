#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=8:00:00
#PBS -N graphormer_finetuning
#PBS -J 0-24

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments
python run_experiment.py transfer_and_finetune NeptuneObserver --jobid=${PBS_ARRAY_INDEX} --config_json='graphormer_finetuning.json'
