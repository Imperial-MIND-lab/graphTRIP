#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=4:00:00
#PBS -N graphormer_seed0
#PBS -J 0-9

module load anaconda3/personal
source activate graphtrip

config_file='experiments/configs/graphtrip.json'
output_dir='outputs/graphtrip_graphormer/psilodep1_wo_pretraining'
seed=0 # this is the initial seed; seed = seed + jobid
jobid=${PBS_ARRAY_INDEX}
config_id=0

cd ~/projects/graphTRIP/scripts
python psilodep1_wo_pretraining.py -c ${config_file} -o ${output_dir} -s ${seed} -j ${jobid} -ci ${config_id}
