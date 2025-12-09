#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=03:00:00
#PBS -N grail_2.0
#PBS -J 0-9

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/scripts

python graphtrip.py -s ${PBS_ARRAY_INDEX} -v
