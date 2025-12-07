#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=08:00:00
#PBS -N secondary_scripts
#PBS -J 0-9

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/scripts

# Run the unsupervised VGAE training script
python validation.py -s ${PBS_ARRAY_INDEX}
