#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=1:00:00
#PBS -N atlas_bound

# Script for running a single job.

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Run the desired script
python atlas_bound.py -o outputs_single/atlas_bound/
