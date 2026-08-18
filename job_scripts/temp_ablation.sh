#!/bin/bash

#PBS -l select=1:ncpus=4:mem=2gb
#PBS -l walltime=00:10:00
#PBS -N temp_ablation
#PBS -J 0-9

# Temporary re-run of the linear regression feature ablation only:
#
#   ablation.py -j 4   linreg_on_clinical_data   clinical only
#
# 1 base job x 10 seeds = 10 jobs: PBS_ARRAY_INDEX 0-9.

# Load environment
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip
cd ~/projects/graphTRIP/scripts

# One seed per array index
SEED=${PBS_ARRAY_INDEX}

python ablation.py -j 4 -s $SEED
