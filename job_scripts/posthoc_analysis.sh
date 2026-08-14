#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=0:30:00
#PBS -N posthoc_analysis

# Script for running the post-hoc analysis after primary and secondary scripts have been run.

# Load environment
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip
cd ~/projects/graphTRIP/scripts

# Run post-hoc analysis for graphTRIP
CORE_MODEL_DIR='outputs/graphtrip/'
MEDUSA_MODEL_DIR='outputs/medusa_graphtrip/'
python posthoc.py --core_model_dir $CORE_MODEL_DIR --medusa_model_dir $MEDUSA_MODEL_DIR
