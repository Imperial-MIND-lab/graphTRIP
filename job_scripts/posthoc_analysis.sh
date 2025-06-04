#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=0:30:00
#PBS -N posthoc_analysis

# Script for running the PLS analysis after primary and secondary scripts have been run.

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Define the input directories
GRAIL_DIR='outputs/x_graphtrip/grail/'
ATTENTION_DIR='outputs/x_graphtrip/attention_weights/'

# Run PLS analysis
python pls_analysis.py --grail_dir=$GRAIL_DIR --attention_dir=$ATTENTION_DIR
