#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=0:30:00
#PBS -N posthoc_analysis

# Script for running the post-hoc analysis after primary scripts have been run.

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Run post-hoc analysis for graphTRIP
WEIGHTS_BASE_DIR='outputs/graphtrip/weights/'
OUTPUT_DIR='outputs/graphtrip/'
python graphtrip_posthoc.py -w $WEIGHTS_BASE_DIR -o $OUTPUT_DIR

# Run post-hoc analysis for x-graphTRIP
WEIGHTS_BASE_DIR='outputs/x_graphtrip/weights/'
OUTPUT_DIR='outputs/x_graphtrip/'
python graphtrip_posthoc.py -w $WEIGHTS_BASE_DIR -o $OUTPUT_DIR -grail_only
