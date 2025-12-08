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
GRAIL_DIR='outputs/graphtrip/grail/'
WEIGHTS_DIR='outputs/graphtrip/weights/'
ATTENTION_DIR='outputs/graphtrip/attention_weights/'
python graphtrip_posthoc.py --grail_dir $GRAIL_DIR --weights_dir $WEIGHTS_DIR --attention_dir $ATTENTION_DIR

# Run post-hoc analysis for x-graphTRIP
GRAIL_DIR='outputs/x_graphtrip/grail/'
python graphtrip_posthoc.py --grail_dir $GRAIL_DIR 

# Run post-hoc analysis for Medusa x-graphTRIP
GRAIL_DIR='outputs/x_graphtrip/medusa_grail/'
python graphtrip_posthoc.py --grail_dir $GRAIL_DIR 
