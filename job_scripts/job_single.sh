#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=8:00:00
#PBS -N xgraphtrip_ssristop_16

# Script for running a single job.

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Define the config file and output directory
CONFIG_FILE='experiments/configs/x_graphtrip_SSRIstop.json'
OUTPUT_DIR='outputs/x_graphtrip_SSRIstop/'

# Run x_graphtrip script with the calculated config_id and jobid
python x_graphtrip.py --jobid=16 --config_file=$CONFIG_FILE --output_dir=$OUTPUT_DIR
