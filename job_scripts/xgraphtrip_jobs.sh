#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=8:00:00
#PBS -N xgraphtrip_jobs_qids
#PBS -J 0-41

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Run selected jobs from the config file
CONFIG_IDS=(185 199 291 0 263 37)

# Get the current job index
JOB_ID=${PBS_ARRAY_INDEX}

# Calculate which config_id and jobid to use
CONFIG_INDEX=89 #$((JOB_ID / 42))
JOB_INDEX=$(JOB_ID % 42)

# Define the config file and output directory
CONFIG_FILE='experiments/configs/x_graphtrip_qids.json'
OUTPUT_DIR='outputs/x_graphtrip_qids/'

# Run x_graphtrip script with the calculated config_id and jobid
python x_graphtrip.py --jobid=$JOB_INDEX --config_id=${CONFIG_IDS[$CONFIG_INDEX]} --config_file=$CONFIG_FILE --output_dir=$OUTPUT_DIR
