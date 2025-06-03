#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=8:00:00
#PBS -N xgraphtrip_jobs_qids
#PBS -J 0-41

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Get the current job index
JOB_ID=${PBS_ARRAY_INDEX}

# Calculate which jobid to use
JOB_INDEX=$((JOB_ID % 42))

# Define the config file and output directory
CONFIG_FILE='experiments/configs/x_graphtrip_qids.json'
OUTPUT_DIR='outputs/x_graphtrip_qids/'

# Run x_graphtrip script with config_id 89 and the calculated jobid
python x_graphtrip.py --jobid=$JOB_INDEX --config_id=89 --config_file=$CONFIG_FILE --output_dir=$OUTPUT_DIR
