#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=01:30:00
#PBS -N tlearners_multiseed
#PBS -J 0-767

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/scripts

# Number of seeds per base job
SEEDS_PER_JOB=2

# Get the current job index
JOB_ID=${PBS_ARRAY_INDEX}

# Calculate base job index and seed
BASE_JOB=$((JOB_ID / SEEDS_PER_JOB))
SEED=$((JOB_ID % SEEDS_PER_JOB))

# First quickly run the VGAE-eval run of the x-graphtrip script
python x_graphtrip.py -s ${SEED}

# Run the experiment with the specific config and seed
python tlearners.py -ci ${BASE_JOB} -s ${SEED}