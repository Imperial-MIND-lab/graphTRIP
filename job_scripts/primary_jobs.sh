#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=12:00:00
#PBS -N primary_jobs
#PBS -J 0-79

# Primary scripts do not depend on other scripts.
# Each base job is run with 10 different seeds (0-9).

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Define job ranges for each script (each base job has 10 seeds)
# Ablation: 5 base jobs (0-4) -> 50 jobs (0-49)
# graphTRIP BDI: 1 base job (8) -> 10 jobs (50-59)
# graphTRIP main: 1 base job (9) -> 10 jobs (60-69)
# x-graphTRIP: 1 base job (10) -> 10 jobs (70-79)
ABLATION_START=0
ABLATION_END=49
GRAPHTRIP_BDI_START=50
GRAPHTRIP_BDI_END=59
GRAPHTRIP_MAIN_START=60
GRAPHTRIP_MAIN_END=69
XGRAPHTRIP_START=70
XGRAPHTRIP_END=79

# Number of seeds per base job
SEEDS_PER_JOB=10

# Get the current job index
JOB_ID=${PBS_ARRAY_INDEX}

# Calculate base job index and seed
BASE_JOB=$((JOB_ID / SEEDS_PER_JOB))
SEED=$((JOB_ID % SEEDS_PER_JOB))

# Run the appropriate script based on the job index
if [ $JOB_ID -ge $ABLATION_START ] && [ $JOB_ID -le $ABLATION_END ]; then
    # Ablation models (5 base jobs × 10 seeds = 50 jobs; 0-49)
    python ablation.py -j $BASE_JOB -s $SEED

elif [ $JOB_ID -ge $GRAPHTRIP_BDI_START ] && [ $JOB_ID -le $GRAPHTRIP_BDI_END ]; then
    # graphTRIP for BDI prediction (1 base job × 10 seeds = 10 jobs; 50-59)
    python graphtrip_bdi.py -s $SEED

elif [ $JOB_ID -ge $GRAPHTRIP_MAIN_START ] && [ $JOB_ID -le $GRAPHTRIP_MAIN_END ]; then
    # graphTRIP, main model (1 base job × 10 seeds = 10 jobs; 60-69)
    python graphtrip.py -s $SEED

elif [ $JOB_ID -ge $XGRAPHTRIP_START ] && [ $JOB_ID -le $XGRAPHTRIP_END ]; then
    # x-graphTRIP (1 base job × 10 seeds = 10 jobs; 70-79)
    python x_graphtrip.py -s $SEED

else
    echo "Invalid job index: $JOB_ID"
    exit 1
fi
