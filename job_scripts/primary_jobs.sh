#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=8:00:00
#PBS -N primary_jobs
#PBS -J 0-89

# Primary scripts do not depend on other scripts.
# Each base job is run with 10 different seeds (0-9).

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Define job ranges for each script (each base job has 10 seeds)
# Ablation: 3 base jobs (0-2) -> 30 jobs (0-29)
# Benchmarks: 3 base jobs (3-5) -> 30 jobs (30-59)
# Drug classifier: 1 base job (6) -> 10 jobs (60-69)
# graphTRIP BDI: 1 base job (7) -> 10 jobs (70-79)
# graphTRIP main: 1 base job (8) -> 10 jobs (80-89)
ABLATION_START=0
ABLATION_END=29
BENCHMARKS_START=30
BENCHMARKS_END=59
DRUG_CLASSIFIER_START=60
DRUG_CLASSIFIER_END=69
GRAPHTRIP_BDI_START=70
GRAPHTRIP_BDI_END=79
GRAPHTRIP_MAIN_START=80
GRAPHTRIP_MAIN_END=89

# Number of seeds per base job
SEEDS_PER_JOB=10

# Get the current job index
JOB_ID=${PBS_ARRAY_INDEX}

# Calculate base job index and seed
BASE_JOB=$((JOB_ID / SEEDS_PER_JOB))
SEED=$((JOB_ID % SEEDS_PER_JOB))

# Run the appropriate script based on the job index
if [ $JOB_ID -ge $ABLATION_START ] && [ $JOB_ID -le $ABLATION_END ]; then
    # Ablation models (3 base jobs × 10 seeds = 30 jobs; 0-29)
    python ablation_models.py --jobid=$BASE_JOB -s $SEED

elif [ $JOB_ID -ge $BENCHMARKS_START ] && [ $JOB_ID -le $BENCHMARKS_END ]; then
    # Benchmarks (3 base jobs × 10 seeds = 30 jobs; 30-59)
    BASE_JOB_RELATIVE=$((BASE_JOB - 3))
    python benchmarks.py --jobid=$BASE_JOB_RELATIVE -s $SEED

elif [ $JOB_ID -ge $DRUG_CLASSIFIER_START ] && [ $JOB_ID -le $DRUG_CLASSIFIER_END ]; then
    # Drug classifier (1 base job × 10 seeds = 10 jobs; 60-69)
    python drug_classifier.py -s $SEED

elif [ $JOB_ID -ge $GRAPHTRIP_BDI_START ] && [ $JOB_ID -le $GRAPHTRIP_BDI_END ]; then
    # graphTRIP for BDI prediction (1 base job × 10 seeds = 10 jobs; 70-79)
    python graphtrip_bdi.py -s $SEED

elif [ $JOB_ID -ge $GRAPHTRIP_MAIN_START ] && [ $JOB_ID -le $GRAPHTRIP_MAIN_END ]; then
    # graphTRIP, main model (1 base job × 10 seeds = 10 jobs; 80-89)
    python graphtrip.py -s $SEED

else
    echo "Invalid job index: $JOB_ID"
    exit 1
fi
