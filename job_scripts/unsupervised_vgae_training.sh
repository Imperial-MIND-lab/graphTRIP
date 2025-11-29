#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=00:30:00
#PBS -N unsupervised_vgae_training
#PBS -J 0-419

# Unsupervised VGAE training performs LOOCV training of the VGAE.
# Each fold (jobid 0-41) is run with 10 different seeds (0-9).
# Total: 42 folds × 10 seeds = 420 jobs (0-419)

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Number of seeds per fold
SEEDS_PER_JOB=10

# Get the current job index
JOB_ID=${PBS_ARRAY_INDEX}

# Calculate fold (jobid) and seed
# jobid determines this_k parameter (0-41 for 42-fold LOOCV)
JOBID=$((JOB_ID / SEEDS_PER_JOB))
SEED=$((JOB_ID % SEEDS_PER_JOB))

# Run unsupervised VGAE training
python unsupervised_vgae.py --jobid=$JOBID -s $SEED
