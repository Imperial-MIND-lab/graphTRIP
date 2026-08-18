#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=08:00:00
#PBS -N temp_ablation
#PBS -J 0-59

# Temporary re-run of scripts/ablation.py for the feature ablations only.
#
# The submitted feature ablations were trained with save_weights=False, so none of them
# can be transferred zero-shot onto psilodep1. This re-runs them with weights and adds
# the FC-only cell of the design:
#
#   ablation.py -j  0   control_mlp_raw         clinical only
#   ablation.py -j  4   linreg_on_clinical_data clinical only
#   ablation.py -j  5   no_node_features        FC + clinical
#   ablation.py -j  6   no_clinical_features    FC + REACT + arm
#   ablation.py -j  9   no_react_no_clinical    FC + arm                       (new)
#   ablation.py -j 10   medusa no_react_no_clinical  FC only                   (new)
#
# graphTRIP outputs land in outputs/ablation/feature_ablation/, next to but separate from
# the model ablations (pca_benchmark, tsne_benchmark, vgae_linreg_head), which are not
# re-run. The Medusa job writes to outputs/medusa_ablation/ alongside its siblings.
#
# 6 base jobs x 10 seeds = 60 jobs: PBS_ARRAY_INDEX 0-59.

# Load environment
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip
cd ~/projects/graphTRIP/scripts

# ablation.py job IDs to re-run, in array order
JOBS=(0 4 5 6 9 10)

# Number of seeds per base job
SEEDS_PER_JOB=10

# Get the current job index
JOB_ID=${PBS_ARRAY_INDEX}

# Calculate base job index and seed
BASE_JOB=$((JOB_ID / SEEDS_PER_JOB))
SEED=$((JOB_ID % SEEDS_PER_JOB))

if [ $BASE_JOB -ge ${#JOBS[@]} ]; then
    echo "Invalid job index: $JOB_ID"
    exit 1
fi

python ablation.py -j ${JOBS[$BASE_JOB]} -s $SEED
