#!/bin/bash

#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=02:00:00
#PBS -N preprocess_features_ketamine

set -euo pipefail

module load anaconda3/personal
source activate graphtrp

project_dir="/rds/general/user/hmt23/home/projects/graphTRP"
cd "${project_dir}"

# Parcellate BOLD and compute node/edge features for each atlas.
# Runs preprocessing/preprocess.py for ds005917 with the Exclusion=0 filter, which
# covers all 36 patients (33 MDD + 3 BP).
for atlas in schaefer100 schaefer200 aal; do
    echo "--- Atlas: ${atlas} ---"
    python -m preprocessing.preprocess ds005917 before "${atlas}"
done
