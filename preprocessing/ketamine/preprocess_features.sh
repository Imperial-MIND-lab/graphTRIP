#!/bin/bash

#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=02:00:00
#PBS -N preprocess_features_ketamine

set -euo pipefail

# conda's shell hook references unset variables, so relax `set -u` around it.
set +u
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip
set -u

project_dir="${PBS_O_WORKDIR:-$(pwd)}"
if [ ! -f "${project_dir}/preprocessing/ketamine/subject_map.csv" ]; then
    echo "ERROR: ${project_dir} is not the graphTRIP project root." >&2
    echo "       Submit from the project root, e.g. qsub preprocessing/ketamine/pilot.sh" >&2
    exit 1
fi
cd "${project_dir}"

# Parcellate BOLD and compute node/edge features for each atlas.
# Runs preprocessing/preprocess.py for ds005917 with the Exclusion=0 filter, which
# covers all 36 patients (33 MDD + 3 BP).
for atlas in schaefer100 schaefer200 aal; do
    echo "--- Atlas: ${atlas} ---"
    python -m preprocessing.preprocess ds005917 before "${atlas}"
done
