#!/bin/bash

#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=00:30:00
#PBS -N postprocess_ketamine
#PBS -J 1-36

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
subject_map="${project_dir}/preprocessing/ketamine/subject_map.csv"

cd "${project_dir}"

bids_id=$(awk -F',' -v idx="${PBS_ARRAY_INDEX}" 'NR==idx+1 {print $1}' "${subject_map}")
s_id=$(awk -F',' -v idx="${PBS_ARRAY_INDEX}" 'NR==idx+1 {print $3}' "${subject_map}")
if [ -z "${bids_id}" ] || [ -z "${s_id}" ]; then
    echo "ERROR: array index ${PBS_ARRAY_INDEX} has no row in ${subject_map}" >&2
    exit 1
fi

echo "Array task ${PBS_ARRAY_INDEX}: ${bids_id} → ${s_id}"

python -m preprocessing.ketamine.postprocess \
    --bids-id "${bids_id}" \
    --s-id "${s_id}"
