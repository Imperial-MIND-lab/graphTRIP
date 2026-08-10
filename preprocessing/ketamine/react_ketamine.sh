#!/bin/bash

#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=00:30:00
#PBS -N react_ketamine
#PBS -J 1-36

set -euo pipefail

# conda's shell hook references unset variables, so relax `set -u` around it.
set +u
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip
set -u

study="ds005917"
session="before"

# Both sets are required -- see react_masks_ketamine.sh for why.
receptor_sets=("Believeau-5" "Believeau-3")

project_dir="${PBS_O_WORKDIR:-$(pwd)}"
if [ ! -f "${project_dir}/preprocessing/ketamine/subject_map.csv" ]; then
    echo "ERROR: ${project_dir} is not the graphTRIP project root." >&2
    echo "       Submit from the project root, e.g. qsub preprocessing/ketamine/pilot.sh" >&2
    exit 1
fi
data_dir="${HOME}/data"
subject_map="${project_dir}/preprocessing/ketamine/subject_map.csv"
mni_dir="${project_dir}/data/raw/${study}/${session}/MNI_2mm"

# Resolve the subject from subject_map.csv, not from a directory listing. A listing
# shifts every index when an upstream subject fails, which silently pairs array task N
# with the wrong subject.
s_id=$(awk -F',' -v idx="${PBS_ARRAY_INDEX}" 'NR==idx+1 {print $3}' "${subject_map}")
bids_id=$(awk -F',' -v idx="${PBS_ARRAY_INDEX}" 'NR==idx+1 {print $1}' "${subject_map}")
if [ -z "${s_id}" ]; then
    echo "ERROR: array index ${PBS_ARRAY_INDEX} has no row in ${subject_map}" >&2
    exit 1
fi

echo "Array task ${PBS_ARRAY_INDEX}: ${s_id} (${bids_id})"

input_file="${data_dir}/${study}/${session}/${s_id}/${session}_rest_preproc.nii.gz"
if [ ! -f "${input_file}" ]; then
    echo "ERROR: cleaned NIfTI not found for ${s_id}: ${input_file}" >&2
    echo "       Stage 2 (postprocess) probably failed for this subject." >&2
    exit 1
fi

for receptor_set in "${receptor_sets[@]}"; do
    echo "--- Receptor set: ${receptor_set} ---"
    output_parent_dir="${mni_dir}/REACT_${receptor_set}"
    atlas_file="${output_parent_dir}/pet_atlas.nii.gz"
    masks_dir="${output_parent_dir}/out_masks"

    output_dir="${output_parent_dir}/${s_id}"
    mkdir -p "${output_dir}"
    cd "${output_parent_dir}"

    react "${input_file}" \
          "${masks_dir}/mask_stage1.nii.gz" \
          "${masks_dir}/mask_stage2.nii.gz" \
          "${atlas_file}" \
          "${output_dir}/${s_id}"
done

echo "Done: ${s_id}"
