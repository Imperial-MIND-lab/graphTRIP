#!/bin/bash

#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=01:00:00
#PBS -N react_masks_ketamine

set -euo pipefail

module load anaconda3/personal
source activate graphtrp

# Parameters
study="ds005917"
session="before"
dataset="5-HT_atlas_2mm"
receptor_sets=("Believeau-5" "Believeau-3")

project_dir="/rds/general/user/hmt23/home/projects/graphTRP"
data_dir="/rds/general/user/hmt23/home/data"
react_data_dir="${project_dir}/data/raw/react_data"
subject_map="${project_dir}/preprocessing/ketamine/subject_map.csv"

# Build the subject list from subject_map.csv rather than a glob, so ordering is
# deterministic and missing subjects are reported instead of silently dropped.
mni_dir="${project_dir}/data/raw/${study}/${session}/MNI_2mm"
mkdir -p "${mni_dir}"
subject_list="${mni_dir}/subject_list.txt"
: > "${subject_list}"

n_missing=0
while IFS=',' read -r bids_id study_id s_id group; do
    nifti="${data_dir}/${study}/${session}/${s_id}/${session}_rest_preproc.nii.gz"
    if [ -f "${nifti}" ]; then
        echo "${nifti}" >> "${subject_list}"
    else
        echo "WARNING: missing cleaned NIfTI for ${s_id} (${bids_id}) -- excluded from masks" >&2
        n_missing=$((n_missing + 1))
    fi
done < <(tail -n +2 "${subject_map}")

# Guards against building masks from a half-finished stage 2. Lower it for a pilot:
#   MIN_SUBJECTS=2 bash react_masks_ketamine.sh
# Masks built from 2 subjects are fine for a smoke test but must NOT be reused for the
# full cohort -- stage 3 has to be re-run once all subjects are through stage 2.
MIN_SUBJECTS="${MIN_SUBJECTS:-10}"

n_found=$(wc -l < "${subject_list}")
echo "Found ${n_found} subjects (${n_missing} missing, minimum ${MIN_SUBJECTS})."
if [ "${n_found}" -lt "${MIN_SUBJECTS}" ]; then
    echo "ERROR: only ${n_found} subjects available; refusing to build masks." >&2
    exit 1
fi

for receptor_set in "${receptor_sets[@]}"; do
    echo "--- Receptor set: ${receptor_set} ---"
    output_parent_dir="${mni_dir}/REACT_${receptor_set}"
    mkdir -p "${output_parent_dir}"
    cd "${output_parent_dir}"

    cp "${subject_list}" ./subject_list.txt
    cp "${react_data_dir}/${dataset}/concatenated/${receptor_set}/pet_atlas.nii.gz" .
    cp "${react_data_dir}/${dataset}/concatenated/${receptor_set}/input_maps.txt" .
    cp "${react_data_dir}/masks/gm_mask.nii.gz" .

    react_masks subject_list.txt pet_atlas.nii.gz gm_mask.nii.gz out_masks
done

echo "Masks built for: ${receptor_sets[*]}"
