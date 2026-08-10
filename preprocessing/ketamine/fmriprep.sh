#!/bin/bash

#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=04:00:00
#PBS -N fmriprep_ketamine
#PBS -J 1-36

# One-time setup (run interactively before submitting this array):
#   apptainer pull ${project_dir}/containers/fmriprep.sif docker://nipreps/fmriprep:24.1.1
#   Register a free FreeSurfer license at https://surfer.nmr.mgh.harvard.edu/registration.html
#   Place it at ${HOME}/.freesurfer/license.txt
#
# Apptainer 1.5.1 is available system-wide at /usr/bin/apptainer -- no module load needed.

set -euo pipefail

project_dir="${PBS_O_WORKDIR:-$(pwd)}"
if [ ! -f "${project_dir}/preprocessing/ketamine/subject_map.csv" ]; then
    echo "ERROR: ${project_dir} is not the graphTRIP project root." >&2
    echo "       Submit from the project root, e.g. qsub preprocessing/ketamine/pilot.sh" >&2
    exit 1
fi
subject_map="${project_dir}/preprocessing/ketamine/subject_map.csv"

bids_dir="${project_dir}/data/raw/ds005917"
output_dir="${project_dir}/data/preprocessed/ds005917"
filter_dir="${project_dir}/preprocessing/ketamine"
sif="${project_dir}/containers/fmriprep.sif"
fs_license="${HOME}/.freesurfer/license.txt"

mkdir -p "${output_dir}"

# Resolve subject for this array task (1-indexed -> CSV row, skipping the header)
bids_id=$(awk -F',' -v idx="${PBS_ARRAY_INDEX}" 'NR==idx+1 {print $1}' "${subject_map}")
if [ -z "${bids_id}" ]; then
    echo "ERROR: array index ${PBS_ARRAY_INDEX} has no row in ${subject_map}" >&2
    exit 1
fi
participant_label="${bids_id#sub-}"   # strip "sub-" prefix for fMRIPrep

echo "Array task ${PBS_ARRAY_INDEX}: ${bids_id} (label: ${participant_label})"

# Use per-job scratch for the work directory to avoid conflicts
work_dir="${TMPDIR}/fmriprep_${participant_label}"
mkdir -p "${work_dir}"

# --output-spaces MNI152NLin6Asym:res-2 is REQUIRED
# Verify with: python -m preprocessing.ketamine.qc grid --s-id S01

apptainer exec --cleanenv \
    -B "${bids_dir}":/data:ro \
    -B "${output_dir}":/output \
    -B "${work_dir}":/work \
    -B "${filter_dir}":/filters:ro \
    -B "$(dirname ${fs_license})":/fs_license:ro \
    "${sif}" \
    fmriprep /data /output participant \
        --participant-label "${participant_label}" \
        --bids-filter-file /filters/bids_filter.json \
        --task-id rest \
        --output-spaces MNI152NLin6Asym:res-2 \
        --fs-license-file /fs_license/license.txt \
        --fs-no-reconall \
        --fd-spike-threshold 0.5 \
        --nthreads 8 \
        --mem-mb 30000 \
        --work-dir /work \
        --notrack

rm -rf "${work_dir}"
