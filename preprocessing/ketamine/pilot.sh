#!/bin/bash
#
# Two-subject end-to-end pilot of the ds005917 pipeline.
#
# Submit from the project root:
#     qsub preprocessing/ketamine/pilot.sh
#
# Runs all five stages sequentially for S01 and S32 in a single job, so every failure
# lands in one log in the order it happened. These two subjects are chosen because they
# exercise the only two non-standard input paths:
#     S01 (sub-MOA101) -- two complete baseline rest runs; tests select_run()
#     S32 (sub-MOA136) -- no ses-b0 T1w; tests the cross-session anatomical lookup
#
# Override the pair with, e.g.:
#     qsub -v PILOT_INDICES="1 5" preprocessing/ketamine/pilot.sh
# where the numbers are row indices into subject_map.csv (1 = S01, 32 = S32).
#
# AFTER a successful pilot, before the full run, reset the raw-data flags:
#     python -m preprocessing.ketamine.create_annotations
# Stage 5 marks the 34 unprocessed subjects as missing_raw_before=1, and the full run
# will not clear that by itself.

#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=10:00:00
#PBS -N ketamine_pilot

set -euo pipefail

PILOT_INDICES="${PILOT_INDICES:-1 32}"

project_dir="${PBS_O_WORKDIR:-$(pwd)}"
if [ ! -f "${project_dir}/preprocessing/ketamine/subject_map.csv" ]; then
    echo "ERROR: ${project_dir} is not the graphTRIP project root." >&2
    echo "       Submit from the project root, e.g. qsub preprocessing/ketamine/pilot.sh" >&2
    exit 1
fi
script_dir="${project_dir}/preprocessing/ketamine"
subject_map="${script_dir}/subject_map.csv"

# Check the one-time HPC prerequisites up front, before the (slow) python imports, so a
# missing container fails in a second rather than minutes into step 3.
for prereq in "${project_dir}/containers/fmriprep.sif" "${HOME}/.freesurfer/license.txt"; do
    if [ ! -f "${prereq}" ]; then
        echo "ERROR: missing prerequisite: ${prereq}" >&2
        echo "       See 'One-time HPC setup' in preprocessing/ketamine/README.md" >&2
        exit 1
    fi
done

cd "${project_dir}"

# conda's shell hook references unset variables, so relax `set -u` around it.
set +u
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip
set -u

# Resolve the S-IDs for reporting and the final QC step.
pilot_subjects=""
for idx in ${PILOT_INDICES}; do
    s_id=$(awk -F',' -v i="${idx}" 'NR==i+1 {print $3}' "${subject_map}")
    pilot_subjects="${pilot_subjects} ${s_id}"
done

echo "############################################################"
echo "# ds005917 pilot"
echo "# indices : ${PILOT_INDICES}"
echo "# subjects:${pilot_subjects}"
echo "# started : $(date)"
echo "############################################################"

banner () { echo; echo "=================== $* ==================="; echo; }

banner "Step 1/7 - BIDS pre-flight"
python -m preprocessing.ketamine.qc bids

banner "Step 2/7 - template grid check"
python -m preprocessing.ketamine.qc grid

banner "Step 3/7 - fMRIPrep"
for idx in ${PILOT_INDICES}; do
    PBS_ARRAY_INDEX="${idx}" bash "${script_dir}/fmriprep.sh"
done

banner "Step 4/7 - post-processing"
for idx in ${PILOT_INDICES}; do
    PBS_ARRAY_INDEX="${idx}" bash "${script_dir}/postprocess.sh"
done

banner "Step 5/7 - REACT masks (pilot-sized)"
MIN_SUBJECTS=1 bash "${script_dir}/react_masks_ketamine.sh"

banner "Step 6/7 - REACT per subject"
for idx in ${PILOT_INDICES}; do
    PBS_ARRAY_INDEX="${idx}" bash "${script_dir}/react_ketamine.sh"
done

banner "Step 7/7 - features + per-subject QC"
bash "${script_dir}/preprocess_features.sh"
for s_id in ${pilot_subjects}; do
    python -m preprocessing.ketamine.qc subject --s-id "${s_id}"
done

# Re-run the grid check now that real derivatives exist -- this is the comparison that
# actually matters, and it could not be made before fMRIPrep had produced output.
banner "Post-hoc - template grid check against real output"
for s_id in ${pilot_subjects}; do
    python -m preprocessing.ketamine.qc grid --s-id "${s_id}"
done

echo
echo "############################################################"
echo "# Pilot finished: $(date)"
echo "# QC figures: ${project_dir}/outputs/qc_ds005917/"
echo "#"
echo "# Before the full run:"
echo "#   python -m preprocessing.ketamine.create_annotations"
echo "############################################################"
