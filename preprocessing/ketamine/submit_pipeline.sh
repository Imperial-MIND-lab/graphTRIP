#!/bin/bash
# Submit the full ketamine preprocessing pipeline as chained PBS jobs.
# Run from the project root on the HPC login node:
#   bash preprocessing/ketamine/submit_pipeline.sh
#
# Prerequisites (one-time, do before running):
#   1. Pull fMRIPrep Singularity image:
#        apptainer pull ./containers/fmriprep.sif docker://nipreps/fmriprep:24.1.1
#   2. Place FreeSurfer license at ${HOME}/.freesurfer/license.txt
#   3. Run create_subject_map.py and create_annotations.py locally (already done if
#      subject_map.csv and data/raw/ds005917/annotations.csv exist).

set -e

SCRIPT_DIR="${PBS_O_WORKDIR:-$(pwd)}/preprocessing/ketamine"

# Stage 1: fMRIPrep (~2-4 h per subject, 36 array tasks)
JOB1=$(qsub "${SCRIPT_DIR}/fmriprep.sh")
echo "Stage 1 fMRIPrep submitted:         $JOB1"

# Array-to-array dependencies use afteranyarray, not afterokarray: with afterokarray a
# single failed subject blocks every downstream stage for the whole cohort. Each stage
# fails loudly for its own subject and the rest proceed; run the QC pass afterwards to
# find who dropped out:
#     python -m preprocessing.ketamine.qc cohort

# Stage 2: nilearn post-processing (~30 min per subject)
JOB2=$(qsub -W depend=afteranyarray:"${JOB1}" "${SCRIPT_DIR}/postprocess.sh")
echo "Stage 2 Post-processing submitted:  $JOB2 (after $JOB1)"

# Stage 3: REACT masks — single job, waits for all postprocess tasks
JOB3=$(qsub -W depend=afteranyarray:"${JOB2}" "${SCRIPT_DIR}/react_masks_ketamine.sh")
echo "Stage 3 REACT masks submitted:      $JOB3 (after $JOB2)"

# Stage 4: REACT per subject — waits for the single masks job. afterok is correct here:
# if the shared masks failed, every downstream result would be meaningless.
JOB4=$(qsub -W depend=afterok:"${JOB3}" "${SCRIPT_DIR}/react_ketamine.sh")
echo "Stage 4 REACT submitted:            $JOB4 (after $JOB3)"

# Stage 5: Parcellation + feature extraction — waits for all REACT tasks
JOB5=$(qsub -W depend=afteranyarray:"${JOB4}" "${SCRIPT_DIR}/preprocess_features.sh")
echo "Stage 5 Feature extraction submitted: $JOB5 (after $JOB4)"

echo ""
echo "Pipeline submitted. Monitor with: qstat -u hmt23"
echo "When it finishes:  python -m preprocessing.ketamine.qc cohort"
