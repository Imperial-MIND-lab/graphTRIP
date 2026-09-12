#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=08:00:00
#PBS -N nested_grail
#PBS -J 0-209

# GRAIL on the nested cross-validation models, one element per (outer fold, seed, patient).
# Run only after temp_nested_cv_graphtrip.sh has finished and
# `python -m scripts.nested_cv --summary` shows the inner models still predict.
#
# Array indices: PBS_ARRAY_INDEX = OUTER * 210 + SEED * 42 + SUB,
# with 5 training seeds and 42 patients per outer fold.
#   -J 0-209    pilot: outer fold 0 only (5 seeds x 42 patients)
#   -J 0-1469   full run: all 7 outer folds
#
# All 42 patients are grailed, not just the 36 inner ones: the patient index is then the
# same in every outer fold, and the held-out patients cost only 14% extra. The selection
# step must drop them, which it does via nested_split.csv.
#
# Measured ~15.5 min per (patient, fold) with the 1000-rotation spin test, so ~1.8 h per
# element over the 7 inner folds.

# Load environment
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip

# Change to working directory
cd $PBS_O_WORKDIR

NUM_SUBS=42
SEEDS_PER_OUTER=5
ELEMENTS_PER_OUTER=$((NUM_SUBS * SEEDS_PER_OUTER))

OUTER=$((PBS_ARRAY_INDEX / ELEMENTS_PER_OUTER))
RELATIVE_INDEX=$((PBS_ARRAY_INDEX % ELEMENTS_PER_OUTER))
SEED=$((RELATIVE_INDEX / NUM_SUBS))
SUB=$((RELATIVE_INDEX % NUM_SUBS))

WEIGHTS_DIR="outputs/graphtrip/nested_cv/outer_${OUTER}/weights/"
OUTPUT_DIR="outputs/graphtrip/nested_cv/outer_${OUTER}/grail/"

python -m scripts.xai_biomarkers -j ${SUB} -s ${SEED} -v \
    --weights_base_dir ${WEIGHTS_DIR} \
    --output_dir ${OUTPUT_DIR}
