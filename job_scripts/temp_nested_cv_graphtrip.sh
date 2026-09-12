#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=02:00:00
#PBS -N nested_cv
#PBS -J 0-4

# Nested cross-validation of graphTRIP: trains the inner ensemble of each (outer fold,
# seed) pair. One outer fold of 6 patients is held out completely, so the biomarker
# selection that runs on the inner 36 has not seen it.
#
# Array indices: PBS_ARRAY_INDEX = OUTER * 5 + SEED, with 5 training seeds per outer fold.
#   -J 0-4     pilot: outer fold 0 only, 5 seeds
#   -J 0-34    full run: all 7 outer folds
# Submit the pilot first and check `python -m scripts.nested_cv --summary` before
# committing the rest: if the inner models do not predict at ~31 training patients, the
# GRAIL stage is not worth running.
#
# Measured ~16 min per element (7 inner folds x 300 epochs), so 2 h is ample.

# Load environment
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip

# Change to working directory
cd $PBS_O_WORKDIR

SEEDS_PER_OUTER=5
OUTER=$((PBS_ARRAY_INDEX / SEEDS_PER_OUTER))
SEED=$((PBS_ARRAY_INDEX % SEEDS_PER_OUTER))

python -m scripts.nested_cv -o ${OUTER} -s ${SEED} -v
