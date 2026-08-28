#!/bin/bash
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=03:00:00
#PBS -N pn_graphtrip
#PBS -J 0-999

# Permutation null for graphtrip: 100 permutations x 10 training seeds.
#
# Array indices: PBS_ARRAY_INDEX = PERM * 10 + SEED, with PERM 0-99 and SEED 0-9.
# The 10 seeds of one permutation share its labels and are ensembled into one null draw.
# Submit through launch.sh, which overrides -J for a permutation subrange and passes
# EXTRA_ARGS. Every step is idempotent, so re-submitting an array is safe.

# Load environment
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip

# Change to working directory
cd $PBS_O_WORKDIR

# Keep the numerical libraries on the one core we asked for
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PERM=$((PBS_ARRAY_INDEX / 10))
SEED=$((PBS_ARRAY_INDEX % 10))
python -m scripts.permutation_null -m graphtrip -p $PERM -s $SEED ${EXTRA_ARGS}
