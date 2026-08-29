#!/bin/bash
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=02:00:00
#PBS -N pn_grail
#PBS -J 0-999

# GRAIL on the graphTRIP permutation-null weights: 100 permutations x 10 training seeds.
#
# Array indices: PBS_ARRAY_INDEX = PERM * 10 + SEED, identical to graphtrip.sh, so
# launch.sh's --perms A-B maps onto -J the same way and --perms 0-0 gives a ten-element
# pilot. Submit through launch.sh with --eval grail, which selects this script and
# overrides -J. The evaluation flags live here rather than in EXTRA_ARGS, so nothing
# that qsub -v has to carry ever contains a space.
#
# Expected ~10 min per element: 42 patients x 7 folds x 69 biomarkers at 25 latent
# samples, with the spin permutation test skipped. Pilot with --perms 0-0 and read the PBS
# epilogue before committing the remaining 990 elements.

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
python -m scripts.permutation_null -m graphtrip -p $PERM -s $SEED \
    --eval_only --evaluations grail ${EXTRA_ARGS}
