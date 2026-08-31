#!/bin/bash
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=01:00:00
#PBS -N pn_medusa_grail
#PBS -J 0-999

# GRAIL on the Medusa permutation-null weights: 100 permutations x 10 training seeds.
#
# Array indices: PBS_ARRAY_INDEX = PERM * 10 + SEED, identical to medusa.sh, so launch.sh's
# --perms A-B maps onto -J the same way and --perms 0-0 gives a ten-element pilot. Submit
# through launch.sh with --eval grail, which selects this script and overrides -J.
#
# All three CFRHead outputs -- the escitalopram head, the psilocybin head and their
# difference (ITE) -- are computed in ONE pass. They share the latent samples, the decoder
# reconstructions and the 69 feature gradients. The output carries a grail_mode column, 
# so it has three rows per (subject, fold).

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
python -m scripts.permutation_null -m medusa -p $PERM -s $SEED \
    --eval_only --evaluations grail ${EXTRA_ARGS}
