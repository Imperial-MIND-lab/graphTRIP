#!/bin/bash
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=12:00:00
#PBS -N pn_selser

# Permutation null for the SELSER-fMRI baseline: 100 permutations x 10 CV splits.
#
# Not an array job. One SELSER fit takes ~14 s single-threaded, so all 1000 runs finish in
# about four hours in a single sequential job -- less than one graphTRIP permutation.
# Submit through launch.sh; --perms sets PERM_START/PERM_END.
#
# SELSER has no weight initialisation or batch stochasticity, so --seed selects the
# cross-validation fold split only. The perm_seed matches the one used by
# scripts/permutation_null.py, so the SELSER and graphTRIP nulls are paired.

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

PERM_START=${PERM_START:-0}
PERM_END=${PERM_END:-99}
CONFIG=experiments/configs/selser.json
OUTPUT_BASE=outputs/selser/permutation_null

echo "SELSER permutation null: perm_seeds ${PERM_START}-${PERM_END}, seeds 0-9."

for PERM in $(seq "$PERM_START" "$PERM_END"); do
    for SEED in $(seq 0 9); do
        python -m scripts.train_selser \
            --config "$CONFIG" \
            --seed "$SEED" \
            --perm_seed "$PERM" \
            --output_dir "${OUTPUT_BASE}/perm_${PERM}/seed_${SEED}"
    done
done
