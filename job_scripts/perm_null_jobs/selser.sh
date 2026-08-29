#!/bin/bash
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=01:30:00
#PBS -N pn_selser
#PBS -J 0-99

# Permutation null for the SELSER-fMRI baseline: 100 permutations x 10 CV splits.
#
# One array element per permutation seed: PBS_ARRAY_INDEX = PERM, seeds 0-9 run inside it.
# Submit through launch.sh, which overrides -J for a permutation subrange.
#
# One SELSER fit is ~24 s of CPU but ~95 s of walltime on the cluster: each invocation
# restarts Python, re-imports the scientific stack and re-reads 42 BOLD timeseries from
# RDS. Run sequentially over all 1000 that overhead dominates and overruns any sane
# walltime; spread over 100 elements each element is ~16 min.
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

CONFIG=experiments/configs/selser.json
OUTPUT_BASE=outputs/selser/permutation_null
PERM=$PBS_ARRAY_INDEX

echo "SELSER permutation null: perm_seed ${PERM}, seeds 0-9."

for SEED in $(seq 0 9); do
    python -m scripts.train_selser \
        --config "$CONFIG" \
        --seed "$SEED" \
        --perm_seed "$PERM" \
        --output_dir "${OUTPUT_BASE}/perm_${PERM}/seed_${SEED}"
done
