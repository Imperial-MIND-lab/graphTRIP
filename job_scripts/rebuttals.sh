#!/bin/bash
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=08:00:00
#PBS -N rebuttals
#PBS -J 0-21

# Load environment
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip

# Change to working directory
cd $PBS_O_WORKDIR

# Array indices:
# 0:     graphtrip leakage test (seed 0)
# 1:     medusa leakage test (seed 0)
# 2-11:  ablation job 5 (no node features), seeds 0-9
# 12-21: ablation job 6 (no clinical features), seeds 0-9

if [ "$PBS_ARRAY_INDEX" -eq 0 ]; then
    python scripts/graphtrip_leakage_test.py -s 0
elif [ "$PBS_ARRAY_INDEX" -eq 1 ]; then
    python scripts/medusa_leakage_test.py -s 0
elif [ "$PBS_ARRAY_INDEX" -ge 2 ] && [ "$PBS_ARRAY_INDEX" -le 11 ]; then
    SEED=$((PBS_ARRAY_INDEX - 2))
    python scripts/ablation.py -j 5 -s $SEED
elif [ "$PBS_ARRAY_INDEX" -ge 12 ] && [ "$PBS_ARRAY_INDEX" -le 21 ]; then
    SEED=$((PBS_ARRAY_INDEX - 12))
    python scripts/ablation.py -j 6 -s $SEED
fi
