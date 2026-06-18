#!/bin/bash
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=04:00:00
#PBS -N rebuttals
#PBS -J 40-59

# Load environment
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip

# Change to working directory
cd $PBS_O_WORKDIR

# Array indices:
# 0-9:   graphtrip leakage test, seeds 0-9
# 10-19: medusa leakage test, seeds 0-9
# 20-29: ablation job 5 (no node features), seeds 0-9
# 30-39: ablation job 6 (no clinical features), seeds 0-9
# 40-49: ablation job 7 (medusa no node features), seeds 0-9
# 50-59: ablation job 8 (medusa no clinical features), seeds 0-9

if [ "$PBS_ARRAY_INDEX" -ge 0 ] && [ "$PBS_ARRAY_INDEX" -le 9 ]; then
    SEED=$((PBS_ARRAY_INDEX))
    python -m scripts.graphtrip_leakage_test -s $SEED
elif [ "$PBS_ARRAY_INDEX" -ge 10 ] && [ "$PBS_ARRAY_INDEX" -le 19 ]; then
    SEED=$((PBS_ARRAY_INDEX - 10))
    python -m scripts.medusa_leakage_test -s $SEED
elif [ "$PBS_ARRAY_INDEX" -ge 20 ] && [ "$PBS_ARRAY_INDEX" -le 29 ]; then
    SEED=$((PBS_ARRAY_INDEX - 20))
    python -m scripts.ablation -j 5 -s $SEED
elif [ "$PBS_ARRAY_INDEX" -ge 30 ] && [ "$PBS_ARRAY_INDEX" -le 39 ]; then
    SEED=$((PBS_ARRAY_INDEX - 30))
    python -m scripts.ablation -j 6 -s $SEED
elif [ "$PBS_ARRAY_INDEX" -ge 40 ] && [ "$PBS_ARRAY_INDEX" -le 49 ]; then
    SEED=$((PBS_ARRAY_INDEX - 40))
    python -m scripts.ablation -j 7 -s $SEED
elif [ "$PBS_ARRAY_INDEX" -ge 50 ] && [ "$PBS_ARRAY_INDEX" -le 59 ]; then
    SEED=$((PBS_ARRAY_INDEX - 50))
    python -m scripts.ablation -j 8 -s $SEED
fi
