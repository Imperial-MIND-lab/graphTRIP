#!/bin/bash
#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=06:00:00
#PBS -N perm_null
#PBS -J 0-199

# Load environment
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip

# Change to working directory
cd $PBS_O_WORKDIR

# Array indices:
# 0-99:    graphtrip, 10 permutations x 10 training seeds
# 100-199: medusa graphtrip, 10 permutations x 10 training seeds
#
# Within each block: PERM = (index % 100) / 10, SEED = index % 10.
# The 10 seeds of one permutation share its labels and are ensembled into one null draw.

if [ "$PBS_ARRAY_INDEX" -ge 0 ] && [ "$PBS_ARRAY_INDEX" -le 99 ]; then
    MODEL=graphtrip
    IDX=$((PBS_ARRAY_INDEX))
elif [ "$PBS_ARRAY_INDEX" -ge 100 ] && [ "$PBS_ARRAY_INDEX" -le 199 ]; then
    MODEL=medusa
    IDX=$((PBS_ARRAY_INDEX - 100))
fi

PERM=$((IDX / 10))
SEED=$((IDX % 10))
python -m scripts.permutation_null -m $MODEL -p $PERM -s $SEED
