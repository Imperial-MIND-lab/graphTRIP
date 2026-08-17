#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=04:00:00
#PBS -N temp_additional_graphtrip
#PBS -J 0-9

# Temporary top-up run of scripts/graphtrip.py for the two "frozen VGAE" analyses:
#   outputs/graphtrip/linreg_on_z/       (ridge head on [z, Condition])
#   outputs/graphtrip/retrain_mlp_on_z/  (new MLP head on [z, Condition])
#
# One job per seed: PBS_ARRAY_INDEX 0-9 -> seeds 0-9.

# Load environment
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip
cd ~/projects/graphTRIP/scripts

SEED=${PBS_ARRAY_INDEX}

python graphtrip.py -s $SEED
