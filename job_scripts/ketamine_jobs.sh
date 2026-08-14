#!/bin/bash
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=04:00:00
#PBS -N ketamine
#PBS -J 0-5

# Load environment
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip

# Change to working directory
cd $PBS_O_WORKDIR

# Array index = config_id in experiments/configs/ketamine.json.
# The grid is target x graph_attrs, with graph_attrs varying fastest:
#   0: MADRS_d2   MADRS_b0, HAM17_b0                     (n=29)
#   1: MADRS_d2   + infusion_order                       (n=29)
#   2: MADRS_d2   + infusion_order, Sex                  (n=29)
#   3: MADRS_d10  MADRS_b0, HAM17_b0                     (n=26)
#   4: MADRS_d10  + infusion_order                       (n=26)
#   5: MADRS_d10  + infusion_order, Sex                  (n=26)
#
# Phase 2 -- 10 seeds for the winning arm -- change the header and the two
# assignments below to:
#   #PBS -J 0-9
#   CONFIG_ID=<winner>
#   SEED=$PBS_ARRAY_INDEX
CONFIG_ID=$PBS_ARRAY_INDEX
SEED=0

python -m scripts.ketamine \
  -c experiments/configs/ketamine.json \
  -o outputs/ketamine/ \
  -ci $CONFIG_ID \
  -s $SEED \
  -v
