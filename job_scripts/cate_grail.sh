#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=04:00:00
#PBS -N interpret_cate_grail
#PBS -J 0-69

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/scripts

# Calculate job_id (0-6) and seed (0-9) from array index (0-69)
# job_id = array_index // 10
# seed = array_index % 10
SEEDS=(0 1 2 3 4 5 6 7 8 9)
MAX_NUM_CONFIGS=7

JOBID=$((PBS_ARRAY_INDEX % MAX_NUM_CONFIGS))
SEED_INDEX=$((PBS_ARRAY_INDEX / MAX_NUM_CONFIGS))
SEED=${SEEDS[$SEED_INDEX]}
echo "JOBID: ${JOBID}, SEED: ${SEED}"

# x-graphTRIP ---------------------------------------------------------------
WEIGHTS_BASE_DIR='outputs/x_graphtrip/vgae_weights/'
MLP_WEIGHTS_DIR='outputs/x_graphtrip/cate_model/'
OUTPUT_DIR='outputs/x_graphtrip/grail/'

python interpret.py -j ${JOBID} -s ${SEED} -v --weights_base_dir ${WEIGHTS_BASE_DIR} --mlp_weights_dir ${MLP_WEIGHTS_DIR} --output_dir ${OUTPUT_DIR}

