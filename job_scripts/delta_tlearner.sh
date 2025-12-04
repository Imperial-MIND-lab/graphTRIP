#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=08:00:00
#PBS -N delta_tlearner
#PBS -J 0-9

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/scripts

TLEARNER_CONFIG_FILE='experiments/configs/tlearner_delta.json'
OUTPUT_DIR='outputs/x_graphtrip_delta/'
SEED=${PBS_ARRAY_INDEX}

# Run the unsupervised VGAE training script
python x_graphtrip.py -t ${TLEARNER_CONFIG_FILE} -o ${OUTPUT_DIR} -s ${SEED} -v
