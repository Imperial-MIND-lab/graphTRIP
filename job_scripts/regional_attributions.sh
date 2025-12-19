#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=6:00:00
#PBS -N regional_attributions_jobs
#PBS -J 0-19

# Regional attribution jobs.
# Each base job is run with 10 different seeds (0-9).
# Total: 2 versions × 10 seeds = 20 jobs (PBS_ARRAY_INDEX 0-19)

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

GRAPHTRIP_START=0
GRAPHTRIP_END=9
MEDUSA_START=10
MEDUSA_END=19

# Run the appropriate script based on PBS_ARRAY_INDEX
if [ $PBS_ARRAY_INDEX -ge $GRAPHTRIP_START ] && [ $PBS_ARRAY_INDEX -le $GRAPHTRIP_END ]; then
    # Measure regional attributions for graphTRIP
    SEED=$((PBS_ARRAY_INDEX - GRAPHTRIP_START))
    WEIGHTS_BASE_DIR='outputs/graphtrip/weights/'
    MLP_WEIGHTS_DIR='outputs/graphtrip/weights/'
    OUTPUT_DIR='outputs/graphtrip/regional_attributions/'
    python xai_regions.py -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR}

elif [ $PBS_ARRAY_INDEX -ge $MEDUSA_START ] && [ $PBS_ARRAY_INDEX -le $MEDUSA_END ]; then
    # Measure regional attributions for Medusa graphTRIP
    SEED=$((PBS_ARRAY_INDEX - MEDUSA_START))
    WEIGHTS_BASE_DIR='outputs/x_graphtrip/tlearner/'
    MLP_WEIGHTS_DIR='outputs/x_graphtrip/tlearner/'
    OUTPUT_DIR='outputs/x_graphtrip/regional_attributions/'
    python xai_regions.py -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --medusa

else
    echo "Invalid job index: $PBS_ARRAY_INDEX"
    exit 1
fi