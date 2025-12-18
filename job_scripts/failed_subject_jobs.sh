#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=6:00:00
#PBS -N subject_grail_jobs
#PBS -J 0-9

# GRAIL interpretation jobs - re-running failed job-seed combinations only.
# Total: 7 graphTRIP combinations + 3 x-graphTRIP medusa combinations = 10 jobs (PBS_ARRAY_INDEX 0-9)

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Define job ranges for each GRAIL version
# Interpretation for graphTRIP: 7 specific combinations (PBS_ARRAY_INDEX 0-6)
# Interpretation for x-graphTRIP medusa: 3 specific combinations (PBS_ARRAY_INDEX 7-9)

INTERPRETATION_GRAPHTRIP_START=0
INTERPRETATION_GRAPHTRIP_END=6
INTERPRETATION_XGRAPHTRIP_MEDUSA_START=7
INTERPRETATION_XGRAPHTRIP_MEDUSA_END=9

# Define job-seed combinations for graphTRIP (PBS_ARRAY_INDEX 0-6)
# Format: array of "job_seed" strings
GRAPHTRIP_COMBINATIONS=("29_1" "35_4" "25_5" "29_5" "27_6" "26_9" "28_9")

# Define job-seed combinations for x-graphTRIP medusa (PBS_ARRAY_INDEX 7-9)
XGRAPHTRIP_MEDUSA_COMBINATIONS=("5_1" "8_8" "9_8")

# Run the appropriate script based on PBS_ARRAY_INDEX
if [ $PBS_ARRAY_INDEX -ge $INTERPRETATION_GRAPHTRIP_START ] && [ $PBS_ARRAY_INDEX -le $INTERPRETATION_GRAPHTRIP_END ]; then
    # Map PBS_ARRAY_INDEX (0-6) directly to graphTRIP combinations
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - INTERPRETATION_GRAPHTRIP_START))
    COMBINATION="${GRAPHTRIP_COMBINATIONS[$RELATIVE_INDEX]}"
    IFS='_' read -r JOBID SEED <<< "$COMBINATION"
    
    WEIGHTS_BASE_DIR='outputs/graphtrip/weights/'
    MLP_WEIGHTS_DIR='outputs/graphtrip/weights/'
    OUTPUT_DIR='outputs/graphtrip/grail/'
    python interpretation.py -j ${JOBID} -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR}

elif [ $PBS_ARRAY_INDEX -ge $INTERPRETATION_XGRAPHTRIP_MEDUSA_START ] && [ $PBS_ARRAY_INDEX -le $INTERPRETATION_XGRAPHTRIP_MEDUSA_END ]; then
    # Map PBS_ARRAY_INDEX (7-9) directly to x-graphTRIP medusa combinations
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - INTERPRETATION_XGRAPHTRIP_MEDUSA_START))
    COMBINATION="${XGRAPHTRIP_MEDUSA_COMBINATIONS[$RELATIVE_INDEX]}"
    IFS='_' read -r JOBID SEED <<< "$COMBINATION"
    
    WEIGHTS_BASE_DIR='outputs/x_graphtrip/tlearner/'
    MLP_WEIGHTS_DIR='outputs/x_graphtrip/tlearner/'
    OUTPUT_DIR='outputs/x_graphtrip/medusa_grail/'
    GRAIL_MODE='medusa'
    python interpretation.py -j ${JOBID} -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --grail_mode ${GRAIL_MODE}

else
    echo "Invalid job index: $PBS_ARRAY_INDEX"
    exit 1
fi