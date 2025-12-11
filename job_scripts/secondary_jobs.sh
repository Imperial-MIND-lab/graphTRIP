#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=08:00:00
#PBS -N secondary_scripts
#PBS -J 0-229

# Secondary scripts do not depend on primary scripts.
# Each base job is run with 10 different seeds (0-9).

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Define job ranges for each script (each base job has 10 seeds)
# Validation: 2 base jobs (0-1) -> 20 jobs (PBS_ARRAY_INDEX 0-19)
# Interpretation for graphTRIP: 7 base jobs (0-6) -> 70 jobs (PBS_ARRAY_INDEX 20-89)
# Interpretation for x-graphTRIP: 7 base jobs (0-6) -> 70 jobs (PBS_ARRAY_INDEX 90-159)
# Interpretation for x-graphTRIP (vgae/cate): 7 base jobs (0-6) -> 70 jobs (PBS_ARRAY_INDEX 160-229)

VALIDATION_START=0
VALIDATION_END=19
INTERPRETATION_GRAPHTRIP_START=20
INTERPRETATION_GRAPHTRIP_END=89
INTERPRETATION_XGRAPHTRIP_START=90
INTERPRETATION_XGRAPHTRIP_END=159
INTERPRETATION_XGRAPHTRIP_VGAE_START=160
INTERPRETATION_XGRAPHTRIP_VGAE_END=229

# Run the appropriate script based on PBS_ARRAY_INDEX
if [ $PBS_ARRAY_INDEX -ge $VALIDATION_START ] && [ $PBS_ARRAY_INDEX -le $VALIDATION_END ]; then
    # Validation (2 base jobs × 10 seeds = 20 jobs; 0-19)
    # Calculate base job ID (0-1) and seed (0-9) from PBS_ARRAY_INDEX
    JOBID=$((PBS_ARRAY_INDEX / 10))
    SEED=$((PBS_ARRAY_INDEX % 10))
    python validation.py -s ${SEED} -j ${JOBID}

elif [ $PBS_ARRAY_INDEX -ge $INTERPRETATION_GRAPHTRIP_START ] && [ $PBS_ARRAY_INDEX -le $INTERPRETATION_GRAPHTRIP_END ]; then
    # Interpretation for graphTRIP (7 base jobs × 10 seeds = 70 jobs; 20-89)
    # Calculate base job ID (0-6) and seed (0-9) from PBS_ARRAY_INDEX
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - INTERPRETATION_GRAPHTRIP_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/graphtrip/weights/'
    MLP_WEIGHTS_DIR='outputs/graphtrip/weights/'
    OUTPUT_DIR='outputs/graphtrip/grail/'
    python interpretation.py -j ${JOBID} -s ${SEED} -v --weights_base_dir ${WEIGHTS_BASE_DIR} --mlp_weights_dir ${MLP_WEIGHTS_DIR} --output_dir ${OUTPUT_DIR}

elif [ $PBS_ARRAY_INDEX -ge $INTERPRETATION_XGRAPHTRIP_START ] && [ $PBS_ARRAY_INDEX -le $INTERPRETATION_XGRAPHTRIP_END ]; then
    # Interpretation for x-graphTRIP (7 base jobs × 10 seeds = 70 jobs; 90-159)
    # Calculate base job ID (0-6) and seed (0-9) from PBS_ARRAY_INDEX
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - INTERPRETATION_XGRAPHTRIP_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/x_graphtrip/tlearner/'
    MLP_WEIGHTS_DIR='outputs/x_graphtrip/tlearner/'
    OUTPUT_DIR='outputs/x_graphtrip/medusa_grail/'
    python interpretation.py -j ${JOBID} -s ${SEED} -v --weights_base_dir ${WEIGHTS_BASE_DIR} --mlp_weights_dir ${MLP_WEIGHTS_DIR} --output_dir ${OUTPUT_DIR}

elif [ $PBS_ARRAY_INDEX -ge $INTERPRETATION_XGRAPHTRIP_VGAE_START ] && [ $PBS_ARRAY_INDEX -le $INTERPRETATION_XGRAPHTRIP_VGAE_END ]; then
    # Interpretation for x-graphTRIP (vgae/cate) (7 base jobs × 10 seeds = 70 jobs; 160-229)
    # Calculate base job ID (0-6) and seed (0-9) from PBS_ARRAY_INDEX
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - INTERPRETATION_XGRAPHTRIP_VGAE_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/x_graphtrip/vgae_weights'
    MLP_WEIGHTS_DIR='outputs/x_graphtrip/cate_model'
    OUTPUT_DIR='outputs/x_graphtrip/grail'
    python interpretation.py -j ${JOBID} -s ${SEED} -v --weights_base_dir ${WEIGHTS_BASE_DIR} --mlp_weights_dir ${MLP_WEIGHTS_DIR} --output_dir ${OUTPUT_DIR}

else
    echo "Invalid job index: $PBS_ARRAY_INDEX"
    exit 1
fi