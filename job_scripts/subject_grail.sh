#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=6:00:00
#PBS -N subject_grail_jobs
#PBS -J 0-1679

# GRAIL interpretation jobs.
# Each base job is run with 10 different seeds (0-9) and 42 job IDs (0-41).
# Total: 4 versions × 42 jobs × 10 seeds = 1680 jobs (PBS_ARRAY_INDEX 0-1679)

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Define job ranges for each GRAIL version (each base job has 10 seeds)
# Interpretation for graphTRIP: 42 base jobs (0-41) × 10 seeds = 420 jobs (PBS_ARRAY_INDEX 0-419)
# Interpretation for x-graphTRIP medusa: 42 base jobs (0-41) × 10 seeds = 420 jobs (PBS_ARRAY_INDEX 420-839)
# Interpretation for x-graphTRIP escitalopram: 42 base jobs (0-41) × 10 seeds = 420 jobs (PBS_ARRAY_INDEX 840-1259)
# Interpretation for x-graphTRIP psilocybin: 42 base jobs (0-41) × 10 seeds = 420 jobs (PBS_ARRAY_INDEX 1260-1679)

INTERPRETATION_GRAPHTRIP_START=0
INTERPRETATION_GRAPHTRIP_END=419
INTERPRETATION_XGRAPHTRIP_MEDUSA_START=420
INTERPRETATION_XGRAPHTRIP_MEDUSA_END=839
INTERPRETATION_XGRAPHTRIP_ESCITALOPRAM_START=840
INTERPRETATION_XGRAPHTRIP_ESCITALOPRAM_END=1259
INTERPRETATION_XGRAPHTRIP_PSILOCYBIN_START=1260
INTERPRETATION_XGRAPHTRIP_PSILOCYBIN_END=1679

# Run the appropriate script based on PBS_ARRAY_INDEX
if [ $PBS_ARRAY_INDEX -ge $INTERPRETATION_GRAPHTRIP_START ] && [ $PBS_ARRAY_INDEX -le $INTERPRETATION_GRAPHTRIP_END ]; then
    # Interpretation for graphTRIP (42 base jobs × 10 seeds = 420 jobs; 0-419)
    # Calculate base job ID (0-41) and seed (0-9) from PBS_ARRAY_INDEX
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - INTERPRETATION_GRAPHTRIP_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/graphtrip/weights/'
    MLP_WEIGHTS_DIR='outputs/graphtrip/weights/'
    OUTPUT_DIR='outputs/graphtrip/grail/'
    python xai_biomarkers.py -j ${JOBID} -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR}

elif [ $PBS_ARRAY_INDEX -ge $INTERPRETATION_XGRAPHTRIP_MEDUSA_START ] && [ $PBS_ARRAY_INDEX -le $INTERPRETATION_XGRAPHTRIP_MEDUSA_END ]; then
    # Interpretation for x-graphTRIP medusa (42 base jobs × 10 seeds = 420 jobs; 420-839)
    # Calculate base job ID (0-41) and seed (0-9) from PBS_ARRAY_INDEX
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - INTERPRETATION_XGRAPHTRIP_MEDUSA_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/x_graphtrip/tlearner/'
    MLP_WEIGHTS_DIR='outputs/x_graphtrip/tlearner/'
    OUTPUT_DIR='outputs/x_graphtrip/medusa_grail/'
    GRAIL_MODE='medusa'
    python xai_biomarkers.py -j ${JOBID} -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --grail_mode ${GRAIL_MODE}

elif [ $PBS_ARRAY_INDEX -ge $INTERPRETATION_XGRAPHTRIP_ESCITALOPRAM_START ] && [ $PBS_ARRAY_INDEX -le $INTERPRETATION_XGRAPHTRIP_ESCITALOPRAM_END ]; then
    # Interpretation for x-graphTRIP escitalopram (42 base jobs × 10 seeds = 420 jobs; 840-1259)
    # Calculate base job ID (0-41) and seed (0-9) from PBS_ARRAY_INDEX
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - INTERPRETATION_XGRAPHTRIP_ESCITALOPRAM_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/x_graphtrip/tlearner/'
    MLP_WEIGHTS_DIR='outputs/x_graphtrip/tlearner/'
    OUTPUT_DIR='outputs/x_graphtrip/grail_escitalopram/'
    GRAIL_MODE='escitalopram'
    python xai_biomarkers.py -j ${JOBID} -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --grail_mode ${GRAIL_MODE} 

elif [ $PBS_ARRAY_INDEX -ge $INTERPRETATION_XGRAPHTRIP_PSILOCYBIN_START ] && [ $PBS_ARRAY_INDEX -le $INTERPRETATION_XGRAPHTRIP_PSILOCYBIN_END ]; then
    # Interpretation for x-graphTRIP psilocybin (42 base jobs × 10 seeds = 420 jobs; 1260-1679)
    # Calculate base job ID (0-41) and seed (0-9) from PBS_ARRAY_INDEX
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - INTERPRETATION_XGRAPHTRIP_PSILOCYBIN_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/x_graphtrip/tlearner/'
    MLP_WEIGHTS_DIR='outputs/x_graphtrip/tlearner/'
    OUTPUT_DIR='outputs/x_graphtrip/grail_psilocybin/'
    GRAIL_MODE='psilocybin'
    python xai_biomarkers.py -j ${JOBID} -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --grail_mode ${GRAIL_MODE} 

else
    echo "Invalid job index: $PBS_ARRAY_INDEX"
    exit 1
fi