#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=08:00:00
#PBS -N secondary_scripts
#PBS -J 0-1699

# Secondary scripts do not depend on primary scripts.
# Each base job is run with 10 different seeds (0-9).

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Define job ranges for each script (each base job has 10 seeds)
# Validation: 2 base jobs (0-1) -> 20 jobs (PBS_ARRAY_INDEX 0-19)
# Interpretation for graphTRIP: 42 base jobs (0-41) × 10 seeds = 420 jobs (PBS_ARRAY_INDEX 20-439)
# Interpretation for medusa: 42 base jobs (0-41) × 10 seeds = 420 jobs (PBS_ARRAY_INDEX 440-859)
# Interpretation for medusa escitalopram: 42 base jobs (0-41) × 10 seeds = 420 jobs (PBS_ARRAY_INDEX 860-1279)
# Interpretation for medusa psilocybin: 42 base jobs (0-41) × 10 seeds = 420 jobs (PBS_ARRAY_INDEX 1280-1699)

VALIDATION_START=0
VALIDATION_END=19
GRAIL_GRAPHTRIP_START=20
GRAIL_GRAPHTRIP_END=439
GRAIL_MEDUSA_START=440
GRAIL_MEDUSA_END=859
GRAIL_ESCITALOPRAM_START=860
GRAIL_ESCITALOPRAM_END=1279
GRAIL_PSILOCYBIN_START=1280
GRAIL_PSILOCYBIN_END=1699

# Run the appropriate script based on PBS_ARRAY_INDEX
if [ $PBS_ARRAY_INDEX -ge $VALIDATION_START ] && [ $PBS_ARRAY_INDEX -le $VALIDATION_END ]; then
    # Validation (2 base jobs × 10 seeds = 20 jobs; 0-19)
    # Calculate base job ID (0-1) and seed (0-9) from PBS_ARRAY_INDEX
    JOBID=$((PBS_ARRAY_INDEX / 10))
    SEED=$((PBS_ARRAY_INDEX % 10))
    python validation.py -s ${SEED} -j ${JOBID}

elif [ $PBS_ARRAY_INDEX -ge $GRAIL_GRAPHTRIP_START ] && [ $PBS_ARRAY_INDEX -le $GRAIL_GRAPHTRIP_END ]; then
    # Interpretation for graphTRIP (42 base jobs × 10 seeds = 420 jobs; 20-439)
    # Calculate base job ID (0-41) and seed (0-9) from PBS_ARRAY_INDEX
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - GRAIL_GRAPHTRIP_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/graphtrip/weights/'
    MLP_WEIGHTS_DIR='outputs/graphtrip/weights/'
    OUTPUT_DIR='outputs/graphtrip/grail/'
    python xai_biomarkers.py -j ${JOBID} -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR}

elif [ $PBS_ARRAY_INDEX -ge $GRAIL_MEDUSA_START ] && [ $PBS_ARRAY_INDEX -le $GRAIL_MEDUSA_END ]; then
    # Interpretation for medusa graphTRIP (42 base jobs × 10 seeds = 420 jobs; 440-859)
    # Calculate base job ID (0-41) and seed (0-9) from PBS_ARRAY_INDEX
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - GRAIL_MEDUSA_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/medusa_graphtrip/weights/'
    MLP_WEIGHTS_DIR='outputs/medusa_graphtrip/weights/'
    OUTPUT_DIR='outputs/medusa_graphtrip/grail/'
    GRAIL_MODE='medusa'
    python xai_biomarkers.py -j ${JOBID} -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --grail_mode ${GRAIL_MODE}

elif [ $PBS_ARRAY_INDEX -ge $GRAIL_ESCITALOPRAM_START ] && [ $PBS_ARRAY_INDEX -le $GRAIL_ESCITALOPRAM_END ]; then
    # Interpretation for medusa graphTRIP escitalopram (42 base jobs × 10 seeds = 420 jobs; 860-1279)
    # Calculate base job ID (0-41) and seed (0-9) from PBS_ARRAY_INDEX
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - GRAIL_ESCITALOPRAM_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/medusa_graphtrip/weights/'
    MLP_WEIGHTS_DIR='outputs/medusa_graphtrip/weights/'
    OUTPUT_DIR='outputs/medusa_graphtrip/grail_escitalopram/'
    GRAIL_MODE='escitalopram'
    python xai_biomarkers.py -j ${JOBID} -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --grail_mode ${GRAIL_MODE} 

elif [ $PBS_ARRAY_INDEX -ge $GRAIL_PSILOCYBIN_START ] && [ $PBS_ARRAY_INDEX -le $GRAIL_PSILOCYBIN_END ]; then
    # Interpretation for medusa graphTRIP psilocybin (42 base jobs × 10 seeds = 420 jobs; 1280-1699)
    # Calculate base job ID (0-41) and seed (0-9) from PBS_ARRAY_INDEX
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - GRAIL_PSILOCYBIN_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/medusa_graphtrip/weights/'
    MLP_WEIGHTS_DIR='outputs/medusa_graphtrip/weights/'
    OUTPUT_DIR='outputs/medusa_graphtrip/grail_psilocybin/'
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