#!/bin/bash

#PBS -l select=1:ncpus=16:mem=8gb
#PBS -l walltime=12:00:00
#PBS -N speedy_grail_jobs
#PBS -J 0-279

# GRAIL interpretation jobs.
# Each base job is run with 10 different seeds (0-9) and 7 job IDs (0-6).
# Total: 4 versions × 7 jobs × 10 seeds = 280 jobs (PBS_ARRAY_INDEX 0-279)

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Define job ranges for each GRAIL version (each base job has 10 seeds)
# Interpretation for graphTRIP: 7 base jobs (0-6) × 10 seeds = 70 jobs (PBS_ARRAY_INDEX 0-69)
# Interpretation for x-graphTRIP medusa: 7 base jobs (0-6) × 10 seeds = 70 jobs (PBS_ARRAY_INDEX 70-139)
# Interpretation for x-graphTRIP escitalopram: 7 base jobs (0-6) × 10 seeds = 70 jobs (PBS_ARRAY_INDEX 140-209)
# Interpretation for x-graphTRIP psilocybin: 7 base jobs (0-6) × 10 seeds = 70 jobs (PBS_ARRAY_INDEX 210-279)

INTERPRETATION_GRAPHTRIP_START=0
INTERPRETATION_GRAPHTRIP_END=69
INTERPRETATION_XGRAPHTRIP_MEDUSA_START=70
INTERPRETATION_XGRAPHTRIP_MEDUSA_END=139
INTERPRETATION_XGRAPHTRIP_ESCITALOPRAM_START=140
INTERPRETATION_XGRAPHTRIP_ESCITALOPRAM_END=209
INTERPRETATION_XGRAPHTRIP_PSILOCYBIN_START=210
INTERPRETATION_XGRAPHTRIP_PSILOCYBIN_END=279

# Resource limits
N_WORKERS=4 
MAX_MEMORY_GB=7      
CPU_LIMIT=3 

# Run the appropriate script based on PBS_ARRAY_INDEX
if [ $PBS_ARRAY_INDEX -ge $INTERPRETATION_GRAPHTRIP_START ] && [ $PBS_ARRAY_INDEX -le $INTERPRETATION_GRAPHTRIP_END ]; then
    # Interpretation for graphTRIP (7 base jobs × 10 seeds = 70 jobs; 0-69)
    # Calculate base job ID (0-6) and seed (0-9) from PBS_ARRAY_INDEX
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - INTERPRETATION_GRAPHTRIP_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/graphtrip/weights/'
    MLP_WEIGHTS_DIR='outputs/graphtrip/weights/'
    OUTPUT_DIR='outputs/graphtrip/grail_1000_perms/'
    python interpretation.py -j ${JOBID} -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --n_workers ${N_WORKERS} \
        --max_memory_gb ${MAX_MEMORY_GB} \
        --cpu_limit ${CPU_LIMIT}

elif [ $PBS_ARRAY_INDEX -ge $INTERPRETATION_XGRAPHTRIP_MEDUSA_START ] && [ $PBS_ARRAY_INDEX -le $INTERPRETATION_XGRAPHTRIP_MEDUSA_END ]; then
    # Interpretation for x-graphTRIP medusa (7 base jobs × 10 seeds = 70 jobs; 70-139)
    # Calculate base job ID (0-6) and seed (0-9) from PBS_ARRAY_INDEX
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - INTERPRETATION_XGRAPHTRIP_MEDUSA_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/x_graphtrip/tlearner/'
    MLP_WEIGHTS_DIR='outputs/x_graphtrip/tlearner/'
    OUTPUT_DIR='outputs/x_graphtrip/medusa_grail_1000_perms/'
    GRAIL_MODE='medusa'
    python interpretation.py -j ${JOBID} -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --grail_mode ${GRAIL_MODE} \
        --n_workers ${N_WORKERS} \
        --max_memory_gb ${MAX_MEMORY_GB} \
        --cpu_limit ${CPU_LIMIT}

elif [ $PBS_ARRAY_INDEX -ge $INTERPRETATION_XGRAPHTRIP_ESCITALOPRAM_START ] && [ $PBS_ARRAY_INDEX -le $INTERPRETATION_XGRAPHTRIP_ESCITALOPRAM_END ]; then
    # Interpretation for x-graphTRIP escitalopram (7 base jobs × 10 seeds = 70 jobs; 140-209)
    # Calculate base job ID (0-6) and seed (0-9) from PBS_ARRAY_INDEX
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - INTERPRETATION_XGRAPHTRIP_ESCITALOPRAM_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/x_graphtrip/tlearner/'
    MLP_WEIGHTS_DIR='outputs/x_graphtrip/tlearner/'
    OUTPUT_DIR='outputs/x_graphtrip/grail_escitalopram_1000_perms/'
    GRAIL_MODE='escitalopram'
    python interpretation.py -j ${JOBID} -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --grail_mode ${GRAIL_MODE} \
        --n_workers ${N_WORKERS} \
        --max_memory_gb ${MAX_MEMORY_GB} \
        --cpu_limit ${CPU_LIMIT}

elif [ $PBS_ARRAY_INDEX -ge $INTERPRETATION_XGRAPHTRIP_PSILOCYBIN_START ] && [ $PBS_ARRAY_INDEX -le $INTERPRETATION_XGRAPHTRIP_PSILOCYBIN_END ]; then
    # Interpretation for x-graphTRIP psilocybin (7 base jobs × 10 seeds = 70 jobs; 210-279)
    # Calculate base job ID (0-6) and seed (0-9) from PBS_ARRAY_INDEX
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - INTERPRETATION_XGRAPHTRIP_PSILOCYBIN_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/x_graphtrip/tlearner/'
    MLP_WEIGHTS_DIR='outputs/x_graphtrip/tlearner/'
    OUTPUT_DIR='outputs/x_graphtrip/grail_psilocybin_1000_perms/'
    GRAIL_MODE='psilocybin'
    python interpretation.py -j ${JOBID} -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --grail_mode ${GRAIL_MODE} \
        --n_workers ${N_WORKERS} \
        --max_memory_gb ${MAX_MEMORY_GB} \
        --cpu_limit ${CPU_LIMIT}

else
    echo "Invalid job index: $PBS_ARRAY_INDEX"
    exit 1
fi