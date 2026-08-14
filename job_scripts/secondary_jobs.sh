#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=08:00:00
#PBS -N secondary_scripts
#PBS -J 0-2009

# Secondary scripts do not depend on primary scripts.
# Each base job is run with 10 different seeds (0-9).

# Load environment
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip
cd ~/projects/graphTRIP/scripts

# Define job ranges for each script (each base job has 10 seeds)
# Validation: 1 base job (0) -> 10 jobs (PBS_ARRAY_INDEX 0-9)
# Interpretation for graphTRIP: 42 base jobs (0-41) × 10 seeds = 420 jobs (PBS_ARRAY_INDEX 10-429)
# Interpretation for medusa: 42 base jobs (0-41) × 10 seeds = 420 jobs (PBS_ARRAY_INDEX 430-849)
# Interpretation for medusa escitalopram: 42 base jobs (0-41) × 10 seeds = 420 jobs (PBS_ARRAY_INDEX 850-1269)
# Interpretation for medusa psilocybin: 42 base jobs (0-41) × 10 seeds = 420 jobs (PBS_ARRAY_INDEX 1270-1689)
# GRAIL on psilodep1, graphTRIP: 16 base jobs (0-15) × 10 seeds = 160 jobs (PBS_ARRAY_INDEX 1690-1849)
# GRAIL on psilodep1, medusa:    16 base jobs (0-15) × 10 seeds = 160 jobs (PBS_ARRAY_INDEX 1850-2009)

VALIDATION_START=0
VALIDATION_END=9
GRAIL_GRAPHTRIP_START=10
GRAIL_GRAPHTRIP_END=429
GRAIL_MEDUSA_START=430
GRAIL_MEDUSA_END=849
GRAIL_ESCITALOPRAM_START=850
GRAIL_ESCITALOPRAM_END=1269
GRAIL_PSILOCYBIN_START=1270
GRAIL_PSILOCYBIN_END=1689
GRAIL_P1_GRAPHTRIP_START=1690
GRAIL_P1_GRAPHTRIP_END=1849
GRAIL_P1_MEDUSA_START=1850
GRAIL_P1_MEDUSA_END=2009

# psilodep1 zero-shot GRAIL settings (see job_scripts/grail_psilodep1.sh)
EVAL_STUDY='psilodep1'
EVAL_TARGET='QIDS_1week'
HARMONISE='QIDS_Before BDI_Before'

# Run the appropriate script based on PBS_ARRAY_INDEX
if [ $PBS_ARRAY_INDEX -ge $VALIDATION_START ] && [ $PBS_ARRAY_INDEX -le $VALIDATION_END ]; then
    # Validation (1 base job × 10 seeds = 10 jobs; 0-9)
    SEED=$((PBS_ARRAY_INDEX))
    python validation.py -s ${SEED}

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

elif [ $PBS_ARRAY_INDEX -ge $GRAIL_P1_GRAPHTRIP_START ] && [ $PBS_ARRAY_INDEX -le $GRAIL_P1_GRAPHTRIP_END ]; then
    # GRAIL for graphTRIP on psilodep1 (16 base jobs × 10 seeds = 160 jobs; 1690-1849)
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - GRAIL_P1_GRAPHTRIP_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/graphtrip/weights/'
    MLP_WEIGHTS_DIR='outputs/graphtrip/weights/'
    OUTPUT_DIR='outputs/graphtrip/grail_psilodep1/'
    python xai_biomarkers.py -j ${JOBID} -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --eval_study ${EVAL_STUDY} \
        --eval_target ${EVAL_TARGET} \
        --harmonise ${HARMONISE}

elif [ $PBS_ARRAY_INDEX -ge $GRAIL_P1_MEDUSA_START ] && [ $PBS_ARRAY_INDEX -le $GRAIL_P1_MEDUSA_END ]; then
    # GRAIL for medusa graphTRIP on psilodep1 (16 base jobs × 10 seeds = 160 jobs; 1850-2009)
    # All psilodep1 patients received psilocybin, so the CFRHead routes every prediction
    # through the psilocybin arm on the real treatment vector; drug_condition stays None.
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - GRAIL_P1_MEDUSA_START))
    JOBID=$((RELATIVE_INDEX / 10))
    SEED=$((RELATIVE_INDEX % 10))
    WEIGHTS_BASE_DIR='outputs/medusa_graphtrip/weights/'
    MLP_WEIGHTS_DIR='outputs/medusa_graphtrip/weights/'
    OUTPUT_DIR='outputs/medusa_graphtrip/grail_psilocybin_psilodep1/'
    python xai_biomarkers.py -j ${JOBID} -s ${SEED} -v \
        --weights_base_dir ${WEIGHTS_BASE_DIR} \
        --mlp_weights_dir ${MLP_WEIGHTS_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --eval_study ${EVAL_STUDY} \
        --eval_target ${EVAL_TARGET} \
        --harmonise ${HARMONISE}

else
    echo "Invalid job index: $PBS_ARRAY_INDEX"
    exit 1
fi