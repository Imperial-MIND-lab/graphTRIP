#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=08:00:00
#PBS -N grail_psilodep1
#PBS -J 0-319

# Load environment
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip
cd ~/projects/graphTRIP/scripts

# Job ranges (each base job is one patient, run with 10 seeds)
# graphTRIP:               16 base jobs (0-15) x 10 seeds = 160 jobs (PBS_ARRAY_INDEX 0-159)
# medusa psilocybin head:  16 base jobs (0-15) x 10 seeds = 160 jobs (PBS_ARRAY_INDEX 160-319)

GRAIL_GRAPHTRIP_START=0
GRAIL_GRAPHTRIP_END=159
GRAIL_MEDUSA_START=160
GRAIL_MEDUSA_END=319

EVAL_STUDY='psilodep1'
EVAL_TARGET='QIDS_1week'
HARMONISE='QIDS_Before BDI_Before'

if [ $PBS_ARRAY_INDEX -ge $GRAIL_GRAPHTRIP_START ] && [ $PBS_ARRAY_INDEX -le $GRAIL_GRAPHTRIP_END ]; then
    # graphTRIP on psilodep1
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - GRAIL_GRAPHTRIP_START))
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

elif [ $PBS_ARRAY_INDEX -ge $GRAIL_MEDUSA_START ] && [ $PBS_ARRAY_INDEX -le $GRAIL_MEDUSA_END ]; then
    # medusa graphTRIP on psilodep1
    RELATIVE_INDEX=$((PBS_ARRAY_INDEX - GRAIL_MEDUSA_START))
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
