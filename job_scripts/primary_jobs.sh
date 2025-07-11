#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=8:00:00
#PBS -N primary_jobs
#PBS -J 0-62

# Primary scripts do not depend on other scripts.

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Define job ranges for each script
ABLATION_START=0
ABLATION_END=2
ATLAS_BOUND=3
BENCHMARKS_START=4
BENCHMARKS_END=6
DRUG_CLASSIFIER=7
GRAPHTRIP_BDI=8
GRAPHTRIP_MAIN=9
PSILODEP1_FINETUNING=10
TLEARNER_ESCITALOPRAM=11
TLEARNER_PSILOCYBIN=12
PSILODEP1_WO_PRETRAINING_START=13
PSILODEP1_WO_PRETRAINING_END=62

# Get the current job index
JOB_ID=${PBS_ARRAY_INDEX}

# Run the appropriate script based on the job index
if [ $JOB_ID -ge $ABLATION_START ] && [ $JOB_ID -le $ABLATION_END ]; then
    # Ablation models (3 jobs; 0-2)
    python ablation_models.py --jobid=$((JOB_ID - ABLATION_START))

elif [ $JOB_ID -eq $ATLAS_BOUND ]; then
    # Atlas-bound (1 job; 3)
    python atlas_bound.py

elif [ $JOB_ID -ge $BENCHMARKS_START ] && [ $JOB_ID -le $BENCHMARKS_END ]; then
    # Benchmarks (3 jobs; 4-6)
    python benchmarks.py --jobid=$((JOB_ID - BENCHMARKS_START))

elif [ $JOB_ID -eq $DRUG_CLASSIFIER ]; then
    # Drug classifier (1 job; 7)
    python drug_classifier.py

elif [ $JOB_ID -eq $GRAPHTRIP_BDI ]; then
    # graphTRIP for BDI prediction (1 job; 8)
    python graphtrip_bdi.py

elif [ $JOB_ID -eq $GRAPHTRIP_MAIN ]; then
    # graphTRIP, main model (1 job; 9)
    python graphtrip.py

elif [ $JOB_ID -eq $PSILODEP1_FINETUNING ]; then
    # Pretraining + finetuning on psilodep1 (1 job; 10)
    python psilodep1_finetuning.py

elif [ $JOB_ID -eq $TLEARNER_ESCITALOPRAM ]; then
    # T-learner for escitalopram (1 job; 11)
    python tlearners.py -c experiments/configs/tlearner_escitalopram.json

elif [ $JOB_ID -eq $TLEARNER_PSILOCYBIN ]; then
    # T-learner for psilocybin (1 job; 12)
    python tlearners.py -c experiments/configs/tlearner_psilocybin.json

elif [ $JOB_ID -ge $PSILODEP1_WO_PRETRAINING_START ] && [ $JOB_ID -le $PSILODEP1_WO_PRETRAINING_END ]; then
    # graphTRIP on psilodep1 without pretraining (50 jobs; 13-62)
    python psilodep1_wo_pretraining.py --jobid=$((JOB_ID - PSILODEP1_WO_PRETRAINING_START))

else
    echo "Invalid job index: $JOB_ID"
    exit 1
fi
