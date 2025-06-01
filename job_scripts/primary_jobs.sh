#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=8:00:00
#PBS -N psilodep1_wo_pretraining
#PBS -J 11-110

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
PSILODEP1_WO_PRETRAINING_START=11
PSILODEP1_WO_PRETRAINING_END=60
PSILODEP1_MLP_START=61
PSILODEP1_MLP_END=110
TLEARNERS=111

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
elif [ $JOB_ID -ge $PSILODEP1_WO_PRETRAINING_START ] && [ $JOB_ID -le $PSILODEP1_WO_PRETRAINING_END ]; then
    # graphTRIP on psilodep1 without pretraining (50 jobs; 11-60)
    python psilodep1_wo_pretraining.py -m graphtrip --jobid=$((JOB_ID - PSILODEP1_WO_PRETRAINING_START))
elif [ $JOB_ID -ge $PSILODEP1_MLP_START ] && [ $JOB_ID -le $PSILODEP1_MLP_END ]; then
    # MLP on psilodep1 without pretraining (50 jobs; 61-110)
    python psilodep1_wo_pretraining.py -m control_mlp --jobid=$((JOB_ID - PSILODEP1_MLP_START))
elif [ $JOB_ID -eq $TLEARNERS ]; then
    # T-learners (1 job; 111)
    python tlearners.py
else
    echo "Invalid job index: $JOB_ID"
    exit 1
fi
