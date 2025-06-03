#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=4:00:00
#PBS -N tlearners_SSRIstop
#PBS -J 0-35

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/experiments

# Define job ranges for each script
PSILOCYBIN_TLEARNER_START=0
PSILOCYBIN_TLEARNER_END=17
ESCITALOPRAM_TLEARNER_START=18
ESCITALOPRAM_TLEARNER_END=35

# Get the current job index
JOB_ID=${PBS_ARRAY_INDEX}

# Run the appropriate script based on the job index
if [ $JOB_ID -ge $PSILOCYBIN_TLEARNER_START ] && [ $JOB_ID -le $PSILOCYBIN_TLEARNER_END ]; then
    # Psilocybin T-learner (3 jobs; 0-2)
    python run_experiment.py train_tlearner NeptuneObserver --jobid=$JOB_ID --config_json='tlearner_P_SSRIstop.json'
elif [ $JOB_ID -ge $ESCITALOPRAM_TLEARNER_START ] && [ $JOB_ID -le $ESCITALOPRAM_TLEARNER_END ]; then
    # Escitalopram T-learner (3 jobs; 3-5)
    python run_experiment.py train_tlearner NeptuneObserver --jobid=$((JOB_ID - ESCITALOPRAM_TLEARNER_START)) --config_json='tlearner_E_SSRIstop.json'
else
    echo "Invalid job index: $JOB_ID"
    exit 1
fi
