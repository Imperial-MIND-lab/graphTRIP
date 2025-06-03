#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=4:00:00
#PBS -N tlearners_SSRIstop_and_BDIonly
#PBS -J 0-35

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/experiments

# Define job ranges for each script
TLEARNER_P_SSRIstop_START=0
TLEARNER_P_SSRIstop_END=8
TLEARNER_E_SSRIstop_START=9
TLEARNER_E_SSRIstop_END=17

TLEARNER_P_BDIonly_START=18
TLEARNER_P_BDIonly_END=26
TLEARNER_E_BDIonly_START=27
TLEARNER_E_BDIonly_END=35

# Get the current job index
JOB_ID=${PBS_ARRAY_INDEX}

# Run the appropriate script based on the job index
if [ $JOB_ID -ge $TLEARNER_P_SSRIstop_START ] && [ $JOB_ID -le $TLEARNER_P_SSRIstop_END ]; then
    # Psilocybin T-learner SSRIstop (0-8)
    python run_experiment.py train_tlearner NeptuneObserver --jobid=$((JOB_ID - TLEARNER_P_SSRIstop_START)) --config_json='tlearner_P_SSRIstop.json'

elif [ $JOB_ID -ge $TLEARNER_E_SSRIstop_START ] && [ $JOB_ID -le $TLEARNER_E_SSRIstop_END ]; then
    # Escitalopram T-learner SSRIstop (9-17)
    python run_experiment.py train_tlearner NeptuneObserver --jobid=$((JOB_ID - TLEARNER_E_SSRIstop_START)) --config_json='tlearner_E_SSRIstop.json'

elif [ $JOB_ID -ge $TLEARNER_P_BDIonly_START ] && [ $JOB_ID -le $TLEARNER_P_BDIonly_END ]; then
    # Psilocybin T-learner BDIonly (18-26)
    python run_experiment.py train_tlearner NeptuneObserver --jobid=$((JOB_ID - TLEARNER_P_BDIonly_START)) --config_json='tlearner_P_BDIonly.json'

elif [ $JOB_ID -ge $TLEARNER_E_BDIonly_START ] && [ $JOB_ID -le $TLEARNER_E_BDIonly_END ]; then
    # Escitalopram T-learner BDIonly (27-35)
    python run_experiment.py train_tlearner NeptuneObserver --jobid=$((JOB_ID - TLEARNER_E_BDIonly_START)) --config_json='tlearner_E_BDIonly.json'

else
    echo "Invalid job index: $JOB_ID"
    exit 1
fi
