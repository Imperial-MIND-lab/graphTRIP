#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=6:00:00
#PBS -N xgraphtrip_screening
#PBS -J 0-179

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/experiments

# Define job ranges for each script
XGRAPHTIP_SSRIstop_START=0
XGRAPHTIP_SSRIstop_END=35
XGRAPHTIP_BDIonly_START=36
XGRAPHTIP_BDIonly_END=179

# Get the current job index
JOB_ID=${PBS_ARRAY_INDEX}

# Run the appropriate script based on the job index
if [ $JOB_ID -ge $XGRAPHTIP_SSRIstop_START ] && [ $JOB_ID -le $XGRAPHTIP_SSRIstop_END ]; then
    # X-learner SSRIstop (36 jobs; 0-35)
    python run_experiment.py train_xlearner NeptuneObserver --jobid=$((JOB_ID - XGRAPHTIP_SSRIstop_START)) --config_json='x_graphtrip_SSRIstop.json'

elif [ $JOB_ID -ge $XGRAPHTIP_BDIonly_START ] && [ $JOB_ID -le $XGRAPHTIP_BDIonly_END ]; then
    # X-learner BDIonly (144 jobs; 36-179)
    python run_experiment.py train_xlearner NeptuneObserver --jobid=$((JOB_ID - XGRAPHTIP_BDIonly_START)) --config_json='x_graphtrip_BDIonly.json'

else
    echo "Invalid job index: $JOB_ID"
    exit 1
fi
