#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=8:00:00
#PBS -N secondary_jobs
#PBS -J 0-42

# Secondary scripts depend on prior execution of primary scripts.

# Load environment
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/scripts

# Define job ranges for each script
HYBRID=0
XGRAPHTRIP_START=1
XGRAPHTRIP_END=42

# Get the current job index
JOB_ID=${PBS_ARRAY_INDEX}

# Run the appropriate script based on the job index
if [ $JOB_ID -eq $HYBRID ]; then
    # Hybrid model (1 job; 0)
    python hybrid.py
elif [ $JOB_ID -ge $XGRAPHTRIP_START ] && [ $JOB_ID -le $XGRAPHTRIP_END ]; then
    # x_graphtrip (42 jobs; 1-42)
    python x_graphtrip.py --jobid=$((JOB_ID - XGRAPHTRIP_START))
else
    echo "Invalid job index: $JOB_ID"
    exit 1
fi
