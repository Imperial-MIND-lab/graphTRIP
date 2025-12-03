#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=02:00:00
#PBS -N cfr_trip
#PBS -J 0-19

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/scripts

NUM_SEEDS=10
if [ ${PBS_ARRAY_INDEX} -lt $NUM_SEEDS ]; then
    CONFIG_JSON='experiments/configs/cfr_trip.json'
    OUTPUT_DIR='outputs/cfr_trip/'
    SEED=${PBS_ARRAY_INDEX}
else
    CONFIG_JSON='experiments/configs/cfr_trip_delta.json'
    OUTPUT_DIR='outputs/cfr_trip_delta/'
    SEED=$((PBS_ARRAY_INDEX - NUM_SEEDS))
fi

python cfr_trip.py -c=${CONFIG_JSON} -o=${OUTPUT_DIR} -s=${SEED} -v
