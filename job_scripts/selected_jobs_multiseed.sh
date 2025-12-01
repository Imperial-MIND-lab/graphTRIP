#!/bin/bash

#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=4:00:00
#PBS -N selected_jobs_multiseed
#PBS -J 0-735  

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/experiments

# Settings
OBSERVER='NeptuneObserver'
EXNAME='train_jointly'
CONFIG_JSON='bdi_graphtrip.json'
SEEDS=(2 3 4 5 6 7 8 9)
SELECTED_CONFIGS=(26 40 60 67 68 69 74 81 82 85 87 88 90 91 92 93 95 102 103 104 107 109 110 111 112 113 114 116 117 118 119 120 123 125 126 127 139 156 158 167 168 170 173 187 196 200 202 204 205 206 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 227 230 231 232 234 235 236 237 238 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255)

# Calculate number of selected configs and seeds
NUM_SELECTED_CONFIGS=${#SELECTED_CONFIGS[@]}
NUM_SEEDS=${#SEEDS[@]}

# Calculate config index and seed index from PBS_ARRAY_INDEX
# PBS_ARRAY_INDEX ranges from 0 to (num_seeds * num_selected_configs - 1)
# For example, with 11 selected configs and 3 seeds: PBS_ARRAY_INDEX 0-32 maps to:
# - config_index: 0-10 (index into SELECTED_CONFIGS array)
# - seed_index: 0-2 (index into SEEDS array)
CONFIG_INDEX=$((PBS_ARRAY_INDEX % NUM_SELECTED_CONFIGS))
SEED_INDEX=$((PBS_ARRAY_INDEX / NUM_SELECTED_CONFIGS))
JOBID=${SELECTED_CONFIGS[$CONFIG_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

# Run the experiment with the specific config and seed
python run_experiment.py ${EXNAME} ${OBSERVER} --jobid=${JOBID} --config_json=${CONFIG_JSON} --seed=${SEED}