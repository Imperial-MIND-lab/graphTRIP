#!/bin/bash
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=08:00:00
#PBS -N permutation_test
#PBS -J 0-49

# Load environment
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip

# Change to working directory
cd $PBS_O_WORKDIR

# Run experiment
python -m experiments.run_experiment \
  permutation_test \
  FileStorageObserver \
  --config_json permutation_test.json \
  --jobid $PBS_ARRAY_INDEX \
  --seed $PBS_ARRAY_INDEX
