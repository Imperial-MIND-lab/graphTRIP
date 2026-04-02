#!/bin/bash
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=08:00:00
#PBS -N flatvae_mlp
#PBS -J 0-9

# Load environment
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip

# Change to working directory
cd $PBS_O_WORKDIR

# Set parameters
CONFIG_FILE="flatvae_mlp.json"
EXPERIMENT="train_jointly"

# Run experiment
python -m experiments.run_experiment \
  $EXPERIMENT \
  FileStorageObserver \
  --config_json $CONFIG_FILE \
  --jobid 0 \
  --seed $PBS_ARRAY_INDEX
