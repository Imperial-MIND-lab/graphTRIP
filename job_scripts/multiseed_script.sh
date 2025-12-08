#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=08:00:00
#PBS -N transfer_vgae
#PBS -J 0-9

module load anaconda3/personal
source activate graphtrip

cd ~/projects/graphTRIP/scripts

python transfer.py -s ${PBS_ARRAY_INDEX} -v
