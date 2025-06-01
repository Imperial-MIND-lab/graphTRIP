#!/bin/bash

#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=02:00:00
#PBS -N preprocessing
#PBS -J 1-6
 
module load anaconda3/personal
source activate graphtrip
cd ~/projects/graphTRIP/

atlases=('schaefer100' 'schaefer200' 'aal')
studies=('psilodep2' 'psilodep1')

# Calculate indices for the current job
atlas_idx=$(( ($PBS_ARRAY_INDEX - 1) % 3 ))
study_idx=$(( ($PBS_ARRAY_INDEX - 1) / 3 ))

study=${studies[$study_idx]}
atlas=${atlases[$atlas_idx]}

python -m preprocessing.preprocess $study before $atlas
