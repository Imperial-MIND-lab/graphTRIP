#!/bin/bash

#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=01:00:00
#PBS -N react_masks

module load anaconda3/personal
source activate graphtrp

# Parameters
study="psilodep1"
session="before"
dataset="5-HT_atlas_2mm"
receptor_set="Believeau-5"

# Move into the output parent directory (where out_masks/ is created)
project_dir="/rds/general/user/hmt23/home/projects/graphTRP"
output_parent_dir="${project_dir}/data/raw/${study}/${session}/MNI_2mm/REACT_${receptor_set}"
mkdir -p $output_parent_dir
cd $output_parent_dir

# Copy subject_list.txt to the output parent directory
cp ${project_dir}/data/raw/${study}/${session}/MNI_2mm/subject_list.txt .

# Copy pet_atlas.nii.gz, input_maps.txt, and gm_mask.nii.gz to the output parent directory
react_data_dir="${project_dir}/data/raw/react_data"
cp ${react_data_dir}/${dataset}/concatenated/${receptor_set}/pet_atlas.nii.gz .
cp ${react_data_dir}/${dataset}/concatenated/${receptor_set}/input_maps.txt .
cp ${react_data_dir}/masks/gm_mask.nii.gz .

# Run REACT masks
react_masks subject_list.txt pet_atlas.nii.gz gm_mask.nii.gz out_masks