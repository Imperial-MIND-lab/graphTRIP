#!/bin/bash

#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=00:30:00
#PBS -N react
#PBS -J 1-16

module load anaconda3/personal
source activate graphtrp

# Parameters
study="psilodep1"
session="before"
receptor_set="Believeau-5"

# Define project directory and output directory
project_dir="/rds/general/user/hmt23/home/projects/graphTRP"
output_parent_dir="${project_dir}/data/raw/${study}/${session}/MNI_2mm/REACT_${receptor_set}"

# Move into the output directory
cd ${output_parent_dir}

# Get all subject directories and store them in an array
mapfile -t subject_dirs < <(ls -d /rds/general/user/hmt23/home/data/${study}/${session}/S* | xargs -n 1 basename)

# Get the current subject directory from the array using PBS array index
subject_id="${subject_dirs[${PBS_ARRAY_INDEX}-1]}"

# Define input file paths (for fMRI, PET atlas, and masks)
input_file="/rds/general/user/hmt23/home/data/${study}/${session}/${subject_id}/${session}_rest_rdsmffms6FWHM_bd_M_V_DV_WMlocal2_modecorr.nii.gz"
atlas_file="${output_parent_dir}/pet_atlas.nii.gz"
masks_dir="${output_parent_dir}/out_masks"

# Create output directory if it doesn't exist
output_dir="${subject_id}"
mkdir -p ${output_dir}

# Run REACT analysis
react ${input_file} ${masks_dir}/mask_stage1.nii.gz ${masks_dir}/mask_stage2.nii.gz ${atlas_file} ${output_dir}/${subject_id}