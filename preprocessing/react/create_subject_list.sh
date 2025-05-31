#!/bin/bash

# Define the parent directory (on HPC)
study="psilodep1"
session="before"
parent_dir="/rds/general/user/hmt23/home/data/${study}/${session}"

# Find all .nii.gz files in subdirectories and save their paths to subject_list.txt
find "$parent_dir" -type f -name "*.nii.gz" > subject_list.txt