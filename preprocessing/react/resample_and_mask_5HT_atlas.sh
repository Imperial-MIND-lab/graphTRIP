#!/bin/bash

# Written by Tim Lawn 16/12/2024

# =============================================================================
# Summary for eventual methods section 
# =============================================================================
# This script performs spatial processing and masking of PET atlas images downloaded 
# from https://nru.dk/index.php/allcategories/category/90-nru-serotonin-atlas-and-clustering:
#
# 1. Spatial Processing:
#    - Resamples 1mm isotropic images to 2mm isotropic resolution using ANTs 
#      ResampleImage with linear interpolation (parameters: 2x2x2 0 0)
#
# 2. Anatomical Masking:
#    - Utilizes the Harvard-Oxford subcortical atlas (25% threshold version)
#    - Binarizes the atlas mask using FSL's fslmaths
#    - Applies the binary mask to remove cerebellar reference regions
#
# Processing is within MNI152-space 
# =============================================================================

# Set the input and output directories
INPUT_DIR="/Users/timlawn/Desktop/Projects/Hanna_Tolle/5-HT_atlas"
OUTPUT_DIR="/Users/timlawn/Desktop/Projects/Hanna_Tolle/5-HT_atlas_2mm"
MASK_DIR="/Users/timlawn/Desktop/Projects/Hanna_Tolle/masks"
HO_MASK="HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$MASK_DIR"

# Copy and prepare the Harvard-Oxford mask
echo "Preparing Harvard-Oxford mask..."
cp /usr/local/fsl/data/atlases/HarvardOxford/$HO_MASK "$MASK_DIR"
BINARY_MASK="$MASK_DIR/HarvardOxford-sub-maxprob-thr25-2mm_binary.nii.gz"
fslmaths "$MASK_DIR/$HO_MASK" -bin "$BINARY_MASK"

# Function to process a single directory
process_directory() {
    local dir=$1
    local out_dir=$2
    
    # Create corresponding output subdirectory
    mkdir -p "$out_dir"
    
    # Process only MNI152 files in the directory
    for file in "$dir"/*MNI152*.nii.gz; do
        if [ -f "$file" ]; then
            # Get the filename without path
            filename=$(basename "$file")
            
            # Create output filenames
            output_file="$out_dir/${filename/.nii.gz/_2mm.nii.gz}"
            masked_output="$out_dir/${filename/.nii.gz/_2mm_masked.nii.gz}"
            
            echo "Processing: $filename"
            
            # Resample to 2mm
            echo "Resampling to 2mm..."
            ResampleImage 3 "$file" "$output_file" 2x2x2 0 0
            
            # Apply binary mask
            echo "Applying mask..."
            fslmaths "$output_file" -mas "$BINARY_MASK" "$masked_output"
            
            echo "Created masked file: ${filename/.nii.gz/_2mm_masked.nii.gz}"
        fi
    done
}

# Process each subdirectory
for dir in "$INPUT_DIR"/5-HT*/; do
    if [ -d "$dir" ]; then
        # Get directory name
        dirname=$(basename "$dir")
        
        echo "Processing directory: $dirname"
        process_directory "$dir" "$OUTPUT_DIR/$dirname"
    fi
done

echo "Processing complete! All MNI152 files have been resampled and masked."
echo "Now run the Python normalization script to normalize the images and compute statistics."