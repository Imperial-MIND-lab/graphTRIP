""" 
Written by Tim Lawn 16/12/2024

Summary for eventual methods section
1. Normalizes masked images to 0-1 range using the same approach as react_normalize
2. Computes and saves summary statistics for each normalized image 
"""

import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from datetime import datetime

def normalize_3d_volume(data):
    """
    Normalize a 3D volume to [0,1] range, ignoring zero-valued voxels.
    This implements the exact normalization from react_normalize.
    """
    # Create a mask of non-zero voxels
    nonzero_mask = data != 0
    
    if not np.any(nonzero_mask):
        return data
    
    # Get non-zero values
    nonzero_data = data[nonzero_mask]
    
    # Calculate min and max of non-zero values
    data_min = np.min(nonzero_data)
    data_max = np.max(nonzero_data)
    
    # Avoid division by zero
    if data_max == data_min:
        return data
    
    # Create output array
    normalized = np.zeros_like(data)
    
    # Normalize only non-zero values
    normalized[nonzero_mask] = (data[nonzero_mask] - data_min) / (data_max - data_min)
    
    return normalized

def compute_statistics(img_path):
    """Compute statistics for a given image."""
    img = nib.load(img_path)
    data = img.get_fdata()
    # Only consider non-zero values (within mask)
    masked_data = data[data != 0]
    
    stats = {
        'filename': os.path.basename(img_path),
        'min': np.min(masked_data),
        'max': np.max(masked_data),
        'mean': np.mean(masked_data),
        'median': np.median(masked_data),
        'std': np.std(masked_data),
        'q1': np.percentile(masked_data, 25),
        'q3': np.percentile(masked_data, 75)
    }
    return stats

def process_directory(input_dir):
    """Process all masked images in a directory."""
    # Find all masked MNI152 images
    masked_files = glob.glob(os.path.join(input_dir, '**/5-HT*/*MNI152*_2mm_masked.nii.gz'), recursive=True)
    
    stats_list = []
    
    for masked_file in masked_files:
        print(f"\nProcessing: {os.path.basename(masked_file)}")
        
        # Create normalized filename
        normalized_file = masked_file.replace('_masked.nii.gz', '_masked_normalized.nii.gz')
        
        # Load input image
        print("Loading image...")
        volume = nib.load(masked_file)
        data = volume.get_fdata(dtype=np.float32)
        
        # Normalize image
        print("Normalizing...")
        if data.ndim == 3:
            print('Processing 3D volume')
            rescaled = normalize_3d_volume(data)
        elif data.ndim == 4:
            print('Processing 4D volume')
            rescaled = np.zeros_like(data)
            for i in range(data.shape[3]):
                print(f'Processing volume # {i} of {data.shape[3] - 1}')
                rescaled[..., i] = normalize_3d_volume(data[..., i])
        else:
            raise ValueError('Number of dimensions of input data must be 3 or 4.')
        
        # Save normalized image
        nib.save(nib.Nifti1Image(rescaled, affine=volume.affine), normalized_file)
        
        # Compute statistics
        print("Computing statistics...")
        stats = compute_statistics(normalized_file)
        stats_list.append(stats)
        
        # Print individual file statistics
        print(f"Statistics for {stats['filename']}:")
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
    
    return stats_list

def main():
    # Set base directory
    base_dir = "/Users/timlawn/Desktop/Projects/Hanna_Tolle/5-HT_atlas_2mm"
    
    # Create output directory for statistics
    stats_dir = os.path.join(base_dir, 'statistics')
    os.makedirs(stats_dir, exist_ok=True)
    
    # Process all files and get statistics
    stats_list = process_directory(base_dir)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(stats_list)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = os.path.join(stats_dir, f'normalization_statistics_{timestamp}.csv')
    df.to_csv(csv_file, index=False)
    
    print(f"\nStatistics have been saved to: {csv_file}")

if __name__ == "__main__":
    main()