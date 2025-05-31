"""
Concatenates multiple 3D atlases into a single 4D volume.
Author: Hanna Tolle
Date: 21.12.2024
License: BSD-3-Clause
"""

import os
import glob
import nibabel as nib
import numpy as np
import argparse

def concat_atlases(atlas_dir: str, output_dir: str, receptor_names: list):
    """
    Concatenates multiple 3D atlases (nii.gz files) from receptor subdirectories into a single 4D volume.

    How to run (alternatives):
    python concat_5HT_atlases.py --atlas_dir /path/to/atlas --output_dir Beliveau-3 --receptors 5HT1A 5HT1B 5HT2A
    python concat_5HT_atlases.py --output_dir Beliveau-3 --receptors 5HT1A 5HT1B 5HT2A

    Parameters:
    ----------
    atlas_dir: str
        Base directory containing receptor subdirectories with 3D atlases.
    output_dir: str
        Name of the output directory (will be created in atlas_dir/concatenated/).
    receptor_names: list
        List of receptor names (subdirectory names) to process.
    """
    # Create output directory
    full_output_dir = os.path.join(atlas_dir, "concatenated", output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    # Collect atlas files from each receptor subdirectory
    atlas_files = []
    for receptor in receptor_names:
        receptor_dir = os.path.join(atlas_dir, receptor)
        files = glob.glob(os.path.join(receptor_dir, '*_2mm_masked_normalized.nii.gz'))
        if not files:
            raise ValueError(f"No .nii.gz files found in {receptor_dir}")
        atlas_files.extend(files)

    # Load all atlas files
    atlases = [nib.load(file) for file in atlas_files]
    
    # Get the shape of the first atlas
    shape = atlases[0].shape

    # Create an empty 4D volume
    concatenated_volume = np.zeros(shape + (len(atlases),))

    # Fill the 4D volume with the atlas data
    for i, atlas in enumerate(atlases):
        concatenated_volume[:, :, :, i] = atlas.get_fdata()

    # Save the concatenated volume
    output_file = os.path.join(full_output_dir, 'pet_atlas.nii.gz')
    nib.save(nib.Nifti1Image(concatenated_volume, atlases[0].affine), output_file)

    # Save receptor names and corresponding input files
    with open(os.path.join(full_output_dir, 'input_maps.txt'), 'w') as f:
        for receptor, file in zip(receptor_names, atlas_files):
            f.write(f"{receptor},{os.path.basename(file)}\n")

    print(f"Files saved to {full_output_dir}:")
    print(f"- Concatenated atlas: pet_atlas.nii.gz")
    print(f"- Input maps list: input_maps.txt")


if __name__ == "__main__":
    # Set default directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    default_atlas_dir = os.path.join(this_dir, '..', 'data', 'raw', 'react_data', '5-HT_atlas_2mm')

    # Parse arguments
    parser = argparse.ArgumentParser(description='Concatenate multiple 3D atlases into a single 4D volume.')
    parser.add_argument('--atlas_dir', type=str, default=default_atlas_dir,
                      help=f'Base directory containing receptor subdirectories with 3D nifti atlases (default: {default_atlas_dir})')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Name of the output directory (will be created in atlas_dir/concatenated/)')
    parser.add_argument('--receptors', nargs='+', type=str, required=True,
                      help='List of receptor names (subdirectory names) to process')
    args = parser.parse_args()

    # Run
    concat_atlases(args.atlas_dir, args.output_dir, args.receptors)
