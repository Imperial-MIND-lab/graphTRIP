# REACT Maps Preprocessing

This directory contains scripts for computing REACT maps using the react-fmri toolbox.

## Prerequisites

1. Download the PET maps from the NRU Serotonin Atlas:
   - Visit: https://nru.dk/index.php/allcategories/category/90-nru-serotonin-atlas-and-clustering
   - Download the required PET maps for your analysis

2. Install the react-fmri toolbox:
   ```bash
   pip install react-fmri
   ```
   For more details, visit: https://github.com/ottaviadipasquale/react-fmri

## Directory Structure

The preprocessing pipeline consists of the following scripts, to be executed in order:

1. `resample_and_mask_5HT_atlas.sh`: Resamples and masks the 5-HT atlas
2. `create_subject_list.sh`: Generates the list of subjects for analysis
3. `react_masks.sh`: Creates the necessary masks for REACT analysis
4. `react.sh`: Performs the main REACT computation

## Usage

1. Before running the scripts, ensure all file paths are correctly configured in each script to match your local setup.

2. Execute the scripts in the following order:
   ```bash
   ./resample_and_mask_5HT_atlas.sh
   ./create_subject_list.sh
   ./react_masks.sh
   ./react.sh
   ```

## Configuration

Each script contains configurable parameters that should be adapted to your specific needs:
- Input/output directories
- Subject fMRI data filepaths

Please review and modify these parameters in each script before execution.
