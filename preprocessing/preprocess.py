"""
Preprocesses neuroimaging data and computes all metrics defined in .metrics.

License: BSD-3-Clause
Author: Hanna M. Tolle
Date: 2024-10-31
"""

import sys
sys.path.append('../')

import os
import numpy as np
from typing import Dict

from .metrics import *
from datasets import BrainGraphDataset


def run(study: str = 'psilodep2', session: str = 'before', atlas: str = 'schaefer100', filter: Dict = {'Exclusion':0}):
    """
    Preprocesses neuroimaging data and computes all metrics defined in .metrics.
    Run from command line: python -m preprocessing.preprocess
    
    Parameters:
    ----------
    study (str): Name of the study that collected the data.
    session (str): Name of the scanning session.
    atlas (str): Name of the brain atlas (parcellation).
    filter (dict): Filter for processing data of selected subjects. Keys of the
                   dictionary are column names of the annotations.csv file.                   
    """
    
    # Load annotations for this dataset
    annotations = load_annotations(study=study, filter=filter)

    # Get subject indices and convert to python indices (starting from 0)
    subjects = get_all_ids(annotations)
    subjects = [sub-1 for sub in subjects]

    # Get node and edge feature functions
    node_feature_funs = get_node_feature_funs()
    edge_feature_funs = get_edge_feature_funs()

    subs_missing_raw = []
    for sub in subjects:

        ## Define output directory
        outdir = get_filepath(study=study, session=session, atlas=atlas, subject=sub)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        ## Compute or load parcellated BOLD data ----------------------- ##
        if os.path.exists(os.path.join(outdir, 'bold.csv')):
            print(f'Loading {os.path.join(outdir, 'bold.csv')}.')
            bold = np.genfromtxt(os.path.join(outdir, 'bold.csv'), delimiter=',') # shape = (numTRs x numROIs)

        else:
            # Parcellate raw voxelwise BOLD data and save
            raw_file = os.path.join(raw_data_dir(), 
                                    study, 
                                    session, 
                                    get_subject_id(sub), 
                                    get_raw_filename(study=study, session=session))
            
            # If the raw file does not exist, skip and add the sub to the list
            if not os.path.exists(raw_file):
                subs_missing_raw.append(sub)
                print(f'Subject {sub}: Raw file {raw_file} not found. Skipping.')
                continue
            
            # Parcellate raw voxelwise BOLD data
            bold = parcellate(raw_file, atlas=atlas)

            # Make sure that no columns are nan or zero
            for i in range(bold.shape[1]):
                if np.isnan(bold[:, i]).any():
                    print(f'Subject {sub}: Replacing NaN in column {i} with the mean of the column to the left and right.')
                    mean_val = (bold[:, i-1] + bold[:, i+1]) / 2
                    bold[:, i] = np.where(np.isnan(bold[:, i]), mean_val, bold[:, i])
                if np.all(bold[:, i] == 0):
                    print(f'Subject {sub}: Replacing all zeros in column {i} with the mean of the column to the left and right.')
                    mean_val = (bold[:, i-1] + bold[:, i+1]) / 2    
                    bold[:, i] = np.where(bold[:, i] == 0, mean_val, bold[:, i])  

            # Save the parcellated BOLD data
            np.savetxt(os.path.join(outdir, 'bold.csv'), bold, delimiter=',')

        ## Compute node features -------------------------------------- ##
        feature_file = os.path.join(outdir, 'node.csv')
        compute_metrics(bold, atlas=atlas, feature_file=feature_file, feature_funs=node_feature_funs)

        ## Compute receptor-enriched maps ---------------------------- ##
        feature_file = os.path.join(outdir, 'node.csv')
        receptor_sets = ['Believeau-5', 'Believeau-3']
        react_dir = os.path.join(project_root(), 'data', 'raw', study, session, 'MNI_2mm')
        for receptor_set in receptor_sets:
            react_subject_dir = os.path.join(react_dir, f'REACT_{receptor_set}', get_subject_id(sub))
            compute_single_metric(feature_file, receptor_enriched_maps, react_subject_dir, 
                                  atlas=atlas, receptor_set=receptor_set)

        ## Compute edge features -------------------------------------- ##
        feature_file = os.path.join(outdir, 'edge.csv')
        compute_metrics(bold, atlas=atlas, feature_file=feature_file, feature_funs=edge_feature_funs)

    # Add a column to the annotations file for the subjects with missing raw data
    annotations = load_annotations(study=study) # Load the annotations file without any filters
    if subs_missing_raw:
        # Create a new column for missing raw data, initialize with 0
        annotations[f'missing_raw_{session}'] = 0
        
        # Set the value to 1 for subjects with missing raw data
        for sub in subs_missing_raw:
            annotations.loc[annotations['Patient'] == sub+1, f'missing_raw_{session}'] = 1
        annotations.to_csv(os.path.join(project_root(), 'data', 'raw', study, 'annotations.csv'), index=False)

    print('Preprocessing completed! Loading this into a torch_geometric dataset object...')

    # Load the dataset into a torch_geometric dataset object
    dataset = BrainGraphDataset(study=study, session=session, atlas=atlas, force_reload=True)
    print('All done!')


if __name__ == "__main__":
    # Get system inputs
    if len(sys.argv) > 1:
        study = sys.argv[1]
        session = sys.argv[2]
        atlas = sys.argv[3]
    else:
        study = 'psilodep2'
        session = 'before'
        atlas = 'schaefer100'
    run(study=study, session=session, atlas=atlas)