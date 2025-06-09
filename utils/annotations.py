"""
Utility functions for handling annotations.

License: BSD-3-Clause
Author: Hanna M. Tolle  
Date: 2024-10-31
"""

import os
import pandas as pd
import numpy as np
from typing import List, Any

from utils.files import get_filepath, raw_data_dir, project_root


def load_annotations(study='psilodep2', filter=None):
    ''' Loads and filters clinical outcomes annotations file.

    Parameters:
    ----------
    study (str): Name of the study.
    filter (dict): Column and value of column based on which annotations
                   should be filtered. The value specifies rows to be included.
                   For example: filter={'Exclusion': 0} will remove rows with
                   Exclusion!=0. Filtering based on multiple columns is possible.

    Returns:
    -------
    annotations (pandas.Dataframe)
    '''
    categories = get_categorical_annotations(study)

    datadir = get_filepath(study=study)
    if not os.path.exists(datadir):
        os.makedirs(datadir, exist_ok=True)

    annotations_file = os.path.join(datadir, 'annotations.csv')

    # Fetch from raw data directory, if processed annotations file does not yet exist
    if not os.path.exists(annotations_file):

        raw_file = os.path.join(raw_data_dir(), study, 'clinical_outcomes.csv')
        if not os.path.exists(raw_file):
            raise FileNotFoundError
        
        annotations = pd.read_csv(raw_file)

        # Process: remove trailing spaces for all columns
        for column in annotations.columns:
            if annotations[column].dtype == 'object':
                annotations[column] = annotations[column].str.strip()

        # Convert specified columns to categorical type
        for cat in categories:
            annotations[cat] = annotations[cat].astype('category')

        # Add a Condition column, if study is psilodep1
        if study == 'psilodep1':
            annotations['Condition'] = 1.0 # all patients are in the psilocybin condition
            annotations['Stop_SSRI'] = 1.0 # all patients are TRD and had prior SSRI use
        elif study == 'psilodep2':
            annotations['Condition_bin01'] = annotations['Condition'].map({'P': 1, 'E': 0})

        # Save processed file as annotations.csv
        annotations.to_csv(annotations_file, index=False)

    else:
        annotations = pd.read_csv(annotations_file)

    # Filter, if applicable
    if filter:
        for k, v in filter.items():
            annotations = annotations[annotations[k] == v]
        annotations = annotations.reset_index(drop=True)
    
    return annotations


def get_all_ids(annotations):
    return annotations['Patient'].tolist()

def patient_ids_to_sample_ids(patient_ids: List[str], study: str):
    '''
    Converts patient IDs to sample IDs.
    - patient ids are the indices in the annotations "Patient" column.
    - sample ids are the data-sample indices for those patients in the full dataset.
    '''
    if study == 'psilodep2':
        prefilter = {'Exclusion': 0}
    elif study == 'psilodep1':
        prefilter = {'Exclusion': 0, 'missing_raw_before': 0}
    else:
        raise ValueError(f'Prefilter not specified for study {study}. \n'
                         'Edit utils.annotations.patient_ids_to_sample_ids() to add your prefilter.')
    annotations = load_annotations(study, prefilter)
    all_patient_ids = annotations['Patient'].tolist()
    sample_ids = [all_patient_ids.index(patient_id) for patient_id in patient_ids]
    return sample_ids

def subject_ids_to_sample_ids(subject_ids: List[str], study: str):
    '''Converts subject IDs to sample IDs.
    - subject ids are stored in the data samples as data.subject; subject_ids = patient_ids - 1.
    - sample ids are the data-sample indices for those subjects in the full dataset.
    '''
    patient_ids = [int(subject_id) + 1 for subject_id in subject_ids]
    return patient_ids_to_sample_ids(patient_ids, study)

def get_list_idx(lst: List, target: Any) -> int:
    '''Returns the first index where a list entry matches an input.'''
    try:
        idx = lst.index(target)
        return idx
    except ValueError:
        print(f'{target} is not in the list.')


def convert_to_numerical(df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    '''
    Converts specified columns to numerical labels.
    
    Parameters:
    ----------
    df (pd.DataFrame): The input DataFrame.
    categories (List[str]): List of column names that should be converted to numerical labels.
    
    Returns:
    -------
    pd.DataFrame: The DataFrame with the specified columns converted to numerical labels.
    '''
    for category in categories:
        df[category] = df[category].replace(get_cat2num_dict(category))
    
    return df

def get_drug_label(drug: str, study: str = 'psilodep2') -> int:
    '''Return the numerical symbol for the given drug.'''
    if study == 'psilodep2':
        if drug == 'psilocybin':
            return 1
        elif drug == 'escitalopram':
            return -1
        else:
            raise ValueError("Unknown drug.")
    else:
        raise ValueError("Unknown study. \n"
                         "Edit utils.annotations.get_drug_label() to add your study.")
    
def get_tr(study: str = 'psilodep2'):
    '''Returns the repetition time (TR) in seconds.'''
    if study == 'psilodep2':
        return 1.25
    elif study == 'psilodep1':
        return 2.0
    else:
        raise ValueError("Unknown study. \n"
                         "Edit utils.annotations.get_tr() to add the TR for your study.")
    
def get_categorical_annotations(study: str = 'psilodep2'):
    '''Returns a list of column names that should be converted to categorical type.'''
    if study == 'psilodep2':
        return ['Condition', 'Gender']
    elif study == 'psilodep1':
        return ['Gender']
    else:
        raise ValueError("Unknown study. \n"
                         "Edit utils.annotations.get_categorical_annotations() to add your study.")
    
def get_cat2num_dict(category: str):
    '''Returns a dictionary with categorical values as keys and numerical values as keys.'''
    if category == 'Condition':
        return {'P': 1, 'E': -1}
    elif category == 'Gender':
        return {'M': 1, 'F': -1}
    else:
        raise ValueError("Unknown category. \n"
                         "Edit utils.annotations.get_cat2num_dict() to add your category.")
    
def load_receptor_maps(atlas: str, receptor_subset: List[str] = None):
    '''Loads receptor maps for the given atlas.'''
    # Load the receptor maps file
    receptor_maps_file = os.path.join(project_root(), 'data', 'raw', 
                                      'receptor_maps', atlas, f'{atlas}_receptor_maps.csv')
    try:
        receptor_maps = pd.read_csv(receptor_maps_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Receptor maps file {receptor_maps} not found.")
    
    # Filter the receptor maps based on the subset
    if receptor_subset:
        assert set(receptor_subset).issubset(set(receptor_maps.columns))
        return receptor_maps[receptor_subset]
    else:
        return receptor_maps
    
def load_receptor_nullmaps(atlas: str, receptors: List[str], 
                           rotation_mode: str = 'same', 
                           n_samples: int = 1000):
    '''Loads null receptor maps for the given atlas.'''
    # Load the spatial-autocorrelation-preserving null maps for each receptor
    receptor_dir = os.path.join(project_root(), 'data', 'raw', 'receptor_maps',  
                                atlas, f'{rotation_mode}_rotation_for_each_receptor')
    nulls = {}
    for receptor in receptors:
        nulls[receptor] = pd.read_csv(os.path.join(receptor_dir, f'spin_{receptor}.csv'), 
                                      nrows=n_samples, 
                                      header=None, 
                                      engine='c').values
    return nulls

def load_rotated_rois(atlas, n_permutations):
    """
    Load rotated ROIs for a given atlas and number of permutations.
    """
    assert n_permutations <= 10000, 'Maximum number of permutations is 10000.'
    spatial_coords_dir = os.path.join(project_root(), 'data', 'raw', 'spatial_coordinates', 'rotated_nullmaps')
    if atlas == 'schaefer100':
        file = os.path.join(spatial_coords_dir, 'rotated_rois_schaefer100.csv')
        # Load rotated ROIs shape: (num_regions, num_rotations) 
        rotated_roi_indices = pd.read_csv(file, header=None).iloc[:, :n_permutations].values 
    else:
        raise ValueError(f'No rotated ROIs available for atlas {atlas}. \n'
                         'Edit utils.annotations.load_rotated_rois() to add your atlas.')
    return rotated_roi_indices

def load_ut_axis(atlas: str):
    '''Loads the UT axis for the given atlas.'''
    if atlas == 'schaefer100':
        ut_axis = np.loadtxt(os.path.join(project_root(), 'data', 'raw', 'transmodal_axis', 'Schaefer100_SA_Axis.csv'))
    elif atlas == 'schaefer200':
        ut_axis = np.loadtxt(os.path.join(project_root(), 'data', 'raw', 'transmodal_axis', 'Schaefer200_SA_Axis.csv'))
    else:
        raise ValueError(f'No UT axis available for atlas {atlas}. \n'
                         'Edit utils.annotations.load_ut_axis() to add your atlas.')
    return ut_axis
