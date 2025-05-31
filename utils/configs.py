"""
Utility functions for managing configuration files.

License: BSD-3-Clause
Author: Hanna M. Tolle
Date: 2024-10-31
"""

import os
import json
import itertools
from typing import List, Dict
from collections.abc import Mapping
import numpy as np
import pandas as pd
import copy


def clean_config(data):
    """
    Recursively cleans a dictionary by removing 'py/object', 'dtype', and 'py/id' keys.
    Also, extracts 'value' when it exists.
    """
    if isinstance(data, dict):
        cleaned_data = {}
        for key, value in data.items():
            if key in ['py/object', 'dtype', 'py/id']:
                continue
            if isinstance(value, dict) and 'value' in value:
                cleaned_data[key] = clean_config(value['value'])
            else:
                cleaned_data[key] = clean_config(value)
        return cleaned_data
    elif isinstance(data, list):
        return [clean_config(item) for item in data]
    else:
        return data

def load_configs_from_json(json_file):
    '''
    Loads configurations from a json file, removing unwanted fields.
    '''
    with open(json_file, 'r') as f:
        configs = json.load(f)
    return configs #clean_config(configs)

def make_config_grid(config_ranges):
    '''
    Generates a grid of configurations from a search space. 
    Parameters:
    ----------
    config_ranges (dict): Possibly nested dict with config names and value options.
              If the value of a config is an iterable (such as a list), 
              and if there should be multiple options (different versions
              of the list), then it should be represented as a nested list. 
              If one config parameter has only 1 value (no options), it still
              needs to be within a list (a list with only 1 item).
    Returns:
    -------
    dict    : Yields one dictionary in the grid of options at a time.
    '''
    def recursive_items(d):
        ''' 
        Helper function to recursively traverse dictionary items 
        and yield leaf node paths and values.
        '''
        for key, value in d.items():
            if isinstance(value, Mapping):
                for sub_key, sub_value in recursive_items(value):
                    yield (key + '.' + sub_key, sub_value)
            else:
                yield (key, value)

    def set_in_dict(d, keys, value):
        '''
        Helper function to set a value in a nested dictionary 
        given a list of keys.
        '''
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    # Flatten the configuration dictionary
    flat_cfg = dict(recursive_items(config_ranges))

    # Generate all combinations of the configuration values
    keys, values = zip(*[(k, v) for k, v in flat_cfg.items() if isinstance(v, list)])
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Reconstruct the nested dictionary structure for each combination
    for combo in combinations:
        nested_combo = {}
        for k, v in flat_cfg.items():
            if not isinstance(v, list):
                set_in_dict(nested_combo, k.split('.'), v)
        for k, v in combo.items():
            set_in_dict(nested_combo, k.split('.'), v)
        yield nested_combo

def fetch_job_config(config_ranges, jobid):
    '''
    Returns the config updates for the given job ID.
    '''
    configs = list(make_config_grid(config_ranges))
    if jobid >= len(configs):
        raise IndexError("Job ID exceeds the number of configurations.")
    return configs[jobid]

def load_weight_filenames(source_dir: str, model_name: str):
    '''Loads filenames of trained model weights from the source directory.'''

    # Check if the source directory exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

    # Fetch the filenames of the model weights
    weight_files = []
    for filename in os.listdir(source_dir):
        if filename.endswith('.pth') and model_name in filename:
            weight_files.append(filename)

    # Sort by k{integer} if present
    def get_k_number(filename):
        import re
        match = re.search(r'k(\d+)_', filename)
        return int(match.group(1)) if match else float('inf')
    
    weight_files.sort(key=get_k_number)
    return weight_files

def get_weight_filenames_from_config(config: dict):
    '''
    Returns the weight filenames from the config.
    '''
    weight_filenames = {}
    num_folds = config['dataset'].get('num_folds', 1)
    if 'vgae_model' in config:
        weight_filenames['vgae'] = [f'k{k}_vgae_weights.pth' for k in range(num_folds)]
    if 'mlp_model' in config:
        weight_filenames['mlp'] = [f'k{k}_mlp_weights.pth' for k in range(num_folds)]
    weight_filenames['test_fold_indices'] = ['test_fold_indices.csv']
    return weight_filenames

def load_ingredient_configs(source_dir: str, ingredients: List[str] = ['dataset', 'vgae_model', 'mlp_model']):
    '''Loads ingredient configurations of a previous run.'''

    # Check if the source directory exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

    # Load the configs of the saved run with default configs
    print("Loading model configurations.")
    config_file = os.path.join(source_dir, 'config.json')
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found.")
    configs = load_configs_from_json(config_file)
    
    config_list = [configs.get(ingredient, {}) for ingredient in ingredients]
    return {ingredient: cfg for ingredient, cfg in zip(ingredients, config_list)}

def sanitize_config(config):
    """
    Recursively convert all numpy data types in the config dictionary to their native Python equivalents.
    """
    if isinstance(config, dict):
        return {k: sanitize_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [sanitize_config(item) for item in config]
    elif isinstance(config, np.integer):
        return int(config)
    elif isinstance(config, np.floating):
        return float(config)
    elif isinstance(config, np.bool_):
        return bool(config)
    else:
        return config

def get_config_from_table(config_table, i):
    '''
    Returns a config dictionary from a config table, as downloaded from neptune.ai.

    Parameters:
    ----------
    config_table (str): Path to the CSV file containing the config table.
    i (int): Index of the row to extract the config from.

    Returns:
    -------
    config (dict): Dictionary containing the config.
    '''
    # Read the CSV file into a dataframe
    df = pd.read_csv(config_table)

    # Initialize the configuration dictionary
    config = {}

    # Iterate over each column in the dataframe
    for col in df.columns:
        if col.startswith('experiment/config/'):
            # Extract the relevant part of the column name after 'experiment/config/'
            keys = col[len('experiment/config/'):].split()

            # Get the value from the i-th row of the current column
            value = df.loc[i, col]

            # If the value is nan, set it to None
            if pd.isna(value):
                value = None

            # If value is a string representation of a list, convert it to an actual list
            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                # Remove brackets and split by comma to convert string to list
                value = value.strip('[]').split(',')
                # Clean up whitespace and convert numeric strings to numbers
                value = [item.strip() for item in value]
                # Remove quotes from string items if they are wrapped in quotes
                value = [item.strip("'") for item in value]
                value = [float(item) if item.replace('.','').isdigit() else item for item in value]

            # Create the nested dictionary based on the keys
            d = config
            for key in keys[:-1]:  # Traverse until the second-to-last key
                if key not in d:
                    d[key] = {}
                d = d[key]
            d[keys[-1]] = value  # Set the final key to the value

    # Convert numpy data types to native Python types
    config = sanitize_config(config)

    return config

def check_if_configs_match(configs, reference_configs, exceptions=[], print_all_mismatches=False):
    '''
    Compares all entries (including nested ones) of the two config dictionaries,
    and returns True if they match, False otherwise. Entries in the exceptions list
    are not compared.
    
    Parameters:
    ----------
    configs (dict): Dictionary to compare
    reference_configs (dict): Reference dictionary to compare against
    exceptions (list): List of keys to ignore in comparison
    print_all_mismatches (bool): If True, prints all mismatches instead of stopping at first one
    
    Returns:
    -------
    bool: True if dictionaries match (excluding exceptions), False otherwise
    '''
    all_match = True
    
    for k, v in configs.items():
        if k in exceptions:
            continue

        if k not in reference_configs:
            print(f"Missing key in reference_configs: {k}")
            if not print_all_mismatches:
                return False
            all_match = False
            continue
        
        if isinstance(v, dict) and isinstance(reference_configs[k], dict):
            # Recursively check nested dictionaries
            nested_match = check_if_configs_match(v, reference_configs[k], exceptions, print_all_mismatches)
            if not nested_match:
                if not print_all_mismatches:
                    return False
                all_match = False
            
        elif v != reference_configs[k]:
            print(f"Mismatch found at key: {k}")
            print(f"  configs value: {v}")
            print(f"  reference value: {reference_configs[k]}")
            if not print_all_mismatches:
                return False
            all_match = False
        
    # Check if reference_configs has any extra keys that configs doesn't have
    for k in reference_configs:
        if k not in exceptions and k not in configs:
            print(f"Missing key in configs: {k}")
            if not print_all_mismatches:
                return False
            all_match = False
            
    return all_match

def fill_missing_configs(config, reference_config, exceptions=None):
    '''Fills missing configs with the reference config.
    
    Parameters:
    -----------
    config: dict
        The config dictionary to fill
    reference_config: dict
        The reference config dictionary to fill from
    exceptions: list of str, optional
        List of dictionary keys that should not be recursively filled if they exist
    '''
    if exceptions is None:
        exceptions = []
        
    for k, v in reference_config.items():
        if isinstance(v, dict):
            if k not in config:
                config[k] = copy.deepcopy(v)
            else:
                if k in exceptions and isinstance(v, dict):
                    continue
                else:
                    fill_missing_configs(config[k], v, exceptions)
        else:
            if k not in config:
                config[k] = v
    return config

def match_ingredient_configs(config: Dict, 
                             previous_config: Dict, 
                             ingredients: List[str], 
                             exceptions: List[str] = None) -> Dict:
    """
    Match ingredient configurations between current and previous configs.
    
    Parameters:
    ----------
    config: Current configuration dictionary
    previous_config: Previous configuration dictionary to match against
    ingredients: List of ingredient names to process
    exceptions: List of config keys to ignore during matching
    
    Returns:
    -------
    config_updates: Updated configuration dictionary
    """
    exceptions = exceptions or []
    config_updates = copy.deepcopy(config)
    
    for ingredient in ingredients:
        if ingredient in config:
            # Copy all configs from previous_config to config_updates
            config[ingredient] = fill_missing_configs(
                config=config[ingredient],
                reference_config=previous_config[ingredient],
                exceptions=exceptions)

            # Ensure that remaining configs match
            if not check_if_configs_match(
                configs=config[ingredient],
                reference_configs=previous_config[ingredient],
                exceptions=exceptions,
                print_all_mismatches=True):
                raise ValueError(f"Previous {ingredient} configs do not match the current ones.")
            
            # Update config_updates
            config_updates[ingredient] = copy.deepcopy(config[ingredient])
            
        else:
            # If missing, copy ingredient config from previous_config
            config_updates[ingredient] = copy.deepcopy(previous_config[ingredient])
            
    return config_updates
