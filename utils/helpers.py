"""
Helper functions for loading and processing results.

License: BSD-3-Clause
Author: Hanna M. Tolle
Date: 2024-10-31
"""

import os
import torch
import pandas as pd
import random
import numpy as np
from torch_geometric.seed import seed_everything
import importlib
import logging
from torch_geometric.data import Data
import copy
from typing import List, Optional
import glob


def rank_models(summary_df):
    '''Ranks models based on their MAE and R-squared values.'''
    summary_df['mae_rank'] = summary_df['mae'].rank()
    summary_df['r_rank'] = summary_df['r'].rank(ascending=False)
    summary_df['combined_rank'] = (summary_df['mae_rank'] + summary_df['r_rank'])/2
    return summary_df

def get_best_model(summary_df):
    '''Returns the best model based on its rank.'''
    best_model = summary_df.loc[summary_df['combined_rank'].idxmin()]
    return best_model

def triu_vector2mat(vector):
    '''Converts a vector of upper triangular elements to a matrix.'''
    vector = vector.squeeze() # Remove any extra dimensions
    if isinstance(vector, torch.Tensor):
        num_nodes = num_triu_edges2num_nodes(len(vector))
        triu_idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
        mat = torch.zeros((num_nodes, num_nodes))
        mat[triu_idx[0], triu_idx[1]] = vector
        mat[triu_idx[1], triu_idx[0]] = vector

    elif isinstance(vector, np.ndarray):
        num_nodes = num_triu_edges2num_nodes(len(vector))
        triu_idx = np.triu_indices(num_nodes, k=1)  # Changed offset to k
        mat = np.zeros((num_nodes, num_nodes))
        mat[triu_idx[0], triu_idx[1]] = vector
        mat[triu_idx[1], triu_idx[0]] = vector
        
    else:
        raise ValueError('Input must be a torch.Tensor or np.ndarray')
    return mat

def triu_vector2mat_torch(vector):
    '''
    Converts a vector of upper triangular elements to a matrix.
    For torch tensors that require gradients.
    '''
    num_nodes = num_triu_edges2num_nodes(len(vector))
    # Create indices on correct device
    triu_idx = torch.triu_indices(num_nodes, num_nodes, offset=1, 
                                device=vector.device)
    
    # Create matrix using scatter
    vector = vector.flatten()
    mat = torch.zeros((num_nodes, num_nodes), 
                     device=vector.device)
    
    # Create a temporary view of the matrix as a 1D tensor for scatter
    mat_flat = mat.view(-1)
    linear_indices = triu_idx[0] * num_nodes + triu_idx[1]
    mat_flat.scatter_(0, linear_indices, vector)
    
    # Reshape back to square matrix
    mat = mat_flat.view(num_nodes, num_nodes)
    
    # Make symmetric
    mat = mat + mat.t()
    return mat

def num_triu_edges2num_nodes(num_edges):
    '''
    Returns the number of nodes from the number of upper triangular edges.
    '''
    return int(0.5 + np.sqrt(0.25 + 2*num_edges))

def normalised_mean(feature_vectors):
    """Compute the normalized mean feature vector.
    Parameters:
    ----------
    feature_vectors (2D numpy array): (num_samples, num_features)                               
    """
    # Normalise each vector to unit norm
    vector_norm = np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    normalised_vectors = feature_vectors / vector_norm
    # Compute the mean of normalised vectors
    mean_vector = np.mean(normalised_vectors, axis=0)
    return mean_vector

def fix_random_seed(seed: int):
    '''Fixes random seed for all used libraries.'''
    # Random
    random.seed(seed)

    # Numpy  
    np.random.seed(seed) 

    # PyTorch
    torch.manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch Geometric
    seed_everything(seed)

def load_experiment(exname: str):
    '''Loads a sacred experiment by name.'''
    try:
        # Dynamically import the experiment module
        experiment_module = importlib.import_module(f'experiments.{exname}')
        # Access the 'ex' attribute of the imported module
        ex = getattr(experiment_module, 'ex')
    except ImportError:
        raise ImportError(f"Failed to import the experiment module 'experiments.{exname}'. Make sure it is implemented and available.")
    except AttributeError:
        raise AttributeError(f"The experiment module 'experiments.{exname}' does not have an attribute 'ex'.") 

    return ex 

def load_match_config(exname: str):
    '''Loads the match_config function from the experiment module.'''
    # Import the experiment module
    try:
        experiment_module = importlib.import_module(f'experiments.{exname}')
    except ImportError:
        raise ImportError(f"Failed to import the experiment module 'experiments.{exname}'. Make sure it is implemented and available.")
    
    # Return match_config function, if implemented
    if hasattr(experiment_module, 'match_config'):
        return experiment_module.match_config
    else:
        return None

def log_info(msg, verbose=True):
    '''Logs information if verbose is True.'''
    if verbose:
        print(msg)

def get_logger(name = 'exlogger'):
    logger = logging.getLogger(name)
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname).1s] %(name)s >> "%(message)s"')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def save_test_indices(test_indices, output_dir):
    '''
    Saves the test fold assignments as a list where each element corresponds 
    to the k-fold number for the corresponding dataset index.
    
    Parameters:
    ----------
    test_indices (list): nested list of test indices for each fold.
    output_dir (str): path to the output directory.
    '''
    # Initialize a list with the same length as the dataset, filled with -1
    fold_assignments = [-1] * sum(len(indices) for indices in test_indices)
    
    # Assign each index to its corresponding fold number
    for k in range(len(test_indices)):
        for idx in test_indices[k]:
            fold_assignments[idx] = k
    
    # Save the list as a csv file
    indices_file = os.path.join(output_dir, 'test_fold_indices.csv')
    df = pd.DataFrame(fold_assignments)
    df.to_csv(indices_file, index=False, header=False)
    
    return indices_file

def check_weights_exist(weights_dir: str, weight_filenames: dict):
    '''Checks if all weight files exist.'''
    all_filenames = [item for sublist in list(weight_filenames.values()) for item in sublist]
    for weight_file in all_filenames:
        if not os.path.exists(os.path.join(weights_dir, weight_file)):
            raise FileNotFoundError(f'Weight file {weight_file} not found in {weights_dir}.')
    return True

def corrmat2data(corrmat, original_data):
    '''
    Creates a new torch_geometric Data object with edge attributes from corrmat
    while preserving all other attributes from the original_data.
    
    Parameters:
    ----------
    corrmat (torch.Tensor): Correlation matrix to extract edge attributes from
    original_data (Data): Original torch_geometric Data object to copy structure from
    
    Returns:
    -------
    Data: New torch_geometric Data object
    '''
    # Extract edge attributes from correlation matrix
    edge_attr = corrmat[original_data.edge_index[0], 
                        original_data.edge_index[1]].reshape(-1, 1)
    
    # Create new Data by copying all attributes except edge_attr and edge_index
    new_data = Data(edge_attr=edge_attr,
                    edge_index=original_data.edge_index,
                    **{k: v for k, v in original_data 
                    if k not in ['edge_attr', 'edge_index']})
    return new_data

def get_modified_data_clone(original_data, **kwargs):
    '''
    Creates a modified copy of the original_data with new features as specified in kwargs.
    
    Parameters:
    ----------
    original_data : torch_geometric.data.Data
        Original data object to be modified
    **kwargs : dict
        New attributes to replace in the copy
        
    Returns:
    -------
    torch_geometric.data.Data
        New data object with modified attributes
    '''
    # Create deep copies of all attributes
    attributes = {}
    for key, value in original_data.items():
        if torch.is_tensor(value):
            # For tensors, use clone()
            attributes[key] = value.clone()
        elif isinstance(value, (list, dict, set)):
            # For Python containers, use deepcopy
            attributes[key] = copy.deepcopy(value)
        else:
            # For simple types (int, float, str), direct assignment is fine
            attributes[key] = value
            
    # Overwrite with new features from kwargs
    for k, v in kwargs.items():
        if k in attributes:
            attributes[k] = v
        else:
            raise ValueError(f'Attribute {k} not found in original data.')
            
    # Create new data object
    new_data = Data(**attributes)
    return new_data

def cohen_d(x, y):
    '''Calculates Cohen's d for two samples.'''
    mean_diff = np.mean(x) - np.mean(y)
    pooled_std = np.sqrt(((len(x) - 1) * np.var(x) + (len(y) - 1) * np.var(y)) / (len(x) + len(y) - 2))
    return mean_diff / pooled_std

# Results loading functions --------------------------------------------------

def load_optimised_original_x(results_dir: str) -> tuple:
    """
    Load the optimised_original_x.npz file from optimised_inputs.py.
    """
    npz_data = np.load(os.path.join(results_dir, 'optimised_original_x.npz'))
    return npz_data['optimised_x_means'], npz_data['original_x_means']

def load_optimised_original_adj(results_dir: str) -> tuple:
    """
    Load the optimised_original_adj.npz file from optimised_inputs.py.
    """
    npz_data = np.load(os.path.join(results_dir, 'optimised_original_adj.npz'))
    return npz_data['optimised_adj_means'], npz_data['original_adj_means']

# Sort GRAIL feature names ---------------------------------------------------
def get_groups(features):
    node_feature_corrs = ['x5-HT1A_corr_x5-HT2A', 'x5-HT1A_corr_x5-HTT', 'x5-HT2A_corr_x5-HTT']
    groups = [
        [f for f in features if f.startswith('modularity')],
        [f for f in features if f.endswith('VIS')],
        [f for f in features if f.endswith('SMN')],
        [f for f in features if f.endswith('DAN')],
        [f for f in features if f.endswith('VAN')],
        [f for f in features if f.endswith('LIM')],
        [f for f in features if f.endswith('FPN')],
        [f for f in features if f.endswith('DMN')],
        [f for f in features if f.endswith('D1')],
        [f for f in features if f.endswith('D2')],
        [f for f in features if f.endswith('DAT')],
        [f for f in features if f.endswith('NAT')],
        [f for f in features if f.endswith('VAChT')],        
        [f for f in features if f.endswith('5-HT1A')],
        [f for f in features if f.endswith('5-HT1B')],
        [f for f in features if f.endswith('5-HT2A')],
        [f for f in features if f.endswith('5-HT4')],
        [f for f in features if f.endswith('5-HTT')],
        [f for f in features if f in node_feature_corrs]
    ] 
    # Sort each group alphabetically
    for group in groups:
        group.sort()
    return groups

def sort_features(features):
    '''Sorts features into groups.'''
    groups = get_groups(features)
    sorted_features = []
    for group in groups:
        sorted_features.extend(group)
    # Add un-assigned features at the end
    unassigned_features = [f for f in features if f not in sorted_features]
    sorted_features.extend(unassigned_features)
    return sorted_features

# Results aggregation functions -----------------------------------------------

def aggregate_prediction_results(
    results_file: str,
    merge_column: str = 'subject_id', 
    subdir_name_pattern: str = 'seed_*'
) -> pd.DataFrame:
    """
    Aggregates prediction CSVs from subdirectories by averaging numeric columns
    aligned by a specific merge column (e.g., subject_id). 
    
    Args:
        results_file: Output CSV file to read from or write to.
        merge_column: The column name used to match rows across files (primary key).
        subdir_name_pattern: Pattern to identify subdirectories containing results.
    
    Returns:
        pd.DataFrame: The aggregated and averaged DataFrame.
    """
    # If the results_file exists, just load and return it
    if os.path.exists(results_file):
        print(f"Output file found at {results_file}. Loading existing results.")
        return pd.read_csv(results_file)

    # Otherwise, aggregate from subdirectories
    base_dir = os.path.dirname(results_file)
    file_name = os.path.basename(results_file)

    # Find subdirectories matching the pattern
    subdirs = [
        d for d in glob.glob(os.path.join(base_dir, subdir_name_pattern))
        if os.path.isdir(d)]

    dataframes = []
    for subdir in subdirs:
        file_path = os.path.join(subdir, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: File not found at {file_path}. Skipping.")
            continue
        try:
            df = pd.read_csv(file_path)
            if merge_column not in df.columns:
                print(f"Warning: '{merge_column}' not found in {file_path}. Skipping.")
                continue
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not dataframes:
        raise ValueError("No valid DataFrames were loaded. Check your paths and filenames.")

    # Aggregate and average
    combined_df = pd.concat(dataframes, ignore_index=True)
    aggregated_df = combined_df.groupby(merge_column).mean(numeric_only=True).reset_index()

    # Save to results_file
    output_dir = os.path.dirname(results_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    aggregated_df.to_csv(results_file, index=False)
    print(f"Aggregated results saved to {results_file}")

    return aggregated_df

def aggregate_importance_scores(
    results_file: str,
    merge_column: str = 'feature',
    subdir_name_pattern: str = 'seed_*'
) -> pd.DataFrame:
    """
    Aggregates importance CSVs from subdirectories by averaging the 'mean'
    values per feature, and computes the std and standard error (se)
    of the means for each feature.

    Args:
        results_file: Output CSV file to read from or write to.
        merge_column: The column name used to match rows across files (primary key, default 'feature').
        subdir_name_pattern: Pattern to identify subdirectories containing results.

    Returns:
        pd.DataFrame: Aggregated DataFrame with columns: feature, mean, std, se.
    """

    # If the results_file exists, just load and return it
    if os.path.exists(results_file):
        print(f"Output file found at {results_file}. Loading existing results.")
        return pd.read_csv(results_file)

    # Otherwise, aggregate from subdirectories
    base_dir = os.path.dirname(results_file)
    file_name = os.path.basename(results_file)

    subdirs = [
        d for d in glob.glob(os.path.join(base_dir, subdir_name_pattern))
        if os.path.isdir(d)
    ]

    dataframes = []
    for subdir in subdirs:
        file_path = os.path.join(subdir, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: File not found at {file_path}. Skipping.")
            continue
        try:
            df = pd.read_csv(file_path)
            # Require at least merge_column and 'mean'
            if merge_column not in df.columns or "mean" not in df.columns:
                print(f"Warning: Required columns not found in {file_path}. Skipping.")
                continue
            # Only keep feature and mean
            df = df[[merge_column, "mean"]]
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not dataframes:
        raise ValueError("No valid DataFrames were loaded. Check your paths and filenames.")

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Group by feature and calculate mean, std, and se for 'mean' values per feature
    agg = combined_df.groupby(merge_column)["mean"].agg(
        mean="mean", std="std", count="count"
    ).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    agg = agg[[merge_column, "mean", "std", "se"]]  # drop 'count'

    # Save to results_file
    output_dir = os.path.dirname(results_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    agg.to_csv(results_file, index=False)
    print(f"Aggregated importance results saved to {results_file}")

    return agg

def load_test_fold_indices(weights_base_dir, subdir_name_pattern='seed_*'):
    """
    Loads test_fold_indices.csv from all subdirectories matching a pattern in weights_base_dir.
    Returns a dictionary mapping subdir names to the loaded indices.
    """
    test_indices_dict = {}
    subdirs = sorted([
        d for d in glob.glob(os.path.join(weights_base_dir, subdir_name_pattern))
        if os.path.isdir(d)
    ])
    for subdir in subdirs:
        test_indices_path = os.path.join(subdir, 'test_fold_indices.csv')
        subdir_name = os.path.basename(subdir)
        if os.path.exists(test_indices_path):
            test_indices = np.loadtxt(test_indices_path, dtype=int)
            test_indices_dict[subdir_name] = test_indices
        else:
            print(f"Warning: {test_indices_path} not found, skipping {subdir_name}")
    return test_indices_dict, subdirs

def aggregate_attention_results(results_base_dir, subdir_name_pattern: str = 'seed_*'):
    """
    Aggregates attention, dominance analysis, and RSN attention results from multiple subdirectories.
    If the aggregated dataframes are already present in results_base_dir, loads them.
    Otherwise, computes the means across subdirectories and saves them (does NOT overwrite).
    
    Returns
    -------
    attn_weights_df: DataFrame
        Averaged mean_attention_weights_original across subdirectories.
    da_df: DataFrame
        Averaged da_receptors_utaxis_stats across subdirectories.
    rsn_attn_df: DataFrame
        Averaged mean_rsn_attention_weights_original across subdirectories.
    """
    attn_weights_file = os.path.join(results_base_dir, 'mean_attention_weights_original.csv')
    da_file = os.path.join(results_base_dir, 'da_receptors_utaxis_stats.csv')
    rsn_attn_file = os.path.join(results_base_dir, 'mean_rsn_attention_weights_original.csv')

    # Try to load if already exists
    files_exist = all([os.path.exists(f) for f in [attn_weights_file, da_file, rsn_attn_file]])
    if files_exist:
        attn_weights_df = pd.read_csv(attn_weights_file)
        da_df = pd.read_csv(da_file, index_col=0)
        rsn_attn_df = pd.read_csv(rsn_attn_file)
        return attn_weights_df, da_df, rsn_attn_df

    # Otherwise, aggregate from all subdirs
    subdirs = sorted([
        d for d in glob.glob(os.path.join(results_base_dir, subdir_name_pattern))
        if os.path.isdir(d)
    ])

    attn_weights_dfs = []
    da_dfs = []
    rsn_attn_dfs = []
    for subdir in subdirs:
        # Mean attention weights, original context
        attn_file = os.path.join(subdir, 'mean_attention_weights_original.csv')
        if os.path.exists(attn_file):
            attn_weights = pd.read_csv(attn_file)
            attn_weights_dfs.append(attn_weights)
        # Dominance analysis results
        da_subfile = os.path.join(subdir, 'da_receptors_utaxis_stats.csv')
        if os.path.exists(da_subfile):
            da_results = pd.read_csv(da_subfile, index_col=0)
            da_dfs.append(da_results)
        # Mean attention weights per resting-state network
        rsn_file = os.path.join(subdir, 'mean_rsn_attention_weights_original.csv')
        if os.path.exists(rsn_file):
            rsn_attn = pd.read_csv(rsn_file)
            rsn_attn_dfs.append(rsn_attn)

    # ATTENTION WEIGHTS
    attn_weights_df = None
    if attn_weights_dfs:
        attn_weights_df = pd.concat(attn_weights_dfs, axis=0, ignore_index=False)
        attn_weights_df = attn_weights_df.groupby(attn_weights_df.index).mean().reset_index(drop=True)
        # Save if not exists
        if not os.path.exists(attn_weights_file):
            attn_weights_df.to_csv(attn_weights_file, index=False)

    # DOMINANCE ANALYSIS (DA)
    da_df = None
    if da_dfs:
        df_concat = pd.concat(da_dfs, axis=0)
        numeric_cols = df_concat.select_dtypes(include=[np.number]).columns
        da_numeric_mean = df_concat[numeric_cols].groupby(df_concat.index).mean()
        non_numeric_cols = [col for col in df_concat.columns if col not in numeric_cols]
        if non_numeric_cols:
            da_non_numeric = df_concat[non_numeric_cols].groupby(df_concat.index).first()
            da_df = pd.concat([da_numeric_mean, da_non_numeric], axis=1)
            da_df = da_df[df_concat.columns]
        else:
            da_df = da_numeric_mean
        # Save if not exists
        if not os.path.exists(da_file):
            da_df.to_csv(da_file) 

    # RSN ATTENTION WEIGHTS
    rsn_attn_df = None
    if rsn_attn_dfs:
        rsn_attn_df = pd.concat(rsn_attn_dfs, axis=0, ignore_index=False)
        rsn_attn_df = rsn_attn_df.groupby(rsn_attn_df.index).mean().reset_index(drop=True)
        if not os.path.exists(rsn_attn_file):
            rsn_attn_df.to_csv(rsn_attn_file, index=False)

    return attn_weights_df, da_df, rsn_attn_df