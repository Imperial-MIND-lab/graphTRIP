"""
Functions for loading models and datasets from experiment folders.

Author: Hanna Tolle
Date: 2025-01-12
License: BSD-3-Clause
"""

import sys
sys.path.append('../')

import os
import importlib
import copy
import torch

from utils.configs import fill_missing_configs
from experiments.ingredients.data_ingredient import load_dataset_from_configs
from models.utils import init_model

def init_vgae(config: dict):
    """
    Builds a new VGAE model according to the configs in the weights directory.
    """
    # Combine all config dictionaries
    combined_params = {'params': config['params'],
                       'node_emb_model_cfg': config['node_emb_model_cfg'],
                       'pooling_cfg': config['pooling_cfg'],
                       'encoder_cfg': config['encoder_cfg'],
                       'node_decoder_cfg': config['node_decoder_cfg'],
                       'edge_decoder_cfg': config['edge_decoder_cfg']}
    
    # Add edge index decoder config for NodeLevelVGAEs
    if config['model_type'] != 'GraphLevelVGAE':
        combined_params['edge_idx_decoder_cfg'] = config['edge_idx_decoder_cfg']
    return init_model(config['model_type'], combined_params)

def load_vgaes(config: dict, weights_dir: str, weight_filenames: list) -> list:
    """
    Loads VGAEs from the weights directory (most contain config.json).

    Parameters:
    ----------
        config (dict): The VGAE configuration dictionary.
        weights_dir (str): The directory containing the VGAE weights.
        weight_filenames (list): The list of VGAE weight filenames.

    Returns:
    -------
        list: The list of loaded VGAEs.
    """       
    vgaes = []
    for weight_file in weight_filenames:
        vgae = init_vgae(config)
        vgae.load_state_dict(torch.load(os.path.join(weights_dir, weight_file)))
        vgaes.append(vgae)
    return vgaes

def init_mlp(model_type: str, params: dict, extra_dim: int, latent_dim):
    """
    Builds a new MLP model according to the configs in the weights directory.
    """
    params = copy.deepcopy(params)
    params['input_dim'] = latent_dim + extra_dim
    if 'hidden_dim' not in params or params['hidden_dim'] is None:
        params['hidden_dim'] = max(latent_dim, extra_dim)
    return init_model(model_type, params)

def load_mlps(config: dict, latent_dim: int, weights_dir: str, weight_filenames: list):
    """
    Loads MLPs from the weights directory (most contain config.json).

    Parameters:
    ----------
        config (dict): The VGAE configuration dictionary.
        latent_dim (int): The latent dimension of the VGAE.
        weights_dir (str): The directory containing the VGAE weights.
        weight_filenames (list): The list of VGAE weight filenames.

    Returns:
    -------
        list: The list of loaded MLPs.
    """
    # Load configs
    extra_dim = config.get('extra_dim', None) or len(config['dataset']['graph_attrs'])
    hidden_dim = config.get('hidden_dim', None) or max(latent_dim, extra_dim)
    model_type = config.get('model_type', None)
    params = config.get('params', {})
    params['hidden_dim'] = hidden_dim

    mlps = []
    for weight_file in weight_filenames:
        mlp = init_mlp(model_type, params, extra_dim, latent_dim)
        mlp.load_state_dict(torch.load(os.path.join(weights_dir, weight_file)))
        mlps.append(mlp)
    return mlps

def load_dataset(config: dict):
    """
    Loads a BrainGraphDataset from the dataset directory.
    """
    # Load default configs and fill in missing configs
    data_ingredient = importlib.import_module('experiments.ingredients.data_ingredient')
    default_config = data_ingredient.data_cfg()
    config = fill_missing_configs(config, reference_config=default_config)

    # Load the dataset
    dataset = load_dataset_from_configs(config)
    return dataset