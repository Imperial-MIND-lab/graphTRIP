"""
Contains functions for initialising models and loading their default configurations.
Depends on: ./model_configs.json.

Licence: BSD-3-Clause
Author: Hanna M. Tolle
"""

import os
from typing import Dict
import json
import models

def init_model(model_type: str, params: Dict = {}):
    """Initialises model."""
    try:
        model_class = getattr(models, model_type)
    except AttributeError:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model_class(**params)
    return model

def get_model_configs(model_type, **kwargs):
    '''
    Get model parameters based on model type and provided kwargs.
    
    Parameters:
    ----------
    model_type (str): Name of the model class
    **kwargs (dict): Dictionary of all available parameters
    
    Returns:
    --------
    Dictionary of parameters specific to the model type
    '''
    # Load default configs from JSON file
    json_path = os.path.join(os.path.dirname(__file__), 'model_configs.json')
    try:
        with open(json_path, 'r') as f:
            model_configs = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model configs file not found at: {json_path}")
    
    if model_type not in model_configs:
        return {}  # Return empty dict for models without configs
        
    config = model_configs[model_type]
    params = {}
    
    # Add required parameters
    for param in config['required_params']:
        if param not in kwargs:
            raise ValueError(f"Missing required parameter '{param}' for model type '{model_type}'")
        params[param] = kwargs[param]
    
    # Add optional parameters with defaults
    for param, default in config['optional_params'].items():
        params[param] = kwargs.get(param, default)
    
    return params

def get_optional_params(model_type, **kwargs):
    '''
    Get optional parameters for a model.
    '''
    # Load default configs from JSON file
    json_path = os.path.join(os.path.dirname(__file__), 'model_configs.json')
    try:
        with open(json_path, 'r') as f:
            model_configs = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model configs file not found at: {json_path}")
    
    if model_type not in model_configs:
        return {}  # Return empty dict for models without configs
        
    config = model_configs[model_type]
    params = {}
    
    # Add optional parameters with defaults
    for param, default in config['optional_params'].items():
        params[param] = kwargs.get(param, default)
    
    return params

def freeze_model(model):
    '''Freezes all parameters in a model.'''
    for param in model.parameters():
        param.requires_grad = False
    return model

def unfreeze_model(model):
    '''Unfreezes all parameters in a model.'''
    for param in model.parameters():
        param.requires_grad = True
    return model
