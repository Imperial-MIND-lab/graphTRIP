"""
Trains a ridge regression model on clinical data only (batch.graph_attr).

Author: Hanna Tolle
Date: 2025-11-19 
License: BSD-3-Clause
"""

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.mlp_ingredient import evaluate_regression
from experiments.ingredients.data_ingredient import *

import os
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from time import time
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger
from utils.plotting import true_vs_pred_scatter


# Create experiment and logger -------------------------------------------------
ex = Experiment('train_linreg_on_clinical', ingredients=[data_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg():
    # Experiment name and ID
    exname = 'train_linreg_on_clinical'
    jobid = 0
    seed = 0
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join('outputs', 'runs', run_name)

    # Logging and saving
    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    save_weights = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Training configurations
    ridge_alpha = 1.0     # Regularization strength for Ridge regression

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # Ensure sure that batch size is -1, val_split is 0
    batch_size = config.get('batch_size', -1)
    val_split = config.get('val_split', 0)
    assert batch_size == -1 , "Batch size must be -1 for this experiment."
    assert val_split == 0 , "Val split must be 0 for this experiment."
    return config

# Helper functions ------------------------------------------------------------
def extract_clinical_data_from_dataloader(dataloader, device):
    '''
    Extracts clinical data from a dataloader (data may be standardized).
    
    Parameters:
    ----------
    dataloader: DataLoader
    device: torch device
    
    Returns:
    -------
    y: numpy array of shape (n_samples,) - target values
    clinical_data: numpy array of shape (n_samples, n_clinical_features) - clinical data
    subject_ids: numpy array of shape (n_samples,) - subject IDs
    '''
    with torch.no_grad():
        batch = next(iter(dataloader)).to(device)
        
        # Convert to numpy
        y = batch.y.cpu().numpy().flatten()
        clinical_data = batch.graph_attr.cpu().numpy()
        subject_ids = batch.subject.cpu().numpy().flatten()
    
    return y, clinical_data, subject_ids

def extract_clinical_data_from_indices(data, indices, device):
    '''
    Extracts original (unstandardized) clinical data from dataset using indices.
    
    Parameters:
    ----------
    data: Dataset
    indices: numpy array of indices
    device: torch device
    
    Returns:
    -------
    y: numpy array of shape (n_samples,) - target values
    clinical_data: numpy array of shape (n_samples, n_clinical_features) - clinical data
    subject_ids: numpy array of shape (n_samples,) - subject IDs
    '''
    subset = data[indices]
    with torch.no_grad():
        batch = next(iter(DataLoader(subset, batch_size=len(subset), shuffle=False))).to(device)
        
        # Convert to numpy
        y = batch.y.cpu().numpy().flatten()
        clinical_data = batch.graph_attr.cpu().numpy()
        subject_ids = batch.subject.cpu().numpy().flatten()
    
    return y, clinical_data, subject_ids

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Set up environment --------------------------------------------------------
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']
    ridge_alpha = _config['ridge_alpha']
    num_folds = _config['dataset']['num_folds']
    graph_attrs = _config['dataset']['graph_attrs']

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []  

    # Load data
    data = load_data()
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Get k-fold dataloaders (use full batch -> batch size=-1)
    train_loaders, _, test_loaders, test_indices, _ = \
        get_kfold_dataloaders(data, batch_size=-1, seed=seed)

    # Train ridge regression models on clinical data only ---------------------
    start_time = time()

    # Initialize outputs dictionary
    all_outputs = init_outputs_dict(data)
    ridge_models = []

    for k in tqdm(range(num_folds), desc='Folds', disable=not verbose):
        # Extract standardized clinical data from dataloaders (for training)
        y_train, clinical_train, subject_train = extract_clinical_data_from_dataloader(
            train_loaders[k], device)
        y_test, clinical_test, subject_test = extract_clinical_data_from_dataloader(
            test_loaders[k], device)
        
        # Extract original (unstandardized) clinical data for outputs
        _, clinical_test_original, _ = extract_clinical_data_from_indices(
            data, test_indices[k], device)
        
        # Train Ridge regression model on clinical data only
        ridge_model = Ridge(alpha=ridge_alpha, random_state=seed)
        ridge_model.fit(clinical_train, y_train)
        ridge_models.append(ridge_model)
        
        # Make predictions on test set
        y_pred_test = ridge_model.predict(clinical_test)
        
        # Store outputs (using original, unstandardized clinical data)
        outputs = {'prediction': y_pred_test, 
                   'label': y_test, 
                   'subject_id': subject_test,
                   'clinical_data': []}
        for sub in range(len(subject_test)):
            outputs['clinical_data'].append(tuple(clinical_test_original[sub, :]))
        update_best_outputs(all_outputs, outputs, graph_attrs)
        
        # Evaluate on test set
        r, p, mae, mae_std = evaluate_regression(
            pd.DataFrame({'prediction': y_pred_test, 'label': y_test}))
        
        # Log metrics for this fold
        ex.log_scalar(f'fold{k}/r', r)
        ex.log_scalar(f'fold{k}/p', p)
        ex.log_scalar(f'fold{k}/mae', mae)
        ex.log_scalar(f'fold{k}/mae_std', mae_std)
        
        if verbose:
            logger.info(f"Fold {k}: r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}")
        
        # Save model weights if requested
        if save_weights:
            import pickle
            model_path = os.path.join(output_dir, f'k{k}_ridge_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(ridge_model, f)

    # Print training time
    end_time = time()
    logger.info(f"Training completed after {(end_time-start_time)/60:.2f} minutes.")

    # Save outputs as csv file
    all_outputs_df = pd.DataFrame(all_outputs)
    data_file = os.path.join(output_dir, 'prediction_results.csv')
    all_outputs_df.to_csv(data_file, index=False)

    # Evaluate overall results
    r, p, mae, mae_std = evaluate_regression(all_outputs_df)
    results = {'seed': seed, 'r': r, 'p': p, 'mae': mae, 'mae_std': mae_std}
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(output_dir, 'final_metrics.csv'), index=False)

    # Log final metrics
    for k, v in results.items():
        ex.log_scalar(f'final_prediction/{k}', v)
    logger.info(f"Final results: r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}.")

    # True vs predicted scatter plot
    save_path = os.path.join(output_dir, 'true_vs_predicted.png')
    true_vs_pred_scatter(all_outputs_df, save_path=save_path)
    image_files.append(save_path)

    # Close all plots
    plt.close('all')

    # Log all images
    for img in image_files:
        if img is not None:
            ex.add_artifact(filename=img)
