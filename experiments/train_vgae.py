'''
Unsupervised training of VGAE models for feature reconstruction.
VGAE models are newly initialised, no pre-trained model loading.

Authors: Hanna M. Tolle
Date: 2025-01-08
License: BSD 3-Clause
'''

import sys
sys.path.append('graphTRIP/')

import matplotlib
matplotlib.use('Agg')

from sacred import Experiment
from experiments.ingredients.data_ingredient import * 
from experiments.ingredients.vgae_ingredient import * 

import os
import torch
import torch.nn
from tqdm import tqdm
from time import time
import copy
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger, save_test_indices
from utils.plotting import plot_vgae_reconstructions, plot_loss_curves
from preprocessing.metrics import get_rsn_mapping
from experiments.ingredients.vgae_ingredient import load_trained_vgaes


# Create experiment and logger -------------------------------------------------
ex = Experiment('train_vgae', ingredients=[data_ingredient, 
                                           vgae_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg():
    # Experiment name and ID
    exname = 'train_vgae'
    jobid = 0
    seed = 291
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join('outputs', 'runs', run_name)

    # Logging and saving
    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    save_weights = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Training configurations
    lr = 0.001            # Learning rate.
    num_epochs = 300      # Number of epochs to train.
    balance_attrs = None  # attrs to balance on for k-fold CV. If None, no balancing.
    this_k = None         # If None, train all folds sequentially. If 0 <= this_k < num_folds, train only fold this_k. If this_k == num_foldss, only load models and evaluate.

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # May add compatibility checks here ...
    return config

# Captured functions -----------------------------------------------------------
@ex.capture
def get_optimizer(vgae, lr):
    '''Creates the optimizer for VGAE training.'''
    return torch.optim.Adam(vgae.parameters(), lr=lr)

@ex.capture
def get_dataloaders(data, balance_attrs, seed):
    if balance_attrs is not None:
        train_loaders, val_loaders, test_loaders, test_indices, mean_std \
            = get_balanced_kfold_dataloaders(data, balance_attrs=balance_attrs, seed=seed)
    else:
        train_loaders, val_loaders, test_loaders, test_indices, mean_std \
            = get_kfold_dataloaders(data, seed=seed)
    return train_loaders, val_loaders, test_loaders, test_indices, mean_std

def save_single_fold_test_indices(test_indices_k, k, output_dir):
    '''
    Saves the test fold assignments for a single fold k.
    
    Parameters:
    ----------
    test_indices_k (list): list of test indices for fold k.
    k (int): fold number.
    output_dir (str): path to the output directory.
    '''
    # Create a list where test_indices_k[i] = k for all i in test_indices_k
    fold_assignments = [k] * len(test_indices_k)
    
    # Save the list as a csv file
    indices_file = os.path.join(output_dir, f'k{k}_test_fold_indices.csv')
    df = pd.DataFrame({'test_idx': test_indices_k, 'fold': fold_assignments})
    df.to_csv(indices_file, index=False, header=False)
    
    return indices_file

def aggregate_test_fold_indices(output_dir, num_folds):
    '''
    Aggregates all k{k}_test_fold_indices.csv files into a single test_fold_indices.csv.
    
    Parameters:
    ----------
    output_dir (str): path to the output directory.
    num_folds (int): number of folds.
    
    Returns:
    -------
    str: path to the aggregated test_fold_indices.csv file.
    '''
    # Find the maximum index across all folds
    max_idx = -1
    for k in range(num_folds):
        fold_file = os.path.join(output_dir, f'k{k}_test_fold_indices.csv')
        if not os.path.exists(fold_file):
            raise FileNotFoundError(f'Test fold indices file not found: {fold_file}')
        
        df = pd.read_csv(fold_file, header=None)
        if len(df) > 0:
            max_idx = max(max_idx, df[0].max())
    
    # Initialize array with -1 (unassigned)
    all_fold_assignments = [-1] * (max_idx + 1)
    
    # Load all individual fold files and assign fold numbers
    for k in range(num_folds):
        fold_file = os.path.join(output_dir, f'k{k}_test_fold_indices.csv')
        df = pd.read_csv(fold_file, header=None)
        test_indices_k = df[0].values.astype(int)
        fold_k = df[1].values.astype(int)
        
        # Store the mapping
        for test_idx, fold in zip(test_indices_k, fold_k):
            if test_idx < len(all_fold_assignments):
                all_fold_assignments[test_idx] = fold
    
    # Save the aggregated list
    indices_file = os.path.join(output_dir, 'test_fold_indices.csv')
    df = pd.DataFrame(all_fold_assignments)
    df.to_csv(indices_file, index=False, header=False)
    
    return indices_file

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Unpack configs
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']
    num_folds = _config['dataset']['num_folds']
    this_k = _config.get('this_k', None)

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []
    
    # Check configurations based on this_k
    if this_k is not None:
        if 0 <= this_k < num_folds:
            # Check if weights already exist for this fold
            weight_file = os.path.join(output_dir, f'k{this_k}_vgae_weights.pth')
            if os.path.exists(weight_file):
                logger.warning(f'Weight file {weight_file} already exists. Exiting to avoid overwriting.')
                return
        elif this_k == num_folds:
            # Check if all weight files exist
            for k in range(num_folds):
                weight_file = os.path.join(output_dir, f'k{k}_vgae_weights.pth')
                if not os.path.exists(weight_file):
                    raise FileNotFoundError(f'Weight file {weight_file} not found. Cannot perform evaluation.')
            
            # Check if all test fold indices files exist
            for k in range(num_folds):
                test_indices_file = os.path.join(output_dir, f'k{k}_test_fold_indices.csv')
                if not os.path.exists(test_indices_file):
                    raise FileNotFoundError(f'Test fold indices file {test_indices_file} not found. Cannot perform evaluation.')
            
            # Aggregate all test fold indices files
            aggregate_test_fold_indices(output_dir, num_folds)
        else:
            raise ValueError(f'Invalid this_k value: {this_k}. Must be None, 0 <= this_k < {num_folds}, or {num_folds + 1}.')    

    # Get dataloaders
    data = load_data()
    train_loaders, val_loaders, test_loaders, test_indices, mean_std \
        = get_dataloaders(data, seed=seed)
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Handle this_k == num_folds case (evaluation only)
    if this_k == num_folds:
        # Load all trained VGAEs
        weight_filenames = [f'k{k}_vgae_weights.pth' for k in range(num_folds)]
        vgaes = load_trained_vgaes(output_dir, weight_filenames, device=device)
        
        # Load aggregated test indices
        test_indices_file = os.path.join(output_dir, 'test_fold_indices.csv')
        if not os.path.exists(test_indices_file):
            raise FileNotFoundError(f'Test fold indices file not found: {test_indices_file}')
        
        # Reconstruct test_indices list from the aggregated file
        fold_assignments = np.loadtxt(test_indices_file, dtype=int)
        test_indices = [[] for _ in range(num_folds)]
        for idx, fold in enumerate(fold_assignments):
            if fold >= 0:
                test_indices[fold].append(idx)
        
        # Continue to evaluation section
        start_time = time()
        vgae_train_loss, vgae_test_loss, vgae_val_loss = {}, {}, {}
    else:
        # Train-test loop ------------------------------------------------------------
        start_time = time()

        vgae_train_loss, vgae_test_loss, vgae_val_loss = {}, {}, {}
        best_vgae_states = []

        # Determine which folds to train
        if this_k is None:
            folds_to_train = range(num_folds)
        else:
            folds_to_train = [this_k]

        for k in tqdm(folds_to_train, desc='Folds', disable=not verbose):

            # Initialise losses
            vgae_train_loss[k], vgae_test_loss[k], vgae_val_loss[k] = [], [], []
            
            # Initialise model and optimizer
            vgae = build_vgae().to(device)
            optimizer = get_optimizer(vgae)

            # Best validation loss and early stopping counter
            best_val_loss = float('inf')
            best_vgae_state = None

            for epoch in range(_config['num_epochs']):
                # Train VGAE
                _ = train_vgae(vgae, train_loaders[k], optimizer, device)
                
                # Compute training loss
                vgae_train_loss_epoch = test_vgae(vgae, train_loaders[k], device)
                vgae_train_loss[k].append(vgae_train_loss_epoch)
                
                # Test VGAE
                vgae_test_loss_epoch = test_vgae(vgae, test_loaders[k], device)
                vgae_test_loss[k].append(vgae_test_loss_epoch)

                # Log training and test losses
                ex.log_scalar(f'training/fold{k}/epoch/vgae_loss', vgae_train_loss_epoch)
                ex.log_scalar(f'test/fold{k}/epoch/vgae_loss', vgae_test_loss_epoch)
                
                # Validate model, if applicable
                if len(val_loaders) > 0:
                    vgae_val_loss_epoch = test_vgae(vgae, val_loaders[k], device)
                    vgae_val_loss[k].append(vgae_val_loss_epoch)

                    # Log validation loss
                    ex.log_scalar(f'validation/fold{k}/epoch/vgae_loss', vgae_val_loss_epoch)
                    
                    # Save the best model if validation loss is at its minimum
                    if vgae_val_loss_epoch < best_val_loss:
                        best_val_loss = vgae_val_loss_epoch
                        best_vgae_state = copy.deepcopy(vgae.state_dict()) 

            # Load best model of this fold 
            if best_vgae_state is not None:
                vgae.load_state_dict(best_vgae_state)

            # Save model weights
            if save_weights:
                torch.save(vgae.state_dict(), os.path.join(output_dir, f'k{k}_vgae_weights.pth'))

            # Keep a list of model states
            if this_k is None:
                best_vgae_states.append(copy.deepcopy(vgae.state_dict()))

            # Save test fold assignments for this specific fold
            if this_k is not None:
                save_single_fold_test_indices(test_indices[k], k, output_dir)

        # Print training time
        end_time = time()
        logger.info(f"VGAE training completed after {(end_time-start_time)/60:.2f} minutes.")

        # Save test fold assignments (only if training all folds)
        if this_k is None:
            test_indices_file = save_test_indices(test_indices, output_dir)
        else:
            # For single fold, we already saved the temporary file
            test_indices_file = None

    # Plot results ----------------------------------------------------------------
    # Load all trained VGAEs
    if this_k == num_folds:
        # VGAEs already loaded above
        pass
    elif this_k is None:
        # Load from states list (all folds trained in this run)
        vgaes = load_vgaes_from_states_list(best_vgae_states, device)
    else:
        # For single fold training, we can't do full evaluation
        # Just return early
        logger.info(f"Training completed for fold {this_k}. Run with this_k={num_folds} to perform full evaluation.")
        return
    
    # Get reconstructions
    adj_orig_rcn, x_orig_rcn, fold_assignments = get_test_reconstructions(
        vgaes, data, test_indices, mean_std=mean_std)
    
    # Evaluate FC reconstructions
    adj_orig_rcn['metrics'] = evaluate_fc_reconstructions(adj_orig_rcn)
    ex.log_scalar('final_reconstruction/fc_corr', np.mean(adj_orig_rcn['metrics']['corr']))
    ex.log_scalar('final_reconstruction/fc_mae', np.mean(adj_orig_rcn['metrics']['mae']))

    # Evaluate node feature reconstructions
    if x_orig_rcn:
        x_orig_rcn['metrics'] = evaluate_x_reconstructions(x_orig_rcn)
        for feature in x_orig_rcn['feature_names']:
            ex.log_scalar(f'final_reconstruction/x_{feature}_corr', x_orig_rcn['metrics']['corr'][feature].mean())
            ex.log_scalar(f'final_reconstruction/x_{feature}_mae', x_orig_rcn['metrics']['mae'][feature].mean())
    
    # Some other details needed for plotting
    rsn_mapping, rsn_labels = get_rsn_mapping(data.atlas)
    save_path = os.path.join(output_dir, 'vgae.png')
    vrange = None
    if _config['dataset']['edge_attrs'][0] == 'functional_connectivity':
        vrange = (-0.7, 0.7)
    image_files += plot_vgae_reconstructions(adj_orig_rcn, 
                                             x_orig_rcn, 
                                             fold_assignments, 
                                             conditions=get_conditions(data),
                                             rsn_mapping=rsn_mapping,
                                             rsn_labels=rsn_labels,
                                             atlas=data.atlas,
                                             vrange=vrange, 
                                             save_path=save_path)

    # Log images
    for img in image_files:
        if img is not None:
            ex.add_artifact(filename=img)

    # Close all plots if not verbose
    if not verbose:
        plt.close()
