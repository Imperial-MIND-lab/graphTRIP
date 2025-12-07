"""
Loads a pre-trained VGAE and trains a new MLP on the same
dataset, using the same testfold indices.

Author: Hanna Tolle
Date: 2025-12-05 
License: BSD-3-Clause
"""

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.data_ingredient import * 
from experiments.ingredients.vgae_ingredient import * 
from experiments.ingredients.mlp_ingredient import * 

import os
import numpy as np
import torch
import torch.nn
from tqdm import tqdm
from time import time
import copy
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import Dict

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger, save_test_indices, check_weights_exist
from utils.plotting import plot_vgae_reconstructions, plot_loss_curves, true_vs_pred_scatter
from utils.configs import load_ingredient_configs, match_ingredient_configs
from preprocessing.metrics import get_rsn_mapping
from models.utils import freeze_model


# Create experiment and logger -------------------------------------------------
ex = Experiment('retrain_mlp', ingredients=[data_ingredient, 
                                              vgae_ingredient, 
                                              mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'retrain_mlp'
    jobid = 0
    seed = 0
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join('outputs', 'runs', run_name)

    # Logging and saving
    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    save_weights = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Directory with pre-trained model weights
    weights_dir = os.path.join('outputs', 'weights')

    # If multiple weights are given, evaluate all but finetune only the first
    weight_filenames = {'vgae': [f'k{k}_vgae_weights.pth' for k in range(dataset['num_folds'])],
                        'mlp': [f'k{k}_mlp_weights.pth' for k in range(dataset['num_folds'])],
                        'test_fold_indices': ['test_fold_indices.csv']}

    # Training configurations (only used if num_epochs > 0)
    num_epochs = 100      # Number of epochs for MLP training.
    vgae_lr = 0.0         # VGAE learning rate.
    mlp_lr = 0.001        # MLP learning rate.
    num_z_samples = 1     # 0 for training MLP on the means of VGAE latent variables.
    alpha = 0             # Loss = alpha*vgae_loss + (1-alpha)*mlp_loss; if alpha=0, VGAE is frozen.
    reinit_pooling = True # Whether to reinitialise the VGAE pooling module

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # Get weights_dir (must be in the config)
    assert 'weights_dir' in config, "weights_dir must be specified in config."
    weights_dir = add_project_root(config['weights_dir'])

    # Load the VGAE and dataset configs from weights_dir
    ingredients = ['vgae_model', 'dataset']     
    previous_config = load_ingredient_configs(weights_dir, ingredients)
    
    # Various MLP related configs may mismatch, but other configs must match
    config_updates = copy.deepcopy(config)
    exceptions = ['target', 'graph_attrs', 'graph_attrs_to_standardise', 'num_graph_attr']
    reinit_pooling = config.get('reinit_pooling', True)
    if reinit_pooling: # new pooling, so pooling configs don't need to match
        exceptions.extend(['pooling_cfg', 'context_attrs'])
    config_updates = match_ingredient_configs(config=config,
                                              previous_config=previous_config,
                                              ingredients=ingredients,
                                              exceptions=exceptions)
        
    # If alpha=0, vgae_lr is set to 0 (issue warning if overridden)
    alpha = config_updates.get('alpha', 0.2)
    vgae_lr = config_updates.get('vgae_lr', 0.0001)
    if alpha == 0:
        if vgae_lr != 0:
            print(f"Warning: alpha=0 overrides vgae_lr={vgae_lr} to 0.")
        config_updates['vgae_lr'] = 0
    elif alpha > 0:
        # Make sure that vgae_lr is not 0
        assert vgae_lr != 0, "VGAE learning rate must be non-zero when alpha > 0."

    # Don't support standardisation of x in pretrained models
    if 'standardise_x' in previous_config['dataset']:
        assert not config_updates['dataset']['standardise_x'], \
            "Standardisation of x in pretrained models is not supported."

    return config_updates

# Captured functions -----------------------------------------------------------
@ex.capture
def get_optimizer(vgae, mlp, vgae_lr, mlp_lr):
    '''Creates the optimizer for the joint training of VGAE and MLP.'''
    if vgae_lr == 0:
        optimizer = torch.optim.Adam(mlp.parameters(), lr=mlp_lr)
    else:
        optimizer = torch.optim.Adam([
            {'params': vgae.parameters(), 'lr': vgae_lr},
            {'params': mlp.parameters(), 'lr': mlp_lr}])
    return optimizer

@ex.capture
def train_mlp_only(vgae, mlp, loader, optimizer, device, num_z_samples):
    vgae.eval()
    mlp.train()
    train_loss = 0.
    for batch in loader: 
        batch = batch.to(device)             
        ytrue = get_labels(batch, num_z_samples)
        optimizer.zero_grad()
        x = get_x_with_vgae(batch, device, vgae, num_z_samples)
        ypred = mlp(x)                     
        loss = mlp.loss(ypred, ytrue)
        loss.backward()                      
        optimizer.step()                     
        train_loss += loss.item()
    return train_loss/len(loader)

@ex.capture
def train_vgae_mlp(vgae, mlp, loader, optimizer, device, num_z_samples, alpha):
    if alpha == 0:
        return train_mlp_only(vgae, mlp, loader, optimizer, device, num_z_samples)
    else:
        return train_joint_vgae_mlp(vgae, mlp, loader, optimizer, 
                                    device=device, num_z_samples=num_z_samples, alpha=alpha)

@ex.capture
def test_vgae_mlp(vgae, mlp, loader, device, num_z_samples):
    vgae_test_loss, mlp_test_loss = test_joint_vgae_mlp(vgae, mlp, loader, 
                                                        device=device, num_z_samples=num_z_samples)
    return vgae_test_loss, mlp_test_loss

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Set up environment --------------------------------------------------------
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']
    alpha = _config['alpha']
    num_epochs = _config['num_epochs']
    num_folds = _config['dataset']['num_folds']
    weights_dir = add_project_root(_config['weights_dir'])
    weight_filenames = _config['weight_filenames']
    reinit_pooling = _config['reinit_pooling']
    check_weights_exist(weights_dir, weight_filenames)

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []    

    # Load data
    data = load_data()
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')
    test_indices = np.loadtxt(os.path.join(weights_dir, weight_filenames['test_fold_indices'][0]), dtype=int)
    train_loaders, val_loaders, test_loaders, mean_std \
            = get_dataloaders_from_test_indices(data, test_indices, seed=seed)

    # Get test-indices list for saving and VGAE reconstructions
    test_indices_list = [np.where(test_indices == fold)[0] for fold in range(num_folds)]

    # Load pretrained VGAEs
    if reinit_pooling:
        vgaes = load_trained_vgaes(weights_dir, weight_filenames['vgae'], 
                                              device=device, exclude_module='pooling')
    else:
        vgaes = load_trained_vgaes(weights_dir, weight_filenames['vgae'], device=device)

    # Train MLP and potentially finetune VGAE -----------------------------------------------------
    start_time = time()

    best_outputs = init_outputs_dict(data)
    vgae_train_loss, vgae_test_loss, vgae_val_loss = {}, {}, {}
    mlp_train_loss, mlp_test_loss, mlp_val_loss = {}, {}, {}

    # Train a single MLP for each fold using this fold's VGAE and train-test split
    for k in tqdm(range(num_folds), desc='Folds', disable=not verbose):

        # Initialise losses
        mlp_train_loss[k], mlp_test_loss[k], mlp_val_loss[k] = [], [], []
        vgae_train_loss[k], vgae_test_loss[k], vgae_val_loss[k] = [], [], []
        
        # Build VGAE and MLP
        vgae = vgaes[k].to(device)
        if alpha == 0: # freeze the VGAE, if not trained
            vgae.eval()
            vgae = freeze_model(vgae)
        mlp = build_mlp(latent_dim = vgae.readout_dim).to(device)

        # If alpha=0, only MLP parameters will be passed to the optimizer
        optimizer = get_optimizer(vgae, mlp)

        # Best validation loss and early stopping counter
        best_val_loss = float('inf')
        best_vgae_state = None
        best_mlp_state = None

        for epoch in range(num_epochs):
            # Train VGAE and MLP
            _ = train_vgae_mlp(vgae, mlp, train_loaders[k], optimizer, device)
            
            # Compute training losses
            vgae_train_loss_epoch, mlp_train_loss_epoch = test_vgae_mlp(vgae, mlp, train_loaders[k], device)
            vgae_train_loss[k].append(vgae_train_loss_epoch)
            mlp_train_loss[k].append(mlp_train_loss_epoch)
            
            # Test VGAE and MLP
            vgae_test_loss_epoch, mlp_test_loss_epoch = test_vgae_mlp(vgae, mlp, test_loaders[k], device)
            vgae_test_loss[k].append(vgae_test_loss_epoch)
            mlp_test_loss[k].append(mlp_test_loss_epoch)

            # Log training and test losses
            ex.log_scalar(f'test/fold{k}/epoch/mlp_loss', mlp_test_loss_epoch)
            ex.log_scalar(f'training/fold{k}/epoch/mlp_loss', mlp_train_loss_epoch)
            if alpha > 0:
                ex.log_scalar(f'training/fold{k}/epoch/vgae_loss', vgae_train_loss_epoch)
                ex.log_scalar(f'test/fold{k}/epoch/vgae_loss', vgae_test_loss_epoch)                    

            # Validate models, if applicable
            if len(val_loaders) > 0:
                vgae_val_loss_epoch, mlp_val_loss_epoch = test_vgae_mlp(vgae, mlp, val_loaders[k], device)
                vgae_val_loss[k].append(vgae_val_loss_epoch)
                mlp_val_loss[k].append(mlp_val_loss_epoch)

                # Log validation losses
                ex.log_scalar(f'validation/fold{k}/epoch/vgae_loss', vgae_val_loss_epoch)
                ex.log_scalar(f'validation/fold{k}/epoch/mlp_loss', mlp_val_loss_epoch)
                
                # Save the best model if validation loss is at its minimum
                total_val_loss = alpha*vgae_val_loss_epoch + (1-alpha)*mlp_val_loss_epoch
                if total_val_loss < best_val_loss:
                    best_val_loss = total_val_loss
                    best_vgae_state = copy.deepcopy(vgae.state_dict()) 
                    best_mlp_state = copy.deepcopy(mlp.state_dict()) 

        # Load best model
        if best_vgae_state is not None:
            vgae.load_state_dict(best_vgae_state)
        if best_mlp_state is not None:
            mlp.load_state_dict(best_mlp_state)

        # Save model weights
        if save_weights:
            torch.save(mlp.state_dict(), os.path.join(output_dir, f'k{k}_mlp_weights.pth'))
            torch.save(vgae.state_dict(), os.path.join(output_dir, f'k{k}_vgae_weights.pth'))

        # Save the test predictions (on VGAE latent means) of the best model
        outputs = get_mlp_outputs_nograd(mlp, test_loaders[k], device, 
                                        get_x=get_x_with_vgae, 
                                        vgae=vgae, num_z_samples=0)
        update_best_outputs(best_outputs, outputs)

    # Print training time
    end_time = time()
    logger.info(f"Training completed after {(end_time-start_time)/60:.2f} minutes.")

    # Save ouputs as csv file
    best_outputs = pd.DataFrame(best_outputs)
    best_outputs = add_drug_condition_to_outputs(best_outputs, _config['dataset']['study'])
    data_file = os.path.join(output_dir, 'prediction_results.csv')
    best_outputs.to_csv(data_file, index=False)

    # Save test fold assignments
    if save_weights:
        _ = save_test_indices(test_indices_list, output_dir)

    # Save final prediction results
    r, p, mae, mae_std = evaluate_regression(best_outputs)
    results = {'seed': seed, 'r': r, 'p': p, 'mae': mae, 'mae_std': mae_std}
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(output_dir, 'final_metrics.csv'), index=False)

    # Log final metrics
    for k, v in results.items():
        ex.log_scalar(f'final_prediction/{k}', v)
    logger.info(f"Final results: r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}.")

    # Plot results ----------------------------------------------------------------
    # Only plot the reconstructions if the VGAE was fine-tuned
    if alpha > 0:
        rsn_mapping, rsn_labels = get_rsn_mapping(data.atlas)
        adj_orig_rcn, x_orig_rcn, fold_assignments = get_test_reconstructions(
            vgaes, data, test_indices_list, mean_std=None, device=device)
        
        # Evaluate reconstructions
        adj_orig_rcn['metrics'] = evaluate_fc_reconstructions(adj_orig_rcn)
        ex.log_scalar('final_reconstruction/fc_corr', np.mean(adj_orig_rcn['metrics']['corr']))
        ex.log_scalar('final_reconstruction/fc_mae', np.mean(adj_orig_rcn['metrics']['mae']))

        if x_orig_rcn is not None and 'feature_names' in x_orig_rcn:
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

        # VGAE Loss curves
        save_path_vgae = os.path.join(output_dir, 'vgae_loss_curves.png')
        plot_loss_curves(vgae_train_loss, vgae_test_loss, vgae_val_loss, save_path=save_path_vgae)
        image_files.append(save_path_vgae)

    # MLP loss curves
    save_path_mlp = os.path.join(output_dir, 'mlp_loss_curves.png')
    plot_loss_curves(mlp_train_loss, mlp_test_loss, mlp_val_loss, save_path=save_path_mlp)
    image_files.append(save_path_mlp)

    # True vs predicted scatter
    save_path = os.path.join(output_dir, 'true_vs_predicted.png')
    true_vs_pred_scatter(best_outputs, save_path=save_path)
    image_files.append(save_path)

    # Close all plots
    plt.close('all')

    # Log all images
    for img in image_files:
        if img is not None:
            ex.add_artifact(filename=img)
