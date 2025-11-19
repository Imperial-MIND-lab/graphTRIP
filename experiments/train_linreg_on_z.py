"""
Loads a pre-trained VGAE, obtains latent representations,
and trains a ridge regression model on the latent representations.

Author: Hanna Tolle
Date: 2025-11-19 
License: BSD-3-Clause
"""

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.data_ingredient import * 
from experiments.ingredients.vgae_ingredient import * 
from experiments.ingredients.mlp_ingredient import * 

import os
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from time import time
import copy
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from typing import Dict

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger, check_weights_exist
from utils.plotting import true_vs_pred_scatter
from utils.configs import load_ingredient_configs, match_ingredient_configs
from models.utils import freeze_model


# Create experiment and logger -------------------------------------------------
ex = Experiment('train_linreg_on_z', ingredients=[data_ingredient, 
                                                  vgae_ingredient])
logger = get_logger()
ex.logger = logger

# Helper functions ------------------------------------------------------------
def extract_latent_representations(vgae, data, device):
    '''
    Extracts latent representations (z) from a VGAE for the given data.
    
    Parameters:
    ----------
    vgae: Trained VGAE model
    data: Dataset or list of data samples
    device: torch device
    
    Returns:
    -------
    z: numpy array of shape (n_samples, latent_dim) - latent representations
    y: numpy array of shape (n_samples,) - target values
    clinical_data: numpy array of shape (n_samples, n_clinical_features) - clinical data
    subject_ids: numpy array of shape (n_samples,) - subject IDs
    '''
    vgae = vgae.eval().to(device)
    with torch.no_grad():
        batch = next(iter(DataLoader(data, batch_size=len(data), shuffle=False))).to(device)
        context = get_context(batch)
        out = vgae(batch)
        z = vgae.readout(out.mu, context, batch.batch)
        
        # Convert to numpy
        z = z.cpu().numpy()
        y = batch.y.cpu().numpy().flatten()
        clinical_data = batch.graph_attr.cpu().numpy()
        subject_ids = batch.subject.cpu().numpy().flatten()
    
    return z, y, clinical_data, subject_ids

# Define configurations --------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'train_linreg_on_z'
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
    weight_filenames = {'vgae': [f'k{k}_vgae_weights.pth' for k in range(dataset['num_folds'])],
                        'test_fold_indices': ['test_fold_indices.csv']} 

    # Training configurations
    ridge_alpha = 1.0     # Regularization strength for Ridge regression
    n_pca_components = 0  # Performs PCA on Z if n_pca_components > 0

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # Get weights_dir (must be in the config)
    assert 'weights_dir' in config, "weights_dir must be specified in config."
    weights_dir = add_project_root(config['weights_dir'])

    # Load the VGAE and dataset configs from weights_dir
    ingredients = ['vgae_model', 'dataset']     
    previous_config = load_ingredient_configs(weights_dir, ingredients)

    # Various dataset related configs may mismatch, but other configs must match
    config_updates = copy.deepcopy(config)
    exceptions = ['graph_attrs', 'target']
    config_updates = match_ingredient_configs(config=config,
                                              previous_config=previous_config,
                                              ingredients=ingredients,
                                              exceptions=exceptions)

    # Other config checks
    num_pretrained_models = previous_config['dataset']['num_folds']
    default_weight_filenames = {'vgae': [f'k{i}_vgae_weights.pth' for i in range(num_pretrained_models)]}
    weight_filenames = config.get('weight_filenames', default_weight_filenames)
    check_weights_exist(weights_dir, weight_filenames)
    config_updates['weight_filenames'] = weight_filenames
    
    return config_updates

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Set up environment --------------------------------------------------------
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']
    ridge_alpha = _config['ridge_alpha']
    n_pca_components = _config['n_pca_components']
    num_folds = _config['dataset']['num_folds']
    weights_dir = add_project_root(_config['weights_dir'])
    weight_filenames = _config['weight_filenames']

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []  

    # Load data
    data = load_data()
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Load pretrained VGAEs
    pretrained_vgaes = load_trained_vgaes(weights_dir, weight_filenames['vgae'], device=device)  

    # Get test fold indices
    test_indices = np.loadtxt(os.path.join(weights_dir, weight_filenames['test_fold_indices'][0]), dtype=int)

    # Train ridge regression models on VGAE latent representations ------------
    start_time = time()

    # Initialize outputs dictionary
    all_outputs = init_outputs_dict(data)
    ridge_models = []
    pca_transformers = []
    pca_var_explained = []

    for k in tqdm(range(num_folds), desc='Folds', disable=not verbose):
        # Get pretrained VGAE for this fold
        vgae = pretrained_vgaes[k].to(device)
        vgae = freeze_model(vgae)
        
        # Get train and test data for this fold
        train_data = data[test_indices != k]
        test_data = data[test_indices == k]
        
        # Extract latent representations for train and test sets
        z_train, y_train, clinical_train, subject_train = extract_latent_representations(
            vgae, train_data, device)
        z_test, y_test, clinical_test, subject_test = extract_latent_representations(
            vgae, test_data, device)

        # We actually want to train on [z, clinical_data] -> the MLP input in train_jointly.py
        x_train = np.concatenate([z_train, clinical_train], axis=1)
        x_test = np.concatenate([z_test, clinical_test], axis=1)
        
        # Apply PCA if requested
        pca = None
        if n_pca_components > 0:
            pca = PCA(n_components=n_pca_components, random_state=seed)
            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)
            logger.info(f"Fold {k}: Applied PCA, reduced from {vgae.readout_dim} to {n_pca_components} dimensions")
            var_expl_dict = {f'PC{i+1}': pca.explained_variance_ratio_[i] for i in range(n_pca_components)}
            pca_var_explained.append(var_expl_dict)
        pca_transformers.append(pca)
        
        # Train Ridge regression model
        ridge_model = Ridge(alpha=ridge_alpha, random_state=seed)
        ridge_model.fit(x_train, y_train)
        ridge_models.append(ridge_model)
        
        # Make predictions on test set
        y_pred_test = ridge_model.predict(x_test)
        
        # Store outputs
        outputs = {'prediction': y_pred_test, 
                   'label': y_test, 
                   'subject_id': subject_test,
                   'clinical_data': []}
        for sub in range(len(subject_test)):
            outputs['clinical_data'].append(tuple(clinical_test[sub, :]))
        update_best_outputs(all_outputs, outputs)
        
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
            
            if pca is not None:
                pca_path = os.path.join(output_dir, f'k{k}_pca_transformer.pkl')
                with open(pca_path, 'wb') as f:
                    pickle.dump(pca, f)

    # Print training time
    end_time = time()
    logger.info(f"Training completed after {(end_time-start_time)/60:.2f} minutes.")

    # Save outputs as csv file
    all_outputs_df = pd.DataFrame(all_outputs)
    data_file = os.path.join(output_dir, 'prediction_results.csv')
    all_outputs_df.to_csv(data_file, index=False)

    if len(pca_var_explained) > 0:
        pca_var_explained_df = pd.DataFrame(pca_var_explained)
        pca_var_explained_df.to_csv(os.path.join(output_dir, 'pca_var_explained.csv'), index=False)

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
