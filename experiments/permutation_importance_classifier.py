"""
Permutation importance for the graph-level logistic regression MLP.

Author: Hanna Tolle
Date: 2025-06-02
License: BSD 3-Clause
"""

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.data_ingredient import * 
from experiments.ingredients.vgae_ingredient import * 
from experiments.ingredients.mlp_ingredient import * 

import os
import torch
import torch.nn
import numpy as np
from scipy.stats import ttest_1samp
import logging
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import pandas as pd

from utils.files import add_project_root
from utils.configs import *
from utils.helpers import fix_random_seed, get_logger, check_weights_exist


# Create experiment and logger -------------------------------------------------
ex = Experiment('permutation_importance_classifier', ingredients=[data_ingredient, 
                                                                  vgae_ingredient, 
                                                                  mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'permutation_importance_classifier'
    jobid = 0
    seed = 291
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join(project_root(), 'outputs', 'runs', run_name)

    # Model weights directory, filenames and number of permutations
    weights_dir = os.path.join('outputs', 'weights', 'final_config_screening', f'job{jobid}_seed{seed}')
    weight_filenames = {'vgae': [f'k{k}_vgae_weights.pth' for k in range(dataset['num_folds'])],
                        'mlp': [f'k{k}_mlp_weights.pth' for k in range(dataset['num_folds'])],
                        'test_fold_indices': ['test_fold_indices.csv']}
    n_repeats = 50
    
    # Manage log level/ verbosity
    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # Get weights_dir (must be in the config)
    assert 'weights_dir' in config, "weights_dir must be specified in config."
    weights_dir = add_project_root(config['weights_dir'])

    # Load the VGAE, MLP and dataset configs from weights_dir
    previous_config = load_ingredient_configs(weights_dir, ['vgae_model', 'mlp_model', 'dataset'])

    # Match configs of relevant ingredients
    ingredients = ['dataset', 'vgae_model', 'mlp_model']
    exceptions = ['num_nodes']
    config_updates = match_ingredient_configs(config=config,
                                              previous_config=previous_config,
                                              ingredients=ingredients,
                                              exceptions=exceptions)
    
    # The MLP must be a logistic regression MLP
    if config_updates['mlp_model']['model_type'] != 'LogisticRegressionMLP':
        raise ValueError("The MLP must be a logistic regression MLP.")

    # Other compatibiltiy checks
    num_folds = config_updates['dataset']['num_folds']
    weight_filenames = config_updates.get('weight_filenames', None)
    if weight_filenames is None:
        weight_filenames =  {'vgae': [f'k{k}_vgae_weights.pth' for k in range(num_folds)],
                             'mlp': [f'k{k}_mlp_weights.pth' for k in range(num_folds)],
                             'test_fold_indices': ['test_fold_indices.csv']}
    check_weights_exist(weights_dir, weight_filenames)
    config_updates['weight_filenames'] = weight_filenames
            
    return config_updates

# Helper functions ------------------------------------------------------------
def get_inputs_and_labels(data, vgaes, testfold_indices, device):
    '''Returns all input features and true labels from the data as NumPy arrays.
    
    Parameters:
    ----------
    data (Dataset): The complete dataset
    vgaes (List[VGAE]): List of trained VGAE models, one for each fold
    testfold_indices (numpy array): Maps each data sample to its test fold
        Maps each data sample to its test fold
    device (torch.device): Device to run computations on
    '''
    num_samples = len(data)
    num_folds = len(vgaes)
    
    # Get labels and clinical data for all samples at once
    batch = next(iter(DataLoader(data, batch_size=num_samples, shuffle=False))).to(device)
    ytrue = batch.y.cpu().numpy()
    clinical_data = batch.graph_attr.cpu().numpy()
    
    # Initialize array for VGAE outputs
    z_readout = np.zeros((num_samples, vgaes[0].readout_dim))
    
    # Process each fold
    for k in range(num_folds):
        # Get indices for samples in this test fold
        fold_indices = np.where(testfold_indices == k)[0]
        
        # Get the subset of data for this fold
        fold_data = data[fold_indices]
        fold_batch = next(iter(DataLoader(fold_data, batch_size=len(fold_indices), shuffle=False))).to(device)
        
        # Get VGAE model for this fold
        vgae = vgaes[k]
        vgae.eval()
        
        with torch.no_grad():
            # Get context and VGAE latent representations for this fold
            context = get_context(fold_batch)
            out = vgae(fold_batch)
            fold_z_readout = vgae.readout(out.mu, context, fold_batch.batch).cpu().numpy()
            
            # Store the results
            z_readout[fold_indices] = fold_z_readout

    return z_readout, clinical_data, ytrue

def compute_score_with_fold_permutation(kfold_models, 
                                        testfold_indices, 
                                        mlp_inputs, 
                                        ytrue, 
                                        device, 
                                        feature_idx=None):
    """
    Computes accuracy score with option to permute a feature within each fold.
    
    Parameters:
    ----------
    kfold_models (List[LatentMLP]): list of trained MLPs models,
        each trained on a different fold.
    testfold_indices (numpy array): maps each data sample to its test fold
    mlp_inputs (numpy array): [z, clinical_data]
    ytrue (numpy array): true binary labels (0 or 1)
    device (torch.device): Device to run computations on
    feature_idx (int or slice): If provided, this feature/features will be permuted within each fold
    """
    num_folds = max(testfold_indices) + 1
    ypreds = np.zeros_like(ytrue)
    
    x_permuted = mlp_inputs.copy()
    for k in range(num_folds):
        # Get the MLP that was tested on this fold
        mlp = kfold_models[k]
        test_indices = np.where(testfold_indices == k)[0]
        
        # If feature_idx provided, permute the specified feature(s) within this fold
        if feature_idx is not None:
            fold_permutation = np.random.permutation(len(test_indices))
            x_permuted[test_indices[:, None], feature_idx] = \
                x_permuted[test_indices[fold_permutation, None], feature_idx]
        
        x = torch.tensor(x_permuted[test_indices, :], dtype=torch.float32).to(device)
        
        mlp.eval()
        with torch.no_grad():
            # Get raw predictions and apply sigmoid
            raw_preds = mlp(x).cpu().numpy().flatten()
            ypreds[test_indices] = (raw_preds > 0.5).astype(int)

    # Compute accuracy
    return np.mean(ypreds == ytrue)

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Set up environment ------------------------------------------------------
    seed = _config['seed']
    verbose = _config['verbose']
    n_repeats = _config['n_repeats']
    output_dir = add_project_root(_config['output_dir'])
    weights_dir = add_project_root(_config['weights_dir'])
    weight_filenames = _config['weight_filenames']

    # Make output directory, get device and fix random seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    device = torch.device(_config['device'])

    # Load data and trained models
    data = load_data()
    vgaes = load_trained_vgaes(weights_dir, weight_filenames['vgae'], device)
    mlps = load_trained_mlps(weights_dir, weight_filenames['mlp'], device, 
                             latent_dims=[vgae.readout_dim for vgae in vgaes])
    testfold_indices = np.loadtxt(os.path.join(weights_dir, weight_filenames['test_fold_indices'][0]), dtype=int)

    # If data has no labels (e.g. X-learner), 
    # load the prediction results from weights_dir and get labels from there
    if data[0].y is None:
        pre_results = pd.read_csv(os.path.join(weights_dir, 'prediction_results.csv'))
        labels = dict(zip(pre_results['subject_id']+1, pre_results['label']))
        addlabel_tfm = AddLabel(labels)
        data.transform = T.Compose([*data.transform.transforms, addlabel_tfm])

    # Get MLP inputs and labels
    z_readout, clinical_data, ytrue = get_inputs_and_labels(data, vgaes, testfold_indices, device)
    mlp_inputs = np.concatenate([z_readout, clinical_data], axis=1)
    
    # Initialize lists to store results
    features_agg = []
    scores_agg = []
    
    # Compute baseline score without permutation
    baseline_score = compute_score_with_fold_permutation(
        mlps, testfold_indices, mlp_inputs, ytrue, device)
    
    # a) Compute Z_whole (permuting all latent features together)
    z_whole_scores = np.zeros(n_repeats)
    for i in range(n_repeats):
        score = compute_score_with_fold_permutation(
            mlps, testfold_indices, mlp_inputs, ytrue, device, 
            feature_idx=slice(0, z_readout.shape[1]))
        z_whole_scores[i] = baseline_score - score
    scores_agg.append(z_whole_scores)
    features_agg.append('Z_whole')
    
    # b) Compute graph attribute importance
    for j in range(clinical_data.shape[1]):
        feature_scores = np.zeros(n_repeats)
        for i in range(n_repeats):
            score = compute_score_with_fold_permutation(
                mlps, testfold_indices, mlp_inputs, ytrue, device, 
                feature_idx=z_readout.shape[1] + j)
            feature_scores[i] = baseline_score - score
        scores_agg.append(feature_scores)
        features_agg.append(data[0].attr_names.graph[j])
    
    # Compute statistics for aggregated scores
    scores_agg = np.array(scores_agg)
    means_agg = np.mean(scores_agg, axis=1)
    stds_agg = np.std(scores_agg, axis=1)
    sems_agg = stds_agg / np.sqrt(n_repeats)
    t_stats_agg, p_values_agg = ttest_1samp(scores_agg, 0, axis=1)
    _, p_values_fdr_agg = fdrcorrection(p_values_agg, alpha=0.05)
    
    # Save aggregated importance scores
    agg_stats = pd.DataFrame({
        'feature': features_agg,
        'mean': means_agg,
        'std': stds_agg,
        'se': sems_agg,
        't_stat': t_stats_agg,
        'p_value': p_values_agg,
        'p_value_fdr': p_values_fdr_agg
    })
    agg_stats.to_csv(os.path.join(output_dir, 'importance_scores_aggregated.csv'), index=False)
    
    # Plot aggregated importance scores
    fig, ax = plt.subplots(1, 1, figsize=(len(features_agg)*0.5, 6))
    bars = ax.bar(features_agg, means_agg, yerr=sems_agg, align='center', capsize=5)
    ax.set_ylabel('Accuracy Decrease')
    ax.set_xticks(range(len(features_agg)))
    ax.set_xticklabels(features_agg, rotation=45, ha='right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add significance markers
    offset = 0.05 * max(means_agg + sems_agg)
    for i, (bar, p_value_fdr) in enumerate(zip(bars, p_values_fdr_agg)):
        if p_value_fdr < 0.05:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + offset,
                   '*', ha='center', va='bottom', color='red', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'importance_scores_aggregated.png'), dpi=300)
    ex.add_artifact(os.path.join(output_dir, 'importance_scores_aggregated.png'))
    if not verbose:
        plt.close()
