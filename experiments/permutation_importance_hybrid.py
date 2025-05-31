'''
Computes permutation importance for a hybrid model (2 VGAEs + MLP) by shuffling:
1. Each clinical score individually
2. Each VGAE's latent vector as a whole

Authors: Hanna M. Tolle
Date: 2025-02-22
License: BSD 3-Clause
'''

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.data_ingredient import *
from experiments.ingredients.vgae_ingredient import *
from experiments.ingredients.mlp_ingredient import *

import os
import torch
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection

from utils.files import add_project_root, rm_project_root
from utils.helpers import fix_random_seed, get_logger, check_weights_exist
from utils.configs import *


# Create experiment and logger -------------------------------------------------
ex = Experiment('permutation_importance_hybrid', ingredients=[data_ingredient, 
                                                               vgae_ingredient, 
                                                               mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'permutation_importance_hybrid'
    jobid = 0
    seed = 291
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join(project_root(), 'outputs', 'runs', run_name)

    # Model weights directory, filenames and number of permutations
    weights_dir = os.path.join('outputs', 'hybrid', 'train_hybrid')
    weight_filenames = {'vgae': [f'k{k}_vgae_weights.pth' for k in range(dataset['num_folds'])],
                        'vgae2': [f'k{k}_vgae_weights.pth' for k in range(dataset['num_folds'])],
                        'mlp': [f'k{k}_mlp_weights.pth' for k in range(dataset['num_folds'])],
                        'test_fold_indices': ['test_fold_indices.csv']}
    n_repeats = 30
    
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
    ingredients = ['vgae_model', 'mlp_model', 'dataset']
    previous_config = load_ingredient_configs(weights_dir, ingredients)

    # Match configs of relevant ingredients
    config_updates = match_ingredient_configs(config=config,
                                              previous_config=previous_config,
                                              ingredients=ingredients,
                                              exceptions=[])
    return config_updates

# Helper functions ------------------------------------------------------------
def get_vgae_weights_dirs(weights_dir):
    '''Loads the config.json file inside weights_dir and 
    returns "weights_dir" and "weights_dir2".'''
    config = load_configs_from_json(os.path.join(weights_dir, 'config.json'))
    # Exact root path may mismatch if the model was trained on a different machine
    vgae_weights_dir = add_project_root(rm_project_root(config['weights_dir']))
    vgae2_weights_dir = add_project_root(rm_project_root(config['weights_dir2']))
    return vgae_weights_dir, vgae2_weights_dir

def load_trained_vgaes2(vgae2_config, weights_dir, weight_filenames, device):
    '''
    Loads trained VGAEs from a directory.
    '''
    vgaes = []
    for weight_file in weight_filenames:
        vgae = build_vgae_from_config(vgae2_config).to(device)
        state_dict = torch.load(os.path.join(weights_dir, weight_file))
        vgae.load_state_dict(state_dict)
        vgaes.append(vgae)
    return vgaes

def get_inputs_split_by_source(batch, batch2, vgae, vgae2, device):
    '''
    Returns the MLP inputs split by source (clinical data, z1, z2).
    '''
    # Get clinical data
    clinical_data = batch.graph_attr

    # Get VGAE latent vectors
    with torch.no_grad():
        # Get z1 (first VGAE)
        context = get_context(batch)
        out = vgae(batch)
        z1 = vgae.readout(out.mu, context, batch.batch)

        # Get z2 (second VGAE)
        context2 = get_context(batch2)
        out2 = vgae2(batch2)
        z2 = vgae2.readout(out2.mu, context2, batch2.batch)

    return clinical_data, z1, z2

def compute_score_with_feature_permutation(kfold_mlps, kfold_vgaes, kfold_vgaes2,
                                         testfold_indices, data, data2,
                                         feature_info, device):
    """
    Computes score with option to permute specific features.
    
    Parameters:
    ----------
    feature_info: dict with keys:
        'type': 'clinical', 'z1', or 'z2', or None (no permutation)
        'index': int or None (None for z1/z2 means shuffle whole vector)
    """
    num_folds = len(kfold_mlps)
    all_predictions = []
    all_labels = []
    
    for k in range(num_folds):
        # Get test indices for this fold
        test_indices = np.where(testfold_indices == k)[0]
        
        # Get test data for this fold
        test_data = data[test_indices]
        test_data2 = data2[test_indices]
        test_loader = DataLoader(test_data, batch_size=len(test_indices))
        test_loader2 = DataLoader(test_data2, batch_size=len(test_indices))
        
        # Get models for this fold
        mlp = kfold_mlps[k].to(device)
        vgae = kfold_vgaes[k].to(device)
        vgae2 = kfold_vgaes2[k].to(device)
        mlp.eval()
        vgae.eval()
        vgae2.eval()

        for batch, batch2 in zip(test_loader, test_loader2):
            batch = batch.to(device)
            batch2 = batch2.to(device)
            
            # Get inputs split by source
            clinical_data, z1, z2 = get_inputs_split_by_source(
                batch, batch2, vgae, vgae2, device)
            
            # Permute features according to feature_info
            perm_idx = torch.randperm(len(test_indices))
            
            if feature_info['type'] == 'clinical':
                idx = feature_info['index']
                clinical_data[:, idx] = clinical_data[perm_idx, idx]
            
            elif feature_info['type'] == 'z1':
                z1 = z1[perm_idx]
            
            elif feature_info['type'] == 'z2':
                z2 = z2[perm_idx]
            
            # Concatenate inputs and get predictions
            x = torch.cat([z1, clinical_data, z2], dim=1) # this is how hybrid is trained
            with torch.no_grad():
                y_pred = mlp(x)
            
            # Get true labels
            y_true = get_labels(batch, num_z_samples=0)
            
            # Store predictions and labels
            all_predictions.extend(y_pred.cpu().numpy().flatten())
            all_labels.extend(y_true.cpu().numpy().flatten())
    
    # Compute score (negative MAE)
    mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_labels)))
    return -mae

def plot_permutation_importance(results_df, save_path):
    """Plot permutation importance scores with error bars and significance markers."""
    fig, ax = plt.subplots(1, 1, figsize=(len(results_df)*0.5, 6))
    
    bars = ax.bar(results_df['feature'], results_df['importance_mean'],
                  yerr=results_df['importance_std'], align='center', capsize=5)
    
    ax.set_ylabel('Importance Scores')
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df['feature'], rotation=45, ha='right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add significance markers
    offset = 0.05 * max(results_df['importance_mean'] + results_df['importance_std'])
    for i, (bar, p_value_fdr) in enumerate(zip(bars, results_df['p_value_fdr'])):
        if p_value_fdr < 0.05:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + offset,
                   '*', ha='center', va='bottom', color='red', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Check weight files ------------------------------------------------------
    weight_filenames = _config['weight_filenames']

    # Hybrid weights_dir, which should contain MLP weights and test fold indices
    weights_dir = add_project_root(_config['weights_dir']) 
    check_weights_exist(weights_dir, {'mlp': weight_filenames['mlp'], 
                                      'test_fold_indices': weight_filenames['test_fold_indices']})

    # VGAE weights_dirs, which should contain VGAE weights
    vgae_weights_dir, vgae2_weights_dir = get_vgae_weights_dirs(weights_dir)
    check_weights_exist(vgae_weights_dir, {'vgae': weight_filenames['vgae']})
    check_weights_exist(vgae2_weights_dir, {'vgae2': weight_filenames['vgae2']})

    # Set up environment ------------------------------------------------------
    output_dir = add_project_root(_config['output_dir'])
    seed = _config['seed']
    verbose = _config['verbose']
    n_repeats = _config['n_repeats']

    # Load configs for vgae2 and data2
    ingredients = ['dataset', 'vgae_model']
    config2 = load_ingredient_configs(vgae2_weights_dir, ingredients)

    # Make sure that only one of the VGAEs has graph_attrs
    graph_attrs = _config['dataset']['graph_attrs']
    if len(graph_attrs) > 0:
        config2['dataset']['graph_attrs'] = []

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)

    # Load data for VGAE1 and VGAE2
    data = load_data()
    data2 = load_dataset_from_configs(config2['dataset'])
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Load trained VGAEs
    vgaes = load_trained_vgaes(vgae_weights_dir, weight_filenames['vgae'], device)
    vgaes2 = load_trained_vgaes2(config2['vgae_model'], vgae2_weights_dir, weight_filenames['vgae2'], device)

    # Load trained MLPs and test fold indices
    latent_dims = [vgaes[0].readout_dim + vgaes2[0].readout_dim]*len(weight_filenames['mlp'])
    mlps = load_trained_mlps(weights_dir, weight_filenames['mlp'], device, latent_dims=latent_dims)
    test_indices = np.loadtxt(os.path.join(weights_dir, weight_filenames['test_fold_indices'][0]), dtype=int)

    # Permutation importance --------------------------------------------------
    # Get feature names
    clinical_features = _config['dataset']['graph_attrs']
    num_clinical = len(clinical_features)
    
    # Initialize results lists
    features = []
    importance_scores = []
    
    # Compute baseline scores (multiple runs for stability)
    baseline_scores = np.zeros(n_repeats)
    for i in range(n_repeats):
        baseline_scores[i] = compute_score_with_feature_permutation(
            mlps, vgaes, vgaes2, test_indices, data, data2,
            {'type': None, 'index': None}, device)
    baseline_score = np.mean(baseline_scores)
    
    # Test clinical features individually
    for i, feat_name in enumerate(clinical_features):
        scores = np.zeros(n_repeats)
        for j in range(n_repeats):
            score = compute_score_with_feature_permutation(
                mlps, vgaes, vgaes2, test_indices, data, data2,
                {'type': 'clinical', 'index': i}, device)
            scores[j] = baseline_score - score
        
        features.append(feat_name)
        importance_scores.append(scores)
    
    # Test each VGAE's latent vector
    for vgae_name, vgae_type in [('VGAE1', 'z1'), ('VGAE2', 'z2')]:
        scores = np.zeros(n_repeats)
        for j in range(n_repeats):
            score = compute_score_with_feature_permutation(
                mlps, vgaes, vgaes2, test_indices, data, data2,
                {'type': vgae_type, 'index': None}, device)
            scores[j] = baseline_score - score
        
        features.append(vgae_name)
        importance_scores.append(scores)
    
    # Compute statistics
    importance_scores = np.array(importance_scores)
    means = np.mean(importance_scores, axis=1)
    stds = np.std(importance_scores, axis=1)
    sems = stds / np.sqrt(n_repeats)
    t_stats, p_values = ttest_1samp(importance_scores, 0, axis=1)
    _, p_values_fdr = fdrcorrection(p_values, alpha=0.05)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'feature': features,
        'importance_mean': means,
        'importance_std': stds,
        'importance_se': sems,
        't_stat': t_stats,
        'p_value': p_values,
        'p_value_fdr': p_values_fdr
    })
    
    # Save results
    save_path_df = os.path.join(output_dir, 'permutation_importance.csv')
    results_df.to_csv(save_path_df, index=False)
    ex.add_artifact(save_path_df)
    
    # Plot results
    save_path = os.path.join(output_dir, 'permutation_importance.png')
    plot_permutation_importance(results_df, save_path)
    ex.add_artifact(save_path)
    
    return results_df