'''
The aim of this experiment is to check if the biomarkers identified by GRAIL
are really predictive of treatment outcome.

Computes all the candidate biomarkers (features) as in grad_align.py,
but on the original brain graphs (not the reconstructed graphs).

Then, computes the correlation of each biomarker with treatment outcome.
Also performs PCA and correlates the patient loadings with treatment outcome.

Dependencies:
- data/raw/receptor_maps/f{atlas}/f{atlas}_receptor_maps.csv

Authors: Hanna M. Tolle
Date: 2025-01-28
License: BSD 3-Clause
'''

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.data_ingredient import * 

import os
import torch
from time import time
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats

from utils.files import add_project_root
from utils.helpers import get_logger, fix_random_seed, sort_features
from utils.configs import load_ingredient_configs, match_ingredient_configs
from utils.annotations import load_receptor_maps
from utils.plotting import COOLWARM
from utils.statsalg import compute_pca
from preprocessing.metrics import compute_modularity_torch, get_rsn_mapping


# Create the experiment -------------------------------------------------------
ex = Experiment('test_biomarkers', ingredients=[data_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations -------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'test_biomarkers'
    jobid = 0
    seed = 291
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join('outputs', 'runs', run_name)

    # Logging and saving
    verbose = False
    save_outputs = True
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Directory with dataset configs
    weights_dir = os.path.join('outputs', 'weights', f'job{jobid}_seed{seed}')
    
    # Experiment-specific configs
    all_rsn_conns = False       # Whether to compute RSN connectivity for all RSN pairs.
    n_pca_components = 10       # Number of PCA components to compute.

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # Get weights_dir (must be in the config)
    assert 'weights_dir' in config, "weights_dir must be specified in config."
    weights_dir = add_project_root(config['weights_dir'])

    # Load the VGAE, MLP and dataset configs from weights_dir
    ingredient_names = ['dataset']
    previous_config = load_ingredient_configs(weights_dir, ingredient_names)

    # Match configs of relevant ingredients
    exceptions = ['num_nodes', 'drug_condition', 'graph_attrs', 'context_attrs']
    config_updates = match_ingredient_configs(config=config,
                                              previous_config=previous_config,
                                              ingredients=ingredient_names,
                                              exceptions=exceptions)
            
    return config_updates

# Helper functions -------------------------------------------------------------

def compute_correlation(x, y):
    """Compute Pearson correlation using PyTorch operations."""
    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)
    numerator = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))
    return numerator / denominator

@ex.capture
def compute_features(adj,
                     rsn_mapping, rsn_names, receptor_maps, 
                     seed, all_rsn_conns,
                     x = None, node_attrs = None):
    """
    Computes feature values.

    Parameters:
    ----------
    adj (torch.Tensor): The adjacency matrix.
    rsn_mapping (torch.Tensor): The mapping of nodes to RSNs.
    rsn_names (list): The names of the RSNs.
    receptor_maps (dict): The receptor maps.
    x (torch.Tensor): The node attributes.
    node_attrs (list): The names of the node attributes.

    Returns:
    --------
    feature_values (dict): The values of the features.
    """
    feature_values = {}
    device = adj.device
    
    # Process each feature one at a time 
    def process_feature(name, value):
        feature_values[name] = value.item()

    # Compute modularity using RSN mapping
    rsn_modularity = compute_modularity_torch(adj, rsn_mapping)
    process_feature('modularity_rsn', rsn_modularity)

    # # Compute Louvain communities (no gradients)
    # adj_np = adj.detach().cpu().numpy()
    # np.fill_diagonal(adj_np, 0) # no self-loops
    # G = nx.from_numpy_array(adj_np)
    # louvain_communities = nx.community.louvain_communities(G, seed=seed)
    
    # # Compute modularity with Louvain communities
    # louvain_modularity = compute_modularity_torch(adj, louvain_communities)
    # process_feature('modularity', louvain_modularity)

    # RSN connectivity
    if all_rsn_conns:
        for i in range(len(rsn_names)):
            mask_i = (rsn_mapping == i)
            for j in range(i, len(rsn_names)):
                mask_j = (rsn_mapping == j)
                if i == j:
                    # For within-RSN connectivity, exclude diagonal elements
                    rows = torch.where(mask_i)[0]
                    n = len(rows)
                    diag_mask = ~torch.eye(n, dtype=bool, device=device)
                    sub_adj = adj[rows][:, rows]
                    feat = torch.mean(sub_adj[diag_mask])
                else:
                    feat = torch.mean(adj[mask_i][:, mask_j])
                process_feature(f'fc_mean_{rsn_names[i]}_{rsn_names[j]}', feat)
    else:
        for i in range(len(rsn_names)):
            mask_i = (rsn_mapping == i)
            feat = torch.mean(adj[mask_i])
            process_feature(f'fc_mean_{rsn_names[i]}', feat)

    # Node strength correlations
    fc_strength = torch.sum(adj, dim=1)
    
    # Correlation with receptor densities
    for receptor, density in receptor_maps.items():
        feat = compute_correlation(fc_strength, density)
        process_feature(f'fc_corr_{receptor}', feat)
    
    if x is not None:        
        # Mean x value in each RSN
        for rsn_idx, rsn in enumerate(rsn_names):
            mask = (rsn_mapping == rsn_idx)
            for attr_idx, attr in enumerate(node_attrs):
                feat = torch.mean(x[mask, attr_idx])
                process_feature(f'x{attr}_mean_{rsn}', feat)
        
        # Correlations between node attributes and receptor densities
        for receptor, density in receptor_maps.items():
            for attr_idx, attr in enumerate(node_attrs):
                feat = compute_correlation(x[:, attr_idx], density)
                process_feature(f'x{attr}_corr_{receptor}', feat)
    
    return feature_values

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):
    # Set up environment ------------------------------------------------------
    seed = _config['seed']
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_outputs = _config['save_outputs']
    node_attrs = _config['dataset']['node_attrs']
    node_attrs = [attr.split('_')[0] for attr in node_attrs]

    # Create output directory, fix seed, get device
    os.makedirs(output_dir, exist_ok=True)             # Create output directory
    fix_random_seed(seed)                              # Fix random seed
    image_files = []                                   # List of image files to save
    device = torch.device(_config['device'])           # Get device
    logger.info(f'Using device: {device}')

    # Load data
    data = load_data()

    # Sampling for each subject --------------------------------------------
    start_time = time()

    num_subs = len(data)
    atlas = _config['dataset']['atlas']
    receptor_maps = load_receptor_maps(atlas)
    rsn_mapping, rsn_names = get_rsn_mapping(atlas)

    # Convert to tensors (needed for gradient computation later)
    rsn_mapping = torch.tensor(rsn_mapping, device=device)
    receptor_maps = {name: torch.tensor(values.values, device=device, dtype=torch.float32)
                     for name, values in receptor_maps.items()}

    # Initialize dataframes for storing results
    feature_value_records = []
    
    for sub in tqdm(range(num_subs), desc='Subjects', disable=not verbose):
        # Get adjacency matrix and node features (if they exist)
        adj = data.get_dense_adj(sub).squeeze()
        x = data[sub].x if data[sub].x is not None else None

        # Compute interpretable features
        features = compute_features(
            adj=adj, 
            rsn_mapping=rsn_mapping, 
            rsn_names=rsn_names, 
            receptor_maps=receptor_maps, 
            x=x, 
            node_attrs=node_attrs)

        # Store feature values
        features['sub'] = sub
        features['y'] = data[sub].y.item() # also store the treatment outcome
        feature_value_records.append(features)
    
    # Convert to dataframe
    feature_value_df = pd.DataFrame(feature_value_records)

    # Sort the features according to categories
    feature_cols = [f for f in feature_value_df.columns if f != 'sub' and f != 'y']
    feature_cols = sort_features(feature_cols)
    sorted_columns = ['sub', 'y'] + feature_cols
    feature_value_df = feature_value_df[sorted_columns]

    # Correlate each feature with treatment outcome
    corr_df = []
    for feature in feature_value_df.columns:
        if feature != 'sub' and feature != 'y':
            corr, pval = stats.pearsonr(feature_value_df[feature], feature_value_df['y'])
            corr_df.append({'feature': feature, 'corr': corr, 'pval': pval})
    corr_df = pd.DataFrame(corr_df)
    corr_df['fdr'] = fdrcorrection(corr_df['pval'].values)[1]
    
    # Save the dataframes
    feature_value_df.to_csv(os.path.join(output_dir, 'feature_values.csv'), index=False)
    corr_df.to_csv(os.path.join(output_dir, 'feature_correlations.csv'), index=False)

    # Perform PCA on the features
    n_components = _config['n_pca_components']
    save_path = os.path.join(output_dir, 'pca.png')
    pca, _ = compute_pca(feature_value_df[feature_cols].values, 
                              max_components=n_components,
                              standardise=True,
                              save_path=save_path)
    
    # Save a df with explained variance for each component
    explained_variance = pd.DataFrame({'component': range(1, n_components+1),
                                       'explained_variance': pca.explained_variance_ratio_})
    explained_variance.to_csv(os.path.join(output_dir, 'explained_variance.csv'), index=False)

    # Plot PC components
    patterns = pca.components_[:n_components]
    y_labels = [f'PC{i+1}' for i in range(n_components)]
    vmax = np.percentile(np.abs(patterns), 99)
    vrange = (-vmax, vmax)
    plt.figure(figsize=(15, 5))
    sns.heatmap(patterns,
                cmap=COOLWARM,
                vmin=vrange[0], 
                vmax=vrange[1],
                center=0,
                xticklabels=feature_cols,
                yticklabels=y_labels)

    # Rotate x-tick labels for better readability
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_components.png'))

    # Compute PC loadings
    pc_loadings = pca.transform(feature_value_df[feature_cols].values)
    pc_loadings_df = pd.DataFrame(pc_loadings, columns=[f'PC{i+1}' for i in range(n_components)])
    pc_loadings_df['sub'] = feature_value_df['sub']
    pc_loadings_df['y'] = feature_value_df['y']
    pc_loadings_df.to_csv(os.path.join(output_dir, 'pc_loadings.csv'), index=False)

    # Correlate PC loadings with treatment outcome
    pca_corr_df = []
    for pc in range(n_components):
        corr, pval = stats.pearsonr(pc_loadings_df[f'PC{pc+1}'], pc_loadings_df['y'])
        pca_corr_df.append({'pc': f'PC{pc+1}', 'corr': corr, 'pval': pval})
    pca_corr_df = pd.DataFrame(pca_corr_df)
    pca_corr_df['fdr'] = fdrcorrection(pca_corr_df['pval'].values)[1]
    pca_corr_df.to_csv(os.path.join(output_dir, 'pca_correlations.csv'), index=False)

    # Save the image files
    for img in image_files:
        if img is not None:
            ex.add_artifact(img)

    # Log the runtime
    run_time = (time()-start_time)/60
    print(f"Experiment completed. Runtime: {run_time:.2f} min.")
    