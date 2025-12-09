'''
Computes GRAIL patterns for all patients with all fold models.
Does not cluster results and only saves mean alignment values.
Additionally, computes fold-wise performance of each model and saves it.

Dependencies:
- data/raw/receptor_maps/f{atlas}/f{atlas}_receptor_maps.csv

Authors: Hanna M. Tolle
Date: 2025-05-03
License: BSD 3-Clause
'''

import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.data_ingredient import * 
from experiments.ingredients.vgae_ingredient import * 
from experiments.ingredients.mlp_ingredient import * 

import os
import torch
from tqdm import tqdm
from time import time
import pandas as pd
import logging
from torch_geometric.data import Batch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import fdrcorrection

from utils.files import add_project_root
from utils.helpers import get_logger, check_weights_exist, fix_random_seed, triu_vector2mat_torch, sort_features
from utils.configs import load_ingredient_configs, match_ingredient_configs
from utils.annotations import load_receptor_maps
from utils.statsalg import get_fold_performance, min_significant_r, calculate_cohens_d
from models.utils import freeze_model
from utils.plotting import plot_heatmap, COOLWARM
from preprocessing.metrics import compute_modularity_torch, get_rsn_mapping


# Create the experiment -------------------------------------------------------
ex = Experiment('grail', ingredients=[data_ingredient, 
                                      vgae_ingredient, 
                                      mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations -------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'grail'
    jobid = 0
    seed = 0
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join('outputs', 'runs', run_name)

    # Logging and saving
    verbose = False
    save_outputs = True
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Directory with pre-trained model weights
    vgae_weights_dir = os.path.join('outputs', 'weights', f'job{jobid}_seed{seed}')
    mlp_weights_dir = None      # If None, VGAE and MLP weights dirs are the same
    weight_filenames = {'vgae': [f'k{k}_vgae_weights.pth' for k in range(dataset['num_folds'])],
                        'mlp': [f'k{k}_mlp_weights.pth' for k in range(dataset['num_folds'])],
                        'test_fold_indices': ['test_fold_indices.csv']}
    
    # Experiment-specific configs
    num_z_samples = 100         # Number of samples to draw from the VGAE latent distribution.
    sigma = 2.0                 # Std of the Gaussian noise added to the latent means.
    all_rsn_conns = False       # Whether to compute RSN connectivity for all RSN pairs.
    medusa = False              # Only works with CFRHead MLP. Grails mlp1-mlp0 instead of ypred.

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # Get weights_dir (must be in the config)
    assert 'vgae_weights_dir' in config, "vgae_weights_dir must be specified in config."
    vgae_weights_dir = add_project_root(config['vgae_weights_dir'])
    mlp_weights_dir = config.get('mlp_weights_dir', None)
    if mlp_weights_dir is None:
        mlp_weights_dir = vgae_weights_dir

    # Load all ingredientconfigs from the mlp weights directory
    previous_config = load_ingredient_configs(mlp_weights_dir, ['vgae_model', 'dataset', 'mlp_model'])

    # Check different configs depending on the prediction head type
    ingredients = ['dataset', 'vgae_model', 'mlp_model']
    exceptions = ['num_nodes', 'drug_condition']
    if 'mlp_model' in config:
        mlp_model_type = config['mlp_model']['model_type']
        if mlp_model_type == 'SklearnLinearModelWrapper':
            ingredients = ['dataset', 'vgae_model'] # Sklearn wrapper has no config
    config_updates = match_ingredient_configs(config=config,
                                              previous_config=previous_config,
                                              ingredients=ingredients,
                                              exceptions=exceptions)
    config_updates['mlp_weights_dir'] = mlp_weights_dir

    # Other compatibiltiy checks
    num_z_samples = config_updates.get('num_z_samples', 100)
    assert num_z_samples > 1, 'num_z_samples must be > 1.'
    if config_updates.get('medusa', False):
        assert config_updates['mlp_model']['model_type'] == 'CFRHead', \
            'GRAIL with medusa only works with CFRHead MLP.'
            
    return config_updates

# Helper functions -------------------------------------------------------------
    
@ex.capture
def get_zs(mu, logvar, vgae, num_z_samples, sigma):
    '''
    Returns multiple latent samples.
    '''
    zs = []
    for _ in range(num_z_samples):
        noise = torch.randn_like(mu) * sigma
        z = vgae.reparameterize(mu + noise, logvar)
        z.requires_grad_(True) 
        zs.append(z)
    return zs

def get_reconstructions(z, vgae, batch):
    '''Returns the reconstructed node and adjacency matrix for a given latent sample.'''
    num_nodes = batch[0].num_nodes
    triu_idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
    x, triu_edges = vgae.decode(z, triu_idx)
    return x, triu_edges

def compute_correlation(x, y):
    """Compute Pearson correlation using PyTorch operations."""
    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)
    numerator = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))
    return numerator / denominator

@ex.capture
def get_alignments_and_features(triu_edges, ypred_grad, z,
                                rsn_mapping, rsn_names, receptor_maps, 
                                seed, all_rsn_conns,
                                x = None, node_attrs = None):
    """
    Computes feature gradients and alignment with ypred_grad.

    Parameters:
    ----------
    triu_edges (torch.Tensor): The upper triangular edges of the adjacency matrix.
    ypred_grad (torch.Tensor): The gradient of the MLP prediction with respect to the latent vector.
    z (torch.Tensor): The latent vector/ matrix.
    rsn_mapping (torch.Tensor): The mapping of nodes to RSNs.
    rsn_names (list): The names of the RSNs.
    receptor_maps (dict): The receptor maps.
    x (torch.Tensor): The node attributes.
    node_attrs (list): The names of the node attributes.

    Returns:
    --------
    alignments (dict): The alignments between the feature gradients and ypred_grad.
    feature_gradients (dict): The gradients of each feature with respect to z.
    """
    alignments = {}
    feature_gradients = {}
    device = triu_edges.device
    
    # Get adjacency matrix
    adj = triu_vector2mat_torch(triu_edges)
    
    # Process each feature one at a time 
    def process_feature(name, value):
        grad = compute_gradient(value, z)
        alignment = compute_gradient_alignment(ypred_grad, grad)
        alignments[name] = alignment
        feature_gradients[name] = grad.detach().cpu().numpy()

    # Compute modularity using RSN mapping
    rsn_modularity = compute_modularity_torch(adj, rsn_mapping)
    process_feature('modularity_rsn', rsn_modularity)

    # Compute Louvain communities (no gradients)
    adj_np = adj.detach().cpu().numpy()
    np.fill_diagonal(adj_np, 0) # no self-loops
    G = nx.from_numpy_array(adj_np)
    louvain_communities = nx.community.louvain_communities(G, seed=seed)
    
    # Compute modularity with Louvain communities
    louvain_modularity = compute_modularity_torch(adj, louvain_communities)
    process_feature('modularity', louvain_modularity)

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
            if receptor.upper() not in node_attrs:
                for attr_idx, attr in enumerate(node_attrs):
                    feat = compute_correlation(x[:, attr_idx], density)
                    process_feature(f'x{attr}_corr_{receptor}', feat)
        
        # Correlations between pairs of node attributes
        for i in range(len(node_attrs)):
            for j in range(i+1, len(node_attrs)):
                feat = compute_correlation(x[:, i], x[:, j])
                process_feature(f'x{node_attrs[i]}_corr_x{node_attrs[j]}', feat)
    
    return alignments, feature_gradients

@ex.capture
def get_ypred_from_z(z, vgae, mlp, batch, medusa=False):
    '''Returns the MLP prediction for a given latent sample.'''
    context = get_context(batch)
    treatment = get_treatment(batch, num_z_samples=0)
    clinical_data = batch.graph_attr
    x = torch.cat([vgae.readout(z, context, batch.batch), clinical_data], dim=1)
    if medusa:
        # CFRNet, but we want to grail treatment effect
        return mlp.mlp1(x) - mlp.mlp0(x)
    elif treatment is None:
        # Standard MLP: no treatment information
        return mlp(x)
    else:
        # CFRNet: use treatment information
        return mlp(x, treatment)

def compute_gradient(output_scalar, z):
    """
    Compute gradient of a scalar output with respect to z.
    Parameters:
    ----------
    output_scalar (torch.Tensor): The scalar output to compute the gradient of.
    z (torch.Tensor): The latent vector.
    """
    grad = torch.autograd.grad(outputs=output_scalar,
                               inputs=z,
                               grad_outputs=torch.ones_like(output_scalar),
                               create_graph=True)[0]
    return grad.flatten()

def compute_gradient_alignment(grad1, grad2):
    """Compute cosine similarity between two gradient vectors."""
    grad1_norm = torch.nn.functional.normalize(grad1.view(1, -1), p=2, dim=1)
    grad2_norm = torch.nn.functional.normalize(grad2.view(1, -1), p=2, dim=1)
    return torch.dot(grad1_norm[0], grad2_norm[0]).item()

def validate_weight_files_and_indices(vgae_weights_dir, mlp_weights_dir, weight_filenames):
    """
    Validates that all weight files exist in their respective directories and that
    test_fold_indices files from both directories match.
    
    Parameters:
    ----------
    vgae_weights_dir : str
        Directory containing VGAE weight files
    mlp_weights_dir : str
        Directory containing MLP weight files
    weight_filenames : dict
        Dictionary with keys 'vgae', 'mlp', and 'test_fold_indices', each containing
        a list of filenames
        
    Raises:
    ------
    FileNotFoundError
        If any weight file or test_fold_indices file is missing
    ValueError
        If test_fold_indices files from both directories do not match
    """
    # Check if all VGAE weight files exist
    for weight_file in weight_filenames['vgae']:
        weight_path = os.path.join(vgae_weights_dir, weight_file)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f'Weight file {weight_file} not found in {vgae_weights_dir}.')
    
    # Check if all MLP weight files exist
    for weight_file in weight_filenames['mlp']:
        weight_path = os.path.join(mlp_weights_dir, weight_file)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f'Weight file {weight_file} not found in {mlp_weights_dir}.')
    
    # Check if test_fold_indices files exist in both directories
    test_fold_indices_filename = weight_filenames['test_fold_indices'][0]
    vgae_indices_path = os.path.join(vgae_weights_dir, test_fold_indices_filename)
    mlp_indices_path = os.path.join(mlp_weights_dir, test_fold_indices_filename)
    
    if not os.path.exists(vgae_indices_path):
        raise FileNotFoundError(f'Test fold indices file {test_fold_indices_filename} not found in {vgae_weights_dir}.')
    if not os.path.exists(mlp_indices_path):
        raise FileNotFoundError(f'Test fold indices file {test_fold_indices_filename} not found in {mlp_weights_dir}.')
    
    # Load and compare test_fold_indices from both directories
    vgae_indices = pd.read_csv(vgae_indices_path, header=None).values.flatten()
    mlp_indices = pd.read_csv(mlp_indices_path, header=None).values.flatten()
    
    if not np.array_equal(vgae_indices, mlp_indices):
        raise ValueError(f'Test fold indices in {vgae_weights_dir} and {mlp_weights_dir} do not match.')

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):
    vgae_weights_dir = add_project_root(_config['vgae_weights_dir'])
    mlp_weights_dir = add_project_root(_config['mlp_weights_dir'])
    weight_filenames = _config['weight_filenames']
    
    # Validate weight files and test_fold_indices
    validate_weight_files_and_indices(vgae_weights_dir, mlp_weights_dir, weight_filenames)

    # Set up environment ------------------------------------------------------
    seed = _config['seed']
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_outputs = _config['save_outputs']
    node_attrs = _config['dataset']['node_attrs']
    node_attrs = [attr.split('_')[0] for attr in node_attrs]
    num_z_samples = _config['num_z_samples']

    # Determine the type of prediction head
    prediction_head_type = _config['mlp_model']['model_type']

    # Create output directory, fix seed, get device
    os.makedirs(output_dir, exist_ok=True)             # Create output directory
    fix_random_seed(seed)                              # Fix random seeds
    device = torch.device(_config['device'])           # Get device
    logger.info(f'Using device: {device}')

    # Load trained models and freeze them
    vgaes = load_trained_vgaes(vgae_weights_dir, weight_filenames['vgae'], device)
    vgaes = [freeze_model(vgae) for vgae in vgaes]
    if prediction_head_type == 'SklearnLinearModelWrapper':
        # Sklearn Wrapper model is automatically frozen
        mlps = load_sklearn_wrapper(mlp_weights_dir, weight_filenames['mlp'], device)
    else:
        mlps = load_trained_mlps(mlp_weights_dir, weight_filenames['mlp'], device, 
                            latent_dims=[vgae.readout_dim for vgae in vgaes])
        mlps = [freeze_model(mlp) for mlp in mlps]

    # Load data
    data = load_data()
    if prediction_head_type == 'CFRHead':
        # Add treatment transform to the dataset (required for CFRHead)
        add_treatment_transform(data)

    # Compute fold-wise performance of each model --------------------------
    if len(vgaes) > 1:
        fold_performance = get_fold_performance(mlp_weights_dir)
        fold_performance.to_csv(os.path.join(output_dir, 'fold_performance.csv'), index=False)
        ex.add_artifact(os.path.join(output_dir, 'fold_performance.csv'))

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

    num_folds = len(vgaes)
    all_fold_alignments = []  # Store alignments for all folds
    
    for k in range(num_folds):
        # Initialize dataframes for storing results
        alignment_records = []
        
        # Load models for this fold
        vgae = vgaes[k].eval()
        mlp = mlps[k].eval()
        
        # Initialize lists for Ridge regression analysis
        all_subject_ridge_betas = []  # List of arrays, one per subject
        all_subject_ridge_r2 = []     # List of arrays, one per subject
        feature_names = None
        
        for sub in tqdm(range(num_subs), desc=f'Fold {k}', disable=not verbose):
            batch = Batch.from_data_list([data[sub]]).to(device)
            out = vgae(batch)
            mu = out.mu.detach()
            logvar = out.logvar.detach()

            # Get latent samples
            zs = get_zs(mu, logvar, vgae)
            
            # Initialize storage for this subject's Ridge regression data
            subject_feature_gradients_list = []   # List of dicts, one per z_sample
            subject_ypred_grads = []              # List of arrays, one per z_sample
            
            for z_idx, z in enumerate(zs):
                # Get MLP prediction and its gradient
                ypred = get_ypred_from_z(z, vgae, mlp, batch)
                ypred_grad = compute_gradient(ypred, z)
                
                # Get reconstructed outputs
                x, triu_edges = get_reconstructions(z, vgae, batch)
                
                # Compute interpretable features and their gradients
                alignments, feature_gradients = get_alignments_and_features(
                    triu_edges=triu_edges, 
                    ypred_grad=ypred_grad, 
                    z=z,
                    rsn_mapping=rsn_mapping, 
                    rsn_names=rsn_names, 
                    receptor_maps=receptor_maps, 
                    x=x, 
                    node_attrs=node_attrs)
                
                # Store feature gradients and ypred_grad for Ridge regression
                subject_feature_gradients_list.append(feature_gradients)
                subject_ypred_grads.append(ypred_grad.detach().cpu().numpy())
                
                # Base record for this sample
                base_record = {'sub': sub,
                             'latent_sample': z_idx,
                             'ypred': ypred.item()}

                # Store alignments
                alignment_record = base_record.copy()
                alignment_record.update(alignments)
                alignment_records.append(alignment_record)
            
            # Fit Ridge regression models for this subject --------------------------------
            # Get consistent feature names
            if feature_names is None:
                feature_names = list(subject_feature_gradients_list[0].keys())
            num_features = len(feature_names)
            
            # Initialize storage for betas and R2 per z_sample
            subject_ridge_betas = np.zeros((num_z_samples, num_features))
            subject_ridge_r2 = np.zeros(num_z_samples)
            
            # Fit Ridge regression for each z_sample
            for z_idx in range(num_z_samples):
                feature_grads = subject_feature_gradients_list[z_idx] 
                X = np.array([feature_grads[feat_name] for feat_name in feature_names])  # (num_features, latent_dim)
                y = subject_ypred_grads[z_idx]  # (latent_dim,)
                
                # Normalize all gradients to unit length
                X_norms = np.linalg.norm(X, axis=1, keepdims=True)
                X_norms[X_norms == 0] = 1  # avoid division by zero
                X_normalized = X / X_norms
                
                # Normalize ypred_grad
                y_norm = np.linalg.norm(y)
                if y_norm > 0:
                    y_normalized = y / y_norm
                else:
                    y_normalized = y
                X_reshaped = X_normalized.T  # (latent_dim, num_features)
                
                # Fit Ridge regression
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_reshaped, y_normalized)
                
                # Get coefficients
                betas = ridge.coef_  # (num_features,)
                subject_ridge_betas[z_idx, :] = betas
                
                # Compute R2 using normalized values
                y_pred = ridge.predict(X_reshaped)
                r2 = r2_score(y_normalized, y_pred)
                subject_ridge_r2[z_idx] = r2
            
            # Store results for this subject
            all_subject_ridge_betas.append(subject_ridge_betas)
            all_subject_ridge_r2.append(subject_ridge_r2)
        
        # Convert alignments to dataframes ---------------------------------------
        alignment_df = pd.DataFrame(alignment_records)

        # Sort the feature columns
        non_feature_cols = ['sub', 'latent_sample', 'ypred']
        feature_cols = [col for col in alignment_df.columns if col not in non_feature_cols]

        # Make sure it matches feature_names
        assert set(feature_cols) == set(feature_names), \
            "Feature names from Ridge regression don't match feature columns from alignments"
        if not feature_cols == feature_names:
            feature_cols = feature_names

        # Compute mean alignments for each subject
        mean_alignments = np.zeros((num_subs, len(feature_cols)))
        for sub in range(num_subs):
            sub_df = alignment_df[alignment_df['sub'] == sub]
            for feature_idx, feature in enumerate(feature_cols):
                mean_alignments[sub, feature_idx] = np.mean(sub_df[feature].to_numpy())
        
        # Save mean alignments for this fold
        mean_alignments_df = pd.DataFrame(mean_alignments, columns=feature_cols)
        mean_alignments_df.to_csv(os.path.join(output_dir, f'k{k}_mean_alignments.csv'), index=False)
        ex.add_artifact(os.path.join(output_dir, f'k{k}_mean_alignments.csv'))

        # Store alignments for correlation analysis
        all_fold_alignments.append(mean_alignments)
        
        # Process Ridge regression results ---------------------------------------
        # all_subject_ridge_betas is a list of arrays, each of shape (num_z_samples, num_features)
        # all_subject_ridge_r2 is a list of arrays, each of length num_z_samples
        
        # For each subject, perform one-sample t-test on betas across z_samples
        ridge_beta_pvals = np.zeros((num_subs, len(feature_cols)))
        ridge_beta_pvals_fdr = np.zeros((num_subs, len(feature_cols)))
        ridge_beta_cohen_ds = np.zeros((num_subs, len(feature_cols)))
        ridge_mean_r2 = np.zeros(num_subs)
        
        for sub in range(num_subs):
            # Get betas for this subject: (num_z_samples, num_features)
            subject_betas = all_subject_ridge_betas[sub]
            
            # Get R2 for this subject
            subject_r2 = all_subject_ridge_r2[sub]
            ridge_mean_r2[sub] = np.mean(subject_r2)
            
            # For each feature, perform one-sample t-test
            pvals_subject = []
            for feat_idx, feat_name in enumerate(feature_cols):              
                # Get betas for this feature across z_samples
                betas_for_feature = subject_betas[:, feat_idx]
                
                # One-sample t-test
                if len(betas_for_feature) > 1:
                    t_stat, p_val = ttest_1samp(betas_for_feature, 0.0, nan_policy='omit')
                    cohen_d = calculate_cohens_d(betas_for_feature)
                else:
                    p_val = 1.0
                    cohen_d = 0.0
                
                ridge_beta_pvals[sub, feat_idx] = p_val
                ridge_beta_cohen_ds[sub, feat_idx] = cohen_d
                pvals_subject.append(p_val)
            
            # Perform FDR correction for this subject across all features
            _, pvals_fdr_subject = fdrcorrection(pvals_subject, alpha=0.05)
            ridge_beta_pvals_fdr[sub, :] = pvals_fdr_subject
        
        # Save Ridge regression results
        ridge_beta_pvals_df = pd.DataFrame(ridge_beta_pvals_fdr, columns=feature_cols)
        ridge_beta_pvals_df.to_csv(os.path.join(output_dir, f'k{k}_ridge_beta_pvals.csv'), index=False)
        ex.add_artifact(os.path.join(output_dir, f'k{k}_ridge_beta_pvals.csv'))
        
        ridge_beta_cohen_ds_df = pd.DataFrame(ridge_beta_cohen_ds, columns=feature_cols)
        ridge_beta_cohen_ds_df.to_csv(os.path.join(output_dir, f'k{k}_ridge_beta_cohen_ds.csv'), index=False)
        ex.add_artifact(os.path.join(output_dir, f'k{k}_ridge_beta_cohen_ds.csv'))
        
        ridge_mean_r2_df = pd.DataFrame(ridge_mean_r2, columns=['mean_r2'])
        ridge_mean_r2_df.to_csv(os.path.join(output_dir, f'k{k}_ridge_mean_r2.csv'), index=False)
        ex.add_artifact(os.path.join(output_dir, f'k{k}_ridge_mean_r2.csv'))

        # Plot mean alignments for this fold
        sorted_features = sort_features(list(mean_alignments_df.columns))
        save_path = os.path.join(output_dir, f'k{k}_mean_alignments.png')
        vmax = np.percentile(np.abs(mean_alignments_df.values), 95)
        plot_heatmap(mean_alignments_df[sorted_features].values, 
                     features=sorted_features, 
                     vrange=(-vmax, vmax), 
                     figsize=(10, 4), 
                     cmap=COOLWARM,
                     save_path=save_path)

    # Compute correlations between folds for each subject
    if len(vgaes) > 1:
        corr_matrices = []
        for sub in range(num_subs):
            # Compute correlation matrix between folds for this subject
            sub_alignments = np.array([fold_alignments[sub] for fold_alignments in all_fold_alignments])
            corr_matrix = np.corrcoef(sub_alignments)
            corr_matrices.append(corr_matrix)

        # Compute mean correlations between folds
        mean_correlations = np.mean(corr_matrices, axis=0)
        np.savetxt(os.path.join(output_dir, 'mean_fold_correlations.csv'), mean_correlations, delimiter=',')
        ex.add_artifact(os.path.join(output_dir, 'mean_fold_correlations.csv'))

        # Check how many correlations are significant
        num_features = mean_alignments_df.shape[1]
        r_critical = min_significant_r(num_features)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(mean_correlations, ax=ax, cmap=COOLWARM, vmin=-1, vmax=1, 
                    square=True, annot=True, cbar=True)
        num_unique_pairs = num_folds * (num_folds - 1) / 2
        num_significant_pairs = np.sum(np.triu(mean_correlations, k=1) > r_critical)
        fraction_significant_pairs = (num_significant_pairs / num_unique_pairs)*100
        ax.set_title(f'Significant corrs (r > {r_critical:.2f}): {fraction_significant_pairs:.2f}%');
        save_path = os.path.join(output_dir, 'mean_fold_correlations.png')
        plt.savefig(save_path)
        ex.add_artifact(save_path)

    # Log the runtime
    run_time = (time()-start_time)/60
    print(f"Experiment completed. Runtime: {run_time:.2f} min.")
    