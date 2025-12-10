'''
Ingredient for building the VGAE model, which learns
a latent representation of the neuroimaging features.

License: BSD 3-Clause
Author: Hanna M. Tolle
'''

import sys
sys.path.append('../../')

from sacred import Ingredient
from .data_ingredient import data_ingredient

import torch
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import copy
import os
from models.utils import init_model, get_optional_params


# Create the ingredient --------------------------------------------------------
vgae_ingredient = Ingredient('vgae_model', ingredients=[data_ingredient])

# Define configurations --------------------------------------------------------
@vgae_ingredient.config
def vgae_cfg(dataset):
    # Shared configurations
    model_type = 'NodeLevelVGAE'
    params = {'node_emb_dim': 32,
              'latent_dim': 32,
              'dropout': 0.25,
              'reg_strength': 0.01,
              'num_nodes': dataset['num_nodes'],
              'num_node_attr': len(dataset['node_attrs']),
              'num_edge_attr': len(dataset['edge_attrs']),
              'num_graph_attr': len(dataset['graph_attrs']),
              'num_cond_attrs': len(dataset['cond_attrs']),
              'num_context_attrs': len(dataset['context_attrs']),
              'max_spd_dist': dataset['max_spd_dist']}

    # Node embedding model
    node_emb_model_cfg = {'model_type': 'NodeEmbeddingGCN'}
    node_emb_model_cfg['params'] = get_optional_params(node_emb_model_cfg['model_type'])

    # Pooling layer
    pooling_cfg = {'model_type': 'GlobalAttentionPooling'}
    pooling_cfg['params'] = get_optional_params(pooling_cfg['model_type'])

    # Encoder model
    encoder_cfg = {'model_type': 'DenseOneLayerEncoder'}
    encoder_cfg['params'] = get_optional_params(encoder_cfg['model_type'])

    # Decoder models (for node and edge attributes and edge indices)
    node_decoder_cfg, edge_decoder_cfg, edge_idx_decoder_cfg = {}, {}, {}
    if params['num_node_attr'] > 0:
        node_decoder_cfg['model_type'] = 'MLPNodeDecoder'
        node_decoder_cfg['params'] = get_optional_params(node_decoder_cfg['model_type'])

    if params['num_edge_attr'] > 0:
        edge_decoder_cfg['model_type'] = 'TanhDecoder' if model_type == 'GraphLevelVGAE' else 'MLPEdgeDecoder'
        edge_decoder_cfg['params'] = get_optional_params(edge_decoder_cfg['model_type'])

# Captured functions -----------------------------------------------------------
@vgae_ingredient.capture
def build_vgae(model_type, params, 
               node_emb_model_cfg, 
               pooling_cfg, 
               encoder_cfg, 
               node_decoder_cfg, 
               edge_decoder_cfg,
               edge_idx_decoder_cfg):
               
    # If node_emb_model is NodeEmbeddingGraphormer, max_spd_dist must be provided
    if node_emb_model_cfg['model_type'] == 'NodeEmbeddingGraphormer' \
        and params['max_spd_dist'] is None:
        raise ValueError("max_spd_dist must be provided for NodeEmbeddingGraphormer")    
    
    # Combine all config dictionaries
    updated_params = copy.deepcopy(params)
    combined_params = {'params': updated_params,
                       'node_emb_model_cfg': node_emb_model_cfg,
                       'pooling_cfg': pooling_cfg,
                       'encoder_cfg': encoder_cfg,
                       'node_decoder_cfg': node_decoder_cfg,
                       'edge_decoder_cfg': edge_decoder_cfg}
    
    # Add edge index decoder config for NodeLevelVGAEs
    if not "GraphLevelVGAE" in model_type:
        combined_params['edge_idx_decoder_cfg'] = edge_idx_decoder_cfg

    return init_model(model_type, combined_params)

def build_vgae_from_config(config):
    valid_args = ['model_type', 'params', 'node_emb_model_cfg', 'pooling_cfg',
                  'encoder_cfg', 'node_decoder_cfg', 'edge_decoder_cfg',
                  'edge_idx_decoder_cfg']
    input_args = {k: config[k] for k in valid_args if k in config}
    return build_vgae(**input_args)

@vgae_ingredient.capture
def load_trained_vgaes(weights_dir, weight_filenames, device=None, exclude_module=None):
    '''
    Loads trained VGAEs from a directory, optionally excluding specific modules.

    Parameters:
    ----------
        weights_dir (str): Directory to load the VGAEs from.
        weight_filenames (list): List of VGAE weight filenames.
        device (torch.device): Device to load the VGAEs to.
        exclude_module (str): Name of the module to exclude (e.g., 'pooling' or 'encoder')

    Returns:
    -------
        list: List of trained VGAEs.
    '''
    if device is None:
        device = torch.device('cpu')
    vgaes = []
    for weight_file in weight_filenames:
        vgae = build_vgae().to(device)
        state_dict = torch.load(os.path.join(weights_dir, weight_file))
        
        if exclude_module:
            # Filter out keys that start with the excluded module name
            filtered_state_dict = {
                k: v for k, v in state_dict.items() 
                if not k.startswith(exclude_module)}
            vgae.load_state_dict(filtered_state_dict, strict=False)
        else:
            vgae.load_state_dict(state_dict)
        vgaes.append(vgae)
    return vgaes

@vgae_ingredient.capture
def load_vgaes_from_states_list(vgae_states_list, device):
    '''Loads VGAE models from a list of states.'''
    vgaes = []
    for state in vgae_states_list:
        vgae = build_vgae().to(device)
        vgae.load_state_dict(state)
        vgaes.append(vgae)
    return vgaes

@vgae_ingredient.capture
def train_vgae(vgae, loader, optimizer, device):
    '''Performs model training for one epoch.'''
    vgae.train()  
    train_loss = 0.
    num_batches = len(loader)
    for batch in loader:  
        batch = batch.to(device)             # Move data to device
        optimizer.zero_grad()                # Empty the gradients
        out = vgae(batch)                    # Get predictions
        loss = vgae.loss(out)                # Compute loss
        loss.backward()                      # Backpropagation
        optimizer.step()                     # Gradient updates
        train_loss += loss.item()

    return train_loss/num_batches 

@vgae_ingredient.capture
def test_vgae(vgae, loader, device):
    '''Performs model evaluation for one epoch.'''
    vgae.eval()  
    test_loss = 0.
    num_batches = len(loader)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)          # Move data to device
            out = vgae(batch)                 # Get predictions
            loss = vgae.loss(out)             # Compute loss
            test_loss += loss.item()

    return test_loss/num_batches

@vgae_ingredient.capture
def get_orig_recon_fc(model, dataset, device=None):
    '''Returns the original and reconstructed adjacency matrices.'''
    # Create a dataloader that loads single data objects
    mini_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Get device
    if device is None:
        device = torch.device('cpu')

    orig_matrices, recon_matrices = [], []
    orig_x, recon_x = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(mini_loader):

            # Move data to device
            batch = batch.to(device)

            # Get VGAE outputs
            out = model(batch)
            recon_edges = out.rcn_edges.detach()
            edges = out.edges.detach()

            # Also get node embeddings, if they exist
            if out.rcn_x is not None:
                recon_x.append(out.rcn_x.detach().cpu().numpy())
                orig_x.append(out.x.detach().cpu().numpy())

            # Triu edge indices of FC
            num_nodes = batch.num_nodes # works because batch_size == 1
            num_fc_triu_edges = num_nodes * (num_nodes - 1) // 2
            num_edge_attr = batch.edge_attr.shape[1]

            # Reshape edges and recon_edges to be of shape [num_edges, num_edge_attr]
            edges = edges.view(-1, num_edge_attr)
            recon_edges = recon_edges.view(-1, num_edge_attr)

            if edges.shape[0] == num_fc_triu_edges:
                edge_index = torch.triu(torch.ones((num_nodes, num_nodes)), diagonal=1).nonzero(as_tuple=False).t()
            else:
                edge_index = batch.edge_index
                triu_mask = edge_index[0] < edge_index[1]
                edge_index = edge_index[:, triu_mask]
            edge_index = edge_index.to(device)

            # Create reconstructed and original dense matrices
            recon = torch.zeros((num_nodes, num_nodes), device=device)
            orig = torch.zeros((num_nodes, num_nodes), device=device)

            recon[edge_index[0], edge_index[1]] = recon_edges[:, 0]
            recon[edge_index[1], edge_index[0]] = recon_edges[:, 0]

            orig[edge_index[0], edge_index[1]] = edges[:, 0]
            orig[edge_index[1], edge_index[0]] = edges[:, 0]

            orig_matrices.append(orig.cpu().numpy())
            recon_matrices.append(recon.cpu().numpy())

    # Stack to numpy arrays
    orig_matrices = np.stack(orig_matrices, axis=-1)  # Shape: (num_nodes, num_nodes, num_samples)
    recon_matrices = np.stack(recon_matrices, axis=-1)  # Shape: (num_nodes, num_nodes, num_samples)
            
    return orig_matrices, recon_matrices, orig_x, recon_x

def get_test_reconstructions(vgaes, dataset, test_indices, 
                             mean_std=None, device='cpu'):
    '''
    Gets FC & node embedding reconstructions for test sets across all folds.
    
    Parameters:
    ----------
    vgaes: List of k trained VGAE models
    dataset: The full dataset
    test_indices: List of k numpy arrays containing test fold indices
    mean_std: Dict containing 'mean' and 'std' lists for each fold
    device: Torch device to use

    Returns:
    -------
    adj_orig_rcn: Dict with 'original' and 'reconstructed' arrays of shape 
                  (n_nodes, n_nodes, n_samples) for adjacency matrices
    x_orig_rcn: Dict with 'original' and 'reconstructed' arrays of shape 
                (n_nodes, n_node_features, n_samples) for node features
    fold_assignments: Array of length n_samples indicating which fold's VGAE was used
    '''
    n_nodes = dataset[0].x.shape[0]
    n_node_features = dataset[0].x.shape[1]
    node_feature_names = dataset[0].attr_names.node
    n_samples = len(dataset)
    
    # Initialize output arrays
    adj_orig = np.zeros((n_nodes, n_nodes, n_samples))
    adj_rcn = np.zeros((n_nodes, n_nodes, n_samples))
    x_orig = np.zeros((n_nodes, n_node_features, n_samples))
    x_rcn = np.zeros((n_nodes, n_node_features, n_samples))
    fold_assignments = np.zeros(n_samples)
    
    # Get reconstructions for each fold
    for fold, vgae in enumerate(vgaes):
        fold_indices = test_indices[fold]
        fold_dataset = dataset[fold_indices]
        
        # Get reconstructions
        fold_orig, fold_recon, fold_orig_x, fold_recon_x = get_orig_recon_fc(vgae, 
                                                                             fold_dataset, 
                                                                             device=device)
        
        # Un-standardize the node embeddings using this fold's mean and std
        x_exists = len(fold_orig_x) > 0 and len(fold_recon_x) > 0
        if mean_std is not None and x_exists:
            fold_mean = mean_std['mean'][fold].numpy()
            fold_std = mean_std['std'][fold].numpy()
            fold_orig_x = [x * fold_std + fold_mean for x in fold_orig_x]
            fold_recon_x = [x * fold_std + fold_mean for x in fold_recon_x]
        
        # Store reconstructions and originals
        for i, sub in enumerate(fold_indices):
            adj_orig[:, :, sub] = fold_orig[:, :, i]
            adj_rcn[:, :, sub] = fold_recon[:, :, i]
            if x_exists:
                x_orig[:, :, sub] = fold_orig_x[i]
                x_rcn[:, :, sub] = fold_recon_x[i]
            fold_assignments[sub] = fold
    
    # Create output dictionaries
    adj_orig_rcn = {'original': adj_orig, 'reconstructed': adj_rcn}
    if x_exists:
        x_orig_rcn = {'original': x_orig, 'reconstructed': x_rcn, 'feature_names': node_feature_names}
    else:
        x_orig_rcn = {}
    
    return adj_orig_rcn, x_orig_rcn, fold_assignments

def get_mean_test_reconstructions(vgaes_dict, data, test_indices_dict=None):
    """
    Compute mean adj_orig_rcn and x_orig_rcn across seeds and folds.

    If test_indices_dict is None, reconstruct all samples with all VGAEs (from all folds, all seeds)
    and average reconstructions for each sample across all VGAEs.

    Parameters
    ----------
    test_indices_dict : dict or None
        Dict mapping seed keys to test indices arrays, where the i-th element indicates 
        the index of the VGAE model (trained with the partical seed) which had
        data sample i in its testfold. E.g., test_indices_dict[seed] = [0, 1, 1, 0]
        means sample 0 and 3 were in the testfold of vgaes_dict[seed][0], and
        sample 1 and 2 were in the testfold of vgaes_dict[seed][1].
        If None, reconstructions are generated for all samples by all VGAEs of all seeds.
    vgaes_dict : dict
        Dict mapping seed keys to CV-fold validated VGAE models, trained with that seed.
    data : object
        Dataset or data object required by get_test_reconstructions.

    Returns
    -------
    mean_adj_orig_rcn : dict
        Dict with 'original' and 'reconstructed' keys containing mean adjacency reconstructions.
    mean_x_orig_rcn : dict
        Dict with 'original', 'reconstructed', and 'feature_names' for node features.
    """
    # Determine number of samples and nodes
    n_samples = len(data)
    n_nodes = data[0].x.shape[0]
    n_node_features = data[0].x.shape[1]

    adj_shape = (n_nodes, n_nodes, n_samples)
    x_shape = (n_nodes, n_node_features, n_samples)

    if test_indices_dict is None:
        # For every VGAE in every seed and every fold, reconstruct all data samples
        all_adj_orig = []
        all_adj_rcn = []
        all_x_orig = []
        all_x_rcn = []
        feature_names = None

        for seed_key in vgaes_dict:
            vgaes = vgaes_dict[seed_key]
            for fold in range(len(vgaes)):
                vgae = vgaes[fold]
                # Use all data for reconstructions (pass all indices in one fold)
                test_indices_list = [np.arange(n_samples)]
                adj_rcn, x_rcn, _ = get_test_reconstructions([vgae], data, test_indices_list, mean_std=None)
                # expected output shape: adj_rcn['original']: (n_nodes, n_nodes, n_samples)
                all_adj_orig.append(adj_rcn['original'])
                all_adj_rcn.append(adj_rcn['reconstructed'])
                all_x_orig.append(x_rcn['original'])
                all_x_rcn.append(x_rcn['reconstructed'])
                if feature_names is None:
                    feature_names = x_rcn.get('feature_names', None)

        mean_adj_orig_rcn = {
            'original': np.mean(all_adj_orig, axis=0),
            'reconstructed': np.mean(all_adj_rcn, axis=0)
        }
        mean_x_orig_rcn = {
            'original': np.mean(all_x_orig, axis=0),
            'reconstructed': np.mean(all_x_rcn, axis=0),
            'feature_names': feature_names
        }
        return mean_adj_orig_rcn, mean_x_orig_rcn

    else:
        # Average each patient's reconstruction only across corresponding fold+seed
        adj_orig_rcn_list = []
        x_orig_rcn_list = []
        for seed_key in test_indices_dict:
            num_folds = max(test_indices_dict[seed_key]) + 1
            test_indices_list = [np.where(test_indices_dict[seed_key] == fold)[0] for fold in range(num_folds)]
            vgae = vgaes_dict[seed_key]
            adj_rcn, x_rcn, _ = get_test_reconstructions(vgae, data, test_indices_list, mean_std=None)
            adj_orig_rcn_list.append(adj_rcn)
            x_orig_rcn_list.append(x_rcn)

        mean_adj_orig_rcn = {
            'original': np.mean([d['original'] for d in adj_orig_rcn_list], axis=0),
            'reconstructed': np.mean([d['reconstructed'] for d in adj_orig_rcn_list], axis=0)
        }
        mean_x_orig_rcn = {
            'original': np.mean([d['original'] for d in x_orig_rcn_list], axis=0),
            'reconstructed': np.mean([d['reconstructed'] for d in x_orig_rcn_list], axis=0),
            'feature_names': x_orig_rcn_list[0]['feature_names']
        }
        return mean_adj_orig_rcn, mean_x_orig_rcn

def evaluate_fc_reconstructions(adj_orig_rcn):
    '''
    Compute correlations and MAE between original and reconstructed FC matrices.
    
    Parameters:
    ----------
    adj_orig_rcn: Dict with 'original' and 'reconstructed' arrays of shape 
                  (n_nodes, n_nodes, n_samples) for adjacency matrices
    
    Returns:
    -------
    dict: Dictionary containing:
        - 'correlations': List of correlations between original and reconstructed matrices
        - 'mae': List of mean absolute errors between original and reconstructed matrices
    '''
    fc_corrs = []
    fc_maes = []
    n_nodes = adj_orig_rcn['original'].shape[0]
    n_samples = adj_orig_rcn['original'].shape[2]
    
    for sub in range(n_samples):
        orig = np.tril(adj_orig_rcn['original'][:, :, sub], -1)
        recon = np.tril(adj_orig_rcn['reconstructed'][:, :, sub], -1)
        # Get values from triangles
        orig_vals = orig[np.tril_indices(n_nodes, k=-1)]
        recon_vals = recon[np.tril_indices(n_nodes, k=-1)]
        # Compute correlation
        r = pearsonr(orig_vals, recon_vals)[0]
        fc_corrs.append(r)
        # Compute MAE
        mae = np.mean(np.abs(orig_vals - recon_vals))
        fc_maes.append(mae)
    
    return {'corr': fc_corrs, 'mae': fc_maes}

def evaluate_x_reconstructions(x_orig_rcn):
    '''
    Compute MAE and correlations between original and reconstructed node features.
    
    Parameters:
    ----------
    x_orig_rcn (dict): Output from get_test_reconstructions
        keys: 'original' (n_nodes, n_features, n_samples), 
              'reconstructed' (n_nodes, n_features, n_samples), 
              'feature_names' (n_features)
    
    Returns:
    -------
    dict: Dictionary containing:
        - 'mae_df': DataFrame of MAE values per feature and sample
        - 'corr_df': DataFrame of correlation values per feature and sample
    '''
    feature_names = x_orig_rcn['feature_names']
    n_samples = x_orig_rcn['original'].shape[2]
    
    # Initialize dataframes
    mae_df = pd.DataFrame(index=range(n_samples), columns=feature_names)
    corr_df = pd.DataFrame(index=range(n_samples), columns=feature_names)
    
    # Compute MAE and correlation of orig vs. recon for each subject
    for i, feature in enumerate(feature_names):
        orig = x_orig_rcn['original'][:, i, :]  # Shape: (n_nodes, n_samples)
        recon = x_orig_rcn['reconstructed'][:, i, :]  # Shape: (n_nodes, n_samples)
        mae_df[feature] = np.mean(np.abs(orig - recon), axis=0)
        corr_df[feature] = np.array([pearsonr(orig[:, sub], recon[:, sub])[0] for sub in range(n_samples)])
    
    return {'corr': corr_df, 'mae': mae_df}
