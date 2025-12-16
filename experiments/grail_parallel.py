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
import json
from tqdm import tqdm
from time import time
import pandas as pd
import logging
from torch_geometric.data import Batch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial

from utils.files import add_project_root
from utils.helpers import get_logger, fix_random_seed, triu_vector2mat_torch
from utils.configs import load_ingredient_configs, match_ingredient_configs
from utils.annotations import load_receptor_maps, load_rotated_rois
from utils.statsalg import get_fold_performance, test_column_significance, compute_permutation_stats
from models.utils import freeze_model
from preprocessing.metrics import compute_modularity_torch, get_rsn_mapping, get_atlas
from statsmodels.stats.multitest import fdrcorrection


# Create the experiment -------------------------------------------------------
ex = Experiment('grail_parallel', ingredients=[data_ingredient, 
                                      vgae_ingredient, 
                                      mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations -------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'grail_parallel'
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
    num_z_samples = 25          # Number of samples to draw from the VGAE latent distribution.
    sigma = 2.0                 # Std of the Gaussian noise added to the latent means.
    all_rsn_conns = False       # Whether to compute RSN connectivity for all RSN pairs.
    medusa = False              # Only works with CFRHead MLP. Grails mlp1-mlp0 instead of ypred.
    this_k = None               # If None, compute all folds sequentially. If int, compute only fold this_k.
    n_permutations = 1000       # Number of permutations to use for null model analysis.
    cohen_d_threshold = 0.8     # Threshold for Cohen's d to consider a feature significant.
    n_workers = 1               # Number of parallel workers for subject processing. Set to 1 for sequential, >1 for parallel.
    use_threads = True           # If True, use ThreadPoolExecutor (safer with CUDA). If False, use ProcessPoolExecutor.

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
                                all_rsn_conns,
                                x = None, node_attrs = None,
                                selected_features = None):
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
    selected_features (list, optional): If provided, only compute alignments for these features.

    Returns:
    --------
    alignments (dict): The alignments between the feature gradients and ypred_grad.
    """
    alignments = {}
    device = triu_edges.device
    
    # Get adjacency matrix
    adj = triu_vector2mat_torch(triu_edges)
    
    # Process each feature one at a time 
    def process_feature(name, value):
        if selected_features is None or name in selected_features:
            grad = compute_gradient(value, z)
            alignment = compute_gradient_alignment(ypred_grad, grad)
            alignments[name] = alignment

    # Compute modularity using RSN mapping
    rsn_modularity = compute_modularity_torch(adj, rsn_mapping)
    process_feature('modularity_rsn', rsn_modularity)

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
    
    return alignments

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

# Subject processing function for parallelization ------------------------------
def process_subject(sub, data, vgae, mlp, device, rsn_mapping, rsn_names, receptor_maps, 
                   rotated_roi_indices, node_attrs, num_z_samples, sigma, 
                   cohen_d_threshold, n_permutations, medusa, all_rsn_conns):
    """
    Process a single subject and return results.
    
    Returns:
    -------
    dict with keys:
        - 'sub': subject index
        - 'regional_importance': numpy array of regional importance scores
        - 'mean_alignments': dict of mean alignments
        - 'selected_features': list of significant features
        - 'feature_names': list of feature names (for first subject only)
    """
    batch = Batch.from_data_list([data[sub]]).to(device)
    out = vgae(batch)
    mu = out.mu.detach()
    logvar = out.logvar.detach()

    # Get latent samples
    zs = get_zs(mu, logvar, vgae, num_z_samples, sigma)
    
    # Get number of nodes and latent dimension
    num_nodes = batch[0].num_nodes
    latent_dim = zs[0].shape[1]
    
    # Initialize storage for regional importance scores across latent samples
    subject_regional_importance = []
    
    # Storage for alignments across latent samples
    subject_alignments = []
    ypred_grads = []
    zs_list = []
    xs_list = []
    triu_edges_list = []
    
    for z in zs:
        # Get MLP prediction and its gradient
        ypred = get_ypred_from_z(z, vgae, mlp, batch, medusa)
        ypred_grad = compute_gradient(ypred, z)
        
        # Store ypred_grad for null distribution computation
        ypred_grads.append(ypred_grad.detach().clone())
        zs_list.append(z)
        
        # Regional gradient analysis
        # Normalize ypred_grad
        ypred_grad_copy = ypred_grad.clone().detach()
        ypred_grad_norm = torch.nn.functional.normalize(ypred_grad_copy, p=2, dim=0)
        
        # Compute regional importance scores
        ypred_grad_norm_reshaped = ypred_grad_norm.view(num_nodes, latent_dim)
        regional_importance = torch.sum(torch.abs(ypred_grad_norm_reshaped), dim=1)
        subject_regional_importance.append(regional_importance.cpu().numpy())
        
        # Get reconstructed outputs
        x, triu_edges = get_reconstructions(z, vgae, batch)
        xs_list.append(x)
        triu_edges_list.append(triu_edges)
        
        # Compute interpretable features and their gradients
        alignments = get_alignments_and_features(
            triu_edges=triu_edges, 
            ypred_grad=ypred_grad, 
            z=z,
            rsn_mapping=rsn_mapping, 
            rsn_names=rsn_names, 
            receptor_maps=receptor_maps,
            all_rsn_conns=all_rsn_conns,
            x=x, 
            node_attrs=node_attrs)
        subject_alignments.append(alignments)
    
    # Compute mean regional importance scores across latent samples for this subject
    subject_regional_importance_mean = np.mean(subject_regional_importance, axis=0)
    
    # First-stage GRAIL feature screening
    # Convert alignments to dataframe and store mean alignments for this subject
    feature_alignments = pd.DataFrame(subject_alignments)
    observed_mean_alignments = feature_alignments.mean(axis=0)
    feature_names = list(subject_alignments[0].keys())
    
    # Perform 1-sample t-test for each feature 
    feature_stats = test_column_significance(feature_alignments, test_type='t-test')
    fdr_p_values = feature_stats['fdr_p_value']
    cohen_ds = feature_stats['cohen_d']
    
    # Identify selected features
    selected_features = [feat for feat, fdr_p, cd in zip(feature_names, fdr_p_values, cohen_ds) 
                        if fdr_p < 0.05 and abs(cd) > cohen_d_threshold]
    
    # Second-stage GRAIL permutation test
    significant_features = []
    if len(selected_features) > 0:
        mean_alignment_null_distributions = []
        
        # Compute mean alignment across latent samples for each spin map
        for perm_idx in range(n_permutations):
            perm_indices = rotated_roi_indices[:, perm_idx]
            
            # Create permuted rsn_mapping and receptor_maps
            rsn_mapping_perm = rsn_mapping[perm_indices]
            receptor_maps_perm = {name: values[perm_indices] 
                                 for name, values in receptor_maps.items()}
            null_alignments = []
            for z, ypred_grad, x, triu_edges in zip(zs_list, ypred_grads, xs_list, triu_edges_list):
                # Compute alignments for non-correlation features
                alignments_perm = get_alignments_and_features(
                    triu_edges=triu_edges,
                    ypred_grad=ypred_grad,
                    z=z,
                    rsn_mapping=rsn_mapping_perm,
                    rsn_names=rsn_names,
                    receptor_maps=receptor_maps_perm,
                    all_rsn_conns=all_rsn_conns,
                    x=x,
                    node_attrs=node_attrs,
                    selected_features=selected_features)

                null_alignments.append(alignments_perm)
            
            # Average across latent samples to get mean alignment for each permutation
            mean_alignment_null_distributions.append(pd.DataFrame(null_alignments).mean(axis=0))
        
        # Perform permutation test for each selected feature
        mean_alignment_null_distributions = pd.DataFrame(mean_alignment_null_distributions)
        permtest_results = []
        for feat in selected_features:
            observed_mean = observed_mean_alignments[feat]
            null_dist = mean_alignment_null_distributions[feat].values
            
            # Two-tailed p-value: proportion of null values >= |observed|
            results = compute_permutation_stats(observed_mean, null_dist, alternative='two-sided')
            base_results = {'feature': feat}
            base_results.update(results)
            permtest_results.append(base_results)
        
        # FDR correction for permutation test p-values
        permtest_results = pd.DataFrame(permtest_results)
        permtest_results['fdr_p_value'] = fdrcorrection(permtest_results['p_value'], alpha=0.05)[1]

        # Identify significant features
        significant_features = list(permtest_results[permtest_results['fdr_p_value'] < 0.05]['feature'])
    
    return {
        'sub': sub,
        'regional_importance': subject_regional_importance_mean,
        'mean_alignments': observed_mean_alignments.to_dict(),
        'selected_features': significant_features,
        'feature_names': feature_names
    }

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
    node_attrs = _config['dataset']['node_attrs']
    node_attrs = [attr.split('_')[0] for attr in node_attrs]
    this_k = _config['this_k']
    cohen_d_threshold = _config['cohen_d_threshold']
    n_permutations = _config['n_permutations']
    n_workers = _config.get('n_workers', 1)
    use_threads = _config.get('use_threads', True)
    num_z_samples = _config.get('num_z_samples', 25)
    sigma = _config.get('sigma', 2.0)
    medusa = _config.get('medusa', False)
    all_rsn_conns = _config.get('all_rsn_conns', False)

    # Validate this_k input
    max_num_folds = _config['dataset']['num_folds']
    if this_k is not None:
        if this_k < 0 or this_k >= max_num_folds:
            raise ValueError(f'Invalid this_k value: {this_k}. Must be None or 0 <= this_k < {max_num_folds}.')

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

    # Get brain region labels
    atlas = get_atlas(data.atlas)
    brain_region_labels = [label.decode('utf-8') if isinstance(label, bytes) else label for label in atlas['labels']]

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
    rotated_roi_indices = load_rotated_rois(atlas, n_permutations=_config['n_permutations'])

    # Convert to tensors (needed for gradient computation later)
    rsn_mapping = torch.tensor(rsn_mapping, device=device)
    receptor_maps = {name: torch.tensor(values.values, device=device, dtype=torch.float32)
                    for name, values in receptor_maps.items()}

    num_folds = len(vgaes)
    
    # Determine which folds to compute
    if this_k is None:
        folds_to_compute = range(num_folds)
    else:
        folds_to_compute = [this_k]
    
    for k in folds_to_compute:
        
        # Load models for this fold
        vgae = vgaes[k].eval()
        mlp = mlps[k].eval()
        
        # Initialize storage for regional gradient weights
        all_subject_regional_importance = []
        
        # Storage for spin-permutation test results
        all_mean_alignments = []  
        all_selected_features = {}  # sub: [selected_features]
        
        # Create partial function with fixed arguments
        process_subject_partial = partial(
            process_subject,
            data=data,
            vgae=vgae,
            mlp=mlp,
            device=device,
            rsn_mapping=rsn_mapping,
            rsn_names=rsn_names,
            receptor_maps=receptor_maps,
            rotated_roi_indices=rotated_roi_indices,
            node_attrs=node_attrs,
            num_z_samples=num_z_samples,
            sigma=sigma,
            cohen_d_threshold=cohen_d_threshold,
            n_permutations=n_permutations,
            medusa=medusa,
            all_rsn_conns=all_rsn_conns
        )
        
        # Process subjects in parallel or sequentially
        if n_workers > 1:
            # Use parallel processing
            Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
            with Executor(max_workers=n_workers) as executor:
                # Submit all tasks
                futures = {executor.submit(process_subject_partial, sub): sub 
                          for sub in range(num_subs)}
                
                # Collect results as they complete
                results_list = []
                for future in tqdm(as_completed(futures), total=num_subs, 
                                 desc=f'Fold {k}', disable=not verbose):
                    try:
                        result = future.result()
                        results_list.append(result)
                    except Exception as exc:
                        sub_idx = futures[future]
                        logger.error(f'Subject {sub_idx} generated an exception: {exc}')
                        raise
            
            # Sort results by subject index to maintain order
            results_list.sort(key=lambda x: x['sub'])
            
            # Extract results
            for result in results_list:
                all_subject_regional_importance.append(result['regional_importance'])
                all_mean_alignments.append(result['mean_alignments'])
                all_selected_features[result['sub']] = result['selected_features']
        else:
            # Sequential processing (original behavior)
            for sub in tqdm(range(num_subs), desc=f'Fold {k}', disable=not verbose):
                result = process_subject_partial(sub)
                all_subject_regional_importance.append(result['regional_importance'])
                all_mean_alignments.append(result['mean_alignments'])
                all_selected_features[result['sub']] = result['selected_features']
        
        # Process alignment results ----------------------------------------------        
        # Save mean alignments from all subjects for this fold
        mean_alignments_df = pd.DataFrame(all_mean_alignments)
        mean_alignments_df.to_csv(os.path.join(output_dir, f'k{k}_mean_alignments.csv'), index=False)
        ex.add_artifact(os.path.join(output_dir, f'k{k}_mean_alignments.csv'))
        
        # Save selected features for each subject as a JSON file
        selected_features_path = os.path.join(output_dir, f'k{k}_selected_features.json')
        with open(selected_features_path, 'w') as f:
            json.dump(all_selected_features, f, indent=2)
        ex.add_artifact(selected_features_path)
        
        # Process regional gradient weights --------------------------------------
        regional_grad_weights = np.array(all_subject_regional_importance) # (num_subs, num_nodes)
        regional_grad_weights_df = pd.DataFrame(regional_grad_weights, columns=brain_region_labels)
        regional_grad_weights_df.to_csv(os.path.join(output_dir, f'k{k}_regional_grad_weights.csv'), index=False)
        ex.add_artifact(os.path.join(output_dir, f'k{k}_regional_grad_weights.csv'))

    # Log the runtime
    run_time = (time()-start_time)/60
    print(f"Experiment completed. Runtime: {run_time:.2f} min.")
    