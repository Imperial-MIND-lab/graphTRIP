'''
Computes regional attributions for all patients with all fold models.

Authors: Hanna M. Tolle
Date: 2025-12-19
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

from utils.files import add_project_root
from utils.helpers import get_logger, fix_random_seed
from utils.configs import load_ingredient_configs, match_ingredient_configs
from models.utils import freeze_model
from preprocessing.metrics import get_atlas


# Create the experiment -------------------------------------------------------
ex = Experiment('regional_attributions', ingredients=[data_ingredient, 
                                                      vgae_ingredient, 
                                                      mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations -------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'regional_attributions'
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
    node_attrs = _config['dataset']['node_attrs']
    node_attrs = [attr.split('_')[0] for attr in node_attrs]

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

    # Sampling for each subject --------------------------------------------
    start_time = time()

    num_subs = len(data)
    num_folds = len(vgaes) 
    for k in range(num_folds):
        
        # Load models for this fold
        vgae = vgaes[k].eval()
        mlp = mlps[k].eval()
        
        for sub in tqdm(range(num_subs), desc=f'Fold {k}', disable=not verbose):
            batch = Batch.from_data_list([data[sub]]).to(device)
            out = vgae(batch)
            mu = out.mu.detach()
            logvar = out.logvar.detach()

            # Get latent samples
            zs = get_zs(mu, logvar, vgae)
            
            # Get dimensions
            num_nodes = batch[0].num_nodes
            latent_dim = zs[0].shape[1]
            num_z_samples = len(zs)

            # Initialize storage for outputs
            all_regional_attributions = np.zeros((num_z_samples, num_nodes))
            all_raw_grads = torch.zeros((num_z_samples, num_nodes, latent_dim), device=device)
            
            for i, z in enumerate(zs):
                ypred = get_ypred_from_z(z, vgae, mlp, batch)
                ypred_grad = compute_gradient(ypred, z) # (num_nodes * latent_dim,)
                
                # Store raw gradient for coherence calculation
                all_raw_grads[i] = ypred_grad.view(num_nodes, latent_dim).detach()

                # Compute regional attributions (percentage energy of total gradient)
                with torch.no_grad():
                    ypred_grad_norm = torch.nn.functional.normalize(ypred_grad, p=2, dim=0)
                    ypred_grad_reshaped = ypred_grad_norm.view(num_nodes, latent_dim)
                    regional_attributions = torch.sum(ypred_grad_reshaped ** 2, dim=1) * 100
                    all_regional_attributions[i] = regional_attributions.cpu().numpy()

            # Compute coherence metrics ----------------------------------------------
            with torch.no_grad():
                # 1. Whole-brain coherence
                flat_grads = all_raw_grads.view(num_z_samples, -1) # (num_z_samples, num_nodes * latent_dim)
                norm_of_mean_grad = torch.norm(torch.mean(flat_grads, dim=0), p=2)  # "signal"
                mean_of_grad_norms = torch.mean(torch.norm(flat_grads, p=2, dim=1)) # "total energy"
                whole_brain_coherence = (norm_of_mean_grad / (mean_of_grad_norms + 1e-8)).cpu().item()

                # 2. Regional coherence (coherence of each node's partial gradient)
                node_vector_mean = torch.mean(all_raw_grads, dim=0)
                node_vector_mean_norm = torch.norm(node_vector_mean, p=2, dim=1) # (num_nodes,)
                
                # Mean of individual norms: (num_nodes,)
                node_unit_norms = torch.norm(all_raw_grads, p=2, dim=2) # (num_z_samples, num_nodes)
                mean_node_norm = torch.mean(node_unit_norms, dim=0) # (num_nodes,)
                regional_coherence = (node_vector_mean_norm / (mean_node_norm + 1e-8)).cpu().numpy() # (num_nodes,)

            # Save outputs --------------------------------------------------------------
            sub_output_dir = os.path.join(output_dir, f'sub_{sub}')
            os.makedirs(sub_output_dir, exist_ok=True)
            
            # Save Attribution Dataframe (Mean across samples)
            regional_attributions_df = pd.DataFrame([np.mean(all_regional_attributions, axis=0)], 
                                                    columns=brain_region_labels)
            attr_path = os.path.join(sub_output_dir, f'k{k}_regional_attributions.csv')
            regional_attributions_df.to_csv(attr_path, index=False)
            ex.add_artifact(attr_path)

            # Save Coherence Dataframe (Single row: whole_brain followed by regions)
            coherence_data = np.insert(regional_coherence, 0, whole_brain_coherence)
            coherence_columns = ['whole_brain'] + brain_region_labels
            coherence_df = pd.DataFrame([coherence_data], columns=coherence_columns)
            
            coh_path = os.path.join(sub_output_dir, f'k{k}_regional_coherence.csv')
            coherence_df.to_csv(coh_path, index=False)
            ex.add_artifact(coh_path)

    # Log the runtime
    run_time = (time()-start_time)/60
    print(f"Experiment completed. Runtime: {run_time:.2f} min.")
    