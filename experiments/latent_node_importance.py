"""
Node-level importance analysis for the graph-level regression MLP.
Assesses the importance of individual brain regions (nodes) by replacing
their latent representations with training-fold averages.

Author: Hanna Tolle
Date: 2024-12-30
License: BSD 3-Clause
"""

import matplotlib
matplotlib.use('Agg') # Uncomment when running in debug mode

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.data_ingredient import * 
from experiments.ingredients.vgae_ingredient import * 
from experiments.ingredients.mlp_ingredient import * 

import os
import torch_geometric.transforms as T
from datasets import AddLabel
import torch
import torch.nn
import numpy as np
import logging
import matplotlib.pyplot as plt
import pandas as pd

from utils.files import add_project_root
from utils.configs import *
from utils.helpers import fix_random_seed, get_logger, check_weights_exist
from utils.annotations import load_annotations
from utils.statsalg import get_fold_performance
from preprocessing.metrics import get_atlas
from utils.plotting import plot_brain_surface


# Create experiment and logger -------------------------------------------------
ex = Experiment('latent_node_importance', ingredients=[data_ingredient, 
                                                       vgae_ingredient, 
                                                       mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'latent_node_importance'
    jobid = 0
    seed = 0
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join(project_root(), 'outputs', 'runs', run_name)

    # Model weights directory, filenames
    weights_dir = os.path.join('outputs', 'weights', 'final_config_screening', f'job{jobid}_seed{seed}')
    weight_filenames = {'vgae': [f'k{k}_vgae_weights.pth' for k in range(dataset['num_folds'])],
                        'mlp': [f'k{k}_mlp_weights.pth' for k in range(dataset['num_folds'])],
                        'test_fold_indices': ['test_fold_indices.csv']}
    
    # Condition filtering
    condition = None  # None, 'E', or 'P' - filter analysis to samples with given treatment condition
    
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

    # Other compatibiltiy checks
    num_folds = config_updates['dataset']['num_folds']
    weight_filenames = config_updates.get('weight_filenames', None)
    if weight_filenames is None:
        weight_filenames =  {'vgae': [f'k{k}_vgae_weights.pth' for k in range(num_folds)],
                             'mlp': [f'k{k}_mlp_weights.pth' for k in range(num_folds)],
                             'test_fold_indices': ['test_fold_indices.csv']}
    check_weights_exist(weights_dir, weight_filenames)
    config_updates['weight_filenames'] = weight_filenames

    # Don't support fixing the treatment condition 
    if 'drug_condition' in config_updates['dataset']:
        assert config_updates['dataset']['drug_condition'] is None, \
        "This experiment does not support fixing the treatment condition."

    # This experiment requires canonical node indices (i.e. one consistent brain atlas for all subjects)
    atlas = config_updates['dataset']['atlas']
    if isinstance(atlas, List) and len(atlas) > 1:
        raise ValueError("This experiment requires a single brain atlas for all subjects.")

    return config_updates

# Helper functions ------------------------------------------------------------
def compute_node_importance_scores(data, vgaes, mlps, testfold_indices, device, num_nodes):
    """
    Computes importance scores for each node by replacing its latent representation
    with the training-fold average.
    
    Parameters:
    ----------
    data (Dataset): The complete dataset
    vgaes (List[VGAE]): List of trained VGAE models, one for each fold
    mlps (List[LatentMLP]): List of trained MLP models, one for each fold
    testfold_indices (numpy array): Maps each data sample to its test fold
    device (torch.device): Device to run computations on
    num_nodes (int): Number of nodes per graph (canonical brain regions)
    
    Returns:
    -------
    node_scores (numpy array): Importance scores for each node [num_folds, num_nodes]
    """
    num_folds = len(vgaes)
    node_scores = np.zeros((num_folds, num_nodes))
    
    for k in range(num_folds):
        # Get train and test indices for this fold
        test_indices = np.where(testfold_indices == k)[0]
        train_indices = np.where(testfold_indices != k)[0]
        
        # Get VGAE and MLP for this fold
        vgae = vgaes[k]
        mlp = mlps[k]
        vgae.eval()
        mlp.eval()
        
        # Get training data and compute average latent representations per node
        train_data = data[train_indices]
        train_batch = next(iter(DataLoader(train_data, batch_size=len(train_indices), shuffle=False))).to(device)
        
        with torch.no_grad():
            # Get training latent means
            train_out = vgae(train_batch) 
            train_mu = train_out.mu  # [num_train_samples * num_nodes, latent_dim]
            
            # Compute train-fold average latent node representations 
            # (assumes canonical node indices and numbers across graphs!)
            num_train_graphs = len(train_indices)
            train_mu_reshaped = train_mu.view(num_train_graphs, num_nodes, -1)  # [num_train_graphs, num_nodes, latent_dim]
            node_avg_mu = train_mu_reshaped.mean(dim=0)  # [num_nodes, latent_dim]
        
        # Get test data
        test_data = data[test_indices]
        test_batch = next(iter(DataLoader(test_data, batch_size=len(test_indices), shuffle=False))).to(device)
        
        with torch.no_grad():
            test_context = get_context(test_batch)
            test_out = vgae(test_batch)  
            test_mu = test_out.mu.clone()  # [num_test_samples * num_nodes, latent_dim]
            test_ytrue = test_batch.y.cpu().numpy().flatten()
            test_clinical = test_batch.graph_attr
            test_treatment = get_treatment(test_batch, num_z_samples=0)
            
            # Compute baseline predictions for this test fold
            baseline_z_readout = vgae.readout(test_mu, test_context, test_batch.batch)
            baseline_mlp_input = torch.cat([baseline_z_readout, test_clinical], dim=1)
            if test_treatment is None:
                baseline_ypred = mlp(baseline_mlp_input)
            else:
                baseline_ypred = mlp(baseline_mlp_input, test_treatment)
            baseline_mae = np.mean(np.abs(baseline_ypred.cpu().numpy().flatten() - test_ytrue))
            
            # For each node, replace its representation with train-fold means and compute score
            num_test_graphs = len(test_indices)
            test_mu_reshaped = test_mu.view(num_test_graphs, num_nodes, -1)  # [num_test_graphs, num_nodes, latent_dim]
            
            for node_i in range(num_nodes):
                # Create modified test mu with node i replaced by training average
                modified_mu_reshaped = test_mu_reshaped.clone()
                modified_mu_reshaped[:, node_i, :] = node_avg_mu[node_i]  # Replace node i for all test graphs
                modified_mu = modified_mu_reshaped.view(-1, test_mu.shape[1])  # Reshape back to [num_test_nodes, latent_dim]
                
                # Apply readout and get MLP predictions
                modified_z_readout = vgae.readout(modified_mu, test_context, test_batch.batch)
                modified_mlp_input = torch.cat([modified_z_readout, test_clinical], dim=1)
                if test_treatment is None:
                    modified_ypred = mlp(modified_mlp_input)
                else:
                    modified_ypred = mlp(modified_mlp_input, test_treatment)
                modified_mae = np.mean(np.abs(modified_ypred.cpu().numpy().flatten() - test_ytrue))
                
                # Importance score: increase in MAE when node is replaced
                # Positive score means node is important (replacing it hurts performance)
                node_scores[k, node_i] = (modified_mae - baseline_mae)
    
    return node_scores

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Set up environment ------------------------------------------------------
    seed = _config['seed']
    verbose = _config['verbose']
    output_dir = add_project_root(_config['output_dir'])
    weights_dir = add_project_root(_config['weights_dir'])
    weight_filenames = _config['weight_filenames']
    is_cfrnet = _config['mlp_model']['model_type'] == 'CFRHead'

    # Make output directory, get device and fix random seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    device = torch.device(_config['device'])

    # Load data and trained models
    data = load_data()
    if is_cfrnet:
        # Add treatment transform to the dataset (required for CFRHead)
        add_treatment_transform(data)
    vgaes = load_trained_vgaes(weights_dir, weight_filenames['vgae'], device)
    mlps = load_trained_mlps(weights_dir, weight_filenames['mlp'], device, 
                             latent_dims=[vgae.readout_dim for vgae in vgaes])
    testfold_indices = np.loadtxt(os.path.join(weights_dir, weight_filenames['test_fold_indices'][0]), dtype=int)
    
    # Filter data by condition if specified
    condition = _config.get('condition', None)
    if condition is not None:
        # Load annotations to get condition mapping
        annotations = load_annotations(study=data.study)
        # Filter annotations to only include patients in the dataset
        # Note: Patient IDs in annotations are 1-indexed, subject IDs in data are 0-indexed
        subject_ids = [sub+1 for sub in data.subject.tolist()]
        annotations = annotations[annotations['Patient'].isin(subject_ids)]
        
        # Get condition mapping: Patient ID -> Condition ('E' or 'P')
        condition_dict = pd.Series(annotations['Condition'].values, index=annotations['Patient']).to_dict()
        
        # Find indices of samples with the specified condition
        condition_mask = np.array([condition_dict.get(sub, None) == condition for sub in subject_ids])
        condition_indices = np.where(condition_mask)[0]
        
        if len(condition_indices) == 0:
            raise ValueError(f"No samples found with condition '{condition}'. Available conditions: {set(condition_dict.values())}")
        logger.info(f"Filtering to condition '{condition}': {len(condition_indices)}/{len(data)} samples")
        
        # Filter data to only include samples with the specified condition
        data = data[condition_indices]
        testfold_indices = testfold_indices[condition_mask]

    # If data has no labels (e.g. X-learner), 
    # load the prediction results from weights_dir and get labels from there
    if data[0].y is None:
        pre_results = pd.read_csv(os.path.join(weights_dir, 'prediction_results.csv'))
        labels = dict(zip(pre_results['subject_id']+1, pre_results['label']))
        addlabel_tfm = AddLabel(labels)
        data.transform = T.Compose([*data.transform.transforms, addlabel_tfm])

    # Compute fold-wise performance of each model --------------------------
    if len(vgaes) > 1:
        fold_performance = get_fold_performance(weights_dir)
        fold_performance.to_csv(os.path.join(output_dir, 'fold_performance.csv'), index=False)
        ex.add_artifact(os.path.join(output_dir, 'fold_performance.csv'))

    # Get number of nodes per graph
    num_nodes = data[0].num_nodes
    
    # Compute node importance scores (per fold) -----------------------------------------
    logger.info("Computing node importance scores...")
    node_scores = compute_node_importance_scores(data, vgaes, mlps, testfold_indices, device, num_nodes)
    # node_scores shape: [num_folds, num_nodes]
    
    # Get atlas labels for brain regions
    atlas_specs = get_atlas(data.atlas)
    decoded_labels = [label.decode('utf-8') if isinstance(label, bytes) else label for label in atlas_specs['labels']]
    
    # Create DataFrame with folds as rows and brain regions as columns
    node_importance_df = pd.DataFrame(node_scores, columns=decoded_labels)
    node_importance_df.index.name = 'fold'
    node_importance_df.to_csv(os.path.join(output_dir, 'node_importance_scores.csv'), index=True)
    logger.info(f"Saved node importance scores to {output_dir}")
    
    # Compute mean importance scores across folds
    mean_node_scores = node_scores.mean(axis=0)  # [num_nodes]
    
    # Compute z-scores: average across folds, then z-score across nodes
    z_scored_scores = (mean_node_scores - mean_node_scores.mean()) / mean_node_scores.std()
    
    # Create brain surface plots --------------------------------------------------------
    # Plot 1: Mean importance scores across folds
    atlas = data.atlas
    vrange_mean = (np.percentile(mean_node_scores, 5), np.percentile(mean_node_scores, 95))
    plot_brain_surface(mean_node_scores, 
                       atlas=atlas, 
                       threshold=None, 
                       cmap='mako', 
                       vrange=vrange_mean, 
                       title='Mean node importance scores (across folds)', 
                       save_path=os.path.join(output_dir, 'mean_node_importance.png'))
    ex.add_artifact(os.path.join(output_dir, 'mean_node_importance.png'))
    
    # Plot 2: Z-scored importance scores
    vrange_z = (np.percentile(z_scored_scores, 5), np.percentile(z_scored_scores, 95))
    plot_brain_surface(z_scored_scores, 
                       atlas=atlas, 
                       threshold=None, 
                       cmap='mako', 
                       vrange=vrange_z, 
                       title='Z-scored node importance scores', 
                       save_path=os.path.join(output_dir, 'zscored_node_importance.png'))
    ex.add_artifact(os.path.join(output_dir, 'zscored_node_importance.png'))
    
    if not verbose:
        plt.close('all')
    
    logger.info("Node importance analysis complete.")
