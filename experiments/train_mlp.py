'''
Trains only an MLP on all features in the dataset.

For example:
For training only on graph_attrs, pass node_attrs, edge_attrs = [], [] inside data_ingredient.
For training on flattened neuroimaging features, pass graph_attrs = [] inside data_ingredient.

Author: Hanna M. Tolle
Date: 2025-01-04
License: BSD 3-Clause
'''

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.data_ingredient import * 
from experiments.ingredients.mlp_ingredient import * 

import os
import torch
import torch.nn
from tqdm import tqdm
from time import time
import copy
import pandas as pd
import logging
import matplotlib.pyplot as plt

from utils.files import project_root, add_project_root
from utils.helpers import fix_random_seed, get_logger, save_test_indices
from utils.plotting import plot_loss_curves, true_vs_pred_scatter


# Create experiment and logger -------------------------------------------------
ex = Experiment('train_mlp', ingredients=[data_ingredient, mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg():
    # Experiment name and ID
    exname = 'train_mlp'
    jobid = 0
    seed = 291
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join(project_root(), 'outputs', 'runs', run_name)

    # Logging and saving
    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    save_weights = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Training configurations
    lr = 0.001            # Learning rate.
    num_epochs = 200      # Number of epochs to train.

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    return config

# Captured functions -----------------------------------------------------------
@ex.capture
def get_optimizer(mlp, lr):
    '''Creates the optimizer for the joint training of VGAE and MLP.'''
    return torch.optim.Adam(list(mlp.parameters()), lr=lr)

@ex.capture
def get_x(batch, device):
    '''
    Get all features from the batch and flatten them.
    All graphs must have the same number of nodes and edges.
    
    Parameters:
    ----------
    batch (torch_geometric.data.Batch): Batch of graphs.
    device (torch.device): Device to use for training.

    Returns:
    -------
    x (torch.Tensor): MLP input 
        (batch_size, 
          num_graph_attrs 
        + num_node_attrs*num_nodes 
        + num_edge_attrs*num_edges).
    '''
    # Get graph, node and edge features and reshape
    batch_size = batch.num_graphs
    graph_features = batch.graph_attr                     # [batch_size, num_graph_attrs]
    node_features = batch.x.view(batch_size, -1)          # [batch_size, num_nodes * num_node_attrs]
    edge_features, _ = get_triu_edges(batch)              # [num_triu_edges*batch_size, num_edge_attr]
    edge_features = edge_features.view(batch_size, -1)    # [batch_size, num_triu_edges * num_edge_attrs]
    
    # Concatenate all features
    x = torch.cat([graph_features, node_features, edge_features], dim=1)
    return x.to(device)

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Unpack configs
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']
    num_folds = _config['dataset']['num_folds']

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []    

    # Get dataloaders
    data = load_data()
    train_loaders, val_loaders, test_loaders, test_indices, _ = get_kfold_dataloaders(data, seed=seed)
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Get the MLP input dimensions (in addition to graph attributes)
    num_nodes = _config['dataset']['num_nodes']
    num_edges = get_num_triu_edges(num_nodes)
    num_node_attrs = get_num_attr('node')
    num_edge_attrs = get_num_attr('edge')
    additional_input_dim = num_node_attrs*num_nodes + num_edge_attrs*num_edges

    # Train-test loop ------------------------------------------------------------
    start_time = time()

    best_outputs = init_outputs_dict(data)
    mlp_train_loss, mlp_test_loss, mlp_val_loss = {}, {}, {}
    best_mlp_states = []

    for k in tqdm(range(num_folds), desc='Folds', disable=not verbose):

        # Initialise losses
        mlp_train_loss[k], mlp_test_loss[k], mlp_val_loss[k] = [], [], []
        
        # Initialise models and optimizer
        mlp = build_mlp(latent_dim=additional_input_dim).to(device)
        optimizer = get_optimizer(mlp)

        # Best validation loss
        best_val_loss = float('inf')
        best_mlp_state = None

        for epoch in range(_config['num_epochs']):
            # Train MLP
            _ = train_mlp(mlp, train_loaders[k], optimizer, device, get_x=get_x)
            
            # Compute training losses
            mlp_train_loss_epoch = test_mlp(mlp, train_loaders[k], device, get_x=get_x)
            mlp_train_loss[k].append(mlp_train_loss_epoch)
            
            # Test MLP
            mlp_test_loss_epoch = test_mlp(mlp, test_loaders[k], device, get_x=get_x)
            mlp_test_loss[k].append(mlp_test_loss_epoch)

            # Log training and test losses
            ex.log_scalar(f'training/fold{k}/epoch/mlp_loss', mlp_train_loss_epoch)
            ex.log_scalar(f'test/fold{k}/epoch/mlp_loss', mlp_test_loss_epoch)

            # Validate models, if applicable
            if len(val_loaders) > 0:
                mlp_val_loss_epoch = test_mlp(mlp, val_loaders[k], device, get_x=get_x)
                mlp_val_loss[k].append(mlp_val_loss_epoch)
                ex.log_scalar(f'validation/fold{k}/epoch/mlp_loss', mlp_val_loss_epoch)
                
                # Save the best model if validation loss is at its minimum
                if mlp_val_loss_epoch < best_val_loss:
                    best_val_loss = mlp_val_loss_epoch
                    best_mlp_state = copy.deepcopy(mlp.state_dict()) 

        # Load best model of this fold 
        if best_mlp_state is not None:
            mlp.load_state_dict(best_mlp_state)

        # Save model weights
        if save_weights:
            torch.save(mlp.state_dict(), os.path.join(output_dir, f'k{k}_mlp_weights.pth'))

        # Keep a list of model states
        best_mlp_states.append(copy.deepcopy(mlp.state_dict()))

        # Save the test predictions of the best model
        outputs = get_mlp_outputs_nograd(mlp, test_loaders[k], device, get_x=get_x)
        update_best_outputs(best_outputs, outputs)

    # Print training time
    end_time = time()
    logger.info(f"Joint training completed after {(end_time-start_time)/60:.2f} minutes.")

    # Save ouputs as csv file
    best_outputs = pd.DataFrame(best_outputs)
    data_file = os.path.join(output_dir, 'prediction_results.csv')
    best_outputs.to_csv(data_file, index=False)

    # Save test fold assignments
    if save_weights:
        test_indices_file = save_test_indices(test_indices, output_dir)

    # Save final prediction results
    r, p, mae, mae_std = evaluate_regression(best_outputs)
    results = {'seed': seed, 'r': r, 'p': p, 'mae': mae, 'mae_std': mae_std}
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(output_dir, 'final_metrics.csv'), index=False)

    # Log final metrics
    for k, v in results.items():
        ex.log_scalar(f'final_prediction/{k}', v)
    logger.info(f"Final results: r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}.")

    # Plot results ----------------------------------------------------------------
    # Loss curves
    plot_loss_curves(mlp_train_loss, mlp_test_loss, mlp_val_loss, save_path=os.path.join(output_dir, 'mlp_loss_curves.png'))
    image_files.append(os.path.join(output_dir, 'mlp_loss_curves.png'))

    # True vs predicted scatter
    title = f'r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
    true_vs_pred_scatter(best_outputs, title=title, save_path=os.path.join(output_dir, 'true_vs_predicted.png'))
    image_files.append(os.path.join(output_dir, 'true_vs_predicted.png'))

    # Log images
    for img in image_files:
        if img is not None:
            ex.add_artifact(filename=img)

    # Close all plots if not verbose
    if not verbose:
        plt.close()
